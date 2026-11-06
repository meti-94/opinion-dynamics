"""Unit and integration tests for all six dataset adapters."""

from __future__ import annotations

from opinion_dynamics.datasets.adapters.anthropic_persuasion import AnthropicPersuasionAdapter
from opinion_dynamics.datasets.adapters.cmv import CMVAdapter, strip_delta_annotations
from opinion_dynamics.datasets.adapters.conspiracyai import ConspiracyAIAdapter
from opinion_dynamics.datasets.adapters.debate import DebateAdapter
from opinion_dynamics.datasets.adapters.debategpt import DebateGPTAdapter
from opinion_dynamics.datasets.adapters.persuasiontrace import PersuasionTraceAdapter


def _party(pid: str, initial: float, final: float) -> dict:
    def block(raw: float) -> dict:
        return {
            "rating_100": (raw - 1) / 5 * 100,
            "raw": raw,
            "scale": [1, 6],
            "rating_provenance": "self_reported",
            "statement": "because",
            "statement_provenance": "human_private",
        }
    return {
        "id": pid, "kind": "human", "role": "participant", "tracked": True,
        "opinion": {"initial": block(initial), "final": block(final), "trace": []},
    }


class TestDebateAdapter:
    def test_stances_and_contexts(self, debate_record) -> None:
        adapter = DebateAdapter()
        ep = adapter.build_episode(debate_record, "P1")
        assert ep.initial.stance_type == "continuous"
        assert ep.initial.stance_provenance == "self_reported"
        assert ep.initial.rating_scale == [1, 6]
        assert ep.initial.context is not None
        assert ep.initial.context.type == "participant_justification"
        assert ep.initial.context.speaker_role == "self"
        assert ep.final.context is not None
        assert ep.final.context.speaker_role == "self"
        assert ep.source.split == debate_record["split"]
        assert ep.metadata.structure_type == "multi_party_rounds"

    def test_changed_is_participant_level(self) -> None:
        adapter = DebateAdapter()
        record = {
            "id": "debate:x",
            "split": "train",
            "source": {"dataset": "DEBATE", "original_id": "x", "license": "x", "url": "x"},
            "proposition": {"text": "T", "original_text": "T", "provenance": "dataset"},
            "structure": {"type": "multi_party_rounds", "control": False},
            "parties": [
                _party("P1", 4, 3),
                _party("P3", 1, 1),
            ],
            "turns": [],
            "outcome": {"changed": True, "delta_100": {"P1": -20.0, "P3": 0.0}},
            "extra": {},
        }
        moved = adapter.build_episode(record, "P1")
        still = adapter.build_episode(record, "P3")
        assert moved.outcome.changed is True
        assert moved.outcome.change_100 == -20.0
        assert still.initial.stance == still.final.stance
        assert still.outcome.change_100 == 0.0
        assert still.outcome.changed is False

    def test_self_other_assignment(self, debate_record) -> None:
        adapter = DebateAdapter()
        ep = adapter.build_episode(debate_record, "P3")
        for turn in ep.discussion.visible_turns:
            if turn.speaker_id == "P3":
                assert turn.speaker_role == "self"
            elif turn.speaker_id is not None:
                assert turn.speaker_role == "other"


class TestCMVAdapter:
    def test_successful_delta_proxy(self, cmv_success) -> None:
        adapter = CMVAdapter()
        ep = adapter.build_episode(cmv_success, "OP")
        assert ep.initial.stance == 100.0
        assert ep.final.stance == 0.0
        assert ep.initial.stance_type == "binary_proxy"
        assert ep.initial.stance_provenance == "dataset_outcome_proxy"
        assert ep.outcome.changed is True

    def test_unsuccessful_proxy(self, cmv_fail) -> None:
        adapter = CMVAdapter()
        ep = adapter.build_episode(cmv_fail, "OP")
        assert ep.initial.stance == 100.0
        assert ep.final.stance == 100.0
        assert ep.outcome.changed is False

    def test_unknown_outcome_final_null(self) -> None:
        adapter = CMVAdapter()
        record = {
            "id": "cmv:x",
            "split": "train",
            "source": {"dataset": "CMV", "original_id": "x", "license": "x", "url": "x"},
            "proposition": {"text": "T", "original_text": "CMV: T", "provenance": "op_title"},
            "structure": {"type": "reddit_thread", "control": False},
            "parties": [{
                "id": "OP", "kind": "human", "role": "op", "tracked": True,
                "opinion": {
                    "initial": {"rating_100": None, "raw": None, "scale": None,
                                "statement": "body", "statement_provenance": "human_turn"},
                    "final": {"rating_100": None, "raw": None, "scale": None,
                              "statement": None, "statement_provenance": "missing"},
                    "trace": [],
                },
            }],
            "turns": [{"idx": 0, "round": None, "channel": "thread", "from": "OP",
                       "to": [], "reply_to": None, "text": "body", "annotations": {}}],
            "outcome": {"changed": None},
            "extra": {},
        }
        ep = adapter.build_episode(record, "OP")
        assert ep.initial.stance == 100.0
        assert ep.final.stance is None

    def test_delta_stripped_and_self_attribution(self, cmv_success) -> None:
        adapter = CMVAdapter()
        ep = adapter.build_episode(cmv_success, "OP")
        assert ep.initial.context is not None
        assert ep.initial.context.speaker_role == "self"
        assert ep.initial.context.text
        # Discussion preserves reply structure.
        assert any(t.reply_to is not None for t in ep.discussion.visible_turns) or len(ep.discussion.visible_turns) >= 1
        award_idxs = {
            t["idx"]
            for t in cmv_success["turns"]
            if (t.get("annotations") or {}).get("delta_marker")
            and not (t.get("annotations") or {}).get("is_op_post")
        }
        assert award_idxs
        assert award_idxs.isdisjoint({t.idx for t in ep.discussion.visible_turns})
        assert any("delta award" in e["reason"] for e in ep.discussion.excluded_turns)
        for turn in ep.discussion.visible_turns:
            assert not turn.text.startswith("∆")
            assert not turn.text.startswith("Δ")
        # The award comment is not kept as final context.
        assert ep.final.context is None or not ep.final.context.text

    def test_strip_delta_helper(self) -> None:
        assert strip_delta_annotations("∆. Thanks").startswith("Thanks")
        assert "argument" in strip_delta_annotations("Δ argument kept")
        assert "keep" in strip_delta_annotations("!delta keep this")


class TestPersuasionTraceAdapter:
    def test_first_last_ratings(self, persuasion_main) -> None:
        adapter = PersuasionTraceAdapter()
        party = next(p for p in persuasion_main["parties"] if p["tracked"])
        ep = adapter.build_episode(persuasion_main, party["id"])
        init_raw = party["opinion"]["initial"]["raw"]
        final_raw = party["opinion"]["final"]["raw"]
        assert abs(ep.initial.raw_rating - init_raw) < 1e-9
        assert abs(ep.final.raw_rating - final_raw) < 1e-9
        # Intermediate ratings not represented as separate episodes.
        assert len(adapter.build_episodes(persuasion_main)) == 1
        # Trace preserved only in extra.
        assert ep.extra.get("opinion_trace") is not None
        assert len(ep.extra["opinion_trace"]) >= 1

    def test_control_identified(self, persuasion_control) -> None:
        adapter = PersuasionTraceAdapter()
        assert adapter.is_control(persuasion_control)
        ep = adapter.build_episodes(persuasion_control)[0]
        assert ep.metadata.control is True
        assert any("CONTROL" in lim for lim in ep.metadata.limitations)

    def test_speaker_identities(self, persuasion_main) -> None:
        adapter = PersuasionTraceAdapter()
        ep = adapter.build_episodes(persuasion_main)[0]
        speakers = {t.speaker_id for t in ep.discussion.visible_turns}
        assert "T" in speakers or ep.agent.participant_id in speakers
        assert "AI" in speakers
        assert all(t.speaker_role in {"self", "other", "unknown"} for t in ep.discussion.visible_turns)


class TestConspiracyAIAdapter:
    def test_opening_context_and_ratings(self, conspiracy_record) -> None:
        adapter = ConspiracyAIAdapter()
        ep = adapter.build_episodes(conspiracy_record)[0]
        assert ep.agent.participant_id == "P1"
        assert ep.initial.context is not None
        assert ep.initial.context.text
        assert ep.initial.context.speaker_role == "self"
        assert ep.proposition.provenance == "rewritten"
        assert 0 <= ep.initial.stance <= 100
        assert 0 <= ep.final.stance <= 100
        # Final context is not fabricated from dialogue turns.
        assert ep.final.context is None or ep.final.context.text is None
        # Turn 0 remains in discussion (duplication by design).
        assert ep.discussion.visible_turns[0].speaker_id == "P1"


class TestAnthropicPersuasionAdapter:
    def test_ai_argument_attribution(self, anthropic_ai) -> None:
        adapter = AnthropicPersuasionAdapter()
        ep = adapter.build_episodes(anthropic_ai)[0]
        assert ep.agent.participant_id == "T"
        assert ep.initial.context is None or ep.initial.context.text is None
        assert len(ep.discussion.visible_turns) == 1
        assert ep.discussion.visible_turns[0].speaker_id == "AI"
        assert ep.discussion.visible_turns[0].speaker_role == "other"
        assert ep.discussion.visible_turns[0].speaker_kind == "llm"
        assert ep.initial.rating_scale == [1, 7]

    def test_human_argument_attribution(self, anthropic_human) -> None:
        adapter = AnthropicPersuasionAdapter()
        ep = adapter.build_episodes(anthropic_human)[0]
        assert ep.discussion.visible_turns[0].speaker_id == "P1"
        assert ep.discussion.visible_turns[0].speaker_kind == "human"
        assert ep.discussion.visible_turns[0].speaker_role == "other"


class TestDebateGPTAdapter:
    def test_human_human(self, debategpt_hh) -> None:
        adapter = DebateGPTAdapter()
        episodes = adapter.build_episodes(debategpt_hh)
        assert len(episodes) == 2
        assert {e.agent.participant_id for e in episodes} == {"P1", "P2"}
        for ep in episodes:
            assert ep.initial.rating_scale == [1, 5]
            assert any("convention" in a.lower() or "chronolog" in a.lower()
                       for a in ep.metadata.assumptions)
            # Phase structure preserved via annotations.
            phases = [t.annotations.get("phase") for t in ep.discussion.visible_turns]
            assert "argument" in phases

    def test_human_ai(self, debategpt_hai) -> None:
        adapter = DebateGPTAdapter()
        episodes = adapter.build_episodes(debategpt_hai)
        assert len(episodes) == 1
        ep = episodes[0]
        assert ep.agent.participant_id == "P1"
        speakers = {t.speaker_id for t in ep.discussion.visible_turns}
        assert "AI" in speakers
        assert "P1" in speakers
        for t in ep.discussion.visible_turns:
            if t.speaker_id == "P1":
                assert t.speaker_role == "self"
            else:
                assert t.speaker_role == "other"
