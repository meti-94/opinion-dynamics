"""Stage-1 initial-stance example tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opinion_dynamics.datasets.dataloader import OpinionDynamicsDataLoader
from opinion_dynamics.datasets.episode import (
    AgentInfo,
    DiscussionBlock,
    Episode,
    EpisodeMetadata,
    OutcomeBlock,
    PropositionInfo,
    SourceInfo,
    StanceBlock,
    VisibleTurn,
)
from opinion_dynamics.datasets.episode import ContextBlock
from opinion_dynamics.tasks.initial_stance import (
    EXAMPLE_FIELDS,
    USER_QUESTION,
    InitialStanceDataLoader,
    build_messages,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "datasets.yaml"


def _episode(
    *,
    dataset: str = "DEBATE",
    original_id: str = "d1",
    participant: str = "P1",
    role: str = "participant",
    proposition: str = "Direct democracy is a good way to govern.",
    context: str | None = "I lean toward agreement.",
    stance: float | None = 60.0,
    stance_type: str = "continuous",
    split: str = "train",
    control: bool = False,
    extra: dict | None = None,
    turns: list | None = None,
) -> Episode:
    return Episode(
        episode_id=f"{dataset.lower()}:{original_id}:{participant}",
        source=SourceInfo(dataset=dataset, original_id=original_id, split=split),
        agent=AgentInfo(participant_id=participant, role=role),
        proposition=PropositionInfo(text=proposition, provenance="dataset"),
        initial=StanceBlock(
            stance=stance,
            raw_rating=None,
            rating_scale=None,
            stance_type=stance_type,
            stance_provenance="self_reported",
            context=None if context is None else ContextBlock(
                text=context,
                speaker_id=participant,
                speaker_role="self",
                type="participant_justification",
                source="human_authored",
            ),
        ),
        discussion=DiscussionBlock(visible_turns=turns or []),
        final=StanceBlock(99.0, None, None, stance_type, "self_reported", context=ContextBlock(
            text="FINAL CONTEXT THAT MUST NOT LEAK",
            speaker_id=participant,
            speaker_role="self",
            type="participant_authored",
            source="human_authored",
        )),
        outcome=OutcomeBlock(changed=True, change_100=39.0),
        metadata=EpisodeMetadata(control=control),
        extra=extra or {},
    )


def test_messages_initial_with_and_without_context() -> None:
    with_text = _episode()
    messages = build_messages(with_text, "initial")
    assert [m["role"] for m in messages] == ["system", "assistant", "user"]
    assert with_text.proposition.text in messages[0]["content"]
    assert messages[1]["content"] == "I lean toward agreement."
    assert messages[-1]["content"] == USER_QUESTION
    assert "FINAL CONTEXT" not in json.dumps(messages)
    assert "DEBATE" not in messages[0]["content"]

    bare = build_messages(_episode(context=None), "initial")
    assert [m["role"] for m in bare] == ["system", "user"]


def test_ensure_alternation_inserts_neutral_user_turn() -> None:
    plain = build_messages(_episode(), "initial", ensure_alternation=False)
    assert plain[1]["role"] == "assistant"
    fixed = build_messages(_episode(), "initial", ensure_alternation=True)
    assert [m["role"] for m in fixed] == ["system", "user", "assistant", "user"]
    assert fixed[1]["content"] == "Go ahead."
    no_context = build_messages(_episode(context=None), "initial", ensure_alternation=True)
    assert [m["role"] for m in no_context] == ["system", "user"]


def test_upto_includes_only_visible_turns_through_index() -> None:
    episode = _episode(turns=[
        VisibleTurn(0, 1, "P2", "other", "human", "public", ["P1"], "OTHER TURN ZERO"),
        VisibleTurn(1, 1, "P1", "self", "human", "public", ["P2"], "SELF TURN ONE"),
        VisibleTurn(2, 1, "P2", "other", "human", "public", ["P1"], "FUTURE TURN TWO"),
    ])
    initial = build_messages(episode, "initial")
    assert "OTHER TURN ZERO" not in json.dumps(initial)
    later = build_messages(episode, 1)
    blob = json.dumps(later)
    assert "[P2] OTHER TURN ZERO" in blob
    assert "SELF TURN ONE" in blob
    assert "FUTURE TURN TWO" not in blob
    assert later[-1]["role"] == "user"


@pytest.fixture(scope="module")
def examples():
    loader = InitialStanceDataLoader.from_config(CONFIG, repo_root=REPO)
    rows = loader.load()
    return loader, rows


def test_no_forbidden_fields_or_discussion_text(examples) -> None:
    loader, rows = examples
    episode_loader = OpinionDynamicsDataLoader.from_config(CONFIG, repo_root=REPO)
    by_id = {ep.episode_id: ep for ep in episode_loader.load()}
    forbidden = {"discussion", "final", "outcome", "visible_turns", "final_context"}
    checked = 0
    for example in rows:
        payload = example.to_dict()
        assert set(payload) == set(EXAMPLE_FIELDS)
        assert forbidden.isdisjoint(payload)
        blob = json.dumps(payload["messages"])
        episode = by_id[example.episode_id]
        assert "FINAL CONTEXT" not in blob
        allowed = f"{example.proposition}\n{example.initial_context or ''}"
        for turn in episode.discussion.visible_turns:
            if len(turn.text) < 80 or turn.text in allowed:
                continue
            assert turn.text not in blob
            checked += 1
            break
    assert checked > 0


def test_no_group_leakage_across_splits(examples) -> None:
    _, rows = examples
    splits: dict[tuple[str, str], set] = {}
    for example in rows:
        splits.setdefault((example.dataset, example.group_id), set()).add(example.split)
    assert all(len(found) == 1 for found in splits.values())
    debate = [ex for ex in rows if ex.dataset == "DEBATE"]
    groups: dict[str, set[str]] = {}
    for example in debate:
        groups.setdefault(example.group_id, set()).add(example.episode_id.split(":")[-1])
    assert any(len(members) > 1 for members in groups.values())


def test_counts_and_target_histograms(examples, capsys) -> None:
    loader, rows = examples
    summary = loader.summary(rows)
    print(json.dumps({k: summary[k] for k in summary if k != "target_histograms"}, indent=2))
    for name, hist in summary["target_histograms"].items():
        top = sorted(hist.items(), key=lambda item: -item[1])[:8]
        print(name, "n=", summary["per_dataset"][name], "hist", top)
    assert summary["per_dataset"]["ConspiracyAI"] > 0
    assert summary["per_dataset"]["DEBATE"] > 0
    assert summary["per_dataset"]["PersuasionTrace"] > 0
    assert summary["per_dataset"]["AnthropicPersuasion"] > 0
    assert summary["per_dataset"]["CMV"] > 0
    assert summary["per_dataset"]["DebateGPT"] > 0
    cmv = [ex for ex in rows if ex.dataset == "CMV"]
    assert {ex.target for ex in cmv} == {100.0}
    assert {ex.target_type for ex in cmv} == {"binary_proxy"}
    assert len({ex.group_id for ex in cmv}) == len(cmv)
    steps = {ex.dataset: ex.scale_step for ex in rows}
    assert steps["DEBATE"] == 20
    assert steps["DebateGPT"] == 25
    assert steps["AnthropicPersuasion"] == 16.7
    assert steps["ConspiracyAI"] == 1
    captured = capsys.readouterr()
    assert "ConspiracyAI" in captured.out


def test_printed_examples_and_message_contract(examples, capsys) -> None:
    _, rows = examples
    picked = []
    for dataset in ("ConspiracyAI", "DEBATE", "CMV", "AnthropicPersuasion", "DebateGPT"):
        picked.append(next(ex for ex in rows if ex.dataset == dataset and (
            ex.initial_context is None if dataset in {"AnthropicPersuasion", "DebateGPT"} else ex.initial_context
        )))
    assert any(ex.initial_context is None for ex in picked)
    for example in picked:
        print("=" * 20, example.dataset, example.episode_id)
        print(json.dumps(example.messages, ensure_ascii=False, indent=2)[:1500])
        assert example.messages[-1]["role"] == "user"
        assert example.proposition in example.messages[0]["content"]
        assert example.dataset not in example.messages[0]["content"]
        assistants = [m for m in example.messages if m["role"] == "assistant"]
        if example.initial_context is None:
            assert assistants == []
        else:
            assert assistants[0]["content"] == example.initial_context
    # The AI argument is a discussion turn and must not be in the prompt.
    anthropic = next(ex for ex in rows if ex.dataset == "AnthropicPersuasion")
    assert anthropic.initial_context is None
    assert len(anthropic.messages) == 2


def test_drop_context_ablation_and_persuasion_control() -> None:
    loader = InitialStanceDataLoader.from_config(
        CONFIG,
        repo_root=REPO,
        datasets=["PersuasionTrace", "DEBATE"],
        max_examples=None,
        drop_context=True,
    )
    rows = loader.load()
    debate = [ex for ex in rows if ex.dataset == "DEBATE"]
    assert debate
    assert all(ex.initial_context is None for ex in debate)
    assert all(m["role"] != "assistant" for ex in debate for m in ex.messages)
    controls = [ex for ex in rows if ex.dataset == "PersuasionTrace"]
    assert controls
    assert all(ex.initial_context is None for ex in controls)
    assert all("Dogs are better than cats." not in ex.proposition for ex in controls)
