"""Chat formatter tests."""

from __future__ import annotations

import pytest

from opinion_dynamics.datasets.adapters.debate import DebateAdapter
from opinion_dynamics.datasets.adapters.cmv import CMVAdapter
from opinion_dynamics.formatting.response_generation import ResponseGenerationFormatter
from opinion_dynamics.formatting.stance_prediction import StancePredictionFormatter


def test_stance_prediction_no_target_leak(debate_record) -> None:
    adapter = DebateAdapter()
    ep = adapter.build_episode(debate_record, "P1")
    messages = StancePredictionFormatter().format(ep)
    assert messages[0]["role"] == "system"
    assert "P1" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    # Final stance must not appear as assistant completion by default.
    assert all(m["role"] != "assistant" for m in messages)
    # Historical speakers keep original IDs.
    assert "P1" in messages[1]["content"]
    user = messages[1]["content"]
    assert "PROPOSITION:" in user
    assert "DISCUSSION:" in user
    assert str(ep.final.stance) not in user or ep.initial.stance == ep.final.stance


def test_stance_prediction_self_other_tags(debate_record) -> None:
    adapter = DebateAdapter()
    ep = adapter.build_episode(debate_record, "P2")
    messages = ep.to_stance_prediction_messages()
    content = messages[1]["content"]
    assert "[SELF: P2]" in content or "SELF refers to P2" in messages[0]["content"]
    # Other participants retain IDs.
    for other in ("P1", "P3", "P4"):
        if any(t.speaker_id == other for t in ep.discussion.visible_turns):
            assert f"[OTHER: {other}]" in content


def test_response_generation_no_future_or_target(debate_record) -> None:
    adapter = DebateAdapter()
    ep = adapter.build_episode(debate_record, "P1")
    fmt = ResponseGenerationFormatter()
    candidates = fmt.candidate_target_indices(ep)
    assert candidates, "P1 should have visible self turns"
    target_idx = candidates[len(candidates) // 2]
    messages, target_text = fmt.format(ep, target_turn_idx=target_idx)
    assert target_text is not None
    user = messages[1]["content"]
    # Target text must not appear in the prompt history.
    assert target_text not in user
    # No future turns after target.
    target_turn = next(t for t in ep.discussion.visible_turns if t.idx == target_idx)
    for t in ep.discussion.visible_turns:
        if t.idx > target_idx:
            assert t.text not in user
    assert target_turn.speaker_id == "P1"


def test_response_generation_rejects_other_speaker(debate_record) -> None:
    adapter = DebateAdapter()
    ep = adapter.build_episode(debate_record, "P1")
    other = next(t for t in ep.discussion.visible_turns if t.speaker_id != "P1")
    with pytest.raises(ValueError, match="not the simulated participant"):
        ResponseGenerationFormatter().format(ep, target_turn_idx=other.idx)


def test_private_messages_not_in_unauthorized_prompt(debate_record) -> None:
    adapter = DebateAdapter()
    # Find a private P2-P3 exchange.
    dyad = [
        t for t in debate_record["turns"]
        if t.get("channel") == "private"
        and t.get("from") in {"P2", "P3"}
        and set(t.get("to") or []) <= {"P2", "P3"}
    ]
    if not dyad:
        pytest.skip("no P2-P3 private dyad")
    secret = dyad[0]["text"]
    ep_p1 = adapter.build_episode(debate_record, "P1")
    messages = StancePredictionFormatter().format(ep_p1)
    assert secret not in messages[1]["content"]


def test_cmv_formatter_uses_binary_proxy(cmv_success) -> None:
    adapter = CMVAdapter()
    ep = adapter.build_episode(cmv_success, "OP")
    messages = StancePredictionFormatter().format(ep)
    assert "binary_proxy" in messages[1]["content"] or "INITIAL STANCE:\n100" in messages[1]["content"]
