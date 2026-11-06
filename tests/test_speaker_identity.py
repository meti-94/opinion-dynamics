"""SELF/OTHER speaker identity tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opinion_dynamics.datasets.adapters.debate import DebateAdapter
from opinion_dynamics.perspectives.speaker_identity import speaker_role_for

REPO_ROOT = Path(__file__).resolve().parents[1]
DEBATE_PATH = REPO_ROOT / "data" / "unified" / "debate.jsonl"


def test_speaker_role_assignment() -> None:
    assert speaker_role_for("P1", "P1") == "self"
    assert speaker_role_for("P1", "P2") == "other"
    assert speaker_role_for("P1", None) == "unknown"
    assert speaker_role_for("P1", "") == "unknown"


@pytest.mark.skipif(not DEBATE_PATH.exists(), reason="debate.jsonl missing")
def test_debate_self_other_flip_across_perspectives() -> None:
    adapter = DebateAdapter()
    with DEBATE_PATH.open() as fh:
        record = json.loads(next(fh))
        while any(t.get("from") is None for t in record["turns"]):
            record = json.loads(next(fh))

    ep_p1 = adapter.build_episode(record, "P1")
    ep_p2 = adapter.build_episode(record, "P2")

    p1_msgs_for_p1 = [t for t in ep_p1.discussion.visible_turns if t.speaker_id == "P1"]
    p1_msgs_for_p2 = [t for t in ep_p2.discussion.visible_turns if t.speaker_id == "P1"]
    assert p1_msgs_for_p1, "P1 should have visible self messages"
    assert all(t.speaker_role == "self" for t in p1_msgs_for_p1)
    assert p1_msgs_for_p2, "P2 should see some of P1's public messages"
    assert all(t.speaker_role == "other" for t in p1_msgs_for_p2)

    # Original IDs preserved; not collapsed into a single OTHER identity.
    other_ids_p1 = {
        t.speaker_id
        for t in ep_p1.discussion.visible_turns
        if t.speaker_role == "other"
    }
    assert "P2" in other_ids_p1 or "P3" in other_ids_p1 or "P4" in other_ids_p1
