"""Visibility policy tests, including DEBATE private/public rules on real records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opinion_dynamics.datasets.adapters.debate import DebateAdapter
from opinion_dynamics.perspectives.visibility import (
    PublicAndPrivatePolicy,
    filter_visible_turns,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEBATE_PATH = REPO_ROOT / "data" / "unified" / "debate.jsonl"


def _load_first_usable_debate_record() -> dict:
    with DEBATE_PATH.open() as fh:
        for line in fh:
            record = json.loads(line)
            # Prefer a record with intact speaker IDs.
            if all(t.get("from") for t in record["turns"]):
                return record
    raise RuntimeError("No usable DEBATE record found")


def _private_pairs(record: dict) -> set[tuple[str, str]]:
    pairs = set()
    for turn in record["turns"]:
        if turn.get("channel") != "private":
            continue
        speaker = turn.get("from")
        recipients = turn.get("to") or []
        if speaker and recipients:
            for r in recipients:
                pairs.add(tuple(sorted([speaker, r])))
    return pairs


@pytest.mark.skipif(not DEBATE_PATH.exists(), reason="debate.jsonl missing")
class TestDebateVisibilityRealData:
    def setup_method(self) -> None:
        self.adapter = DebateAdapter()
        self.record = _load_first_usable_debate_record()

    def test_private_dyad_visible_only_to_participants(self) -> None:
        # Find a private conversation between two parties that excludes the others.
        pairs = _private_pairs(self.record)
        assert pairs, "expected at least one private dyad"
        pair = next(iter(pairs))
        a, b = pair
        others = [p for p in ("P1", "P2", "P3", "P4") if p not in pair]
        assert len(others) == 2

        # Collect private turns strictly between a and b.
        dyad_idxs = {
            t["idx"]
            for t in self.record["turns"]
            if t.get("channel") == "private"
            and t.get("from") in pair
            and set(t.get("to") or []).issubset(pair)
            and set(t.get("to") or [])
        }
        assert dyad_idxs, "expected private dyad turns"

        for pid in (a, b):
            visible, _ = self.adapter.get_visible_history(self.record, pid)
            visible_idxs = {t.idx for t in visible}
            assert dyad_idxs.issubset(visible_idxs), f"{pid} should see private dyad {pair}"

        for pid in others:
            visible, excluded = self.adapter.get_visible_history(self.record, pid)
            visible_idxs = {t.idx for t in visible}
            assert dyad_idxs.isdisjoint(visible_idxs), f"{pid} must not see private dyad {pair}"
            excluded_idxs = {e["idx"] for e in excluded}
            assert dyad_idxs.issubset(excluded_idxs)

    def test_p2_p3_private_hidden_from_p1_and_p4(self) -> None:
        # Synthetic-style assertion over the real record's P2-P3 private turns if present.
        dyad = {
            t["idx"]
            for t in self.record["turns"]
            if t.get("channel") == "private"
            and t.get("from") in {"P2", "P3"}
            and set(t.get("to") or []) <= {"P2", "P3"}
            and set(t.get("to") or [])
        }
        if not dyad:
            pytest.skip("this record has no P2-P3 private dyad")

        for pid in ("P2", "P3"):
            visible_idxs = {t.idx for t in self.adapter.get_visible_history(self.record, pid)[0]}
            assert dyad.issubset(visible_idxs)
        for pid in ("P1", "P4"):
            visible_idxs = {t.idx for t in self.adapter.get_visible_history(self.record, pid)[0]}
            assert dyad.isdisjoint(visible_idxs)

    def test_public_messages_visible_to_all_participants(self) -> None:
        public_idxs = {
            t["idx"] for t in self.record["turns"] if t.get("channel") == "public"
        }
        assert public_idxs
        for pid in self.adapter.get_participants(self.record):
            visible_idxs = {t.idx for t in self.adapter.get_visible_history(self.record, pid)[0]}
            assert public_idxs.issubset(visible_idxs)

    def test_four_participant_episodes(self) -> None:
        episodes = self.adapter.build_episodes(self.record)
        assert len(episodes) == 4
        assert {e.agent.participant_id for e in episodes} == {"P1", "P2", "P3", "P4"}


class TestPublicAndPrivatePolicyUnit:
    def test_private_between_p2_p3(self) -> None:
        policy = PublicAndPrivatePolicy()
        turn = {
            "idx": 1,
            "channel": "private",
            "from": "P2",
            "to": ["P3"],
            "text": "secret",
        }
        assert policy.is_visible(turn, "P2").visible
        assert policy.is_visible(turn, "P3").visible
        assert not policy.is_visible(turn, "P1").visible
        assert not policy.is_visible(turn, "P4").visible

    def test_public_visible_to_everyone(self) -> None:
        policy = PublicAndPrivatePolicy()
        turn = {"idx": 0, "channel": "public", "from": "P1", "to": ["P4"], "text": "tweet"}
        for pid in ("P1", "P2", "P3", "P4"):
            assert policy.is_visible(turn, pid).visible

    def test_corrupt_private_excluded(self) -> None:
        policy = PublicAndPrivatePolicy()
        turn = {"idx": 0, "channel": "private", "from": None, "to": [], "text": "?"}
        for pid in ("P1", "P2", "P3", "P4"):
            assert not policy.is_visible(turn, pid).visible

    def test_filter_preserves_order(self) -> None:
        policy = PublicAndPrivatePolicy()
        turns = [
            {"idx": 0, "channel": "public", "from": "P1", "to": ["P2"], "text": "a"},
            {"idx": 1, "channel": "private", "from": "P2", "to": ["P3"], "text": "b"},
            {"idx": 2, "channel": "private", "from": "P1", "to": ["P2"], "text": "c"},
        ]
        visible, excluded = filter_visible_turns(turns, "P1", policy)
        assert [t["idx"] for t in visible] == [0, 2]
        assert [e["idx"] for e in excluded] == [1]
