"""DEBATE adapter: multi-party rounds with public tweets and private dyadic chat."""

from __future__ import annotations

from typing import Any, Optional

from opinion_dynamics.datasets.adapters.base import BaseDatasetAdapter
from opinion_dynamics.datasets.episode import (
    ContextBlock,
    ContextSource,
    ContextType,
    StanceBlock,
    VisibleTurn,
)
from opinion_dynamics.perspectives.visibility import PublicAndPrivatePolicy


class DebateAdapter(BaseDatasetAdapter):
    """Adapter for the DEBATE (Chuang et al.) unified JSONL records.

    Visibility policy (confirmed):
    - Public tweets are visible to all four participants.
    - Private messages are visible only to the sender and intended recipients.
    - Private turns with missing speaker and empty recipients are excluded.
    """

    dataset_name = "DEBATE"
    default_visibility_policy = PublicAndPrivatePolicy()

    ASSUMPTIONS = [
        "Public-channel tweets are treated as visible to all participants in the record, "
        "even when addressed to a single round partner.",
        "Private justifications accompanying ratings are never visible to other participants.",
    ]

    def get_participants(self, record: dict[str, Any]) -> list[str]:
        return [p["id"] for p in self.get_tracked_parties(record) if p.get("id")]

    def get_initial_stance(self, record: dict[str, Any], participant_id: str) -> StanceBlock:
        party = self.get_party(record, participant_id)
        block = (party.get("opinion") or {}).get("initial") or {}
        context = self.get_initial_context(record, participant_id)
        return self.normalize_continuous_stance(
            raw=block.get("raw"),
            scale=block.get("scale") or [1, 6],
            context=context,
            rating_provenance=block.get("rating_provenance"),
        )

    def get_final_stance(self, record: dict[str, Any], participant_id: str) -> StanceBlock:
        party = self.get_party(record, participant_id)
        block = (party.get("opinion") or {}).get("final") or {}
        context = self.get_final_context(record, participant_id)
        return self.normalize_continuous_stance(
            raw=block.get("raw"),
            scale=block.get("scale") or [1, 6],
            context=context,
            rating_provenance=block.get("rating_provenance"),
        )

    def get_initial_context(
        self, record: dict[str, Any], participant_id: str
    ) -> Optional[ContextBlock]:
        party = self.get_party(record, participant_id)
        block = (party.get("opinion") or {}).get("initial") or {}
        statement = block.get("statement")
        if not statement:
            return self.build_context(
                text=None,
                speaker_id=participant_id,
                agent_id=participant_id,
                context_type=ContextType.MISSING.value,
                source=ContextSource.MISSING.value,
                speaker_kind=party.get("kind"),
            )
        return self.build_context(
            text=statement,
            speaker_id=participant_id,
            agent_id=participant_id,
            context_type=ContextType.PARTICIPANT_JUSTIFICATION.value,
            source=ContextSource.HUMAN_AUTHORED.value,
            speaker_kind=party.get("kind"),
        )

    def get_final_context(
        self, record: dict[str, Any], participant_id: str
    ) -> Optional[ContextBlock]:
        party = self.get_party(record, participant_id)
        block = (party.get("opinion") or {}).get("final") or {}
        statement = block.get("statement")
        if not statement:
            return self.build_context(
                text=None,
                speaker_id=participant_id,
                agent_id=participant_id,
                context_type=ContextType.MISSING.value,
                source=ContextSource.MISSING.value,
                speaker_kind=party.get("kind"),
            )
        return self.build_context(
            text=statement,
            speaker_id=participant_id,
            agent_id=participant_id,
            context_type=ContextType.PARTICIPANT_AUTHORED.value,
            source=ContextSource.HUMAN_AUTHORED.value,
            speaker_kind=party.get("kind"),
        )

    def get_visible_history(
        self, record: dict[str, Any], participant_id: str
    ) -> tuple[list[VisibleTurn], list[dict[str, Any]]]:
        return self.turns_to_visible(
            record,
            participant_id,
            self.default_visibility_policy,
        )

    def perspective_notes(
        self, record: dict[str, Any], participant_id: str
    ) -> tuple[list[str], list[str]]:
        assumptions = list(self.ASSUMPTIONS)
        limitations: list[str] = []
        null_speakers = sum(1 for t in (record.get("turns") or []) if t.get("from") is None)
        if null_speakers:
            limitations.append(
                f"{null_speakers} turn(s) have missing speaker identity; "
                "assigned speaker_role=unknown where included."
            )
        empty_private = sum(
            1
            for t in (record.get("turns") or [])
            if (t.get("channel") or "").lower() == "private"
            and t.get("from") is None
            and not (t.get("to") or [])
        )
        if empty_private:
            limitations.append(
                f"{empty_private} private turn(s) lack sender and recipients and are "
                "excluded from all participant perspectives."
            )
        return assumptions, limitations
