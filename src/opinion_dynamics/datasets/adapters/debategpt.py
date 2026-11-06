"""DebateGPT adapter: phased debates with human–human and human–AI configurations."""

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
from opinion_dynamics.perspectives.visibility import FullHistoryPolicy


class DebateGPTAdapter(BaseDatasetAdapter):
    """Adapter for DebateGPT unified records.

    Episodes are generated only for tracked participants.
    Within-phase Pro-before-Con ordering is a dataset convention, not verified chronology.
    """

    dataset_name = "DebateGPT"
    default_visibility_policy = FullHistoryPolicy()

    ASSUMPTIONS = [
        "Full debate transcript is visible to each tracked participant.",
        "Within each phase, the Pro turn is listed before the Con turn by convention; "
        "this is not verified chronological ordering (phases were written in parallel).",
    ]

    LIMITATIONS = [
        "Initial/final participant-authored reasoning is unavailable in the source data.",
        "Human-Human personalized treatments track only P1 (opponent ratings absent).",
    ]

    def get_participants(self, record: dict[str, Any]) -> list[str]:
        return [p["id"] for p in self.get_tracked_parties(record) if p.get("id")]

    def get_initial_stance(self, record: dict[str, Any], participant_id: str) -> StanceBlock:
        party = self.get_party(record, participant_id)
        block = (party.get("opinion") or {}).get("initial") or {}
        return self.normalize_continuous_stance(
            raw=block.get("raw"),
            scale=block.get("scale") or [1, 5],
            context=self.get_initial_context(record, participant_id),
            rating_provenance=block.get("rating_provenance"),
        )

    def get_final_stance(self, record: dict[str, Any], participant_id: str) -> StanceBlock:
        party = self.get_party(record, participant_id)
        block = (party.get("opinion") or {}).get("final") or {}
        return self.normalize_continuous_stance(
            raw=block.get("raw"),
            scale=block.get("scale") or [1, 5],
            context=self.get_final_context(record, participant_id),
            rating_provenance=block.get("rating_provenance"),
        )

    def get_initial_context(
        self, record: dict[str, Any], participant_id: str
    ) -> Optional[ContextBlock]:
        party = self.get_party(record, participant_id)
        block = (party.get("opinion") or {}).get("initial") or {}
        if block.get("statement"):
            return self.build_context(
                text=block["statement"],
                speaker_id=participant_id,
                agent_id=participant_id,
                context_type=ContextType.PARTICIPANT_JUSTIFICATION.value,
                source=ContextSource.HUMAN_AUTHORED.value,
                speaker_kind=party.get("kind"),
            )
        return self.build_context(
            text=None,
            speaker_id=participant_id,
            agent_id=participant_id,
            context_type=ContextType.RATING_ONLY.value
            if block.get("raw") is not None
            else ContextType.MISSING.value,
            source=ContextSource.MISSING.value,
            speaker_kind=party.get("kind"),
        )

    def get_final_context(
        self, record: dict[str, Any], participant_id: str
    ) -> Optional[ContextBlock]:
        party = self.get_party(record, participant_id)
        block = (party.get("opinion") or {}).get("final") or {}
        if block.get("statement"):
            return self.build_context(
                text=block["statement"],
                speaker_id=participant_id,
                agent_id=participant_id,
                context_type=ContextType.PARTICIPANT_AUTHORED.value,
                source=ContextSource.HUMAN_AUTHORED.value,
                speaker_kind=party.get("kind"),
            )
        return self.build_context(
            text=None,
            speaker_id=participant_id,
            agent_id=participant_id,
            context_type=ContextType.RATING_ONLY.value
            if block.get("raw") is not None
            else ContextType.MISSING.value,
            source=ContextSource.MISSING.value,
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
        limitations = list(self.LIMITATIONS)
        treatment = (record.get("extra") or {}).get("treatment")
        if treatment:
            limitations.append(f"Treatment configuration: {treatment}.")
        return list(self.ASSUMPTIONS), limitations
