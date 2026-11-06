"""ConspiracyAI adapter: human–GPT-4 dialogue with pre/post conspiracy belief ratings."""

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


class ConspiracyAIAdapter(BaseDatasetAdapter):
    """Adapter for ConspiracyAI unified records.

    Initial context is the participant's opening message (also present as turn 0).
    Final context is null unless an actual post-discussion authored statement exists
    (none do in the current unified file).
    """

    dataset_name = "ConspiracyAI"
    default_visibility_policy = FullHistoryPolicy()

    ASSUMPTIONS = [
        "Full human–AI dialogue is visible to the tracked participant.",
        "Initial context duplicates turn 0 text by design (opening message).",
        "Proposition provenance 'rewritten' is preserved from the unified record.",
    ]

    def get_participants(self, record: dict[str, Any]) -> list[str]:
        return [p["id"] for p in self.get_tracked_parties(record) if p.get("id")]

    def get_initial_stance(self, record: dict[str, Any], participant_id: str) -> StanceBlock:
        party = self.get_party(record, participant_id)
        block = (party.get("opinion") or {}).get("initial") or {}
        return self.normalize_continuous_stance(
            raw=block.get("raw"),
            scale=block.get("scale") or [0, 100],
            context=self.get_initial_context(record, participant_id),
            rating_provenance=block.get("rating_provenance"),
        )

    def get_final_stance(self, record: dict[str, Any], participant_id: str) -> StanceBlock:
        party = self.get_party(record, participant_id)
        block = (party.get("opinion") or {}).get("final") or {}
        return self.normalize_continuous_stance(
            raw=block.get("raw"),
            scale=block.get("scale") or [0, 100],
            context=self.get_final_context(record, participant_id),
            rating_provenance=block.get("rating_provenance"),
        )

    def get_initial_context(
        self, record: dict[str, Any], participant_id: str
    ) -> Optional[ContextBlock]:
        party = self.get_party(record, participant_id)
        block = (party.get("opinion") or {}).get("initial") or {}
        statement = block.get("statement")
        if not statement:
            turns = record.get("turns") or []
            if turns and turns[0].get("from") == participant_id:
                statement = turns[0].get("text")
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

    def get_final_context(
        self, record: dict[str, Any], participant_id: str
    ) -> Optional[ContextBlock]:
        party = self.get_party(record, participant_id)
        block = (party.get("opinion") or {}).get("final") or {}
        statement = block.get("statement")
        # Do not promote mid-dialogue turns to final context.
        if not statement:
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
        limitations = [
            "No post-discussion participant-authored final statement in the source data; "
            "final context is rating-only/missing.",
        ]
        return list(self.ASSUMPTIONS), limitations
