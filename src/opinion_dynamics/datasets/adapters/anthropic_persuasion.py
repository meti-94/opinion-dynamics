"""AnthropicPersuasion adapter: one-shot written argument with pre/post ratings."""

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
from opinion_dynamics.perspectives.visibility import OneShotArgumentPolicy


class AnthropicPersuasionAdapter(BaseDatasetAdapter):
    """Adapter for Anthropic Persuasion Experiment unified records.

    Discussion is a single written argument attributed to its actual author (AI or P1).
    Initial/final contexts are null unless participant-authored text exists.
    """

    dataset_name = "AnthropicPersuasion"
    default_visibility_policy = OneShotArgumentPolicy()

    ASSUMPTIONS = [
        "The single written argument is fully visible to the reader (tracked target).",
        "Argument authorship is taken from turn.from (AI or human P1); never invented.",
        "Control rows (undisputed factual claims) remain available and are flagged.",
    ]

    def get_participants(self, record: dict[str, Any]) -> list[str]:
        return [p["id"] for p in self.get_tracked_parties(record) if p.get("id")]

    def get_initial_stance(self, record: dict[str, Any], participant_id: str) -> StanceBlock:
        party = self.get_party(record, participant_id)
        block = (party.get("opinion") or {}).get("initial") or {}
        return self.normalize_continuous_stance(
            raw=block.get("raw"),
            scale=block.get("scale") or [1, 7],
            context=self.get_initial_context(record, participant_id),
            rating_provenance=block.get("rating_provenance"),
        )

    def get_final_stance(self, record: dict[str, Any], participant_id: str) -> StanceBlock:
        party = self.get_party(record, participant_id)
        block = (party.get("opinion") or {}).get("final") or {}
        return self.normalize_continuous_stance(
            raw=block.get("raw"),
            scale=block.get("scale") or [1, 7],
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
        limitations: list[str] = []
        if (record.get("structure") or {}).get("control"):
            limitations.append("Control record: undisputed factual claim.")
        if (record.get("extra") or {}).get("persuasiveness_metric_mismatch"):
            limitations.append(
                "Published persuasiveness_metric disagrees with recomputed final-initial; "
                "episode uses recomputed values."
            )
        return list(self.ASSUMPTIONS), limitations
