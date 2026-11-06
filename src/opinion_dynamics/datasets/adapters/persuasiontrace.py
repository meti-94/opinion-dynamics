"""PersuasionTrace adapter: human–LLM dialogue with first/last belief ratings."""

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


class PersuasionTraceAdapter(BaseDatasetAdapter):
    """Adapter for PersuasionTrace unified records.

    Canonical episodes use only the first (pre-dialogue) and last belief ratings.
    Intermediate ratings remain in extra.opinion_trace only.
    Control dialogues (off-topic) are flagged via metadata.control / structure.control.
    """

    dataset_name = "PersuasionTrace"
    default_visibility_policy = FullHistoryPolicy()

    ASSUMPTIONS = [
        "Full human–AI dialogue is visible to the tracked target.",
        "Initial stance is the first available belief rating (pre-dialogue); "
        "final stance is the last available belief rating.",
        "Intermediate serial-question ratings are excluded from the canonical episode.",
    ]

    def get_participants(self, record: dict[str, Any]) -> list[str]:
        return [p["id"] for p in self.get_tracked_parties(record) if p.get("id")]

    def get_initial_stance(self, record: dict[str, Any], participant_id: str) -> StanceBlock:
        party = self.get_party(record, participant_id)
        block = (party.get("opinion") or {}).get("initial") or {}
        # Prefer explicit initial; fall back to first available rating in trace if needed.
        raw = block.get("raw")
        scale = block.get("scale") or [0, 1]
        if raw is None:
            trace = (party.get("opinion") or {}).get("trace") or []
            if trace:
                raw = trace[0].get("raw")
        return self.normalize_continuous_stance(
            raw=raw,
            scale=scale,
            context=self.get_initial_context(record, participant_id),
            rating_provenance=block.get("rating_provenance") or "self_reported",
        )

    def get_final_stance(self, record: dict[str, Any], participant_id: str) -> StanceBlock:
        party = self.get_party(record, participant_id)
        block = (party.get("opinion") or {}).get("final") or {}
        raw = block.get("raw")
        scale = block.get("scale") or [0, 1]
        if raw is None:
            trace = (party.get("opinion") or {}).get("trace") or []
            if trace:
                raw = trace[-1].get("raw")
        return self.normalize_continuous_stance(
            raw=raw,
            scale=scale,
            context=self.get_final_context(record, participant_id),
            rating_provenance=block.get("rating_provenance") or "self_reported",
        )

    def get_initial_context(
        self, record: dict[str, Any], participant_id: str
    ) -> Optional[ContextBlock]:
        party = self.get_party(record, participant_id)
        block = (party.get("opinion") or {}).get("initial") or {}
        statement = block.get("statement")
        if statement:
            return self.build_context(
                text=statement,
                speaker_id=participant_id,
                agent_id=participant_id,
                context_type=ContextType.PARTICIPANT_JUSTIFICATION.value,
                source=ContextSource.HUMAN_AUTHORED.value,
                speaker_kind=party.get("kind"),
            )
        # Rating exists without authored text.
        if block.get("raw") is not None:
            return self.build_context(
                text=None,
                speaker_id=participant_id,
                agent_id=participant_id,
                context_type=ContextType.RATING_ONLY.value,
                source=ContextSource.MISSING.value,
                speaker_kind=party.get("kind"),
            )
        return self.build_context(
            text=None,
            speaker_id=participant_id,
            agent_id=participant_id,
            context_type=ContextType.MISSING.value,
            source=ContextSource.MISSING.value,
            speaker_kind=party.get("kind"),
        )

    def get_final_context(
        self, record: dict[str, Any], participant_id: str
    ) -> Optional[ContextBlock]:
        party = self.get_party(record, participant_id)
        block = (party.get("opinion") or {}).get("final") or {}
        statement = block.get("statement")
        if statement:
            return self.build_context(
                text=statement,
                speaker_id=participant_id,
                agent_id=participant_id,
                context_type=ContextType.PARTICIPANT_AUTHORED.value,
                source=ContextSource.HUMAN_AUTHORED.value,
                speaker_kind=party.get("kind"),
            )
        if block.get("raw") is not None:
            return self.build_context(
                text=None,
                speaker_id=participant_id,
                agent_id=participant_id,
                context_type=ContextType.RATING_ONLY.value,
                source=ContextSource.MISSING.value,
                speaker_kind=party.get("kind"),
            )
        return self.build_context(
            text=None,
            speaker_id=participant_id,
            agent_id=participant_id,
            context_type=ContextType.MISSING.value,
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

    def is_control(self, record: dict[str, Any]) -> bool:
        return bool((record.get("structure") or {}).get("control"))

    def perspective_notes(
        self, record: dict[str, Any], participant_id: str
    ) -> tuple[list[str], list[str]]:
        assumptions = list(self.ASSUMPTIONS)
        limitations: list[str] = []
        if self.is_control(record):
            limitations.append(
                "CONTROL dialogue: proposition.text is the discussion topic "
                f"({(record.get('proposition') or {}).get('text')!r}). Belief ratings "
                "were collected about a different measured claim stored in "
                "extra.measured_proposition / proposition.original_text."
            )
        return assumptions, limitations
