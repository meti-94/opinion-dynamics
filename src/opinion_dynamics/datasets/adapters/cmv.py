"""CMV adapter: Reddit threads with binary Δ outcome proxies."""

from __future__ import annotations

import re
from typing import Any, Optional

from opinion_dynamics.datasets.adapters.base import BaseDatasetAdapter
from opinion_dynamics.datasets.episode import (
    ContextBlock,
    ContextSource,
    ContextType,
    OutcomeBlock,
    StanceBlock,
    StanceProvenance,
    StanceType,
    VisibleTurn,
)
from opinion_dynamics.perspectives.visibility import FullHistoryPolicy


# Award / annotation markers to strip from visible text. Argument content is preserved.
_DELTA_PATTERNS = [
    re.compile(r"^\s*[∆Δ]\s*[.\-]?\s*", re.UNICODE),
    re.compile(r"^\s*!+\s*[Dd]elta\b\s*[.\-]?\s*"),
    re.compile(r"(?i)\bdelta\s+awarded\b"),
]


def _is_delta_award_turn(turn: dict[str, Any]) -> bool:
    """OP comment that awards a Δ. The opening post is kept even if it mentions the word."""
    annotations = turn.get("annotations") or {}
    return bool(annotations.get("delta_marker")) and not annotations.get("is_op_post")


def _is_delta_award_text(record: dict[str, Any], text: str) -> bool:
    return any(
        _is_delta_award_turn(turn) and (turn.get("text") or "") == text
        for turn in record.get("turns") or []
    )


def strip_delta_annotations(text: str) -> str:
    """Remove literal Δ award markers without fabricating or rewriting arguments."""
    if not text:
        return text
    cleaned = text
    for pat in _DELTA_PATTERNS:
        cleaned = pat.sub("", cleaned)
    # Also strip isolated leading delta tokens that appear mid-cleanup leftovers.
    cleaned = re.sub(r"^\s*[∆Δ]\s*", "", cleaned)
    return cleaned.lstrip() if cleaned != text else cleaned


class CMVAdapter(BaseDatasetAdapter):
    """Adapter for ChangeMyView / Winning-Arguments unified records.

    Binary stance proxy on a 0–100 scale (not a measured rating):
      successful Δ   -> initial=100, final=0
      unsuccessful   -> initial=100, final=100
      unknown/null   -> initial=100, final=null
    """

    dataset_name = "CMV"
    default_visibility_policy = FullHistoryPolicy()

    ASSUMPTIONS = [
        "Binary stance proxy on a 0–100 scale: initial=100 means OP starts holding the proposition; "
        "final=0 for successful Δ indicates a dataset-labelled change, not measured disagreement.",
        "Reddit thread comments are visible to the OP, except OP comments that award a Δ.",
        "Initial context (OP post body) is also present as turn 0 in the discussion.",
        "Δ-award comments are removed from the discussion and are not used as final context, "
        "because they state that the opinion changed.",
    ]

    def get_participants(self, record: dict[str, Any]) -> list[str]:
        return [p["id"] for p in self.get_tracked_parties(record) if p.get("id")]

    def _proxy_stances(self, record: dict[str, Any]) -> tuple[float, Optional[float]]:
        changed = (record.get("outcome") or {}).get("changed")
        initial = 100.0
        if changed is True:
            return initial, 0.0
        if changed is False:
            return initial, 100.0
        return initial, None

    def get_initial_stance(self, record: dict[str, Any], participant_id: str) -> StanceBlock:
        initial, _ = self._proxy_stances(record)
        return StanceBlock(
            stance=initial,
            raw_rating=None,
            rating_scale=None,
            stance_type=StanceType.BINARY_PROXY.value,
            stance_provenance=StanceProvenance.DATASET_OUTCOME_PROXY.value,
            context=self.get_initial_context(record, participant_id),
        )

    def get_final_stance(self, record: dict[str, Any], participant_id: str) -> StanceBlock:
        _, final = self._proxy_stances(record)
        return StanceBlock(
            stance=final,
            raw_rating=None,
            rating_scale=None,
            stance_type=StanceType.BINARY_PROXY.value,
            stance_provenance=StanceProvenance.DATASET_OUTCOME_PROXY.value,
            context=self.get_final_context(record, participant_id),
        )

    def get_initial_context(
        self, record: dict[str, Any], participant_id: str
    ) -> Optional[ContextBlock]:
        party = self.get_party(record, participant_id)
        block = (party.get("opinion") or {}).get("initial") or {}
        statement = block.get("statement")
        if not statement:
            # Fall back to turn 0 if annotated as OP post.
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
        if statement and _is_delta_award_text(record, statement):
            statement = None
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
            text=strip_delta_annotations(statement),
            speaker_id=participant_id,
            agent_id=participant_id,
            context_type=ContextType.PARTICIPANT_AUTHORED.value,
            source=ContextSource.HUMAN_AUTHORED.value,
            speaker_kind=party.get("kind"),
        )

    def get_visible_history(
        self, record: dict[str, Any], participant_id: str
    ) -> tuple[list[VisibleTurn], list[dict[str, Any]]]:
        kept: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        for turn in record.get("turns") or []:
            if _is_delta_award_turn(turn):
                excluded.append({
                    "idx": turn.get("idx"),
                    "round": turn.get("round"),
                    "channel": turn.get("channel"),
                    "from": turn.get("from"),
                    "to": list(turn.get("to") or []),
                    "reason": "delta award comment excluded; it states that the opinion changed",
                    "text_preview": (turn.get("text") or "")[:120],
                })
            else:
                kept.append(turn)

        visible, other_excluded = self.turns_to_visible(
            {**record, "turns": kept},
            participant_id,
            self.default_visibility_policy,
            text_transform=lambda text, _turn: strip_delta_annotations(text),
        )
        return visible, excluded + other_excluded

    def build_outcome(
        self,
        record: dict[str, Any],
        participant_id: str,
        initial: StanceBlock,
        final: StanceBlock,
    ) -> OutcomeBlock:
        outcome = record.get("outcome") or {}
        changed = outcome.get("changed")
        change = None
        if initial.stance is not None and final.stance is not None:
            change = float(final.stance) - float(initial.stance)
        return OutcomeBlock(
            changed=changed if changed is None else bool(changed),
            change_100=change,
            persuader_relative_delta_100=outcome.get("persuader_relative_delta_100"),
            delta_100=None,
        )

    def build_proposition(self, record: dict[str, Any]):
        prop = super().build_proposition(record)
        # Proposition text in unified already has CMV: stripped; keep as-is.
        return prop

    def perspective_notes(
        self, record: dict[str, Any], participant_id: str
    ) -> tuple[list[str], list[str]]:
        limitations = [
            "Binary stance proxy must never be treated as a continuous 0–100 human rating.",
        ]
        if not (record.get("outcome") or {}).get("changed") is False:
            party = self.get_party(record, participant_id)
            final_stmt = ((party.get("opinion") or {}).get("final") or {}).get("statement")
            if not final_stmt:
                limitations.append("No final OP reply available; final context is missing.")
            elif _is_delta_award_text(record, final_stmt):
                limitations.append(
                    "Final OP reply awards a Δ and was removed from the interaction and final context."
                )
        return list(self.ASSUMPTIONS), limitations
