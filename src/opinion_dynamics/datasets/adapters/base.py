"""Base dataset adapter interface.

Adapters convert unified JSONL records into participant-level Episode objects.
Dataset-specific parsing, visibility, and context attribution live in subclasses.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence, Union

from opinion_dynamics.datasets.episode import (
    AgentInfo,
    ContextBlock,
    ContextSource,
    ContextType,
    DiscussionBlock,
    Episode,
    EpisodeMetadata,
    OutcomeBlock,
    PropositionInfo,
    SourceInfo,
    SpeakerRole,
    StanceBlock,
    StanceProvenance,
    StanceType,
    VisibleTurn,
)
from opinion_dynamics.datasets.stance import change_100, normalize_rating, stance_changed
from opinion_dynamics.perspectives.speaker_identity import party_kind_lookup, speaker_role_for
from opinion_dynamics.perspectives.visibility import (
    VisibilityPolicy,
    filter_visible_turns,
)


PathLike = Union[str, Path]


class BaseDatasetAdapter(ABC):
    """Common interface for dataset-specific adapters."""

    dataset_name: str = "BASE"
    default_visibility_policy: Optional[VisibilityPolicy] = None

    # ------------------------------------------------------------------ I/O
    def load(self, path: PathLike) -> list[dict[str, Any]]:
        """Load unified JSONL records from disk."""
        records: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
        return records

    def iter_records(self, path: PathLike) -> Iterator[dict[str, Any]]:
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    # ------------------------------------------------------------------ abstract API
    @abstractmethod
    def get_participants(self, record: dict[str, Any]) -> list[str]:
        """Return tracked participant IDs for which episodes should be generated."""

    @abstractmethod
    def get_initial_stance(self, record: dict[str, Any], participant_id: str) -> StanceBlock:
        ...

    @abstractmethod
    def get_final_stance(self, record: dict[str, Any], participant_id: str) -> StanceBlock:
        ...

    @abstractmethod
    def get_initial_context(
        self, record: dict[str, Any], participant_id: str
    ) -> Optional[ContextBlock]:
        ...

    @abstractmethod
    def get_final_context(
        self, record: dict[str, Any], participant_id: str
    ) -> Optional[ContextBlock]:
        ...

    @abstractmethod
    def get_visible_history(
        self, record: dict[str, Any], participant_id: str
    ) -> tuple[list[VisibleTurn], list[dict[str, Any]]]:
        ...

    # ------------------------------------------------------------------ helpers
    def get_party(self, record: dict[str, Any], participant_id: str) -> dict[str, Any]:
        for party in record.get("parties") or []:
            if party.get("id") == participant_id:
                return party
        raise KeyError(f"Participant {participant_id!r} not found in {record.get('id')}")

    def get_tracked_parties(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        return [p for p in (record.get("parties") or []) if p.get("tracked")]

    def normalize_continuous_stance(
        self,
        *,
        raw: Optional[float],
        scale: Optional[Sequence[float]],
        context: Optional[ContextBlock],
        rating_provenance: Optional[str] = None,
    ) -> StanceBlock:
        stance = normalize_rating(raw, scale)
        if raw is None:
            provenance = StanceProvenance.NONE.value
            stance_type = StanceType.CONTINUOUS.value
        else:
            provenance = (
                rating_provenance
                if rating_provenance in {
                    StanceProvenance.SELF_REPORTED.value,
                    StanceProvenance.DATASET_OUTCOME_PROXY.value,
                }
                else StanceProvenance.SELF_REPORTED.value
            )
            stance_type = StanceType.CONTINUOUS.value
        return StanceBlock(
            stance=stance,
            raw_rating=raw if raw is None else float(raw),
            rating_scale=list(scale) if scale is not None else None,
            stance_type=stance_type,
            stance_provenance=provenance,
            context=context,
        )

    def build_context(
        self,
        *,
        text: Optional[str],
        speaker_id: Optional[str],
        agent_id: str,
        context_type: str,
        source: str,
        speaker_kind: Optional[str] = None,
    ) -> Optional[ContextBlock]:
        cleaned = (text or "").strip() or None
        if cleaned is None and context_type == ContextType.MISSING.value:
            return ContextBlock(
                text=None,
                speaker_id=speaker_id,
                speaker_role=speaker_role_for(agent_id, speaker_id),
                type=ContextType.MISSING.value,
                source=ContextSource.MISSING.value,
                speaker_kind=speaker_kind,
            )
        if cleaned is None:
            return None
        return ContextBlock(
            text=cleaned,
            speaker_id=speaker_id,
            speaker_role=speaker_role_for(agent_id, speaker_id),
            type=context_type,
            source=source,
            speaker_kind=speaker_kind,
        )

    def turns_to_visible(
        self,
        record: dict[str, Any],
        participant_id: str,
        policy: VisibilityPolicy,
        *,
        text_transform=None,
    ) -> tuple[list[VisibleTurn], list[dict[str, Any]]]:
        kinds = party_kind_lookup(record.get("parties") or [])
        raw_turns = list(record.get("turns") or [])
        visible_raw, excluded = filter_visible_turns(raw_turns, participant_id, policy)
        visible: list[VisibleTurn] = []
        for turn in visible_raw:
            text = turn.get("text") or ""
            if text_transform is not None:
                text = text_transform(text, turn)
            speaker_id = turn.get("from")
            visible.append(
                VisibleTurn(
                    idx=int(turn.get("idx")),
                    round=turn.get("round"),
                    speaker_id=speaker_id,
                    speaker_role=speaker_role_for(participant_id, speaker_id),
                    speaker_kind=kinds.get(speaker_id) if speaker_id else None,
                    channel=turn.get("channel"),
                    recipient=list(turn.get("to") or []),
                    text=text,
                    reply_to=turn.get("reply_to"),
                    visible_to_agent=True,
                    annotations=dict(turn.get("annotations") or {}),
                )
            )
        return visible, excluded

    def make_episode_id(self, record: dict[str, Any], participant_id: str) -> str:
        original = (record.get("source") or {}).get("original_id") or record.get("id")
        dataset_key = self.dataset_name.lower().replace(" ", "")
        return f"{dataset_key}:{original}:{participant_id}"

    def build_source(self, record: dict[str, Any]) -> SourceInfo:
        src = record.get("source") or {}
        return SourceInfo(
            dataset=src.get("dataset") or self.dataset_name,
            original_id=str(src.get("original_id") or record.get("id")),
            split=record.get("split"),
            license=src.get("license"),
            url=src.get("url"),
        )

    def build_proposition(self, record: dict[str, Any]) -> PropositionInfo:
        prop = record.get("proposition") or {}
        return PropositionInfo(
            text=prop.get("text") or "",
            original_text=prop.get("original_text"),
            provenance=prop.get("provenance"),
            group_key=prop.get("group_key"),
        )

    def build_outcome(
        self,
        record: dict[str, Any],
        participant_id: str,
        initial: StanceBlock,
        final: StanceBlock,
    ) -> OutcomeBlock:
        outcome = record.get("outcome") or {}
        delta_map = outcome.get("delta_100") or {}
        change = None
        if participant_id in delta_map and delta_map[participant_id] is not None:
            change = float(delta_map[participant_id])
        else:
            change = change_100(initial.stance, final.stance)

        # Participant-level. Prefer the per-person flag written by preprocessing.
        # Do not use outcome.changed: on multi-party records that flag is either
        # null or "anyone in the group moved".
        party = self.get_party(record, participant_id)
        person_changed = (party.get("opinion") or {}).get("changed")
        if person_changed is not None:
            changed = bool(person_changed)
        elif change is not None:
            changed = abs(change) > 1e-9
        else:
            changed = stance_changed(initial.stance, final.stance)
        return OutcomeBlock(
            changed=changed,
            change_100=change,
            persuader_relative_delta_100=outcome.get("persuader_relative_delta_100"),
            delta_100=dict(delta_map) if delta_map else None,
        )

    def build_metadata(
        self,
        record: dict[str, Any],
        *,
        initial: StanceBlock,
        final: StanceBlock,
        visibility_policy: str,
        visibility_filtered: bool,
        assumptions: Optional[list[str]] = None,
        limitations: Optional[list[str]] = None,
    ) -> EpisodeMetadata:
        structure = record.get("structure") or {}
        has_init_ctx = bool(initial.context and initial.context.text)
        has_final_ctx = bool(final.context and final.context.text)
        return EpisodeMetadata(
            structure_type=structure.get("type"),
            visibility_policy=visibility_policy,
            visibility_filtered=visibility_filtered,
            has_initial_rating=initial.stance is not None,
            has_final_rating=final.stance is not None,
            has_initial_context=has_init_ctx,
            has_final_context=has_final_ctx,
            control=bool(structure.get("control")),
            assumptions=list(assumptions or []),
            limitations=list(limitations or []),
        )

    def build_agent(self, record: dict[str, Any], participant_id: str) -> AgentInfo:
        party = self.get_party(record, participant_id)
        return AgentInfo(
            participant_id=participant_id,
            kind=party.get("kind") or "human",
            role=party.get("role") or "participant",
        )

    def build_episode(self, record: dict[str, Any], participant_id: str) -> Episode:
        """Assemble a canonical episode for one tracked participant."""
        initial_context = self.get_initial_context(record, participant_id)
        final_context = self.get_final_context(record, participant_id)

        initial = self.get_initial_stance(record, participant_id)
        if initial.context is None:
            initial.context = initial_context
        final = self.get_final_stance(record, participant_id)
        if final.context is None:
            final.context = final_context

        visible_turns, excluded = self.get_visible_history(record, participant_id)
        policy_name = getattr(self.default_visibility_policy, "name", "unspecified")

        assumptions, limitations = self.perspective_notes(record, participant_id)

        return Episode(
            episode_id=self.make_episode_id(record, participant_id),
            source=self.build_source(record),
            agent=self.build_agent(record, participant_id),
            proposition=self.build_proposition(record),
            initial=initial,
            discussion=DiscussionBlock(
                visible_turns=visible_turns,
                excluded_turns=excluded,
            ),
            final=final,
            outcome=self.build_outcome(record, participant_id, initial, final),
            metadata=self.build_metadata(
                record,
                initial=initial,
                final=final,
                visibility_policy=policy_name,
                visibility_filtered=bool(excluded),
                assumptions=assumptions,
                limitations=limitations,
            ),
            extra=self.build_extra(record, participant_id),
        )

    def build_episodes(self, record: dict[str, Any]) -> list[Episode]:
        return [self.build_episode(record, pid) for pid in self.get_participants(record)]

    def build_extra(self, record: dict[str, Any], participant_id: str) -> dict[str, Any]:
        """Preserve dataset-specific metadata without discarding useful fields."""
        extra = dict(record.get("extra") or {})
        structure = record.get("structure") or {}
        party = self.get_party(record, participant_id)
        return {
            **extra,
            "unified_id": record.get("id"),
            "structure_notes": structure.get("notes"),
            "n_rounds": structure.get("n_rounds"),
            "party_persona": party.get("persona"),
            "party_assigned_stance": party.get("assigned_stance"),
            "party_model": party.get("model"),
            "opinion_trace": (party.get("opinion") or {}).get("trace"),
            "all_party_ids": [p.get("id") for p in (record.get("parties") or [])],
        }

    def perspective_notes(
        self, record: dict[str, Any], participant_id: str
    ) -> tuple[list[str], list[str]]:
        return [], []

    def iter_episodes(self, path: PathLike) -> Iterator[Episode]:
        for record in self.iter_records(path):
            yield from self.build_episodes(record)
