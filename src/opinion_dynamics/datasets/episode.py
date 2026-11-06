"""Canonical participant-level episode schema.

Episodes are the unit of output for the opinion-dynamics DataLoader. Each episode
represents one tracked participant's perspective on one original record.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Optional


class StanceType(str, Enum):
    CONTINUOUS = "continuous"
    BINARY_PROXY = "binary_proxy"


class StanceProvenance(str, Enum):
    SELF_REPORTED = "self_reported"
    DATASET_OUTCOME_PROXY = "dataset_outcome_proxy"
    NONE = "none"


class ContextType(str, Enum):
    PARTICIPANT_AUTHORED = "participant_authored"
    PARTICIPANT_JUSTIFICATION = "participant_justification"
    RATING_ONLY = "rating_only"
    MISSING = "missing"


class SpeakerRole(str, Enum):
    SELF = "self"
    OTHER = "other"
    UNKNOWN = "unknown"


class ContextSource(str, Enum):
    HUMAN_AUTHORED = "human_authored"
    LLM_AUTHORED = "llm_authored"
    MISSING = "missing"
    DATASET = "dataset"


@dataclass
class SourceInfo:
    dataset: str
    original_id: str
    split: Optional[str] = None
    license: Optional[str] = None
    url: Optional[str] = None


@dataclass
class AgentInfo:
    participant_id: str
    kind: str = "human"
    role: str = "participant"


@dataclass
class PropositionInfo:
    text: str
    original_text: Optional[str] = None
    provenance: Optional[str] = None
    group_key: Optional[str] = None


@dataclass
class ContextBlock:
    text: Optional[str]
    speaker_id: Optional[str]
    speaker_role: str
    type: str
    source: str
    speaker_kind: Optional[str] = None


@dataclass
class StanceBlock:
    stance: Optional[float]
    raw_rating: Optional[float]
    rating_scale: Optional[list[float]]
    stance_type: str
    stance_provenance: str
    context: Optional[ContextBlock] = None


@dataclass
class VisibleTurn:
    idx: int
    round: Optional[int]
    speaker_id: Optional[str]
    speaker_role: str
    speaker_kind: Optional[str]
    channel: Optional[str]
    recipient: list[str]
    text: str
    reply_to: Optional[int] = None
    visible_to_agent: bool = True
    annotations: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscussionBlock:
    visible_turns: list[VisibleTurn] = field(default_factory=list)
    excluded_turns: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class OutcomeBlock:
    changed: Optional[bool] = None
    change_100: Optional[float] = None
    persuader_relative_delta_100: Optional[float] = None
    delta_100: Optional[dict[str, float]] = None


@dataclass
class EpisodeMetadata:
    structure_type: Optional[str] = None
    visibility_policy: Optional[str] = None
    visibility_filtered: bool = False
    has_initial_rating: bool = False
    has_final_rating: bool = False
    has_initial_context: bool = False
    has_final_context: bool = False
    control: bool = False
    assumptions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


@dataclass
class Episode:
    """Canonical participant-level episode."""

    episode_id: str
    source: SourceInfo
    agent: AgentInfo
    proposition: PropositionInfo
    initial: StanceBlock
    discussion: DiscussionBlock
    final: StanceBlock
    outcome: OutcomeBlock
    metadata: EpisodeMetadata
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Episode":
        return cls(
            episode_id=data["episode_id"],
            source=SourceInfo(**data["source"]),
            agent=AgentInfo(**data["agent"]),
            proposition=PropositionInfo(**{
                k: data["proposition"].get(k)
                for k in ("text", "original_text", "provenance", "group_key")
            }),
            initial=_stance_from_dict(data["initial"]),
            discussion=DiscussionBlock(
                visible_turns=[VisibleTurn(**_turn_kwargs(t)) for t in data.get("discussion", {}).get("visible_turns", [])],
                excluded_turns=list(data.get("discussion", {}).get("excluded_turns", [])),
            ),
            final=_stance_from_dict(data["final"]),
            outcome=OutcomeBlock(**{
                k: data.get("outcome", {}).get(k)
                for k in ("changed", "change_100", "persuader_relative_delta_100", "delta_100")
            }),
            metadata=EpisodeMetadata(**{
                k: data.get("metadata", {}).get(k, _metadata_default(k))
                for k in (
                    "structure_type", "visibility_policy", "visibility_filtered",
                    "has_initial_rating", "has_final_rating", "has_initial_context",
                    "has_final_context", "control", "assumptions", "limitations",
                )
            }),
            extra=dict(data.get("extra") or {}),
        )

    def to_stance_prediction_messages(self) -> list[dict[str, str]]:
        from opinion_dynamics.formatting.stance_prediction import StancePredictionFormatter

        return StancePredictionFormatter().format(self)

    def to_response_generation_messages(
        self,
        target_turn_idx: int,
    ) -> tuple[list[dict[str, str]], Optional[str]]:
        from opinion_dynamics.formatting.response_generation import ResponseGenerationFormatter

        return ResponseGenerationFormatter().format(self, target_turn_idx=target_turn_idx)


def _metadata_default(key: str) -> Any:
    defaults = {
        "structure_type": None,
        "visibility_policy": None,
        "visibility_filtered": False,
        "has_initial_rating": False,
        "has_final_rating": False,
        "has_initial_context": False,
        "has_final_context": False,
        "control": False,
        "assumptions": [],
        "limitations": [],
    }
    return defaults[key]


def _context_from_dict(data: Optional[dict[str, Any]]) -> Optional[ContextBlock]:
    if data is None:
        return None
    return ContextBlock(**{
        k: data.get(k)
        for k in ("text", "speaker_id", "speaker_role", "type", "source", "speaker_kind")
    })


def _stance_from_dict(data: dict[str, Any]) -> StanceBlock:
    return StanceBlock(
        stance=data.get("stance"),
        raw_rating=data.get("raw_rating"),
        rating_scale=data.get("rating_scale"),
        stance_type=data.get("stance_type", StanceType.CONTINUOUS.value),
        stance_provenance=data.get("stance_provenance", StanceProvenance.NONE.value),
        context=_context_from_dict(data.get("context")),
    )


def _turn_kwargs(data: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "idx", "round", "speaker_id", "speaker_role", "speaker_kind", "channel",
        "recipient", "text", "reply_to", "visible_to_agent", "annotations",
    }
    out = {k: data.get(k) for k in allowed if k in data or k in ("visible_to_agent", "annotations", "recipient")}
    out.setdefault("recipient", list(data.get("recipient") or []))
    out.setdefault("visible_to_agent", True)
    out.setdefault("annotations", dict(data.get("annotations") or {}))
    return out


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj
