"""Dataset / episode corpus summary statistics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from opinion_dynamics.datasets.episode import Episode


@dataclass
class CorpusSummary:
    original_record_count: int = 0
    episode_count: int = 0
    unique_participants: int = 0
    records_per_dataset: dict[str, int] = field(default_factory=dict)
    episodes_per_dataset: dict[str, int] = field(default_factory=dict)
    episodes_with_initial_rating: int = 0
    episodes_with_final_rating: int = 0
    episodes_with_both_ratings: int = 0
    binary_proxy_episodes: int = 0
    continuous_rating_episodes: int = 0
    episodes_with_initial_context: int = 0
    episodes_without_initial_context: int = 0
    conversation_length: dict[str, Any] = field(default_factory=dict)
    structure_type_distribution: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)
    control_episodes: int = 0
    exclusions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_record_count": self.original_record_count,
            "episode_count": self.episode_count,
            "unique_participants": self.unique_participants,
            "records_per_dataset": dict(self.records_per_dataset),
            "episodes_per_dataset": dict(self.episodes_per_dataset),
            "episodes_with_initial_rating": self.episodes_with_initial_rating,
            "episodes_with_final_rating": self.episodes_with_final_rating,
            "episodes_with_both_ratings": self.episodes_with_both_ratings,
            "binary_proxy_episodes": self.binary_proxy_episodes,
            "continuous_rating_episodes": self.continuous_rating_episodes,
            "episodes_with_initial_context": self.episodes_with_initial_context,
            "episodes_without_initial_context": self.episodes_without_initial_context,
            "conversation_length": dict(self.conversation_length),
            "structure_type_distribution": dict(self.structure_type_distribution),
            "split_counts": dict(self.split_counts),
            "control_episodes": self.control_episodes,
            "exclusions": list(self.exclusions),
        }

    def __str__(self) -> str:
        lines = [
            "=== Opinion Dynamics Corpus Summary ===",
            f"Original records          : {self.original_record_count}",
            f"Participant episodes      : {self.episode_count}",
            f"Unique participants       : {self.unique_participants}",
            f"Records / dataset         : {self.records_per_dataset}",
            f"Episodes / dataset        : {self.episodes_per_dataset}",
            f"Initial ratings           : {self.episodes_with_initial_rating}",
            f"Final ratings             : {self.episodes_with_final_rating}",
            f"Both ratings              : {self.episodes_with_both_ratings}",
            f"Binary proxy              : {self.binary_proxy_episodes}",
            f"Continuous                : {self.continuous_rating_episodes}",
            f"With initial context      : {self.episodes_with_initial_context}",
            f"Without initial context   : {self.episodes_without_initial_context}",
            f"Control episodes          : {self.control_episodes}",
            f"Splits                    : {self.split_counts}",
            f"Structure types           : {self.structure_type_distribution}",
            f"Conversation length       : {self.conversation_length}",
            f"Exclusions reported       : {len(self.exclusions)}",
        ]
        return "\n".join(lines)


def summarize_episodes(
    episodes: Iterable[Episode],
    *,
    original_record_count: Optional[int] = None,
    records_per_dataset: Optional[dict[str, int]] = None,
    exclusions: Optional[list[dict[str, Any]]] = None,
) -> CorpusSummary:
    eps = list(episodes)
    lengths = [len(e.discussion.visible_turns) for e in eps]
    # Distinct simulated agents = (dataset, original record, participant).
    participants = {
        (e.source.dataset, e.source.original_id, e.agent.participant_id) for e in eps
    }

    summary = CorpusSummary(
        original_record_count=original_record_count
        if original_record_count is not None
        else len({(e.source.dataset, e.source.original_id) for e in eps}),
        episode_count=len(eps),
        unique_participants=len(participants),
        records_per_dataset=dict(records_per_dataset or Counter(e.source.dataset for e in eps)),
        episodes_per_dataset=dict(Counter(e.source.dataset for e in eps)),
        episodes_with_initial_rating=sum(1 for e in eps if e.metadata.has_initial_rating),
        episodes_with_final_rating=sum(1 for e in eps if e.metadata.has_final_rating),
        episodes_with_both_ratings=sum(
            1 for e in eps if e.metadata.has_initial_rating and e.metadata.has_final_rating
        ),
        binary_proxy_episodes=sum(1 for e in eps if e.initial.stance_type == "binary_proxy"),
        continuous_rating_episodes=sum(1 for e in eps if e.initial.stance_type == "continuous"),
        episodes_with_initial_context=sum(1 for e in eps if e.metadata.has_initial_context),
        episodes_without_initial_context=sum(1 for e in eps if not e.metadata.has_initial_context),
        structure_type_distribution=dict(Counter(e.metadata.structure_type for e in eps)),
        split_counts=dict(Counter(e.source.split for e in eps)),
        control_episodes=sum(1 for e in eps if e.metadata.control),
        exclusions=list(exclusions or []),
    )
    if lengths:
        lengths_sorted = sorted(lengths)
        summary.conversation_length = {
            "min": lengths_sorted[0],
            "max": lengths_sorted[-1],
            "mean": sum(lengths) / len(lengths),
            "median": lengths_sorted[len(lengths) // 2],
            "distribution": dict(Counter(lengths)),
        }
    else:
        summary.conversation_length = {"min": None, "max": None, "mean": None, "median": None}
    # Prefer true original record counts when provided.
    if records_per_dataset:
        summary.records_per_dataset = dict(records_per_dataset)
        summary.original_record_count = sum(records_per_dataset.values())
    return summary
