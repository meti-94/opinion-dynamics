"""General OpinionDynamicsDataLoader.

Registers adapters, loads unified JSONL, builds participant-level episodes,
and provides filtering, summary, inspection, export, and Hugging Face conversion.
Contains no dataset-specific parsing logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Union

from opinion_dynamics.datasets.adapters import ADAPTER_REGISTRY, get_adapter
from opinion_dynamics.datasets.adapters.base import BaseDatasetAdapter
from opinion_dynamics.datasets.episode import Episode
from opinion_dynamics.inspection.comparison import compare_original_and_episode
from opinion_dynamics.inspection.inspect import (
    inspect_episode,
    inspect_perspective,
    inspect_visibility,
)
from opinion_dynamics.inspection.summary import CorpusSummary, summarize_episodes

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


PathLike = Union[str, Path]


def drop_unchanged_duplicate_posts(
    episodes: list[Episode],
) -> tuple[list[Episode], list[Episode]]:
    """Drop unchanged CMV episodes when the same post has a changed episode.

    CMV stores each post as a successful thread plus one or more unsuccessful
    siblings. Those are separate episode ids for one discussion. When a post
    has both outcomes, keep the episodes where the participant changed. A post
    with only one outcome is kept as-is.
    """
    groups: dict[str, list[Episode]] = {}
    for episode in episodes:
        if episode.source.dataset != "CMV":
            continue
        post_id = str((episode.extra or {}).get("conversation_id") or "")
        if not post_id:
            continue
        groups.setdefault(post_id, []).append(episode)

    drop_ids: set[int] = set()
    for group in groups.values():
        has_change = any(ep.outcome.changed is True for ep in group)
        if has_change and len(group) > 1:
            for ep in group:
                if ep.outcome.changed is not True:
                    drop_ids.add(id(ep))
    kept = [ep for ep in episodes if id(ep) not in drop_ids]
    dropped = [ep for ep in episodes if id(ep) in drop_ids]
    return kept, dropped


class OpinionDynamicsDataLoader:
    """Load and manage participant-level opinion-dynamics episodes."""

    def __init__(
        self,
        *,
        repo_root: Optional[PathLike] = None,
        exclude_control: bool = False,
        splits: Optional[Sequence[str]] = None,
        validate: bool = True,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self.exclude_control = exclude_control
        self.splits = list(splits) if splits is not None else ["train", "val", "test"]
        self.validate = validate

        self._adapters: dict[str, BaseDatasetAdapter] = {}
        self._paths: dict[str, Path] = {}
        self._records: dict[str, list[dict[str, Any]]] = {}
        self._episodes: list[Episode] = []
        self._episode_index: dict[str, Episode] = {}
        self._record_index: dict[tuple[str, str], dict[str, Any]] = {}
        self._exclusions: list[dict[str, Any]] = []
        self._loaded = False

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, config_path: PathLike, *, repo_root: Optional[PathLike] = None) -> "OpinionDynamicsDataLoader":
        path = Path(config_path)
        root = Path(repo_root) if repo_root else path.resolve().parents[1] if path.name else Path.cwd()
        # Prefer repo root as parent of configs/
        if (path.parent.name == "configs") and repo_root is None:
            root = path.resolve().parents[1]
        if yaml is None:
            raise ImportError("PyYAML is required to load configs; pip install pyyaml")
        with path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        options = cfg.get("options") or {}
        loader = cls(
            repo_root=root,
            exclude_control=bool(options.get("exclude_control", False)),
            splits=options.get("splits") or ["train", "val", "test"],
            validate=bool(options.get("validate", True)),
        )
        for name, entry in (cfg.get("datasets") or {}).items():
            if not entry or not entry.get("enabled", True):
                continue
            adapter_name = entry.get("adapter") or name
            data_path = Path(entry["path"])
            if not data_path.is_absolute():
                data_path = root / data_path
            loader.register(adapter_name, data_path, dataset_key=name)
        return loader

    def register(
        self,
        adapter: Union[str, BaseDatasetAdapter],
        path: PathLike,
        *,
        dataset_key: Optional[str] = None,
    ) -> "OpinionDynamicsDataLoader":
        if isinstance(adapter, str):
            adapter_obj = get_adapter(adapter)
            key = dataset_key or adapter_obj.dataset_name
        else:
            adapter_obj = adapter
            key = dataset_key or adapter_obj.dataset_name
        self._adapters[key] = adapter_obj
        self._paths[key] = Path(path)
        return self

    # ------------------------------------------------------------------ loading
    def load(self, *, force: bool = False) -> list[Episode]:
        if self._loaded and not force:
            return list(self._episodes)

        episodes: list[Episode] = []
        exclusions: list[dict[str, Any]] = []
        records_map: dict[str, list[dict[str, Any]]] = {}
        record_index: dict[tuple[str, str], dict[str, Any]] = {}

        for dataset_key, adapter in self._adapters.items():
            path = self._paths[dataset_key]
            if not path.exists():
                exclusions.append({
                    "dataset": dataset_key,
                    "reason": f"file not found: {path}",
                    "count": None,
                })
                continue
            records = adapter.load(path)
            kept_records: list[dict[str, Any]] = []
            for record in records:
                split = record.get("split")
                if self.splits and split not in self.splits:
                    exclusions.append({
                        "dataset": dataset_key,
                        "original_id": (record.get("source") or {}).get("original_id"),
                        "reason": f"split {split!r} not in requested splits {self.splits}",
                    })
                    continue
                if self.exclude_control and (record.get("structure") or {}).get("control"):
                    exclusions.append({
                        "dataset": dataset_key,
                        "original_id": (record.get("source") or {}).get("original_id"),
                        "reason": "control record excluded by exclude_control=True",
                    })
                    continue
                kept_records.append(record)
                original_id = str((record.get("source") or {}).get("original_id") or record.get("id"))
                record_index[(adapter.dataset_name, original_id)] = record

                for episode in adapter.build_episodes(record):
                    if self.validate:
                        self.validate_episode(episode)
                    episodes.append(episode)

            records_map[dataset_key] = kept_records

        episodes, dropped = drop_unchanged_duplicate_posts(episodes)
        if dropped:
            exclusions.append({
                "dataset": "CMV",
                "reason": (
                    "unchanged episode dropped because the same post also has an "
                    "episode where the participant changed their mind"
                ),
                "count": len(dropped),
                "episode_ids": [e.episode_id for e in dropped],
            })

        self._records = records_map
        self._episodes = episodes
        self._episode_index = {e.episode_id: e for e in episodes}
        self._record_index = record_index
        self._exclusions = exclusions
        self._loaded = True
        return list(self._episodes)

    # ------------------------------------------------------------------ access
    @property
    def episodes(self) -> list[Episode]:
        if not self._loaded:
            self.load()
        return list(self._episodes)

    def get_episode(self, episode_id: str) -> Episode:
        if not self._loaded:
            self.load()
        if episode_id not in self._episode_index:
            raise KeyError(f"Unknown episode_id: {episode_id}")
        return self._episode_index[episode_id]

    def get_original_record(self, episode: Union[str, Episode]) -> dict[str, Any]:
        if not self._loaded:
            self.load()
        if isinstance(episode, str):
            episode = self.get_episode(episode)
        key = (episode.source.dataset, episode.source.original_id)
        if key not in self._record_index:
            raise KeyError(f"Original record not found for {key}")
        return self._record_index[key]

    def sibling_episodes(self, episode: Union[str, Episode]) -> list[Episode]:
        """All participant episodes derived from the same original record."""
        if isinstance(episode, str):
            episode = self.get_episode(episode)
        return [
            e for e in self.episodes
            if e.source.dataset == episode.source.dataset
            and e.source.original_id == episode.source.original_id
        ]

    # ------------------------------------------------------------------ filter / summary
    def filter(
        self,
        *,
        dataset: Optional[str] = None,
        split: Optional[str] = None,
        participant_id: Optional[str] = None,
        structure_type: Optional[str] = None,
        stance_type: Optional[str] = None,
        has_initial_rating: Optional[bool] = None,
        has_final_rating: Optional[bool] = None,
        has_initial_context: Optional[bool] = None,
        control: Optional[bool] = None,
        predicate: Optional[Callable[[Episode], bool]] = None,
    ) -> list[Episode]:
        result = []
        for ep in self.episodes:
            if dataset is not None and ep.source.dataset.lower() != dataset.lower():
                continue
            if split is not None and ep.source.split != split:
                continue
            if participant_id is not None and ep.agent.participant_id != participant_id:
                continue
            if structure_type is not None and ep.metadata.structure_type != structure_type:
                continue
            if stance_type is not None and ep.initial.stance_type != stance_type:
                continue
            if has_initial_rating is not None and ep.metadata.has_initial_rating != has_initial_rating:
                continue
            if has_final_rating is not None and ep.metadata.has_final_rating != has_final_rating:
                continue
            if has_initial_context is not None and ep.metadata.has_initial_context != has_initial_context:
                continue
            if control is not None and ep.metadata.control != control:
                continue
            if predicate is not None and not predicate(ep):
                continue
            result.append(ep)
        return result

    def summary(self, episodes: Optional[Iterable[Episode]] = None) -> CorpusSummary:
        if not self._loaded:
            self.load()
        eps = list(episodes) if episodes is not None else self._episodes
        records_per_dataset = {k: len(v) for k, v in self._records.items()}
        return summarize_episodes(
            eps,
            records_per_dataset=records_per_dataset,
            exclusions=self._exclusions,
        )

    # ------------------------------------------------------------------ inspection
    def inspect(self, episode: Union[str, Episode], *, max_text: int = 400) -> str:
        if isinstance(episode, str):
            episode = self.get_episode(episode)
        return inspect_episode(episode, max_text=max_text)

    def inspect_perspective(self, episode: Union[str, Episode], *, max_text: int = 200) -> str:
        if isinstance(episode, str):
            episode = self.get_episode(episode)
        return inspect_perspective(self.sibling_episodes(episode), max_text=max_text)

    def inspect_visibility(self, episode: Union[str, Episode]) -> str:
        if isinstance(episode, str):
            episode = self.get_episode(episode)
        return inspect_visibility(episode)

    def compare_original_and_episode(
        self,
        episode_id: str,
        *,
        max_text: int = 300,
    ) -> str:
        episode = self.get_episode(episode_id)
        record = self.get_original_record(episode)
        return compare_original_and_episode(record, episode, max_text=max_text)

    # ------------------------------------------------------------------ export / HF
    def export_jsonl(self, path: PathLike, episodes: Optional[Iterable[Episode]] = None) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        eps = list(episodes) if episodes is not None else self.episodes
        with out.open("w", encoding="utf-8") as fh:
            for ep in eps:
                fh.write(json.dumps(ep.to_dict(), ensure_ascii=False) + "\n")
        return out

    @staticmethod
    def load_episodes_jsonl(path: PathLike) -> list[Episode]:
        episodes = []
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    episodes.append(Episode.from_dict(json.loads(line)))
        return episodes

    def to_huggingface(self, episodes: Optional[Iterable[Episode]] = None):
        try:
            from datasets import Dataset
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Hugging Face datasets is required: pip install datasets"
            ) from exc
        eps = list(episodes) if episodes is not None else self.episodes
        rows = [ep.to_dict() for ep in eps]
        return Dataset.from_list(rows)

    # ------------------------------------------------------------------ validation
    @staticmethod
    def validate_episode(episode: Episode) -> None:
        if not episode.episode_id:
            raise ValueError("episode_id is required")
        if not episode.source.dataset:
            raise ValueError("source.dataset is required")
        if not episode.agent.participant_id:
            raise ValueError("agent.participant_id is required")
        if episode.proposition.text is None:
            raise ValueError("proposition.text is required")
        for turn in episode.discussion.visible_turns:
            if turn.speaker_role not in {"self", "other", "unknown"}:
                raise ValueError(f"invalid speaker_role: {turn.speaker_role}")
            if not turn.visible_to_agent:
                raise ValueError("visible_turns must have visible_to_agent=True")
        # Chronological order by idx
        idxs = [t.idx for t in episode.discussion.visible_turns]
        if idxs != sorted(idxs):
            raise ValueError(f"visible turns not in chronological order: {idxs}")
