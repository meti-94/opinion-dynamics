"""Stage-1 initial-stance examples.

The model speaks as the tracked participant and reports that person's initial
agreement with the proposition. Discussion turns, the final stance, the final
context, and outcome fields are not read into these examples.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from opinion_dynamics.datasets.dataloader import OpinionDynamicsDataLoader
from opinion_dynamics.datasets.episode import Episode

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


PathLike = Union[str, Path]

SYSTEM_TEMPLATE = (
    "You are a real person with your own opinions, speaking as yourself in the first person. "
    "You have a view on the following statement.\n\n"
    "Statement: {proposition}\n\n"
    "Answer as you genuinely feel, based on your own beliefs, experience and way of expressing yourself, "
    "not as you think you should."
)
USER_QUESTION = (
    "How much do you agree with the statement? "
    "Reply with a single whole number from 0 (completely disagree) "
    "to 100 (completely agree) inside <answer></answer> tags."
)
ALTERNATION_USER = "Go ahead."

# One raw Likert step, expressed on the 0-100 scale. Unlisted datasets have
# no fixed step (continuous sliders, or the CMV binary proxy).
SCALE_STEP = {
    "DEBATE": 20.0,
    "DebateGPT": 25.0,
    "AnthropicPersuasion": 16.7,
    "ConspiracyAI": 1.0,
}

# Traces that give away a CMV delta award inside the post itself.
_DELTA_MARK = re.compile(r"[∆Δ]|!delta|delta\s+awarded", re.IGNORECASE)
_EDIT_AND_DELTA = re.compile(r"\bedit\b[\s\S]{0,200}delta", re.IGNORECASE)

EXAMPLE_FIELDS = (
    "episode_id",
    "group_id",
    "dataset",
    "split",
    "role",
    "messages",
    "proposition",
    "initial_context",
    "target",
    "scale_step",
    "target_type",
)


@dataclass
class InitialStanceExample:
    episode_id: str
    group_id: str
    dataset: str
    split: Optional[str]
    role: str
    messages: list[dict[str, str]]
    proposition: str
    initial_context: Optional[str]
    target: float
    scale_step: Optional[float]
    target_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_messages(
    episode: Episode,
    upto: Union[str, int] = "initial",
    *,
    proposition: Optional[str] = None,
    ensure_alternation: bool = False,
    drop_context: bool = False,
) -> list[dict[str, str]]:
    """Build chat messages for one episode.

    ``upto="initial"`` is stage 1: proposition and, when present, the
    participant's own initial text. No discussion turn is read.

    ``upto=<turn idx>`` includes visible turns with ``idx <= upto``. Turns by
    the tracked participant are assistant messages. Other speakers are user
    messages prefixed with their speaker id. Later stages use this; stage 1
    does not.
    """
    text = episode.proposition.text if proposition is None else proposition
    context = None if drop_context else _own_initial_text(episode)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(proposition=text)},
    ]
    if context:
        messages.append({"role": "assistant", "content": context})
    if upto != "initial":
        if not isinstance(upto, int):
            raise ValueError(f"upto must be 'initial' or a turn index, got {upto!r}")
        messages.extend(_turns_through(episode, upto))
    messages.append({"role": "user", "content": USER_QUESTION})
    if ensure_alternation:
        messages = _ensure_alternation(messages)
    return messages


def _own_initial_text(episode: Episode) -> Optional[str]:
    context = episode.initial.context
    if context is None or not context.text:
        return None
    return context.text


def _turns_through(episode: Episode, upto: int) -> list[dict[str, str]]:
    agent = episode.agent.participant_id
    out: list[dict[str, str]] = []
    for turn in episode.discussion.visible_turns:
        if turn.idx > upto or not turn.visible_to_agent:
            continue
        if turn.speaker_id == agent:
            out.append({"role": "assistant", "content": turn.text})
        else:
            label = turn.speaker_id or "UNKNOWN"
            out.append({"role": "user", "content": f"[{label}] {turn.text}"})
    return out


def _ensure_alternation(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Insert one neutral user turn before the first assistant message."""
    out: list[dict[str, str]] = []
    inserted = False
    for message in messages:
        if message["role"] == "assistant" and not inserted:
            if not out or out[-1]["role"] != "user":
                out.append({"role": "user", "content": ALTERNATION_USER})
            inserted = True
        out.append(message)
    return out


def _group_id(episode: Episode) -> str:
    dataset = episode.source.dataset
    if dataset == "CMV":
        post = (episode.extra or {}).get("conversation_id")
        if post:
            return str(post)
        return episode.source.original_id.split("/")[0]
    if dataset == "AnthropicPersuasion":
        argument = (episode.extra or {}).get("argument_id")
        if argument:
            return str(argument)
        return episode.source.original_id
    return episode.source.original_id


def _proposition_for(episode: Episode) -> Optional[str]:
    """Return the proposition to show, or None when the episode is excluded."""
    if episode.source.dataset == "PersuasionTrace" and episode.metadata.control:
        measured = (episode.extra or {}).get("measured_proposition")
        if not measured:
            return None
        return str(measured)
    return episode.proposition.text


def _post_has_delta_trace(proposition: str, context: Optional[str]) -> bool:
    blob = f"{proposition}\n{context or ''}"
    return bool(_DELTA_MARK.search(blob) or _EDIT_AND_DELTA.search(blob))


class InitialStanceDataLoader:
    """Build stage-1 initial-stance examples from participant episodes."""

    def __init__(
        self,
        *,
        repo_root: PathLike,
        datasets: Optional[Sequence[str]] = None,
        splits: Optional[Sequence[str]] = None,
        max_examples: Optional[int] = None,
        drop_context: bool = False,
        ensure_alternation: bool = False,
        paths: Optional[dict[str, Path]] = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.datasets = list(datasets) if datasets is not None else list(SCALE_STEP) + [
            "CMV",
            "PersuasionTrace",
        ]
        # Preserve a stable dataset order matching the config when possible.
        self.splits = list(splits) if splits is not None else ["train", "val", "test"]
        self.max_examples = max_examples
        self.drop_context = drop_context
        self.ensure_alternation = ensure_alternation
        self.paths = paths or {}
        self.exclusions: list[dict[str, Any]] = []
        self._examples: list[InitialStanceExample] = []

    @classmethod
    def from_config(
        cls,
        config_path: PathLike,
        *,
        repo_root: Optional[PathLike] = None,
        datasets: Optional[Sequence[str]] = None,
        splits: Optional[Sequence[str]] = None,
        max_examples: Optional[int] = None,
        drop_context: bool = False,
        ensure_alternation: bool = False,
    ) -> "InitialStanceDataLoader":
        path = Path(config_path)
        root = Path(repo_root) if repo_root is not None else path.resolve().parents[1]
        if yaml is None:
            raise ImportError("PyYAML is required to load configs; pip install pyyaml")
        with path.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        wanted = {name.lower() for name in datasets} if datasets is not None else None
        paths: dict[str, Path] = {}
        names: list[str] = []
        for name, entry in (cfg.get("datasets") or {}).items():
            if not entry or not entry.get("enabled", True):
                continue
            if wanted is not None and name.lower() not in wanted and str(entry.get("adapter", "")).lower() not in wanted:
                continue
            data_path = Path(entry["path"])
            if not data_path.is_absolute():
                data_path = root / data_path
            paths[entry.get("adapter") or name] = data_path
            names.append(entry.get("adapter") or name)
        return cls(
            repo_root=root,
            datasets=names,
            splits=splits if splits is not None else (cfg.get("options") or {}).get("splits"),
            max_examples=max_examples,
            drop_context=drop_context,
            ensure_alternation=ensure_alternation,
            paths=paths,
        )

    def load(self) -> list[InitialStanceExample]:
        episodes = self._load_episodes()
        self.exclusions = []
        built: list[InitialStanceExample] = []
        for episode in episodes:
            example = self._build_one(episode)
            if example is not None:
                built.append(example)
        built = self._dedupe_cmv_posts(built)
        built.sort(key=lambda ex: (ex.dataset, ex.split or "", ex.episode_id))
        self._assert_group_splits(built)
        if self.max_examples is not None:
            built = built[: self.max_examples]
        self._examples = built
        return list(built)

    def _load_episodes(self) -> list[Episode]:
        loader = OpinionDynamicsDataLoader(
            repo_root=self.repo_root,
            splits=self.splits,
            exclude_control=False,
            validate=True,
        )
        for name, path in self.paths.items():
            loader.register(name, path)
        return loader.load()

    def _build_one(self, episode: Episode) -> Optional[InitialStanceExample]:
        proposition = _proposition_for(episode)
        if proposition is None:
            self._exclude(episode, "control episode has no measured proposition")
            return None
        context = None if self.drop_context else _own_initial_text(episode)
        if episode.source.dataset == "CMV" and _post_has_delta_trace(proposition, _own_initial_text(episode)):
            self._exclude(episode, "CMV post contains a delta trace")
            return None
        if episode.initial.stance_type == "binary_proxy":
            if episode.initial.stance is None:
                self._exclude(episode, "missing initial stance")
                return None
            target = float(episode.initial.stance)
            target_type = "binary_proxy"
        else:
            if episode.initial.stance is None:
                self._exclude(episode, "missing initial stance")
                return None
            target = float(episode.initial.stance)
            target_type = episode.initial.stance_type
        # Stage 1 passes upto="initial", so discussion, final stance, final
        # context and outcome are not read here.
        messages = build_messages(
            episode,
            "initial",
            proposition=proposition,
            ensure_alternation=self.ensure_alternation,
            drop_context=self.drop_context,
        )
        return InitialStanceExample(
            episode_id=episode.episode_id,
            group_id=_group_id(episode),
            dataset=episode.source.dataset,
            split=episode.source.split,
            role=episode.agent.role,
            messages=messages,
            proposition=proposition,
            initial_context=context,
            target=target,
            scale_step=SCALE_STEP.get(episode.source.dataset),
            target_type=target_type,
        )

    def _dedupe_cmv_posts(self, examples: list[InitialStanceExample]) -> list[InitialStanceExample]:
        """Keep one CMV episode per post. Other datasets are unchanged."""
        chosen: dict[str, str] = {}
        for example in examples:
            if example.dataset != "CMV":
                continue
            current = chosen.get(example.group_id)
            if current is None or example.episode_id < current:
                chosen[example.group_id] = example.episode_id
        kept: list[InitialStanceExample] = []
        for example in examples:
            if example.dataset != "CMV":
                kept.append(example)
                continue
            if example.episode_id != chosen[example.group_id]:
                self._exclude_example(example, "duplicate CMV post; kept one episode")
                continue
            kept.append(example)
        return kept

    def _exclude(self, episode: Episode, reason: str) -> None:
        self.exclusions.append({
            "episode_id": episode.episode_id,
            "dataset": episode.source.dataset,
            "reason": reason,
        })

    def _exclude_example(self, example: InitialStanceExample, reason: str) -> None:
        self.exclusions.append({
            "episode_id": example.episode_id,
            "dataset": example.dataset,
            "reason": reason,
        })

    @staticmethod
    def _assert_group_splits(examples: Sequence[InitialStanceExample]) -> None:
        splits: dict[tuple[str, str], set[Optional[str]]] = defaultdict(set)
        for example in examples:
            splits[(example.dataset, example.group_id)].add(example.split)
        leaked = {key: value for key, value in splits.items() if len(value) > 1}
        if leaked:
            sample = next(iter(leaked.items()))
            raise AssertionError(f"group_id appears in more than one split: {sample}")

    def summary(self, examples: Optional[Sequence[InitialStanceExample]] = None) -> dict[str, Any]:
        rows = list(examples) if examples is not None else self._examples
        by_dataset: dict[str, list[InitialStanceExample]] = defaultdict(list)
        for example in rows:
            by_dataset[example.dataset].append(example)
        histograms = {
            name: dict(Counter(ex.target for ex in group))
            for name, group in sorted(by_dataset.items())
        }
        return {
            "n_examples": len(rows),
            "per_dataset": {name: len(group) for name, group in sorted(by_dataset.items())},
            "per_split": dict(Counter(ex.split for ex in rows)),
            "with_initial_context": sum(ex.initial_context is not None for ex in rows),
            "without_initial_context": sum(ex.initial_context is None for ex in rows),
            "binary_proxy": sum(ex.target_type == "binary_proxy" for ex in rows),
            "target_histograms": histograms,
            "exclusions": dict(Counter(item["reason"] for item in self.exclusions)),
        }
