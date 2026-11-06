"""OpinionDynamicsDataLoader integration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opinion_dynamics.datasets.dataloader import (
    OpinionDynamicsDataLoader,
    drop_unchanged_duplicate_posts,
)
from opinion_dynamics.datasets.episode import (
    AgentInfo,
    DiscussionBlock,
    Episode,
    EpisodeMetadata,
    OutcomeBlock,
    PropositionInfo,
    SourceInfo,
    StanceBlock,
)


@pytest.fixture(scope="module")
def small_loader(repo_root, tmp_path_factory) -> OpinionDynamicsDataLoader:
    """Loader over a tiny multi-dataset sample for fast tests."""
    tmp = tmp_path_factory.mktemp("unified_sample")
    samples = {}
    for name, n in [
        ("debate", 2),
        ("cmv", 3),
        ("persuasiontrace", 3),
        ("conspiracyai", 2),
        ("anthropicpersuasion", 2),
        ("debategpt", 3),
    ]:
        src = repo_root / "data" / "unified" / f"{name}.jsonl"
        dst = tmp / f"{name}.jsonl"
        lines = []
        with src.open() as fh:
            for i, line in enumerate(fh):
                if i >= n:
                    break
                lines.append(line)
        dst.write_text("".join(lines), encoding="utf-8")
        samples[name] = dst

    loader = OpinionDynamicsDataLoader(repo_root=repo_root, exclude_control=False)
    loader.register("DEBATE", samples["debate"])
    loader.register("CMV", samples["cmv"])
    loader.register("PersuasionTrace", samples["persuasiontrace"])
    loader.register("ConspiracyAI", samples["conspiracyai"])
    loader.register("AnthropicPersuasion", samples["anthropicpersuasion"])
    loader.register("DebateGPT", samples["debategpt"])
    loader.load()
    return loader


def test_from_config_and_load(repo_root) -> None:
    # Use config but only enable DEBATE via a temp config for speed.
    cfg = repo_root / "configs" / "datasets.yaml"
    loader = OpinionDynamicsDataLoader.from_config(cfg, repo_root=repo_root)
    # Restrict to DEBATE only by clearing others after construction.
    path = loader._paths["DEBATE"]
    loader = OpinionDynamicsDataLoader(repo_root=repo_root)
    loader.register("DEBATE", path)
    episodes = loader.load()
    assert len(episodes) == 263 * 4  # all four participants
    summary = loader.summary()
    assert summary.original_record_count == 263
    assert summary.episode_count == 1052


def test_filter_and_get(small_loader) -> None:
    debate_eps = small_loader.filter(dataset="DEBATE")
    assert debate_eps
    assert all(e.source.dataset == "DEBATE" for e in debate_eps)
    p1 = small_loader.filter(dataset="DEBATE", participant_id="P1")
    assert p1 and all(e.agent.participant_id == "P1" for e in p1)
    ep = small_loader.get_episode(p1[0].episode_id)
    assert ep.episode_id == p1[0].episode_id
    binary = small_loader.filter(stance_type="binary_proxy")
    assert binary and all(e.initial.stance_type == "binary_proxy" for e in binary)
    non_control = small_loader.filter(control=False)
    assert all(not e.metadata.control for e in non_control)


def test_inspect_methods(small_loader) -> None:
    ep = small_loader.filter(dataset="DEBATE", participant_id="P1")[0]
    text = small_loader.inspect(ep)
    assert ep.episode_id in text
    assert "PROPOSITION" in text
    persp = small_loader.inspect_perspective(ep.episode_id)
    assert "Perspective comparison" in persp
    vis = small_loader.inspect_visibility(ep.episode_id)
    assert "Visibility inspection" in vis
    cmp = small_loader.compare_original_and_episode(ep.episode_id)
    assert "Original vs Episode" in cmp


def test_export_reload_roundtrip(small_loader, tmp_path) -> None:
    out = tmp_path / "episodes.jsonl"
    small_loader.export_jsonl(out)
    reloaded = OpinionDynamicsDataLoader.load_episodes_jsonl(out)
    assert len(reloaded) == len(small_loader.episodes)
    for a, b in zip(small_loader.episodes, reloaded):
        assert a.episode_id == b.episode_id
        assert a.initial.stance == b.initial.stance
        assert a.final.stance == b.final.stance
        assert len(a.discussion.visible_turns) == len(b.discussion.visible_turns)
        assert a.to_dict() == b.to_dict()


def test_to_huggingface(small_loader) -> None:
    pytest.importorskip("datasets")
    ds = small_loader.to_huggingface(small_loader.filter(dataset="DEBATE"))
    assert len(ds) == len(small_loader.filter(dataset="DEBATE"))
    assert "episode_id" in ds.column_names
    assert "discussion" in ds.column_names


def _cmv_episode(episode_id: str, post_id: str, changed: bool) -> Episode:
    return Episode(
        episode_id=episode_id,
        source=SourceInfo(dataset="CMV", original_id=episode_id, split="train"),
        agent=AgentInfo(participant_id="OP"),
        proposition=PropositionInfo(text="t"),
        initial=StanceBlock(1.0, None, None, "binary_proxy", "dataset_outcome_proxy"),
        discussion=DiscussionBlock(),
        final=StanceBlock(0.0 if changed else 1.0, None, None, "binary_proxy", "dataset_outcome_proxy"),
        outcome=OutcomeBlock(changed=changed, change_100=-1.0 if changed else 0.0),
        metadata=EpisodeMetadata(),
        extra={"conversation_id": post_id},
    )


def test_drop_unchanged_cmv_sibling_when_post_also_changed() -> None:
    changed = _cmv_episode("cmv:post:success", "post", True)
    unchanged = _cmv_episode("cmv:post:fail", "post", False)
    only = _cmv_episode("cmv:solo:fail", "solo", False)
    kept, dropped = drop_unchanged_duplicate_posts([changed, unchanged, only])
    assert [e.episode_id for e in kept] == ["cmv:post:success", "cmv:solo:fail"]
    assert [e.episode_id for e in dropped] == ["cmv:post:fail"]


def test_exclude_control(repo_root, tmp_path) -> None:
    src = repo_root / "data" / "unified" / "persuasiontrace.jsonl"
    sample = tmp_path / "pt.jsonl"
    lines = []
    with src.open() as fh:
        for i, line in enumerate(fh):
            if i >= 20:
                break
            lines.append(line)
    sample.write_text("".join(lines))
    loader = OpinionDynamicsDataLoader(repo_root=repo_root, exclude_control=True)
    loader.register("PersuasionTrace", sample)
    eps = loader.load()
    assert all(not e.metadata.control for e in eps)
    assert any(ex["reason"].startswith("control") for ex in loader._exclusions)


def test_episode_validation_ordering() -> None:
    # Chronological validation should catch unsorted turns.
    from opinion_dynamics.datasets.episode import (
        AgentInfo, DiscussionBlock, Episode, EpisodeMetadata, OutcomeBlock,
        PropositionInfo, SourceInfo, StanceBlock, VisibleTurn,
    )
    ep = Episode(
        episode_id="x",
        source=SourceInfo(dataset="X", original_id="1"),
        agent=AgentInfo(participant_id="P1"),
        proposition=PropositionInfo(text="t"),
        initial=StanceBlock(1, 1, [1, 5], "continuous", "self_reported"),
        discussion=DiscussionBlock(visible_turns=[
            VisibleTurn(1, 1, "P1", "self", "human", "public", [], "b"),
            VisibleTurn(0, 1, "P2", "other", "human", "public", [], "a"),
        ]),
        final=StanceBlock(1, 1, [1, 5], "continuous", "self_reported"),
        outcome=OutcomeBlock(),
        metadata=EpisodeMetadata(),
    )
    with pytest.raises(ValueError, match="chronological"):
        OpinionDynamicsDataLoader.validate_episode(ep)
