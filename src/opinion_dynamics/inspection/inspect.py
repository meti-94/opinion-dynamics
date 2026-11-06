"""Human-readable episode inspection utilities."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from opinion_dynamics.datasets.episode import Episode


def _trunc(text: Optional[str], n: int = 400) -> str:
    if text is None:
        return "(none)"
    text = text.strip()
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def inspect_episode(episode: Episode, *, max_text: int = 400) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"EPISODE  : {episode.episode_id}")
    lines.append(f"SOURCE   : {episode.source.dataset} / {episode.source.original_id}")
    lines.append(f"SPLIT    : {episode.source.split}")
    lines.append(f"LICENSE  : {episode.source.license}")
    lines.append("")
    lines.append(f"AGENT    : {episode.agent.participant_id} "
                 f"(kind={episode.agent.kind}, role={episode.agent.role})")
    lines.append("")
    lines.append("PROPOSITION:")
    lines.append(f"  text       : {_trunc(episode.proposition.text, max_text)}")
    lines.append(f"  provenance : {episode.proposition.provenance}")
    lines.append("")
    lines.append("INITIAL:")
    lines.append(
        f"  stance={episode.initial.stance} raw={episode.initial.raw_rating} "
        f"scale={episode.initial.rating_scale} type={episode.initial.stance_type} "
        f"prov={episode.initial.stance_provenance}"
    )
    ctx = episode.initial.context
    if ctx:
        lines.append(
            f"  context type={ctx.type} speaker={ctx.speaker_id}/{ctx.speaker_role} "
            f"source={ctx.source}"
        )
        lines.append(f"  context text: {_trunc(ctx.text, max_text)}")
    else:
        lines.append("  context: (none)")
    lines.append("")
    lines.append(f"DISCUSSION ({len(episode.discussion.visible_turns)} visible turns):")
    for turn in episode.discussion.visible_turns:
        lines.append(
            f"  [{turn.idx}] round={turn.round} channel={turn.channel} "
            f"{turn.speaker_id} ({turn.speaker_role}/{turn.speaker_kind}) "
            f"-> {turn.recipient} reply_to={turn.reply_to}"
        )
        lines.append(f"      {_trunc(turn.text, max_text)}")
    if episode.discussion.excluded_turns:
        lines.append("")
        lines.append(f"EXCLUDED ({len(episode.discussion.excluded_turns)}):")
        for ex in episode.discussion.excluded_turns:
            lines.append(
                f"  [{ex.get('idx')}] channel={ex.get('channel')} "
                f"{ex.get('from')}->{ex.get('to')} :: {ex.get('reason')}"
            )
    lines.append("")
    lines.append("FINAL:")
    lines.append(
        f"  stance={episode.final.stance} raw={episode.final.raw_rating} "
        f"scale={episode.final.rating_scale} type={episode.final.stance_type} "
        f"prov={episode.final.stance_provenance}"
    )
    fctx = episode.final.context
    if fctx:
        lines.append(
            f"  context type={fctx.type} speaker={fctx.speaker_id}/{fctx.speaker_role} "
            f"source={fctx.source}"
        )
        lines.append(f"  context text: {_trunc(fctx.text, max_text)}")
    else:
        lines.append("  context: (none)")
    lines.append("")
    lines.append(
        f"OUTCOME: changed={episode.outcome.changed} "
        f"change_100={episode.outcome.change_100} "
        f"persuader_rel={episode.outcome.persuader_relative_delta_100}"
    )
    lines.append("")
    lines.append("METADATA:")
    lines.append(
        f"  structure={episode.metadata.structure_type} "
        f"visibility={episode.metadata.visibility_policy} "
        f"filtered={episode.metadata.visibility_filtered} "
        f"control={episode.metadata.control}"
    )
    if episode.metadata.assumptions:
        lines.append("ASSUMPTIONS:")
        for a in episode.metadata.assumptions:
            lines.append(f"  - {a}")
    if episode.metadata.limitations:
        lines.append("LIMITATIONS:")
        for lim in episode.metadata.limitations:
            lines.append(f"  - {lim}")
    lines.append("=" * 72)
    return "\n".join(lines)


def inspect_perspective(episodes: Iterable[Episode], *, max_text: int = 200) -> str:
    """Compare the same original record from multiple participants' perspectives."""
    eps = list(episodes)
    if not eps:
        return "(no episodes)"
    lines = [
        f"=== Perspective comparison for {eps[0].source.dataset}:{eps[0].source.original_id} ===",
        f"{len(eps)} participant episode(s)",
        "",
    ]
    for ep in eps:
        lines.append(f"--- Agent {ep.agent.participant_id} ({ep.episode_id}) ---")
        lines.append(
            f"  initial={ep.initial.stance} final={ep.final.stance} "
            f"visible_turns={len(ep.discussion.visible_turns)} "
            f"excluded={len(ep.discussion.excluded_turns)}"
        )
        role_counts: dict[str, int] = {}
        speakers: dict[str, str] = {}
        for t in ep.discussion.visible_turns:
            role_counts[t.speaker_role] = role_counts.get(t.speaker_role, 0) + 1
            if t.speaker_id:
                speakers[t.speaker_id] = t.speaker_role
        lines.append(f"  speaker_roles: {role_counts}")
        lines.append(f"  id->role: {speakers}")
        # Show first few turns briefly
        for t in ep.discussion.visible_turns[:5]:
            lines.append(
                f"    [{t.idx}] {t.speaker_id}/{t.speaker_role} "
                f"ch={t.channel}: {_trunc(t.text, max_text)}"
            )
        if len(ep.discussion.visible_turns) > 5:
            lines.append(f"    ... ({len(ep.discussion.visible_turns) - 5} more)")
        lines.append("")
    return "\n".join(lines)


def inspect_visibility(episode: Episode) -> str:
    lines = [
        f"=== Visibility inspection: {episode.episode_id} ===",
        f"Policy: {episode.metadata.visibility_policy}",
        f"Filtered: {episode.metadata.visibility_filtered}",
        "",
        f"INCLUDED ({len(episode.discussion.visible_turns)}):",
    ]
    for t in episode.discussion.visible_turns:
        lines.append(
            f"  + [{t.idx}] {t.channel} {t.speaker_id}->{t.recipient} "
            f"({t.speaker_role})"
        )
    lines.append("")
    lines.append(f"EXCLUDED ({len(episode.discussion.excluded_turns)}):")
    if not episode.discussion.excluded_turns:
        lines.append("  (none)")
    for ex in episode.discussion.excluded_turns:
        lines.append(
            f"  - [{ex.get('idx')}] {ex.get('channel')} "
            f"{ex.get('from')}->{ex.get('to')}: {ex.get('reason')}"
        )
    if episode.metadata.assumptions:
        lines.append("")
        lines.append("Policy assumptions:")
        for a in episode.metadata.assumptions:
            lines.append(f"  - {a}")
    return "\n".join(lines)
