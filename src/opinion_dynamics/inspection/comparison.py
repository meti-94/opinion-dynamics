"""Compare an original unified record against a generated episode."""

from __future__ import annotations

import json
from typing import Any, Optional

from opinion_dynamics.datasets.episode import Episode
from opinion_dynamics.inspection.inspect import _trunc


def compare_original_and_episode(
    record: dict[str, Any],
    episode: Episode,
    *,
    max_text: int = 300,
) -> str:
    pid = episode.agent.participant_id
    party = next((p for p in record.get("parties") or [] if p.get("id") == pid), None)
    lines = [
        f"=== Original vs Episode: {episode.episode_id} ===",
        "",
        "SOURCE RECORD:",
        f"  id={record.get('id')} split={record.get('split')}",
        f"  dataset={(record.get('source') or {}).get('dataset')}",
        f"  n_parties={len(record.get('parties') or [])} n_turns={len(record.get('turns') or [])}",
        f"  structure={(record.get('structure') or {}).get('type')} "
        f"control={(record.get('structure') or {}).get('control')}",
        f"  proposition: {_trunc((record.get('proposition') or {}).get('text'), max_text)}",
        "",
        "TRACKED PARTY (original):",
    ]
    if party is None:
        lines.append("  (party not found)")
    else:
        init = (party.get("opinion") or {}).get("initial") or {}
        final = (party.get("opinion") or {}).get("final") or {}
        lines.append(
            f"  id={party.get('id')} kind={party.get('kind')} role={party.get('role')} "
            f"tracked={party.get('tracked')}"
        )
        lines.append(
            f"  initial raw={init.get('raw')} rating_100={init.get('rating_100')} "
            f"stmt_prov={init.get('statement_provenance')}"
        )
        lines.append(f"  initial statement: {_trunc(init.get('statement'), max_text)}")
        lines.append(
            f"  final raw={final.get('raw')} rating_100={final.get('rating_100')} "
            f"stmt_prov={final.get('statement_provenance')}"
        )
        lines.append(f"  final statement: {_trunc(final.get('statement'), max_text)}")
        trace = (party.get("opinion") or {}).get("trace") or []
        lines.append(f"  opinion.trace length={len(trace)}")

    lines.extend([
        "",
        "GENERATED EPISODE:",
        f"  agent={episode.agent.participant_id}",
        f"  initial stance={episode.initial.stance} "
        f"(raw={episode.initial.raw_rating}, type={episode.initial.stance_type})",
        f"  final stance={episode.final.stance} "
        f"(raw={episode.final.raw_rating}, type={episode.final.stance_type})",
        f"  visible_turns={len(episode.discussion.visible_turns)} / "
        f"original_turns={len(record.get('turns') or [])}",
        f"  excluded_turns={len(episode.discussion.excluded_turns)}",
        f"  outcome.changed={episode.outcome.changed} "
        f"change_100={episode.outcome.change_100}",
        "",
        "TURN COVERAGE:",
    ])

    original_idxs = {t.get("idx") for t in (record.get("turns") or [])}
    visible_idxs = {t.idx for t in episode.discussion.visible_turns}
    excluded_idxs = {e.get("idx") for e in episode.discussion.excluded_turns}
    lines.append(f"  original idxs : {sorted(original_idxs)}")
    lines.append(f"  visible idxs  : {sorted(visible_idxs)}")
    lines.append(f"  excluded idxs : {sorted(x for x in excluded_idxs if x is not None)}")
    missing = original_idxs - visible_idxs - excluded_idxs
    if missing:
        lines.append(f"  WARNING unaccounted idxs: {sorted(missing)}")
    else:
        lines.append("  All original turns are either visible or explicitly excluded.")
    return "\n".join(lines)
