"""Shared helpers for Hugging Face-compatible chat formatters."""

from __future__ import annotations

from typing import Optional

from opinion_dynamics.datasets.episode import ContextBlock, Episode, VisibleTurn


def format_speaker_tag(turn: VisibleTurn) -> str:
    role = (turn.speaker_role or "unknown").upper()
    speaker = turn.speaker_id or "UNKNOWN"
    return f"[{role}: {speaker}]"


def format_context_block(label: str, context: Optional[ContextBlock]) -> str:
    if context is None or not context.text:
        return f"{label}:\n(none)"
    role = (context.speaker_role or "unknown").upper()
    speaker = context.speaker_id or "UNKNOWN"
    return f"{label}:\n[{role}: {speaker}]\n{context.text}"


def format_discussion(turns: list[VisibleTurn]) -> str:
    if not turns:
        return "DISCUSSION:\n(none)"
    lines = ["DISCUSSION:"]
    for turn in turns:
        lines.append("")
        lines.append(format_speaker_tag(turn))
        if turn.channel:
            meta = f"(channel={turn.channel}"
            if turn.round is not None:
                meta += f", round={turn.round}"
            meta += ")"
            lines.append(meta)
        lines.append(turn.text)
    return "\n".join(lines)


def system_prompt_for(episode: Episode) -> str:
    pid = episode.agent.participant_id
    return (
        f"You are simulating participant {pid}.\n\n"
        f"SELF refers to {pid}.\n"
        "OTHER refers to all other participants.\n\n"
        "Preserve the distinction between participants.\n"
        f"Only use information available to {pid}."
    )


class BaseFormatter:
    def render_proposition(self, episode: Episode) -> str:
        return f"PROPOSITION:\n{episode.proposition.text}"
