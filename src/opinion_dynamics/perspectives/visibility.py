"""Visibility policies for participant-specific information filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol


@dataclass
class VisibilityDecision:
    visible: bool
    reason: str


class VisibilityPolicy(Protocol):
    name: str

    def is_visible(self, turn: dict[str, Any], participant_id: str) -> VisibilityDecision:
        ...


@dataclass
class PublicAndPrivatePolicy:
    """DEBATE-style policy.

    - Public channel: visible to all participants in the record.
    - Private channel: visible only to sender and intended recipients.
    - Private turns with missing sender and empty recipients are invisible to everyone.
    """

    name: str = "public_and_private"

    def is_visible(self, turn: dict[str, Any], participant_id: str) -> VisibilityDecision:
        channel = (turn.get("channel") or "").lower()
        sender = turn.get("from")
        recipients = list(turn.get("to") or [])

        if channel == "public":
            return VisibilityDecision(True, "public channel visible to all participants")

        if channel == "private":
            if sender is None and not recipients:
                return VisibilityDecision(
                    False,
                    "private turn missing speaker and recipients; excluded",
                )
            if sender == participant_id or participant_id in recipients:
                return VisibilityDecision(
                    True,
                    "private turn visible to sender or intended recipient",
                )
            return VisibilityDecision(
                False,
                "private turn not addressed to this participant",
            )

        # Fallback: treat unknown channels as fully visible.
        return VisibilityDecision(True, f"channel '{channel or 'unknown'}' assumed fully visible")


@dataclass
class FullHistoryPolicy:
    """All turns in the record are visible to the tracked participant."""

    name: str = "full_history"

    def is_visible(self, turn: dict[str, Any], participant_id: str) -> VisibilityDecision:
        return VisibilityDecision(True, "full conversation history assumed visible")


@dataclass
class OneShotArgumentPolicy:
    """AnthropicPersuasion: the single written argument is visible to the reader."""

    name: str = "one_shot_argument"

    def is_visible(self, turn: dict[str, Any], participant_id: str) -> VisibilityDecision:
        return VisibilityDecision(True, "one-shot argument visible to the reader")


def filter_visible_turns(
    turns: list[dict[str, Any]],
    participant_id: str,
    policy: VisibilityPolicy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split turns into visible and excluded lists, preserving original order."""
    visible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for turn in turns:
        decision = policy.is_visible(turn, participant_id)
        if decision.visible:
            visible.append(turn)
        else:
            excluded.append({
                "idx": turn.get("idx"),
                "round": turn.get("round"),
                "channel": turn.get("channel"),
                "from": turn.get("from"),
                "to": list(turn.get("to") or []),
                "reason": decision.reason,
                "text_preview": (turn.get("text") or "")[:120],
            })
    return visible, excluded
