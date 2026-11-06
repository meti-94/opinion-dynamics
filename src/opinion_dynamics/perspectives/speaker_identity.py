"""Dynamic SELF/OTHER speaker identity assignment relative to a simulated participant."""

from __future__ import annotations

from typing import Optional

from opinion_dynamics.datasets.episode import SpeakerRole


def speaker_role_for(agent_id: str, speaker_id: Optional[str]) -> str:
    """Assign self/other/unknown relative to the simulated participant.

    Original speaker IDs are never rewritten; only the relative role changes.
    """
    if speaker_id is None or speaker_id == "":
        return SpeakerRole.UNKNOWN.value
    if speaker_id == agent_id:
        return SpeakerRole.SELF.value
    return SpeakerRole.OTHER.value


def party_kind_lookup(parties: list[dict]) -> dict[str, Optional[str]]:
    """Map party id -> kind (human/llm)."""
    return {p.get("id"): p.get("kind") for p in parties if p.get("id")}
