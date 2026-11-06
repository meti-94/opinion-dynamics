"""Next-response-generation chat formatter.

Builds a prompt containing only turns strictly before the target SELF turn.
The target response is returned separately and never leaked into the input.
"""

from __future__ import annotations

from typing import Optional

from opinion_dynamics.datasets.episode import Episode, VisibleTurn
from opinion_dynamics.formatting.base import (
    BaseFormatter,
    format_context_block,
    format_discussion,
    system_prompt_for,
)


class ResponseGenerationFormatter(BaseFormatter):
    def format(
        self,
        episode: Episode,
        *,
        target_turn_idx: int,
    ) -> tuple[list[dict[str, str]], Optional[str]]:
        pid = episode.agent.participant_id
        turns = episode.discussion.visible_turns

        target: Optional[VisibleTurn] = None
        history: list[VisibleTurn] = []
        for turn in turns:
            if turn.idx == target_turn_idx:
                target = turn
                break
            history.append(turn)
        else:
            raise ValueError(
                f"target_turn_idx={target_turn_idx} not found in visible turns of "
                f"{episode.episode_id}"
            )

        if target.speaker_id != pid:
            raise ValueError(
                f"target_turn_idx={target_turn_idx} is spoken by {target.speaker_id}, "
                f"not the simulated participant {pid}"
            )

        # Hard guarantee: no future turns after the target.
        assert all(t.idx < target_turn_idx for t in history)

        user_content = "\n\n".join([
            self.render_proposition(episode),
            format_context_block("INITIAL CONTEXT", episode.initial.context),
            format_discussion(history),
            f"Generate {pid}'s next response.",
        ])

        messages = [
            {"role": "system", "content": system_prompt_for(episode)},
            {"role": "user", "content": user_content},
        ]
        return messages, target.text

    def candidate_target_indices(self, episode: Episode) -> list[int]:
        """Return visible turn indices authored by the simulated participant."""
        pid = episode.agent.participant_id
        return [
            t.idx
            for t in episode.discussion.visible_turns
            if t.speaker_id == pid and t.speaker_role == "self"
        ]
