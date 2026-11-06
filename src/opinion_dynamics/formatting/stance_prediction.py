"""Stance-prediction chat formatter.

Historical conversation remains contextual evidence inside the user message.
The assistant completion (final stance) is reserved for supervised targets and
is not included in the returned messages by default.
"""

from __future__ import annotations

from typing import Any, Optional

from opinion_dynamics.datasets.episode import Episode
from opinion_dynamics.formatting.base import (
    BaseFormatter,
    format_context_block,
    format_discussion,
    system_prompt_for,
)


class StancePredictionFormatter(BaseFormatter):
    def format(
        self,
        episode: Episode,
        *,
        include_assistant_target: bool = False,
    ) -> list[dict[str, str]]:
        pid = episode.agent.participant_id
        initial_stance = episode.initial.stance
        stance_line = (
            "INITIAL STANCE:\n(none)"
            if initial_stance is None
            else (
                f"INITIAL STANCE:\n{initial_stance}"
                + (
                    f" (type={episode.initial.stance_type}, "
                    f"provenance={episode.initial.stance_provenance})"
                )
            )
        )

        user_content = "\n\n".join([
            self.render_proposition(episode),
            stance_line,
            format_context_block("INITIAL CONTEXT", episode.initial.context),
            format_discussion(episode.discussion.visible_turns),
            f"TASK:\nEstimate the final stance of {pid}.",
        ])

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt_for(episode)},
            {"role": "user", "content": user_content},
        ]
        if include_assistant_target and episode.final.stance is not None:
            messages.append({
                "role": "assistant",
                "content": str(episode.final.stance),
            })
        return messages

    def target(self, episode: Episode) -> Optional[float]:
        return episode.final.stance
