"""Task-specific example builders. Stage 1 is initial-stance prediction."""

from opinion_dynamics.tasks.initial_stance import (
    InitialStanceDataLoader,
    InitialStanceExample,
    build_messages,
)

__all__ = [
    "InitialStanceDataLoader",
    "InitialStanceExample",
    "build_messages",
]
