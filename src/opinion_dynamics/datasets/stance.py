"""Stance normalization helpers."""

from __future__ import annotations

from typing import Optional, Sequence, Union

Number = Union[int, float]


def normalize_rating(
    raw_rating: Optional[Number],
    scale: Optional[Sequence[Number]],
) -> Optional[float]:
    """Map a raw rating onto 0–100 using linear scaling.

    Returns None when the rating or scale is missing, or when the scale is degenerate.
    Never fabricates values.
    """
    if raw_rating is None or scale is None or len(scale) != 2:
        return None
    min_value, max_value = float(scale[0]), float(scale[1])
    if max_value == min_value:
        return None
    return ((float(raw_rating) - min_value) / (max_value - min_value)) * 100.0


def change_100(
    initial: Optional[Number],
    final: Optional[Number],
) -> Optional[float]:
    if initial is None or final is None:
        return None
    return float(final) - float(initial)


def stance_changed(
    initial: Optional[Number],
    final: Optional[Number],
    *,
    explicit: Optional[bool] = None,
) -> Optional[bool]:
    """Prefer an explicit dataset label when provided; otherwise compare numeric stances."""
    if explicit is not None:
        return bool(explicit)
    if initial is None or final is None:
        return None
    return float(final) != float(initial)
