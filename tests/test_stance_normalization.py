"""Stance normalization unit tests."""

from opinion_dynamics.datasets.stance import change_100, normalize_rating, stance_changed


def test_normalize_likert_1_6() -> None:
    assert normalize_rating(1, [1, 6]) == 0.0
    assert normalize_rating(6, [1, 6]) == 100.0
    assert abs(normalize_rating(4, [1, 6]) - 60.0) < 1e-9


def test_normalize_0_1() -> None:
    assert normalize_rating(0.0, [0, 1]) == 0.0
    assert normalize_rating(1.0, [0, 1]) == 100.0
    assert abs(normalize_rating(0.6625, [0, 1]) - 66.25) < 1e-9


def test_normalize_missing() -> None:
    assert normalize_rating(None, [1, 6]) is None
    assert normalize_rating(3, None) is None
    assert normalize_rating(3, [5, 5]) is None


def test_change_and_changed() -> None:
    assert change_100(20, 50) == 30
    assert change_100(None, 50) is None
    assert stance_changed(20, 50) is True
    assert stance_changed(20, 20) is False
    assert stance_changed(None, 20) is None
    assert stance_changed(20, 20, explicit=True) is True
