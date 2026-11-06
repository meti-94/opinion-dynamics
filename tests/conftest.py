"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
UNIFIED = REPO_ROOT / "data" / "unified"


def _first_record(name: str, predicate=None) -> dict:
    path = UNIFIED / f"{name}.jsonl"
    with path.open() as fh:
        for line in fh:
            record = json.loads(line)
            if predicate is None or predicate(record):
                return record
    raise RuntimeError(f"No matching record in {path}")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def debate_record() -> dict:
    return _first_record("debate", lambda r: all(t.get("from") for t in r["turns"]))


@pytest.fixture(scope="session")
def cmv_success() -> dict:
    return _first_record("cmv", lambda r: r["outcome"]["changed"] is True)


@pytest.fixture(scope="session")
def cmv_fail() -> dict:
    return _first_record("cmv", lambda r: r["outcome"]["changed"] is False)


@pytest.fixture(scope="session")
def persuasion_control() -> dict:
    return _first_record("persuasiontrace", lambda r: r["structure"]["control"] is True)


@pytest.fixture(scope="session")
def persuasion_main() -> dict:
    return _first_record("persuasiontrace", lambda r: r["structure"]["control"] is False)


@pytest.fixture(scope="session")
def conspiracy_record() -> dict:
    return _first_record("conspiracyai")


@pytest.fixture(scope="session")
def anthropic_ai() -> dict:
    return _first_record(
        "anthropicpersuasion",
        lambda r: any(p.get("id") == "AI" for p in r["parties"]),
    )


@pytest.fixture(scope="session")
def anthropic_human() -> dict:
    return _first_record(
        "anthropicpersuasion",
        lambda r: any(p.get("id") == "P1" and not p.get("tracked") for p in r["parties"]),
    )


@pytest.fixture(scope="session")
def debategpt_hh() -> dict:
    return _first_record(
        "debategpt",
        lambda r: sum(1 for p in r["parties"] if p.get("tracked")) == 2,
    )


@pytest.fixture(scope="session")
def debategpt_hai() -> dict:
    return _first_record(
        "debategpt",
        lambda r: any(p.get("id") == "AI" for p in r["parties"]),
    )
