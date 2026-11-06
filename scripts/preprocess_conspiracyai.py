#!/usr/bin/env python3
"""Convert the ConspiracyAI (Costello, Pennycook & Rand 2024) dialogues into the
unified opinion-dynamics schema (schema_version 1.0).

Source: the original Aug-2024 Dryad release ``Data 8.28.24/AllDataForPublication.PPI.8.28.24.csv``
(the Sept-2026 "correction" release strips all free text and is therefore not usable here).

Each output record is one *Treatment* participant who completed a 3-round dialogue
with GPT-4 Turbo about a conspiracy they described, with a self-reported belief
rating (0-100) before and after the dialogue.

Usage:
    python scripts/preprocess_conspiracyai.py                 # 'complete' filter (~1,749 rows)
    python scripts/preprocess_conspiracyai.py --filter valid  # also drop invalid-conspiracy / bad-summary / attention-fail rows
    python scripts/preprocess_conspiracyai.py --filter analysis  # only the authors' published analysis sample
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data/ConspiracyAI/Data 8.28.24/AllDataForPublication.PPI.8.28.24.csv"
DEFAULT_FILTERED = REPO_ROOT / "data/ConspiracyAI/Data 8.28.24/AllDataFiltered.Cluster.PPI.8.28.24.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data/unified/conspiracyai.jsonl"

DATASET_NAME = "ConspiracyAI"
DATASET_KEY = DATASET_NAME.lower()
SOURCE_URL = "https://doi.org/10.5061/dryad.v6wwpzh4h"
SOURCE_LICENSE = "CC0-1.0"
AI_MODEL = "gpt-4-turbo"  # paper: GPT-4 Turbo

# The CSV was written by R: only the literal "NA" token (or empty) means missing.
# Do not add "None"/"null" here -- participants occasionally type those as real answers.
NA_VALUES = {"", "NA"}

# (column, party, round)
TURN_COLUMNS = [
    ("userResponse_combined", "P1", 1),
    ("GPTResponse", "AI", 1),
    ("userSecondResponse", "P1", 2),
    ("GPTResponse2", "AI", 2),
    ("userThirdResponse", "P1", 3),
    ("GPTResponse3", "AI", 3),
]

STUDY_LABEL = {"1": "1", "2": "2a", "3": "2b"}

# "The assertion is that X" / "The individual asserts that X" -> "X"
WRAPPER_RE = re.compile(
    r"^(?:The|This)\s+"
    r"(?:assertion|statement|claim|theory|belief|conspiracy theory|argument|"
    r"individual|participant|person|user|speaker|respondent|writer)\s+"
    r"(?:is|asserts|posits|states|suggests|claims|believes|argues|holds|expresses|"
    r"contends|proposes|maintains|revolves around the idea|centers on the idea|"
    r"is making the claim)\s+that\s+",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- helpers
def is_na(v: str | None) -> bool:
    return v is None or v.strip() in NA_VALUES


def to_float(v: str | None) -> float | None:
    if is_na(v):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def to_int(v: str | None) -> int | None:
    f = to_float(v)
    return None if f is None else int(round(f))


def flag(v: str | None) -> bool | None:
    """Binary 0/1 column -> bool (None if NA)."""
    if is_na(v):
        return None
    return v.strip() == "1"


def clean_text(v: str) -> str:
    return v.replace("\r\n", "\n").replace("\r", "\n").strip()


def assign_split(record_id: str, train: float, val: float) -> str:
    """Deterministic hash-based split so re-runs are stable."""
    h = int(hashlib.sha1(record_id.encode("utf-8")).hexdigest(), 16) / 16**40
    if h < train:
        return "train"
    if h < train + val:
        return "val"
    return "test"


def age_bucket(age_years: str | None) -> str | None:
    a = to_int(age_years)
    if a is None:
        return None
    if a < 30:
        return "18-29"
    if a < 45:
        return "30-44"
    if a < 60:
        return "45-59"
    return "60+"


def cluster_label(v: str | None) -> str | None:
    """'JFK  (15.14%)' -> 'JFK'."""
    if is_na(v):
        return None
    return re.sub(r"\s*\([\d.]+%\)\s*$", "", v).strip()


def slugify(text: str, max_words: int = 8) -> str:
    words = re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
    return "-".join(words[:max_words]) or "unknown"


def normalise_proposition(restatement: str) -> tuple[str, str]:
    """Return (declarative text, provenance). Strips meta wrappers such as
    'The assertion is that ...' when it is safe to do so."""
    m = WRAPPER_RE.match(restatement)
    if m:
        body = restatement[m.end():].strip()
        if len(body.split()) >= 3:
            body = body[0].upper() + body[1:]
            return body, "rewritten"
    return restatement, "dataset"


# --------------------------------------------------------------------------- filtering
def passes_filter(row: dict, mode: str, analysis_ids: set[tuple[str, str]]) -> bool:
    if row.get("ExperimentalCondition") != "Treatment":
        return False
    if any(is_na(row.get(col)) for col, _, _ in TURN_COLUMNS):
        return False
    if to_float(row.get("Pre_Belief_Specific")) is None or to_float(row.get("Post_Belief_Specific")) is None:
        return False
    if mode == "complete":
        return True
    if mode == "analysis":
        return (row["StudyNumber"], row["X"]) in analysis_ids
    if mode == "valid":
        if flag(row.get("ProvidedInvalidConspiracy")) is True:
            return False
        if flag(row.get("AIProvidedInaccurateSummary")) is True:
            return False
        if flag(row.get("FailedAttentionCheck")) is True:
            return False
        return True
    raise ValueError(mode)


# --------------------------------------------------------------------------- conversion
def convert_row(row: dict, analysis_ids: set[tuple[str, str]], train: float, val: float) -> dict:
    study = row["StudyNumber"]
    original_id = f"s{study}-{row['X']}"
    record_id = f"{DATASET_KEY}:{original_id}"

    pre = to_float(row["Pre_Belief_Specific"])
    post = to_float(row["Post_Belief_Specific"])
    delta = post - pre

    restatement = clean_text(row["conRestatement"])
    prop_text, prop_prov = normalise_proposition(restatement)
    cluster = cluster_label(row.get("dbscan_cluster"))
    group_key = cluster if cluster and cluster.lower() != "not classified" else slugify(prop_text)

    turns = []
    for idx, (col, party, rnd) in enumerate(TURN_COLUMNS):
        turns.append(
            {
                "idx": idx,
                "round": rnd,
                "channel": "dialogue",
                "from": party,
                "to": ["AI" if party == "P1" else "P1"],
                "reply_to": idx - 1 if idx > 0 else None,
                "text": clean_text(row[col]),
                "annotations": {},
            }
        )
    last_idx = turns[-1]["idx"]

    persona = {
        "age_bucket": age_bucket(row.get("AgeYears")),
        "gender": None if is_na(row.get("GenderCat")) else row["GenderCat"],
        "party": None if is_na(row.get("PartyCat")) else row["PartyCat"],
        "education": None if is_na(row.get("Education_Cat")) else row["Education_Cat"],
    }
    if all(v is None for v in persona.values()):
        persona = None

    def rating(value: float | None, statement: str | None, statement_prov: str) -> dict:
        return {
            "rating_100": value,
            "raw": value,
            "scale": [0, 100],
            "rating_provenance": "self_reported" if value is not None else "none",
            "statement": statement,
            "statement_provenance": statement_prov,
            "consistency": None,
        }

    participant = {
        "id": "P1",
        "kind": "human",
        "role": "participant",
        "tracked": True,
        "model": None,
        "persona": persona,
        "assigned_stance": None,
        "opinion": {
            "initial": rating(pre, turns[0]["text"], "human_turn"),
            "final": rating(post, None, "missing"),
            "change_100": delta,
            "trace": [
                {"after_turn_idx": last_idx, "rating_100": post, "raw": post, "provenance": "self_reported"}
            ],
        },
    }
    ai = {
        "id": "AI",
        "kind": "llm",
        "role": "persuader",
        "tracked": False,
        "model": AI_MODEL,
        "persona": None,
        "assigned_stance": "oppose",
        "opinion": {
            "initial": rating(None, None, "missing"),
            "final": rating(None, None, "missing"),
            "change_100": None,
            "trace": [],
        },
    }

    follow_up_10d = to_float(row.get("followUp_SpecificConspiracyDV_1"))
    follow_up_2mo = to_float(row.get("followUp2_SpecificConspiracyDV_1"))

    extra = {
        "study": STUDY_LABEL.get(study, study),
        "study_number_raw": study,
        "topic_cluster": cluster,
        "follow_up": {
            "rating_100_10day": follow_up_10d,
            "rating_100_2month": follow_up_2mo,
        },
        "general_conspiracy_belief": {  # BCTI mean, 0-100
            "pre": to_float(row.get("Pre_Belief_General")),
            "post": to_float(row.get("Post_Belief_General")),
        },
        "importance_0_8": to_int(row.get("Importance")),
        "sureness_pre": to_float(row.get("Sureness_1")),
        "gpt_plausibility_rating_1_5": to_int(row.get("GPT_CoT_PlausibilityRating")),
        "quality": {
            "provided_valid_conspiracy": flag(row.get("ProvidedValidConspiracy")),
            "ai_summary_inaccurate": flag(row.get("AIProvidedInaccurateSummary")),
            "failed_attention_check": flag(row.get("FailedAttentionCheck")),
            "low_confidence_pre_lt_50": flag(row.get("LowConfidence")),
            "exclusion_reason_screening": None if is_na(row.get("exc1")) else row["exc1"],
            "in_authors_analysis_sample": (study, row["X"]) in analysis_ids,
        },
        "word_counts": {
            "P1": [to_int(row.get(f"ParticipantResponseWordCount_r{i}")) for i in (1, 2, 3)],
            "AI": [to_int(row.get(f"GPTResponseWordCount_r{i}")) for i in (1, 2, 3)],
        },
    }

    return {
        "schema_version": "1.0",
        "id": record_id,
        "split": assign_split(record_id, train, val),
        "source": {
            "dataset": DATASET_NAME,
            "original_id": original_id,
            "license": SOURCE_LICENSE,
            "url": SOURCE_URL,
        },
        "proposition": {
            "text": prop_text,
            "original_text": restatement,
            "group_key": group_key,
            "provenance": prop_prov,
        },
        "structure": {
            "type": "human_llm_dialogue",
            "n_rounds": 3,
            "control": False,
            "notes": (
                "Participant's open-ended description of the conspiracy (turn 0) was shown to "
                "GPT-4 Turbo, which was instructed to persuade them against it over 3 rounds. "
                "Belief rated 0-100 before turn 0 and after the final turn."
            ),
        },
        "parties": [participant, ai],
        "turns": turns,
        "outcome": {
            "changed": delta != 0,
            "delta_100": {"P1": delta},
            "persuader_relative_delta_100": -delta,  # AI opposes the proposition
        },
        "extra": extra,
    }


# --------------------------------------------------------------------------- validation
def validate(rec: dict) -> None:
    assert rec["id"].startswith(f"{DATASET_KEY}:")
    assert rec["split"] in {"train", "val", "test"}
    assert rec["proposition"]["text"].strip()
    assert len(rec["turns"]) == 6
    for i, t in enumerate(rec["turns"]):
        assert t["idx"] == i and t["text"].strip(), rec["id"]
    p1 = rec["parties"][0]["opinion"]
    assert 0 <= p1["initial"]["rating_100"] <= 100 and 0 <= p1["final"]["rating_100"] <= 100, rec["id"]


# --------------------------------------------------------------------------- main
def load_analysis_ids(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        print(f"[warn] filtered analysis file not found: {path}", file=sys.stderr)
        return set()
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return {(r["StudyNumber"], r["X"]) for r in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--filtered-input", type=Path, default=DEFAULT_FILTERED,
                    help="Authors' analysis-sample CSV, used for --filter analysis and the in_authors_analysis_sample flag")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--filter", choices=["complete", "valid", "analysis"], default="complete",
                    help="complete: full 3-round transcript + pre/post rating (default); "
                         "valid: additionally drop invalid-conspiracy, inaccurate-summary and attention-check failures; "
                         "analysis: only rows in the authors' published analysis sample")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    analysis_ids = load_analysis_ids(args.filtered_input)

    with args.input.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))

    records = []
    seen = set()
    for row in rows:
        if not passes_filter(row, args.filter, analysis_ids):
            continue
        rec = convert_row(row, analysis_ids, args.train_frac, args.val_frac)
        assert rec["id"] not in seen, f"duplicate id {rec['id']}"
        seen.add(rec["id"])
        validate(rec)
        records.append(rec)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # summary
    n = len(records)
    splits = {s: sum(r["split"] == s for r in records) for s in ("train", "val", "test")}
    studies = {}
    for r in records:
        studies[r["extra"]["study"]] = studies.get(r["extra"]["study"], 0) + 1
    deltas = [r["outcome"]["delta_100"]["P1"] for r in records]
    mean_delta = sum(deltas) / n if n else float("nan")
    rewritten = sum(r["proposition"]["provenance"] == "rewritten" for r in records)
    valid = sum(r["extra"]["quality"]["provided_valid_conspiracy"] is True for r in records)
    in_analysis = sum(r["extra"]["quality"]["in_authors_analysis_sample"] for r in records)

    print(f"input rows            : {len(rows)}")
    print(f"filter                : {args.filter}")
    print(f"records written       : {n} -> {args.output}")
    print(f"splits                : {splits}")
    print(f"by study              : {dict(sorted(studies.items()))}")
    print(f"mean belief change    : {mean_delta:+.2f} (0-100 scale)")
    print(f"propositions rewritten: {rewritten}")
    print(f"flagged valid consp.  : {valid}")
    print(f"in authors' analysis  : {in_analysis}")


if __name__ == "__main__":
    main()
