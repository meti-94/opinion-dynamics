#!/usr/bin/env python3
"""
Anthropic "Measuring the Persuasiveness of Language Models" dataset
(https://huggingface.co/datasets/Anthropic/persuasion) -> unified opinion-dynamics schema.

Input: data/AnthropicPersuasion/persuasion_data.csv (3,939 rows). One row = one participant who
rated agreement with a claim on a 1-7 scale, read ONE written argument supporting the claim
(by a Claude model with one of four prompt types, or by a human), and rated the claim again.
Each argument was shown to 3 participants. A small control arm uses undisputed factual claims.

One output record = one (participant, argument) reading:
  proposition  = claim                      structure.type = one_shot_argument
  parties      = T (human reader, tracked)  +  persuader (AI with model name, or P1 for a human writer)
  turns        = the single argument        opinion.initial/final = 1-7 ratings -> 0-100

Usage:
  python scripts/preprocess_anthropic_persuasion.py                    # all rows
  python scripts/preprocess_anthropic_persuasion.py --exclude-control  # drop the factual-claim control arm
  python scripts/preprocess_anthropic_persuasion.py --download         # fetch the CSV from the Hub first
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/AnthropicPersuasion"
DEFAULT_INPUT = DATA_DIR / "persuasion_data.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data/unified/anthropicpersuasion.jsonl"

HF_REPO_ID = "Anthropic/persuasion"
DATASET_NAME = "AnthropicPersuasion"
DATASET_KEY = DATASET_NAME.lower()
SOURCE_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}"
SOURCE_LICENSE = "CC-BY-NC-SA-4.0"
SCALE_MIN, SCALE_MAX = 1, 7
RATING_RE = re.compile(r"^\s*(\d)\s*-\s*(.*)$")


# --------------------------------------------------------------------------- helpers
def parse_rating(v: str) -> tuple[int | None, str | None]:
    """'5 - Somewhat support' -> (5, 'Somewhat support')."""
    m = RATING_RE.match(str(v)) if isinstance(v, str) else None
    if not m:
        return None, None
    return int(m.group(1)), m.group(2).strip()


def to_100(raw: int | None) -> float | None:
    return None if raw is None else round((raw - SCALE_MIN) / (SCALE_MAX - SCALE_MIN) * 100, 4)


def assign_split(key: str, train: float, val: float) -> str:
    """Deterministic 80/10/10. Keyed on the *argument*, so the 3 readers of one argument
    (and the identical argument text) never straddle splits."""
    h = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) / 16**40
    return "train" if h < train else "val" if h < train + val else "test"


def sha(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def rating_block(raw: int | None) -> dict:
    return {
        "rating_100": to_100(raw),
        "raw": raw,
        "scale": [SCALE_MIN, SCALE_MAX],
        "rating_provenance": "self_reported" if raw is not None else "none",
        "statement": None,
        "statement_provenance": "missing",
        "consistency": None,
    }


def empty_opinion() -> dict:
    return {"initial": rating_block(None), "final": rating_block(None), "change_100": None, "trace": []}


# --------------------------------------------------------------------------- conversion
def convert_row(i: int, row: pd.Series, train: float, val: float) -> dict:
    claim = str(row["claim"]).strip()
    argument = str(row["argument"]).replace("\r\n", "\n").strip()
    source = str(row["source"]).strip()
    prompt_type = None if pd.isna(row["prompt_type"]) else str(row["prompt_type"]).strip()
    control = source.lower() == "control"
    is_human_writer = source.lower() == "human"

    init_raw, init_label = parse_rating(row["rating_initial"])
    final_raw, final_label = parse_rating(row["rating_final"])
    change_100 = None if init_raw is None or final_raw is None else round(to_100(final_raw) - to_100(init_raw), 4)
    metric = None if pd.isna(row["persuasiveness_metric"]) else int(row["persuasiveness_metric"])
    delta_raw = None if init_raw is None or final_raw is None else final_raw - init_raw

    argument_id = sha(claim + "\n" + argument)
    original_id = f"r{i:04d}"
    record_id = f"{DATASET_KEY}:{original_id}"

    persuader_id = "P1" if is_human_writer else "AI"
    target = {
        "id": "T", "kind": "human", "role": "target", "tracked": True, "model": None,
        "persona": None, "assigned_stance": None,
        "opinion": {
            "initial": rating_block(init_raw),
            "final": rating_block(final_raw),
            "change_100": change_100,
            "trace": [] if final_raw is None else [
                {"after_turn_idx": 0, "rating_100": to_100(final_raw), "raw": final_raw, "provenance": "self_reported"}
            ],
        },
    }
    persuader = {
        "id": persuader_id,
        "kind": "human" if is_human_writer else "llm",
        "role": "persuader",
        "tracked": False,
        "model": None if (is_human_writer or control) else source,
        "persona": None,
        # arguments were written in favour of the claim; control arm has no persuasive stance
        "assigned_stance": None if control else "support",
        "opinion": empty_opinion(),
    }

    turn = {
        "idx": 0, "round": 1, "channel": "dialogue",
        "from": persuader_id, "to": ["T"], "reply_to": None,
        "text": argument,
        "annotations": {"source": source, "prompt_type": prompt_type,
                        "target_belief_final": {"rating_100": to_100(final_raw), "raw": final_raw,
                                                "measure": "post_argument"}},
    }

    return {
        "schema_version": "1.0",
        "id": record_id,
        "split": assign_split(argument_id, train, val),
        "source": {"dataset": DATASET_NAME, "original_id": original_id,
                   "license": SOURCE_LICENSE, "url": SOURCE_URL},
        "proposition": {"text": claim, "original_text": claim, "group_key": claim, "provenance": "claim"},
        "structure": {
            "type": "one_shot_argument",
            "n_rounds": 1,
            "control": control,
            "notes": (
                "Participant rated agreement with the claim (1=Strongly oppose .. 7=Strongly support), read one "
                "~250-word argument in favour of the claim, then rated again. No reply from the participant. "
                + ("CONTROL arm: undisputed factual claim, used to estimate rating noise." if control else
                   f"Argument written by {'a human' if is_human_writer else source}"
                   + (f" with prompt type '{prompt_type}'." if prompt_type else "."))
            ),
        },
        "parties": [target, persuader],
        "turns": [turn],
        "outcome": {
            "changed": None if change_100 is None else change_100 != 0,
            "delta_100": {} if change_100 is None else {"T": change_100},
            "persuader_relative_delta_100": None if control or change_100 is None else change_100,
        },
        "extra": {
            "source": source,
            "prompt_type": prompt_type,
            "argument_id": argument_id,  # shared by the 3 readers of the same argument
            "worker_id": str(row["worker_id"]),
            "rating_labels": {"initial": init_label, "final": final_label},
            "delta_raw_1_7": delta_raw,
            "persuasiveness_metric_published": metric,
            "persuasiveness_metric_mismatch": (metric is not None and delta_raw is not None and metric != delta_raw),
        },
    }


def validate(rec: dict) -> None:
    assert rec["split"] in {"train", "val", "test"} and rec["proposition"]["text"]
    assert len(rec["turns"]) == 1 and rec["turns"][0]["text"]
    for k in ("initial", "final"):
        r = rec["parties"][0]["opinion"][k]["rating_100"]
        assert r is None or 0 <= r <= 100, rec["id"]


# --------------------------------------------------------------------------- main
def download(dest: Path) -> Path:
    from huggingface_hub import hf_hub_download
    hf_hub_download(HF_REPO_ID, "README.md", repo_type="dataset", local_dir=str(dest))
    return Path(hf_hub_download(HF_REPO_ID, "persuasion_data.csv", repo_type="dataset", local_dir=str(dest)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--download", action="store_true", help=f"download {HF_REPO_ID} into {DATA_DIR} first")
    ap.add_argument("--exclude-control", action="store_true", help="drop the factual-claim control arm")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    if args.download or not args.input.exists():
        download(DATA_DIR)

    df = pd.read_csv(args.input)
    records = []
    for i, row in df.iterrows():
        rec = convert_row(int(i), row, args.train_frac, args.val_frac)
        if args.exclude_control and rec["structure"]["control"]:
            continue
        validate(rec)
        records.append(rec)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(records)
    deltas = [r["outcome"]["delta_100"]["T"] for r in records if r["outcome"]["delta_100"]]
    print(f"input rows            : {len(df)}")
    print(f"records written       : {n} -> {args.output}")
    print(f"splits                : {dict(Counter(r['split'] for r in records))}  (keyed on argument)")
    print(f"sources               : {dict(Counter(r['extra']['source'] for r in records))}")
    print(f"prompt types          : {dict(Counter(r['extra']['prompt_type'] for r in records))}")
    print(f"claims / arguments    : {len({r['proposition']['text'] for r in records})} / {len({r['extra']['argument_id'] for r in records})}")
    print(f"control records       : {sum(r['structure']['control'] for r in records)}")
    print(f"metric mismatches     : {sum(r['extra']['persuasiveness_metric_mismatch'] for r in records)} (published metric != final-initial)")
    print(f"mean delta            : {sum(deltas)/len(deltas):+.2f} (0-100); changed share: {sum(d != 0 for d in deltas)/len(deltas):.1%}")


if __name__ == "__main__":
    main()
