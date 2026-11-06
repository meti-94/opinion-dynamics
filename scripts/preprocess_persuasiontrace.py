#!/usr/bin/env python3
"""
PersuasionTrace (Moore et al. 2026, https://arxiv.org/abs/2606.05330) -> unified opinion-dynamics schema.

Input: the exported human rounds in data/PersuationTrace/results/human_target=True*/*.jsonl.
Each file is one JSON line holding a list of rounds; a round is one human *target* talking to an
LLM *persuader* (gpt-5 family) about one proposition. The target reports a belief in [0, 1]
before the dialogue, after reading every persuader message ("serial question"), and after the
dialogue. That is what makes this dataset special: we get a belief reading per turn, not only
pre/post. In the unified record these readings live in two places:

  * parties[T].opinion.trace       -> one entry per reading (after_turn_idx = the persuader turn
                                      that was just read), plus the final reading on the last turn
  * turns[i].annotations            -> the same reading attached to the turn it follows
                                      ("target_belief_after"), so per-turn agreement is visible
                                      inline; the final reading is "target_belief_final"

The change (opinion.change_100 / outcome.delta_100) is still final - initial.

Where available, the LLM-judge rhetoric scores (logos / pathos / ethos, 0-2) from
annotations/rhetoric_cluster_k2_sample.jsonl are joined onto the persuader turns as
annotations["rhetoric"].

Usage:
  python scripts/preprocess_persuasiontrace.py                     # all unique human rounds
  python scripts/preprocess_persuasiontrace.py --filter complete   # initial+final belief, >=2 turns, >=1 serial reading
  python scripts/preprocess_persuasiontrace.py --exclude-control   # drop the unrelated-topic control dialogues
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/PersuationTrace"
DEFAULT_OUTPUT = REPO_ROOT / "data/unified/persuasiontrace.jsonl"
RHETORIC_FILE = DATA_DIR / "annotations/rhetoric_cluster_k2_sample.jsonl"

DATASET_NAME = "PersuasionTrace"
DATASET_KEY = DATASET_NAME.lower()
SOURCE_URL = "https://arxiv.org/abs/2606.05330"
SOURCE_LICENSE = "MIT"
BELIEF_SCALE = [0, 1]


# --------------------------------------------------------------------------- helpers
def as_bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if v is None or v == "None":
        return None
    return str(v).strip().lower() == "true"


def as_float(v) -> float | None:
    if v is None or v == "None":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def as_int(v) -> int | None:
    f = as_float(v)
    return None if f is None else int(f)


def belief_100(v: float | None) -> float | None:
    return None if v is None else round(v * 100, 4)


def clean_text(v) -> str:
    return (v or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def assign_split(record_id: str, train: float, val: float) -> str:
    h = int(hashlib.sha1(record_id.encode("utf-8")).hexdigest(), 16) / 16**40
    return "train" if h < train else "val" if h < train + val else "test"


def round_fingerprint(rd: dict) -> str:
    """Rounds were exported more than once; identical content -> identical id."""
    # (older exports lack some keys such as persuader_id, so only use fields present everywhere)
    payload = [str(rd.get("human_target_id")), rd.get("proposition"),
               as_float(rd.get("target_initial_belief")), as_float(rd.get("target_final_belief")),
               rd.get("messages")]
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def rating(v: float | None) -> dict:
    return {
        "rating_100": belief_100(v),
        "raw": v,
        "scale": BELIEF_SCALE,
        "rating_provenance": "self_reported" if v is not None else "none",
        "statement": None,
        "statement_provenance": "missing",
        "consistency": None,
    }


# --------------------------------------------------------------------------- rhetoric annotations
def load_rhetoric(path: Path) -> dict[tuple, dict]:
    """(source_path, line_index, round_index, message_index) -> {logos, pathos, ethos, judge_model}."""
    out: dict[tuple, dict] = {}
    if not path.exists():
        return out
    model = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("type") == "meta":
                model = row.get("model")
                continue
            if row.get("type") != "annotation" or row.get("error") or not row.get("parsed"):
                continue
            t = row["target"]
            if t.get("unit_type") != "message":
                continue
            key = (t["source_path"], t["line_index"], t["round_index"], t["message_index"])
            scores = {k: row["parsed"][k]["score"] for k in ("logos", "pathos", "ethos") if k in row["parsed"]}
            scores["judge_model"] = model
            out[key] = scores
    return out


# --------------------------------------------------------------------------- conversion
def convert_round(rd: dict, source_path: str, line_index: int, round_index: int,
                  rhetoric: dict, train: float, val: float) -> dict | None:
    cond = rd.get("condition") or {}
    roles = cond.get("roles") or {}
    control = bool(as_bool(cond.get("control_dialogue")))
    supports = as_bool(rd.get("persuader_supports_proposition"))
    initial = as_float(rd.get("target_initial_belief"))
    final = as_float(rd.get("target_final_belief"))
    serial = [as_float(v) for v in (rd.get("serial_questions") or [])]
    persuader_model = roles.get("llm_persuader") if roles.get("llm_persuader") not in (None, "None") else None

    # ---- turns
    turns, n_dropped = [], 0
    persuader_turn_idx: list[int] = []  # turn idx of each persuader message, in order
    for msg_i, m in enumerate(rd.get("messages") or []):
        text = clean_text(m.get("content"))
        if not text:
            n_dropped += 1
            continue
        role = m.get("role")
        party = "AI" if role == "persuader" else "T"
        idx = len(turns)
        if party == "AI":
            persuader_turn_idx.append(idx)
        turns.append({
            "idx": idx, "round": None, "channel": "dialogue",
            "from": party, "to": ["T" if party == "AI" else "AI"],
            "reply_to": idx - 1 if idx > 0 else None,
            "text": text,
            "annotations": {"source_message_index": msg_i},
        })
    if not turns:
        return None
    # round = index of the persuader message that opened the exchange (1-based)
    rnd = 0
    for t in turns:
        if t["from"] == "AI":
            rnd += 1
        t["round"] = rnd or None

    # ---- per-turn belief readings
    trace = []
    for k, v in enumerate(serial):
        if v is None:
            continue
        if k < len(persuader_turn_idx):
            after = persuader_turn_idx[k]  # read after the k-th persuader message
        else:
            after = turns[-1]["idx"]
        turns[after]["annotations"]["target_belief_after"] = {
            "rating_100": belief_100(v), "raw": v, "measure": "serial_question"}
        trace.append({"after_turn_idx": after, "rating_100": belief_100(v), "raw": v,
                      "provenance": "self_reported"})
    last_idx = turns[-1]["idx"]
    if final is not None:
        turns[last_idx]["annotations"]["target_belief_final"] = {
            "rating_100": belief_100(final), "raw": final, "measure": "post_dialogue"}
        trace.append({"after_turn_idx": last_idx, "rating_100": belief_100(final), "raw": final,
                      "provenance": "self_reported"})

    # ---- rhetoric scores on persuader turns
    for t in turns:
        key = (source_path, line_index, round_index, t["annotations"]["source_message_index"])
        if key in rhetoric:
            t["annotations"]["rhetoric"] = rhetoric[key]

    change_100 = None if initial is None or final is None else round((final - initial) * 100, 4)
    rel = None
    if change_100 is not None and supports is not None and not control:
        rel = change_100 if supports else -change_100

    target = {
        "id": "T", "kind": "human", "role": "target", "tracked": True, "model": None,
        "persona": None, "assigned_stance": None,
        "opinion": {"initial": rating(initial), "final": rating(final), "change_100": change_100, "trace": trace},
    }
    persuader = {
        "id": "AI", "kind": "llm", "role": "persuader", "tracked": False, "model": persuader_model,
        "persona": None,
        "assigned_stance": None if supports is None else ("support" if supports else "oppose"),
        "opinion": {"initial": rating(None), "final": rating(None), "change_100": None, "trace": []},
    }

    fp = round_fingerprint(rd)
    record_id = f"{DATASET_KEY}:{fp}"
    bn = rd.get("bayesian_network") or {}
    measured = clean_text(rd.get("proposition"))
    during_raw = rd.get("proposition_during_round")
    during = None if during_raw in (None, "None", "") else clean_text(during_raw)
    # Control rounds discuss `proposition_during_round` (e.g. "Dogs are better than cats")
    # while `proposition` is the claim whose belief was rated. The dialogue proposition
    # is the discussion topic; the rated claim is kept as original_text / extra.
    if during:
        proposition = during
        original_text = measured
        provenance = "discussion_topic"
    else:
        proposition = measured
        original_text = measured
        provenance = "dataset"
    return {
        "schema_version": "1.0",
        "id": record_id,
        "split": assign_split(record_id, train, val),
        "source": {"dataset": DATASET_NAME, "original_id": fp, "license": SOURCE_LICENSE, "url": SOURCE_URL},
        "proposition": {"text": proposition, "original_text": original_text, "group_key": proposition,
                        "provenance": provenance},
        "structure": {
            "type": "human_llm_dialogue",
            "n_rounds": len(persuader_turn_idx),
            "control": control,
            "notes": (
                "LLM persuader vs human target. Target belief in the proposition is reported on a 0-1 slider "
                "before the dialogue, after reading each persuader message (serial question; stored on that "
                "persuader turn as annotations.target_belief_after and in opinion.trace) and after the dialogue "
                "(annotations.target_belief_final on the last turn). "
                + ("CONTROL: the dialogue is about proposition_during_round (used as proposition.text); "
                   "belief ratings were collected on a different measured claim (proposition.original_text)."
                   if control else
                   "Persuader was assigned to " + ("support" if supports else "oppose") + " the proposition.")
            ),
        },
        "parties": [target, persuader],
        "turns": turns,
        "outcome": {
            "changed": None if change_100 is None else change_100 != 0,
            "delta_100": {} if change_100 is None else {"T": change_100},
            "persuader_relative_delta_100": rel,
        },
        "extra": {
            "condition": {
                "llm_persuader": persuader_model,
                "control_dialogue": control,
                "participant_proposition": as_bool(cond.get("participant_proposition")),
                "proposition_source": None if cond.get("proposition_source") in (None, "None") else cond.get("proposition_source"),
                "use_audio": as_bool(cond.get("use_audio")),
                "show_transcript": as_bool(cond.get("show_transcript")),
                "turn_limit": as_int(cond.get("turn_limit")),
                "minimum_turns": as_int(cond.get("minimum_turns")),
                "node_belief_survey": as_bool(cond.get("enable_node_belief_survey")),
            },
            "proposition_during_round": during,
            "measured_proposition": measured if during else None,
            "persuader_supports_proposition": supports,
            "timed_out": as_bool(rd.get("timed_out")),
            "target_ended_round": as_bool(rd.get("target_ended_round")),
            "human_target_id": None if rd.get("human_target_id") in (None, "None") else str(rd.get("human_target_id")),
            "serial_questions_raw": serial,
            "belief_nodes": bn.get("belief_nodes"),
            "target_initial_node_beliefs": None if rd.get("target_initial_node_beliefs") in (None, "None") else rd.get("target_initial_node_beliefs"),
            "target_final_node_beliefs": None if rd.get("target_final_node_beliefs") in (None, "None") else rd.get("target_final_node_beliefs"),
            "n_dropped_empty_messages": n_dropped,
            "source_files": [f"{source_path}#line{line_index}#round{round_index}"],
        },
    }


def is_complete(rec: dict) -> bool:
    op = rec["parties"][0]["opinion"]
    return (op["initial"]["rating_100"] is not None and op["final"]["rating_100"] is not None
            and len(rec["turns"]) >= 2
            and any(t["annotations"].get("target_belief_after") for t in rec["turns"]))


def validate(rec: dict) -> None:
    assert rec["split"] in {"train", "val", "test"} and rec["proposition"]["text"]
    assert rec["turns"] and all(t["idx"] == i and t["text"] for i, t in enumerate(rec["turns"]))
    for e in rec["parties"][0]["opinion"]["trace"]:
        assert 0 <= e["rating_100"] <= 100 and 0 <= e["after_turn_idx"] < len(rec["turns"])


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--filter", choices=["all", "complete"], default="all",
                    help="complete: initial+final belief, >=2 turns and >=1 per-turn reading")
    ap.add_argument("--exclude-control", action="store_true", help="drop control (unrelated-topic) dialogues")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    files = sorted(glob.glob(str(args.input_dir / "results" / "human_target=True*" / "*.jsonl")))
    if not files:
        raise SystemExit(f"no human_target=True result files under {args.input_dir / 'results'}")
    rhetoric = load_rhetoric(args.input_dir / "annotations/rhetoric_cluster_k2_sample.jsonl")

    records: dict[str, dict] = {}
    n_rounds = n_dup = n_empty = n_filtered = 0
    for path in files:
        source_path = os.path.relpath(path, args.input_dir)
        with open(path, encoding="utf-8") as f:
            for line_index, line in enumerate(f):
                payload = json.loads(line)
                rounds = payload if isinstance(payload, list) else [payload]
                for round_index, rd in enumerate(rounds):
                    n_rounds += 1
                    rec = convert_round(rd, source_path, line_index, round_index, rhetoric,
                                        args.train_frac, args.val_frac)
                    if rec is None:
                        n_empty += 1
                        continue
                    if rec["id"] in records:  # re-exported duplicate: keep first, merge provenance/annotations
                        n_dup += 1
                        kept = records[rec["id"]]
                        kept["extra"]["source_files"].extend(rec["extra"]["source_files"])
                        for t_old, t_new in zip(kept["turns"], rec["turns"]):
                            if "rhetoric" in t_new["annotations"] and "rhetoric" not in t_old["annotations"]:
                                t_old["annotations"]["rhetoric"] = t_new["annotations"]["rhetoric"]
                        continue
                    if args.exclude_control and rec["structure"]["control"]:
                        n_filtered += 1
                        continue
                    if args.filter == "complete" and not is_complete(rec):
                        n_filtered += 1
                        continue
                    validate(rec)
                    records[rec["id"]] = rec

    out = list(records.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- summary
    n = len(out)
    splits = Counter(r["split"] for r in out)
    deltas = [r["outcome"]["delta_100"].get("T") for r in out if r["outcome"]["delta_100"]]
    rel = [r["outcome"]["persuader_relative_delta_100"] for r in out if r["outcome"]["persuader_relative_delta_100"] is not None]
    n_readings = sum(1 for r in out for t in r["turns"] if "target_belief_after" in t["annotations"])
    n_rhet = sum(1 for r in out for t in r["turns"] if "rhetoric" in t["annotations"])
    print(f"input rounds          : {n_rounds} in {len(files)} files  (duplicates {n_dup}, empty {n_empty}, filtered {n_filtered})")
    print(f"records written       : {n} -> {args.output}")
    print(f"splits                : {dict(splits)}")
    print(f"control / treatment   : {sum(r['structure']['control'] for r in out)} / {sum(not r['structure']['control'] for r in out)}")
    print(f"persuader models      : {dict(Counter(r['parties'][1]['model'] for r in out))}")
    print(f"turns                 : {sum(len(r['turns']) for r in out)}  per-turn belief readings: {n_readings}  rhetoric-annotated turns: {n_rhet}")
    print(f"propositions          : {len({r['proposition']['text'] for r in out})}")
    if deltas:
        print(f"mean delta (final-init): {sum(deltas)/len(deltas):+.2f}  mean |delta|: {sum(abs(d) for d in deltas)/len(deltas):.2f}  (0-100)")
    if rel:
        print(f"mean persuader-relative delta (treatment only): {sum(rel)/len(rel):+.2f}")


if __name__ == "__main__":
    main()
