#!/usr/bin/env python3
"""
DEBATE (https://huggingface.co/datasets/seantw/DEBATE_LLM) -> unified opinion-dynamics schema.

Builds on the two helper scripts in this folder, so parsing, usability filtering and the
train/val/test rule are exactly the same as the per-conversation JSON exports:

  show_debate_conversation.py : parse_conversation / is_usable / delta_of / list_csvs
  export_conversations_json.py: to_record (the intermediate JSON record) / split_of

Pipeline per CSV:  CSV --parse_conversation--> conv --to_record--> record --to_unified--> JSONL line

CSVs are downloaded with hf_hub_download into data/DEBATE_LLM/ (repo layout is preserved:
golden/{depth,breadth}/<topic>/*.csv, raw/...). Files already present there are reused.

Usage:
  python scripts/preprocess_debate.py                       # golden subset -> data/unified/debate.jsonl
  python scripts/preprocess_debate.py --subset raw          # every raw session
  python scripts/preprocess_debate.py --kind depth          # only depth topics
  python scripts/preprocess_debate.py --keep-incomplete     # keep groups missing an initial/final rating
  python scripts/preprocess_debate.py --json-dir data/DEBATE_LLM/json   # also dump debate_<ID>.json files
  python scripts/preprocess_debate.py --files a.csv b.csv   # local CSVs instead of the Hub listing

License reminder: DEBATE Dataset Research-Only License (Non-Commercial, v1.0).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_conversations_json import to_record, split_of  # noqa: E402
from show_debate_conversation import REPO_ID, is_usable, list_csvs, parse_conversation  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/DEBATE_LLM"
DEFAULT_OUTPUT = REPO_ROOT / "data/unified/debate.jsonl"

DATASET_NAME = "DEBATE"
SOURCE_URL = f"https://huggingface.co/datasets/{REPO_ID}"
SOURCE_LICENSE = "DEBATE Dataset Research-Only License (Non-Commercial, v1.0)"
SLIDER_MIN, SLIDER_MAX = 1, 6
PERSONA_FIELDS = {"gender": "gender", "education": "education",
                  "politicalIdentity": "party", "politicalViews": "political_views"}


# --------------------------------------------------------------------------- helpers
def rating_100(v):
    return None if v is None else round((v - SLIDER_MIN) / (SLIDER_MAX - SLIDER_MIN) * 100, 4)


def age_bucket(v):
    try:
        a = int(float(v))
    except (TypeError, ValueError):
        return None
    return "18-29" if a < 30 else "30-44" if a < 45 else "45-59" if a < 60 else "60+"


def personas_of(csv_path: str, worker_ids: dict) -> dict:
    """Coarse demographics from the exit_survey rows (not used by the reference parser)."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if "field" not in df.columns:
        return {a: None for a in worker_ids}
    ex = df[df["event_type"].astype(str).str.strip().str.lower() == "exit_survey"]
    out = {}
    for alias, w in worker_ids.items():
        rows = ex[ex["worker_id"].astype(str) == str(w)]
        answers = {str(r["field"]): str(r["text"]) for _, r in rows.iterrows() if pd.notna(r["text"])}
        persona = {"age_bucket": age_bucket(answers.get("age"))}
        persona.update({dst: answers.get(src) or None for src, dst in PERSONA_FIELDS.items()})
        out[alias] = None if all(v is None for v in persona.values()) else persona
    return out


def opinion_block(entry: dict) -> dict:
    rating = entry.get("rating")
    statement = (entry.get("statement") or "").strip() or None
    return {
        "rating_100": rating_100(rating),
        "raw": rating,
        "scale": [SLIDER_MIN, SLIDER_MAX],
        "rating_provenance": "self_reported" if rating is not None else "none",
        "statement": statement,
        "statement_provenance": "human_private" if statement else "missing",
        "consistency": None,
    }


# --------------------------------------------------------------------------- record -> unified
def to_unified(rec: dict, personas: dict) -> dict:
    """rec is the dict produced by export_conversations_json.to_record."""
    # ---- turns: per round, the public tweets first, then each private dyad in order
    turns = []
    for rnd in rec["rounds"]:
        r = rnd["round"] or None
        tweet_idx = {}
        for m in rnd["public"]:
            tweet_idx[m["from"]] = len(turns)
            turns.append({"idx": len(turns), "round": r, "channel": "public",
                          "from": m["from"], "to": [m["to"]] if m["to"] else [], "reply_to": None,
                          "text": m["text"], "annotations": {"event_type": "tweet"}})
        for dyad in rnd["private"]:
            pair = dyad["between"]
            prev = None
            for m in dyad["messages"]:
                other = next((p for p in pair if p != m["from"]), None)
                if prev is None and other in tweet_idx:
                    prev = tweet_idx[other]  # first chat message answers the partner's tweet
                turns.append({"idx": len(turns), "round": r, "channel": "private",
                              "from": m["from"], "to": [other] if other else [], "reply_to": prev,
                              "text": m["text"], "annotations": {"event_type": "message_sent"}})
                prev = len(turns) - 1
    last_idx = turns[-1]["idx"] if turns else None

    # ---- parties
    parties, deltas = [], {}
    for alias in rec["participants"]:
        ini, fin = opinion_block(rec["initial"][alias]), opinion_block(rec["final"][alias])
        change = rec["final"][alias].get("change")
        change_100 = None if change is None else round(change / (SLIDER_MAX - SLIDER_MIN) * 100, 4)
        if change_100 is not None:
            deltas[alias] = change_100
        # Per participant. The record-level flag used to be any(delta != 0), which
        # marked people who did not move as changed whenever a teammate did.
        person_changed = None if change_100 is None else change_100 != 0
        trace = []
        if fin["rating_100"] is not None and last_idx is not None:
            trace.append({"after_turn_idx": last_idx, "rating_100": fin["rating_100"],
                          "raw": fin["raw"], "provenance": "self_reported"})
        parties.append({
            "id": alias, "kind": "human", "role": "participant", "tracked": True, "model": None,
            "persona": personas.get(alias), "assigned_stance": None,
            "opinion": {"initial": ini, "final": fin, "change_100": change_100,
                        "changed": person_changed, "trace": trace},
        })

    src = rec["source_file"]
    parts = src.split("/")
    subset = parts[0] if parts[0] in ("golden", "raw") else None
    kind = parts[1] if len(parts) > 2 and parts[1] in ("depth", "breadth") else None
    return {
        "schema_version": "1.0",
        "id": f"{DATASET_NAME.lower()}:{rec['id']}",
        "split": split_of(src),
        "source": {"dataset": DATASET_NAME, "original_id": rec["id"],
                   "license": SOURCE_LICENSE, "url": SOURCE_URL},
        "proposition": {"text": rec["topic"], "original_text": rec["topic"],
                        "group_key": rec["topic"], "provenance": "dataset"},
        "structure": {
            "type": "multi_party_rounds",
            "n_rounds": len(rec["rounds"]),
            "control": False,
            "notes": ("4 human participants; private 6-point rating (" + rec["rating_scale"] +
                      ") with free-text justification before and after. Each round: one public tweet "
                      "per person (to that round's partner) followed by a private one-on-one chat; "
                      "a new partner every round. Empty/duplicated message_received rows dropped."),
        },
        "parties": parties,
        "turns": turns,
        "outcome": {
            # Not a per-person label. Who moved is opinion.changed / delta_100.
            "changed": None,
            "any_changed": any(d != 0 for d in deltas.values()) if deltas else None,
            "delta_100": deltas,
            "persuader_relative_delta_100": None,
        },
        "extra": {"subset": subset, "kind": kind, "source_file": src,
                  "rating_scale": rec["rating_scale"], "worker_ids": rec["worker_ids"]},
    }


# --------------------------------------------------------------------------- sources
def resolve_sources(args):
    """Yield (repo_name, local_path). Uses files already in data/DEBATE_LLM, downloads the rest."""
    if args.files:
        return [(f, f) for f in args.files]
    local = sorted(DATA_DIR.glob(f"{args.subset}/*/*/*.csv"))
    if local and not args.refresh:
        names = [str(p.relative_to(DATA_DIR)) for p in local]
        if args.kind != "any":
            names = [n for n in names if f"/{args.kind}/" in n]
    else:
        names = list_csvs(args.subset, args.kind, "")
    names = names[: args.max_files] if args.max_files else names
    return [(n, None) for n in names]


def fetch(name: str, local: str | None) -> str:
    if local:
        return local
    p = DATA_DIR / name
    if p.exists():
        return str(p)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO_ID, name, repo_type="dataset", local_dir=str(DATA_DIR))


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", choices=["golden", "raw"], default="golden")
    ap.add_argument("--kind", choices=["depth", "breadth", "any"], default="any")
    ap.add_argument("--max-files", type=int, default=0, help="0 = all")
    ap.add_argument("--files", nargs="*", help="local CSVs instead of the Hub listing")
    ap.add_argument("--refresh", action="store_true", help="re-list files from the Hub even if local copies exist")
    ap.add_argument("--keep-incomplete", action="store_true",
                    help="also keep groups missing an initial/final rating for someone")
    ap.add_argument("--json-dir", help="also write one debate_<ID>.json per conversation (export_conversations_json format)")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    sources = resolve_sources(args)
    if args.json_dir:
        os.makedirs(args.json_dir, exist_ok=True)

    records, skipped, seen = [], 0, set()
    for name, local in sources:
        path = fetch(name, local)
        conv = parse_conversation(path)
        conv["file"] = name
        if not args.keep_incomplete and not is_usable(conv, need_change=False):
            skipped += 1
            continue
        if args.keep_incomplete:  # same gap-filling as export_conversations_json
            for w in conv["participants"]:
                for k in ("initial", "post"):
                    conv["stance"].setdefault(w, {}).setdefault(k, {"value": None, "text": ""})
        rec = to_record(conv)
        if args.json_dir:
            with open(os.path.join(args.json_dir, f"debate_{rec['id']}.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
        uni = to_unified(rec, personas_of(path, rec["worker_ids"]))
        assert uni["id"] not in seen, f"duplicate id {uni['id']}"
        seen.add(uni["id"])
        assert all(t["text"] for t in uni["turns"]) and uni["turns"], uni["id"]
        records.append(uni)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(records)
    splits = {s: sum(r["split"] == s for r in records) for s in ("train", "val", "test")}
    kinds = {k: sum(r["extra"]["kind"] == k for r in records) for k in ("depth", "breadth")}
    deltas = [d for r in records for d in r["outcome"]["delta_100"].values()]
    n_turns = sum(len(r["turns"]) for r in records)
    n_pub = sum(t["channel"] == "public" for r in records for t in r["turns"])
    print(f"source               : {args.subset}/{args.kind}  ({len(sources)} CSVs, {skipped} skipped as incomplete)")
    print(f"records written      : {n} -> {args.output}")
    print(f"splits               : {splits}")
    print(f"kinds                : {kinds}   topics: {len({r['proposition']['text'] for r in records})}")
    print(f"participants         : {sum(len(r['parties']) for r in records)}  with both ratings: {len(deltas)}")
    print(f"turns                : {n_turns}  (tweets {n_pub}, chat {n_turns - n_pub})")
    if deltas:
        print(f"mean |change|        : {sum(abs(d) for d in deltas) / len(deltas):.2f} (0-100); "
              f"changed share: {sum(d != 0 for d in deltas) / len(deltas):.1%}")


if __name__ == "__main__":
    main()
