#!/usr/bin/env python3
"""
Save every DEBATE_LLM conversation as ONE JSON file in the current directory.

Each JSON looks like:
{
  "id": "01JTH0DYKP0PY9KX7PJP8GCVQ0",
  "topic": "...",
  "source_file": "golden/breadth/.../....csv",
  "rating_scale": "1 (strongly disagree) .. 6 (strongly agree)",
  "participants": ["P1", "P2", "P3", "P4"],
  "worker_ids": {"P1": "5f171GCVQ0", ...},
  "initial": {"P1": {"rating": 4, "statement": "..."}, ...},
  "rounds": [
    {"round": 1,
     "public":  [{"from": "P1", "to": "P4", "text": "..."}, ...],
     "private": [{"between": ["P1", "P4"],
                  "messages": [{"from": "P1", "text": "..."}, ...]}, ...]},
    {"round": 2, ...}, {"round": 3, ...}
  ],
  "final": {"P1": {"rating": 3, "statement": "...", "change": -1}, ...}
}

Public = the "tweet" each person posts per round; private = the one-on-one chat that follows.
Ratings exist only before (initial) and after (final) the discussion.

Usage (needs show_debate_conversation.py in the same folder):
  pip install huggingface_hub pandas
  python export_conversations_json.py                      # all golden conversations -> ./debate_<ID>.json
  python export_conversations_json.py --max-files 5        # just try a few
  python export_conversations_json.py --out-dir debate_json
  python export_conversations_json.py --files my.csv       # local CSVs
  python export_conversations_json.py --split test --out-dir debate_test   # held-out ~10% of conversations

License reminder: DEBATE Dataset Research-Only License (Non-Commercial, v1.0).
"""
import argparse
import hashlib
import json
import os
import re

from huggingface_hub import hf_hub_download

from show_debate_conversation import (REPO_ID, delta_of, is_usable, list_csvs,
                                      parse_conversation)

SCALE = "1 (strongly disagree) .. 6 (strongly agree)"
ROUNDS = (1, 2, 3)


def split_of(key: str) -> str:
    """Same rule as build_training_data.py: ~80% train, 10% val, 10% test, per conversation."""
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"


def conversation_id(name: str) -> str:
    m = re.search(r"_([A-Za-z0-9]{26})\.csv$", name)
    return m.group(1) if m else hashlib.md5(name.encode()).hexdigest()[:26]


def to_record(conv: dict) -> dict:
    A, S = conv["aliases"], conv["stance"]
    alias = lambda w: A.get(w) if isinstance(w, str) else None

    def num(x):
        return None if x is None else (int(x) if float(x).is_integer() else float(x))

    initial, final = {}, {}
    for w in conv["participants"]:
        i, p, d = S[w]["initial"], S[w]["post"], delta_of(conv, w)
        initial[A[w]] = {"rating": num(i["value"]), "statement": i["text"]}
        final[A[w]] = {"rating": num(p["value"]), "statement": p["text"], "change": num(d)}

    round_ids = sorted(set(ROUNDS) | {m["round"] for m in conv["messages"] if m["round"] > 0})
    rounds = []
    for r in round_ids:
        msgs = [m for m in conv["messages"] if m["round"] == r]
        public = [{"from": alias(m["sender"]), "to": alias(m["recipient"]), "text": m["text"]}
                  for m in msgs if m["kind"] == "tweet"]
        dyads = {}
        for m in (m for m in msgs if m["kind"] == "chat"):
            pair = tuple(sorted([alias(m["sender"]), alias(m["recipient"])], key=lambda a: a or "Z"))
            dyads.setdefault(pair, []).append({"from": alias(m["sender"]), "text": m["text"]})
        private = [{"between": list(pair), "messages": ms} for pair, ms in dyads.items()]
        rounds.append({"round": r, "public": public, "private": private})

    return {
        "id": conversation_id(conv["file"]),
        "topic": conv["topic"],
        "source_file": conv["file"],
        "rating_scale": SCALE,
        "participants": [A[w] for w in conv["participants"]],
        "worker_ids": {A[w]: w for w in conv["participants"]},
        "initial": initial,
        "rounds": rounds,
        "final": final,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", choices=["golden", "raw"], default="golden")
    ap.add_argument("--kind", choices=["depth", "breadth", "any"], default="any")
    ap.add_argument("--max-files", type=int, default=0, help="0 = all")
    ap.add_argument("--files", nargs="*", help="local CSVs instead of downloading")
    ap.add_argument("--out-dir", default=".", help="where to write the JSON files (default: current dir)")
    ap.add_argument("--split", choices=["all", "train", "val", "test"], default="all",
                    help="export only one portion (the dataset has no official split; this is our own "
                         "deterministic per-conversation 80/10/10 split)")
    ap.add_argument("--keep-incomplete", action="store_true",
                    help="also export groups missing an initial/final rating for someone")
    args = ap.parse_args()

    if args.files:
        sources = [(f, f) for f in args.files]
    else:
        names = list_csvs(args.subset, args.kind, "")
        names = names[: args.max_files] if args.max_files else names
        sources = [(n, None) for n in names]

    os.makedirs(args.out_dir, exist_ok=True)
    written = skipped = 0
    for name, local in sources:
        if args.split != "all" and split_of(name) != args.split:
            continue                       # decided from the file name, so nothing is downloaded
        path = local or hf_hub_download(REPO_ID, name, repo_type="dataset")
        conv = parse_conversation(path)
        conv["file"] = name
        if not args.keep_incomplete and not is_usable(conv, need_change=False):
            skipped += 1
            continue
        if args.keep_incomplete:   # fill gaps so to_record doesn't crash
            for w in conv["participants"]:
                for k in ("initial", "post"):
                    conv["stance"].setdefault(w, {}).setdefault(k, {"value": None, "text": ""})
        rec = to_record(conv)
        out = os.path.join(args.out_dir, f"debate_{rec['id']}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        written += 1
    print(f"Wrote {written} JSON files to '{os.path.abspath(args.out_dir)}' ({skipped} skipped as incomplete).")


if __name__ == "__main__":
    main()
