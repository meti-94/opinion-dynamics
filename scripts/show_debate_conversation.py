#!/usr/bin/env python3
"""
Download ONE multi-party conversation from the DEBATE dataset
(https://huggingface.co/datasets/seantw/DEBATE_LLM) and print it together with
each participant's stance dynamics.

Dataset layout (from the dataset card):
  golden/{depth,breadth}/.../*.csv   curated, complete groups (4 participants each)
  raw/{depth,breadth}/<topic>/*.csv  everything, including dropouts
Each CSV = one group of 4 people discussing one topic. Key columns:
  event_type   Initial Opinion | tweet | message_sent | message_received |
               Post Opinion | exit_survey
  worker_id, sender_id, recipient_id, chat_round_order, event_order, text,
  sliderValue  (stance rating on a 1-6 scale, set at Initial/Post Opinion)

Stance is rated privately only BEFORE and AFTER the discussion, so the "dynamic"
is initial -> post per participant, with the 3 rounds of dyadic chat in between.

Usage:
  pip install huggingface_hub pandas
  python show_debate_conversation.py                  # first golden conversation where someone changed stance
  python show_debate_conversation.py --any            # don't require a stance change
  python show_debate_conversation.py --kind breadth --index 5
  python show_debate_conversation.py --topic "direct democracy"
  python show_debate_conversation.py --file local.csv # use a CSV you already have

License reminder: DEBATE Dataset Research-Only License (Non-Commercial, v1.0).
"""
import argparse
import re
import statistics
import sys
import textwrap
from collections import OrderedDict

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "seantw/DEBATE_LLM"
SLIDER_COLUMNS = ["sliderValue", "slider_value", "SLIDER_VALUE"]
WIDTH = 100


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def clean_text(x) -> str:
    """NaN -> '', and strip the '[SLIDER_VALUE=X]' marker some exports put in text."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return re.sub(r"\[SLIDER_VALUE=[^\]]*\]", "", str(x)).strip()


def slider_of(row):
    """Return the stance rating of an opinion row, or None."""
    for col in SLIDER_COLUMNS:
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
    m = re.search(r"\[SLIDER_VALUE=(\d+(?:\.\d+)?)\]", str(row.get("text", "")))
    return float(m.group(1)) if m else None


def topic_from_filename(path: str) -> str:
    stem = path.rsplit("/", 1)[-1].removesuffix(".csv")
    m = re.match(r"^\d{8}_\d{6}_(.+)_[A-Za-z0-9]{26}$", stem)  # YYYYMMDD_HHMMSS_TOPIC_UNIQUEID
    return (m.group(1) if m else stem).replace("_", " ")


def parse_conversation(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if "event_order" in df.columns:
        df = df.sort_values("event_order", kind="stable")
    etype = df["event_type"].astype(str).str.strip().str.lower()

    # --- stance: Initial Opinion / Post Opinion --------------------------------
    stance, order = {}, []
    for idx, row in df[etype.isin(["initial opinion", "post opinion"])].iterrows():
        w = row["worker_id"]
        if w not in order:
            order.append(w)
        key = "initial" if etype[idx] == "initial opinion" else "post"
        entry = stance.setdefault(w, {}).setdefault(key, {"value": None, "text": ""})
        val, txt = slider_of(row), clean_text(row.get("text"))
        if val is not None:      # a participant can have several rows; keep the last real value
            entry["value"] = val
        if txt:
            entry["text"] = txt

    # --- dialogue: tweets + chat. message_received just duplicates message_sent --
    messages = []
    for idx, row in df[etype.isin(["tweet", "message_sent"])].iterrows():
        text = clean_text(row.get("text"))
        if not text:             # empty rows appear when consecutive messages get merged
            continue
        rnd = row.get("chat_round_order")
        messages.append({
            "round": int(rnd) if pd.notna(rnd) else 0,
            "kind": "tweet" if etype[idx] == "tweet" else "chat",
            "sender": row.get("sender_id", row["worker_id"]),
            "recipient": row.get("recipient_id"),
            "text": text,
        })
        for w in (messages[-1]["sender"], messages[-1]["recipient"]):
            if isinstance(w, str) and w not in order:
                order.append(w)

    aliases = {w: f"P{i + 1}" for i, w in enumerate(order)}
    return {"topic": topic_from_filename(csv_path), "file": csv_path,
            "participants": order, "aliases": aliases,
            "stance": stance, "messages": messages}


def delta_of(conv, w):
    s = conv["stance"].get(w, {})
    a, b = s.get("initial", {}).get("value"), s.get("post", {}).get("value")
    return None if a is None or b is None else b - a


def is_usable(conv, need_change: bool) -> bool:
    ws = conv["participants"]
    if len(ws) < 2 or len(conv["messages"]) < 6:
        return False
    deltas = [delta_of(conv, w) for w in ws]
    if any(d is None for d in deltas):
        return False
    return (not need_change) or any(d != 0 for d in deltas)


# --------------------------------------------------------------------------- #
# Display
# --------------------------------------------------------------------------- #
def wrap(prefix: str, text: str, indent: int = 4) -> str:
    return textwrap.fill(text, width=WIDTH, initial_indent=" " * indent + prefix,
                         subsequent_indent=" " * (indent + len(prefix)))


def arrow(d):
    return "more AGREE" if d > 0 else "more DISAGREE" if d < 0 else "unchanged"


def print_conversation(conv):
    A, S = conv["aliases"], conv["stance"]
    val = lambda w, k: S.get(w, {}).get(k, {}).get("value")
    fmt = lambda v: "?" if v is None else f"{v:g}"

    print("=" * WIDTH)
    print(f"TOPIC : {conv['topic']}")
    print(f"FILE  : {conv['file']}")
    print("Stance scale: 1 (strongly disagree) .. 6 (strongly agree), rated before and after")
    print("=" * WIDTH)

    # 1) stance dynamics per participant
    print("\nSTANCE DYNAMICS")
    print(f"  {'party':<6}{'initial':>8}{'post':>7}{'delta':>7}   direction")
    for w in conv["participants"]:
        d = delta_of(conv, w)
        print(f"  {A[w]:<6}{fmt(val(w, 'initial')):>8}{fmt(val(w, 'post')):>7}"
              f"{('' if d is None else ('0' if d == 0 else f'{d:+g}')):>7}   {'' if d is None else arrow(d)}")

    ini = [val(w, "initial") for w in conv["participants"] if val(w, "initial") is not None]
    pst = [val(w, "post") for w in conv["participants"] if val(w, "post") is not None]
    if len(ini) > 1 and len(pst) > 1:
        sd0, sd1 = statistics.pstdev(ini), statistics.pstdev(pst)
        print(f"\n  group mean {statistics.mean(ini):.2f} -> {statistics.mean(pst):.2f}"
              f" | spread (sd) {sd0:.2f} -> {sd1:.2f}"
              f" ({'converged' if sd1 < sd0 else 'diverged' if sd1 > sd0 else 'same spread'})")

    # 2) what each person said about their view, before vs after
    print("\nSTATED REASONING (private free text accompanying the rating)")
    for w in conv["participants"]:
        print(f"  {A[w]}")
        print(wrap("before: ", S.get(w, {}).get("initial", {}).get("text") or "(none)"))
        print(wrap("after : ", S.get(w, {}).get("post", {}).get("text") or "(none)"))

    # 3) dialogue by round, grouped by dyad
    print("\nDIALOGUE")
    rounds = OrderedDict()
    for m in conv["messages"]:
        rounds.setdefault(m["round"], []).append(m)
    for r, msgs in rounds.items():
        print(f"\n--- Round {r} " + "-" * (WIDTH - 12))
        tweets = [m for m in msgs if m["kind"] == "tweet"]
        if tweets:
            print("  Public posts:")
            for m in tweets:
                print(wrap(f"{A.get(m['sender'], '?')} -> {A.get(m['recipient'], '?')}: ", m["text"]))
        dyads = OrderedDict()
        for m in (m for m in msgs if m["kind"] == "chat"):
            dyads.setdefault(tuple(sorted([m["sender"], str(m["recipient"])], key=lambda w: A.get(w, "Z"))), []).append(m)
        for (x, y), ms in dyads.items():
            print(f"\n  Chat {A.get(x, '?')} <-> {A.get(y, '?')}"
                  f"   (initial stance {fmt(val(x, 'initial'))} vs {fmt(val(y, 'initial'))})")
            for m in ms:
                print(wrap(f"{A.get(m['sender'], '?')}: ", m["text"]))
    print()


# --------------------------------------------------------------------------- #
# Download / selection
# --------------------------------------------------------------------------- #
def list_csvs(subset: str, kind: str, topic: str):
    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    csvs = sorted(f for f in files if f.endswith(".csv") and f.startswith(f"{subset}/"))
    if kind != "any":
        csvs = [f for f in csvs if f"/{kind}/" in f]
    if topic:
        t = topic.lower().replace(" ", "_")
        csvs = [f for f in csvs if t in f.lower().replace(" ", "_")]
    if not csvs:
        print(f"No CSVs under '{subset}/' matching your filters. Repo files start with:")
        for f in files[:30]:
            print("  ", f)
        sys.exit(1)
    return csvs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", choices=["golden", "raw"], default="golden")
    ap.add_argument("--kind", choices=["depth", "breadth", "any"], default="any")
    ap.add_argument("--topic", default="", help="substring of the topic name")
    ap.add_argument("--index", type=int, default=0, help="start scanning from this file index")
    ap.add_argument("--any", action="store_true", help="don't require at least one stance change")
    ap.add_argument("--max-scan", type=int, default=60, help="max files to download while searching")
    ap.add_argument("--file", help="parse a local CSV instead of downloading")
    args = ap.parse_args()

    if args.file:
        print_conversation(parse_conversation(args.file))
        return

    csvs = list_csvs(args.subset, args.kind, args.topic)
    print(f"{len(csvs)} candidate files in '{args.subset}'. Scanning for a suitable conversation...")
    for f in csvs[args.index: args.index + args.max_scan]:
        path = hf_hub_download(REPO_ID, f, repo_type="dataset")   # cached after the first download
        conv = parse_conversation(path)
        conv["file"] = f
        if is_usable(conv, need_change=not args.any):
            print_conversation(conv)
            return
    print("No conversation met the criteria; try --any or a different --index/--kind.")


if __name__ == "__main__":
    main()
