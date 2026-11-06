#!/usr/bin/env python3
"""
ChangeMyView "Winning Arguments" corpus (Tan et al., WWW 2016; ConvoKit `winning-args-corpus`)
-> unified opinion-dynamics schema.

Raw data: data/CMV/winning-args-corpus/ (ConvoKit format, downloaded with
`convokit.download("winning-args-corpus", data_dir="data/CMV")`). 3,051 r/ChangeMyView posts with
their full comment trees (293k comments). The only opinion signal is binary: within each post, some
top-level comment threads are labelled `success=1` (OP awarded a delta, i.e. changed their view) and
a topic-matched sibling thread is labelled `success=0` (no delta). Everything else is unlabelled.

Unit of a record = one labelled argument thread (root comment + the OP/challenger exchange
beneath it), i.e. the "rooted path-unit" of the original paper:
  proposition     = OP title (leading "CMV:" stripped)   structure.type = reddit_thread
  parties         = OP (tracked; initial statement = OP post body, final = OP's last reply in thread)
                    C1 (the challenger who wrote the root comment; CMV rules require challenging OP)
  turns           = turn 0 is the OP post, then the thread comments in time order (reply_to resolved)
  outcome.changed = success == 1 ; no numeric ratings exist (rating_100 = null everywhere)

Splits: the corpus carries the paper's conversation-level train/heldout flag. heldout -> "test";
from the train conversations a deterministic ~12% (hash of conversation id) is carved out as "val"
so that all threads of one post share a split.

Usage:
  python scripts/preprocess_cmv.py                       # 8,106 labelled threads
  python scripts/preprocess_cmv.py --include-unlabelled  # + every other top-level thread (changed=null)
  python scripts/preprocess_cmv.py --download            # fetch the corpus into data/CMV first
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/CMV"
CORPUS_NAME = "winning-args-corpus"
DEFAULT_CORPUS = DATA_DIR / CORPUS_NAME
DEFAULT_OUTPUT = REPO_ROOT / "data/unified/cmv.jsonl"

DATASET_NAME = "CMV"
DATASET_KEY = DATASET_NAME.lower()
SOURCE_URL = "https://convokit.cornell.edu/documentation/winning.html"
SOURCE_LICENSE = "Reddit user content; research use (Tan et al. 2016 / ConvoKit)"

TITLE_PREFIX_RE = re.compile(r"^\s*\[?\s*cmv\s*\]?\s*[:\-–—]?\s*", re.I)
FOOTNOTE_RE = re.compile(r"(\n\s*_+\s*)?\n+\s*(&gt;|>)\s*\*?\s*Hello, users of CMV!.*$", re.S)
DELTA_RE = re.compile(r"(∆|Δ|&amp;#8710;|&#8710;|!delta|\bdelta\b)", re.I)
REDDIT_URL = "https://www.reddit.com/r/changemyview/comments/{}/"


# --------------------------------------------------------------------------- helpers
def clean_text(text: str | None) -> str:
    return html.unescape(text or "").replace("\r\n", "\n").strip()


def clean_title(title: str) -> str:
    t = TITLE_PREFIX_RE.sub("", clean_text(title)).strip()
    return t or clean_text(title)


def clean_op_body(body: str) -> tuple[str, bool]:
    cleaned = FOOTNOTE_RE.sub("", body or "")
    stripped = cleaned != (body or "")
    return clean_text(cleaned), stripped


def split_of(conv_meta: dict, conv_id: str, val_share_of_train: float) -> str:
    if int(conv_meta.get("train", 1)) == 0:
        return "test"
    h = int(hashlib.sha1(conv_id.encode("utf-8")).hexdigest(), 16) / 16**40
    return "val" if h < val_share_of_train else "train"


def empty_rating(statement: str | None, prov: str) -> dict:
    return {
        "rating_100": None, "raw": None, "scale": None, "rating_provenance": "none",
        "statement": statement, "statement_provenance": prov if statement else "missing",
        "consistency": None,
    }


def utt_annotations(u, is_op: bool) -> dict:
    m = u.meta
    text = u.text or ""
    return {
        "reddit_id": u.id,
        "timestamp": u.timestamp,
        "success": m.get("success"),
        "pair_ids": list(m.get("pair_ids") or []),
        "score": m.get("score"), "ups": m.get("ups"), "downs": m.get("downs"),
        "controversiality": m.get("controversiality"), "gilded": m.get("gilded"),
        "edited": m.get("edited"),
        "delta_marker": bool(is_op and DELTA_RE.search(text)),
        "deleted": text.strip() in {"[deleted]", "[removed]"},
    }


# --------------------------------------------------------------------------- conversion
def thread_utterances(root, children: dict, label) -> list:
    """Root comment plus all descendants carrying the same success label (unlabelled roots: the
    whole subtree). Returned in (timestamp, id) order."""
    out, stack, seen = [], [root], set()
    while stack:
        x = stack.pop()
        if x.id in seen:
            continue
        seen.add(x.id)
        out.append(x)
        for ch in children.get(x.id, []):
            if label is None or ch.meta.get("success") == label:
                stack.append(ch)
    return sorted(out, key=lambda u: (u.timestamp or 0, u.id))


def convert_thread(conv, op_utt, root, thread: list, split: str, paired_ids: list[str]) -> dict:
    meta = conv.meta
    op_user = meta["op-userID"]
    label = root.meta.get("success")
    conv_id, root_id = conv.id, root.id

    title = clean_title(meta["op-title"])
    op_body, footnote_stripped = clean_op_body(meta["op-text-body"])

    # party ids: OP, C1 = root author, C2.. for anyone else that slipped into the subtree
    party_of = {op_user: "OP", root.speaker.id: "C1"}
    for u in thread:
        party_of.setdefault(u.speaker.id, f"C{len(party_of)}")  # OP + C1 already present -> C2, ...

    op_text = op_body or clean_text(op_utt.text) or title  # 5 posts are title-only
    turns = [{
        "idx": 0, "round": None, "channel": "thread", "from": "OP", "to": [],
        "reply_to": None, "text": op_text,
        "annotations": {**utt_annotations(op_utt, True), "is_op_post": True,
                        "footnote_stripped": footnote_stripped, "body_empty": not op_body},
    }]
    idx_of = {op_utt.id: 0}
    for u in thread:
        idx = len(turns)
        idx_of[u.id] = idx
        parent_idx = idx_of.get(u.reply_to)
        frm = party_of[u.speaker.id]
        to = [turns[parent_idx]["from"]] if parent_idx is not None else ["OP"]
        if to == [frm]:  # self-reply (e.g. continuation) -> address the other side
            to = ["C1" if frm == "OP" else "OP"]
        turns.append({
            "idx": idx, "round": None, "channel": "thread", "from": frm, "to": to,
            "reply_to": parent_idx, "text": clean_text(u.text) or "[empty]",
            "annotations": utt_annotations(u, frm == "OP"),
        })

    op_turns = [t for t in turns[1:] if t["from"] == "OP"]
    challenger_turns = [t for t in turns[1:] if t["from"] != "OP"]
    final_statement = op_turns[-1]["text"] if op_turns else None
    changed = None if label is None else label == 1

    parties = [{
        "id": "OP", "kind": "human", "role": "op", "tracked": True, "model": None,
        "persona": None, "assigned_stance": "support",
        "opinion": {
            "initial": empty_rating(op_body or None, "human_turn"),
            "final": empty_rating(final_statement, "human_turn"),
            "change_100": None, "trace": [],
        },
    }]
    for user, pid in party_of.items():
        if pid == "OP":
            continue
        parties.append({
            "id": pid, "kind": "human", "role": "commenter", "tracked": False, "model": None,
            "persona": None, "assigned_stance": "oppose",
            "opinion": {"initial": empty_rating(None, "missing"), "final": empty_rating(None, "missing"),
                        "change_100": None, "trace": []},
        })

    reply_parents = Counter(t["reply_to"] for t in turns[1:])
    is_chain = all(v == 1 for v in reply_parents.values())

    return {
        "schema_version": "1.0",
        "id": f"{DATASET_KEY}:{conv_id}:{root_id}",
        "split": split,
        "source": {"dataset": DATASET_NAME, "original_id": f"{conv_id}/{root_id}",
                   "license": SOURCE_LICENSE, "url": SOURCE_URL},
        "proposition": {"text": title, "original_text": clean_text(meta["op-title"]),
                        "group_key": conv_id, "provenance": "op_title"},
        "structure": {
            "type": "reddit_thread",
            "n_rounds": len(challenger_turns),
            "control": False,
            "notes": (
                "r/ChangeMyView post (turn 0) followed by one top-level challenger comment and the "
                "OP<->challenger exchange beneath it. Binary outcome only: changed=true if the OP awarded "
                "a delta in this thread (success=1), false for the paper's topic-matched unsuccessful "
                "sibling thread (success=0), null if the thread was not labelled. No numeric ratings."
            ),
        },
        "parties": parties,
        "turns": turns,
        "outcome": {"changed": changed, "delta_100": {}, "persuader_relative_delta_100": None},
        "extra": {
            "conversation_id": conv_id,
            "root_comment_id": root_id,
            "success": label,
            "pair_ids": list(root.meta.get("pair_ids") or []),
            "paired_record_ids": paired_ids,
            "original_train_flag": int(meta.get("train", 1)),
            "op_user": op_user,
            "root_author": root.speaker.id,
            "n_op_turns": len(op_turns),
            "n_challenger_turns": len(challenger_turns),
            "thread_is_chain": is_chain,
            "delta_marker_in_op_turn": any(t["annotations"]["delta_marker"] for t in op_turns),
            "conversation_n_utterances": len(list(conv.iter_utterances())),
            "reddit_url": REDDIT_URL.format(conv_id.split("_", 1)[-1]),
        },
    }


def validate(rec: dict) -> None:
    assert rec["split"] in {"train", "val", "test"} and rec["proposition"]["text"]
    assert len(rec["turns"]) >= 2 and all(t["text"] for t in rec["turns"])
    ids = {p["id"] for p in rec["parties"]}
    assert all(t["from"] in ids and all(x in ids for x in t["to"]) for t in rec["turns"]), rec["id"]
    for t in rec["turns"]:
        assert t["reply_to"] is None or 0 <= t["reply_to"] < t["idx"], rec["id"]


# --------------------------------------------------------------------------- main
def download(dest: Path) -> Path:
    import os
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())  # python.org builds lack CA certs
    from convokit import download as ck_download
    return Path(ck_download(CORPUS_NAME, data_dir=str(dest)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="ConvoKit corpus directory")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--download", action="store_true", help=f"download {CORPUS_NAME} into {DATA_DIR} first")
    ap.add_argument("--include-unlabelled", action="store_true",
                    help="also emit threads with no success label (outcome.changed = null)")
    ap.add_argument("--val-share-of-train", type=float, default=0.12,
                    help="fraction of the paper's train conversations hashed into 'val' (~10%% of all)")
    args = ap.parse_args()

    if args.download or not (args.corpus / "utterances.jsonl").exists():
        download(DATA_DIR)

    from convokit import Corpus
    corpus = Corpus(filename=str(args.corpus))

    records = []
    for conv in corpus.iter_conversations():
        children = defaultdict(list)
        for u in conv.iter_utterances():
            children[u.reply_to].append(u)
        op_utt = corpus.get_utterance(conv.id)
        split = split_of(conv.meta, conv.id, args.val_share_of_train)

        roots = [r for r in children.get(conv.id, [])
                 if r.meta.get("success") is not None or args.include_unlabelled]
        # opposite-label threads in the same post sharing a pair_id
        by_pair = defaultdict(list)
        for r in roots:
            for p in r.meta.get("pair_ids") or []:
                by_pair[p].append(r)

        for root in sorted(roots, key=lambda u: (u.timestamp or 0, u.id)):
            label = root.meta.get("success")
            paired = sorted({f"{DATASET_KEY}:{conv.id}:{o.id}"
                             for p in (root.meta.get("pair_ids") or []) for o in by_pair[p]
                             if o.id != root.id and o.meta.get("success") != label})
            thread = thread_utterances(root, children, label)
            rec = convert_thread(conv, op_utt, root, thread, split, paired)
            validate(rec)
            records.append(rec)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(records)
    print(f"conversations (posts) : {len(list(corpus.iter_conversations()))}")
    print(f"records (threads)     : {n} -> {args.output}")
    print(f"splits                : {dict(Counter(r['split'] for r in records))}")
    print(f"outcome.changed       : {dict(Counter(r['outcome']['changed'] for r in records))}")
    print(f"turns per record      : {dict(sorted(Counter(len(r['turns']) for r in records).items()))}")
    print(f"records with OP reply : {sum(r['extra']['n_op_turns'] > 0 for r in records)}")
    print(f"non-chain threads     : {sum(not r['extra']['thread_is_chain'] for r in records)}")
    print(f"paired records        : {sum(bool(r['extra']['paired_record_ids']) for r in records)}")
    print(f"delta marker (succ=1) : {sum(r['extra']['delta_marker_in_op_turn'] for r in records if r['outcome']['changed'])}"
          f" / {sum(1 for r in records if r['outcome']['changed'])}")


if __name__ == "__main__":
    main()
