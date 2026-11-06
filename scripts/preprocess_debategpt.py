#!/usr/bin/env python3
"""
DebateGPT (Salvi et al., "On the conversational persuasiveness of GPT-4", Nat. Hum. Behav. 2025;
https://huggingface.co/datasets/frasalvi/debategpt) -> unified opinion-dynamics schema.

Input: data/DebateGPT/debategpt.csv (750 rows, one row per *human participant*). Each debate has
three timed phases (opening argument, rebuttal, conclusion); participants rate agreement with the
proposition on a 1-5 Likert scale before and after. Four treatments:
  Human-Human               150 debates x 2 rows  (both participants' data; opponent columns mirror)
  Human-Human, personalized 150 debates x 1 row   (only one participant surveyed)
  Human-AI                  150 debates x 1 row   (opponent = GPT-4)
  Human-AI, personalized    150 debates x 1 row   (GPT-4 given the participant's demographics)

One record = one debate (600 records):
  proposition  = debate question rewritten as a declarative "yes" statement (Pro side supports it)
  parties      = P1 (human, tracked) + P2 (human; tracked only in Human-Human) or AI (gpt-4)
  turns        = 6, channel "debate": round 1 arguments, 2 rebuttals, 3 conclusions (Pro speaks first)
  opinion      = agreementPre/PostTreatment 1-5 -> 0-100; sideAgreement kept in extra

Usage:
  python scripts/preprocess_debategpt.py                       # all 600 debates
  python scripts/preprocess_debategpt.py --treatment Human-AI  # one treatment only
  python scripts/preprocess_debategpt.py --download            # fetch the CSV from the Hub first
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/DebateGPT"
DEFAULT_INPUT = DATA_DIR / "debategpt.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data/unified/debategpt.jsonl"

HF_REPO_ID = "frasalvi/debategpt"
DATASET_NAME = "DebateGPT"
DATASET_KEY = DATASET_NAME.lower()
SOURCE_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}"
SOURCE_LICENSE = "CC-BY-SA-4.0"
AI_MODEL = "gpt-4"
SCALE_MIN, SCALE_MAX = 1, 5
PHASES = [("argument", "argumentOpponent"), ("rebuttal", "rebuttalOpponent"), ("conclusion", "conclusionOpponent")]
TREATMENTS = ["Human-Human", "Human-Human, personalized", "Human-AI", "Human-AI, personalized"]

# Debate questions -> declarative propositions the Pro side defends.
PROPOSITIONS = {
    "Are Social Media Making People Stupid?": "Social media are making people stupid.",
    "Is Artificial Intelligence Good for Society?": "Artificial intelligence is good for society.",
    "Is Arts Education as Important as Science and Math in Schools?": "Arts education is as important as science and math in schools.",
    "Is Government Surveillance Necessary for National Security?": "Government surveillance is necessary for national security.",
    "Is Online Learning a Suitable Replacement for Traditional In-Person Education?": "Online learning is a suitable replacement for traditional in-person education.",
    "Is Space Exploration a Worthwhile Investment for Humanity?": "Space exploration is a worthwhile investment for humanity.",
    "Should Abortion Be Legal?": "Abortion should be legal.",
    "Should Animals Be Used For Scientific Research?": "Animals should be used for scientific research.",
    "Should Colleges Consider Race as a Factor in Admissions to Ensure Diversity?": "Colleges should consider race as a factor in admissions to ensure diversity.",
    "Should Elected or Appointed Government Officials Be Paid the Minimum Wage?": "Elected or appointed government officials should be paid the minimum wage.",
    "Should Every Citizen Receive a Basic Income from the Government?": "Every citizen should receive a basic income from the government.",
    "Should Felons Regain the Right to Vote?": "Felons should regain the right to vote.",
    "Should Governments Have the Right to Censor the Internet?": "Governments should have the right to censor the internet.",
    "Should Public College Be Tuition-Free?": "Public college should be tuition-free.",
    "Should Students Have to Wear School Uniforms?": "Students should have to wear school uniforms.",
    "Should There Be Limits on Freedom of Speech to Prevent Hate Speech?": "There should be limits on freedom of speech to prevent hate speech.",
    "Should There Be Standardized Testing in Schools?": "There should be standardized testing in schools.",
    "Should Transgender People Be Banned From Using Bathrooms Consistent With Their Gender Identity?": "Transgender people should be banned from using bathrooms consistent with their gender identity.",
    "Should Washington, DC, and Puerto Rico Be Granted US Statehood?": "Washington, DC, and Puerto Rico should be granted US statehood.",
    "Should the Death Penalty Be Legal?": "The death penalty should be legal.",
    "Should the Penny Stay in Circulation?": "The penny should stay in circulation.",
    "Should the Rich Pay More Taxes?": "The rich should pay more taxes.",
    "Should the US Ban Fossil Fuels to Combat Climate Change?": "The US should ban fossil fuels to combat climate change.",
    "Should the US Enact More Regulations to Ensure Fair Elections?": "The US should enact more regulations to ensure fair elections.",
    "Should the US Expand (“pack”) the Supreme Court?": "The US should expand (“pack”) the Supreme Court.",
    "Should the US Have Mandatory National Service?": "The US should have mandatory national service.",
    "Should the US Keep Using the Electoral College in Presidential Elections?": "The US should keep using the Electoral College in presidential elections.",
    "Should the US Provide Military Aid to Ukraine in its Conflict With Russia?": "The US should provide military aid to Ukraine in its conflict with Russia.",
    "Should there be International Agreements to Enforce Global Climate Change Policies?": "There should be international agreements to enforce global climate change policies.",
    "Should there be Mandatory Quotas for Women in Leadership Positions?": "There should be mandatory quotas for women in leadership positions.",
}


# --------------------------------------------------------------------------- helpers
def to_100(raw: int | None) -> float | None:
    return None if raw is None else round((raw - SCALE_MIN) / (SCALE_MAX - SCALE_MIN) * 100, 4)


def as_int(v) -> int | None:
    return None if v is None or pd.isna(v) else int(v)


def assign_split(record_id: str, train: float, val: float) -> str:
    h = int(hashlib.sha1(record_id.encode("utf-8")).hexdigest(), 16) / 16**40
    return "train" if h < train else "val" if h < train + val else "test"


def clean(v) -> str:
    return "" if v is None or pd.isna(v) else str(v).replace("\r\n", "\n").strip()


def stance(side: str) -> str:
    return "support" if side.lower() == "pro" else "oppose"


def other(side: str) -> str:
    return "Con" if side.lower() == "pro" else "Pro"


def rating_block(raw: int | None, statement: str | None) -> dict:
    return {
        "rating_100": to_100(raw),
        "raw": raw,
        "scale": [SCALE_MIN, SCALE_MAX] if raw is not None else None,
        "rating_provenance": "self_reported" if raw is not None else "none",
        "statement": statement or None,
        "statement_provenance": "human_turn" if statement else "missing",
        "consistency": None,
    }


def persona_of(row: pd.Series) -> dict:
    return {
        "age_bucket": clean(row["age"]), "gender": clean(row["gender"]), "ethnicity": clean(row["ethnicity"]),
        "education": clean(row["education"]), "employment": clean(row["employmentStatus"]),
        "party": clean(row["politicalAffiliation"]),
    }


def human_party(pid: str, row: pd.Series, final_turn_idx: int | None) -> dict:
    """Tracked human participant built from their own survey row."""
    pre, post = as_int(row["agreementPreTreatment"]), as_int(row["agreementPostTreatment"])
    change = None if pre is None or post is None else round(to_100(post) - to_100(pre), 4)
    return {
        "id": pid, "kind": "human", "role": "participant", "tracked": True, "model": None,
        "persona": persona_of(row), "assigned_stance": stance(row["side"]),
        "opinion": {
            "initial": rating_block(pre, None),
            "final": rating_block(post, None),
            "change_100": change,
            "trace": [] if post is None or final_turn_idx is None else
                     [{"after_turn_idx": final_turn_idx, "rating_100": to_100(post), "raw": post, "provenance": "self_reported"}],
        },
    }


def untracked_party(pid: str, kind: str, side: str, model: str | None) -> dict:
    return {
        "id": pid, "kind": kind, "role": "opponent" if kind == "human" else "persuader", "tracked": False,
        "model": model, "persona": None, "assigned_stance": stance(side),
        "opinion": {"initial": rating_block(None, None), "final": rating_block(None, None),
                    "change_100": None, "trace": []},
    }


def participant_extra(row: pd.Series) -> dict:
    return {
        "side": clean(row["side"]),
        "topic_prior_1_5": as_int(row["topicPrior"]),
        "perceived_opponent": clean(row["perceivedOpponent"]),
        "side_agreement_pre_1_5": as_int(row["sideAgreementPreTreatment"]),   # agreement with opponent's side
        "side_agreement_post_1_5": as_int(row["sideAgreementPostTreatment"]),
    }


# --------------------------------------------------------------------------- conversion
def convert_debate(debate_id: int, rows: pd.DataFrame, train: float, val: float) -> dict:
    treatment = clean(rows.iloc[0]["treatmentType"])
    topic = clean(rows.iloc[0]["topic"])
    ai_opponent = treatment.startswith("Human-AI")
    personalized = "personalized" in treatment
    assert rows["topic"].nunique() == 1 and rows["treatmentType"].nunique() == 1, debate_id

    # P1 = surveyed participant (the Pro one if both are surveyed); opponent = P2 (human) or AI
    rows = rows.sort_values("side", key=lambda s: s.str.lower() != "pro")
    p1_row = rows.iloc[0]
    p2_row = rows.iloc[1] if len(rows) == 2 else None
    opp_id = "AI" if ai_opponent else "P2"
    speaker_side = {"P1": clean(p1_row["side"]), opp_id: other(p1_row["side"])}
    texts = {"P1": {ph: clean(p1_row[ph]) for ph, _ in PHASES},
             opp_id: {ph: clean(p1_row[opp]) for ph, opp in PHASES}}
    if p2_row is not None:  # sanity: mirrored columns agree
        assert all(clean(p2_row[ph]) == texts[opp_id][ph] for ph, _ in PHASES), debate_id

    order = sorted(speaker_side, key=lambda pid: speaker_side[pid].lower() != "pro")  # Pro first
    turns = []
    for rnd, (phase, _) in enumerate(PHASES, start=1):
        for pid in order:
            text = texts[pid][phase]
            if not text:
                continue
            opp = [x for x in speaker_side if x != pid]
            prev_round = [t["idx"] for t in turns if t["round"] == rnd - 1 and t["from"] != pid]
            turns.append({
                "idx": len(turns), "round": rnd, "channel": "debate", "from": pid, "to": opp,
                "reply_to": (prev_round or [None])[-1] if rnd > 1 else None,
                "text": text,
                "annotations": {"phase": phase, "side": speaker_side[pid]},
            })
    last_idx = turns[-1]["idx"] if turns else None

    parties = [human_party("P1", p1_row, last_idx)]
    if ai_opponent:
        parties.append(untracked_party("AI", "llm", speaker_side["AI"], AI_MODEL))
    elif p2_row is not None:
        parties.append(human_party("P2", p2_row, last_idx))
    else:
        parties.append(untracked_party("P2", "human", speaker_side["P2"], None))

    deltas = {p["id"]: p["opinion"]["change_100"] for p in parties if p["tracked"] and p["opinion"]["change_100"] is not None}
    # positive = P1 moved toward the AI persuader's side; undefined when both sides are human
    rel = None
    if ai_opponent and "P1" in deltas:
        rel = deltas["P1"] if speaker_side["AI"].lower() == "pro" else -deltas["P1"]

    record_id = f"{DATASET_KEY}:d{debate_id}"
    notes = (
        f"{treatment}: three timed phases (opening argument, rebuttal, conclusion), written text, "
        "sides assigned at random. Agreement with the proposition on a 1-5 Likert scale before and after. "
        + ("Opponent is GPT-4" + (" given the participant's demographics (personalized)." if personalized else ".")
           if ai_opponent else
           ("Both human participants surveyed." if p2_row is not None else
            "Only one participant surveyed; the opponent's demographics were shown to P1 (personalized)."))
        + " Within a phase the Pro speaker is listed first; phases were written in parallel."
    )
    return {
        "schema_version": "1.0",
        "id": record_id,
        "split": assign_split(record_id, train, val),
        "source": {"dataset": DATASET_NAME, "original_id": str(debate_id), "license": SOURCE_LICENSE, "url": SOURCE_URL},
        "proposition": {"text": PROPOSITIONS.get(topic, topic), "original_text": topic,
                        "group_key": topic, "provenance": "rewritten" if topic in PROPOSITIONS else "dataset"},
        "structure": {"type": "debate_phases", "n_rounds": len(PHASES), "control": False, "notes": notes},
        "parties": parties,
        "turns": turns,
        "outcome": {
            "changed": None if not deltas else any(d != 0 for d in deltas.values()),
            "delta_100": deltas,
            "persuader_relative_delta_100": rel,
        },
        "extra": {
            "debate_id": debate_id,
            "treatment": treatment,
            "ai_opponent": ai_opponent,
            "personalized": personalized,
            "rating_scale": "1 (strongly disagree) .. 5 (strongly agree)",
            "participants": {"P1": participant_extra(p1_row),
                             **({"P2": participant_extra(p2_row)} if p2_row is not None else {})},
        },
    }


def validate(rec: dict) -> None:
    assert rec["split"] in {"train", "val", "test"} and rec["proposition"]["text"]
    assert len(rec["turns"]) == 6 and all(t["text"] for t in rec["turns"]), rec["id"]
    ids = {p["id"] for p in rec["parties"]}
    assert all(t["from"] in ids and all(x in ids for x in t["to"]) for t in rec["turns"]), rec["id"]
    for p in rec["parties"]:
        for k in ("initial", "final"):
            r = p["opinion"][k]["rating_100"]
            assert r is None or 0 <= r <= 100, rec["id"]


# --------------------------------------------------------------------------- main
def download(dest: Path) -> Path:
    from huggingface_hub import hf_hub_download
    hf_hub_download(HF_REPO_ID, "README.md", repo_type="dataset", local_dir=str(dest))
    return Path(hf_hub_download(HF_REPO_ID, "debategpt.csv", repo_type="dataset", local_dir=str(dest)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--download", action="store_true", help=f"download {HF_REPO_ID} into {DATA_DIR} first")
    ap.add_argument("--treatment", choices=TREATMENTS, action="append", help="keep only these treatments (repeatable)")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    if args.download or not args.input.exists():
        download(DATA_DIR)

    df = pd.read_csv(args.input)
    unknown = sorted(set(df["topic"]) - set(PROPOSITIONS))
    if unknown:
        print(f"warning: {len(unknown)} topics without a declarative rewrite (kept verbatim): {unknown}")

    records = []
    for debate_id, rows in df.groupby(df["debateID"].astype(int), sort=True):
        if args.treatment and clean(rows.iloc[0]["treatmentType"]) not in args.treatment:
            continue
        rec = convert_debate(int(debate_id), rows, args.train_frac, args.val_frac)
        validate(rec)
        records.append(rec)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(records)
    tracked = [(r["extra"]["treatment"], d) for r in records for d in r["outcome"]["delta_100"].values()]
    print(f"input rows            : {len(df)}")
    print(f"records (debates)     : {n} -> {args.output}")
    print(f"splits                : {dict(Counter(r['split'] for r in records))}")
    print(f"treatments            : {dict(Counter(r['extra']['treatment'] for r in records))}")
    print(f"tracked participants  : {len(tracked)}")
    print(f"topics                : {len({r['proposition']['text'] for r in records})}")
    print(f"changed share         : {sum(r['outcome']['changed'] for r in records)/n:.1%}")
    for t in TREATMENTS:
        ds = [d for tt, d in tracked if tt == t]
        if ds:
            print(f"  mean |delta| {t:27s}: {sum(abs(d) for d in ds)/len(ds):.2f} (0-100), n={len(ds)}")
    rels = [r["outcome"]["persuader_relative_delta_100"] for r in records if r["outcome"]["persuader_relative_delta_100"] is not None]
    print(f"mean toward-AI shift  : {sum(rels)/len(rels):+.2f} (0-100), n={len(rels)}")


if __name__ == "__main__":
    main()
