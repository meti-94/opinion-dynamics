# Opinion-dynamics corpus

Six public datasets in which people state an opinion on a proposition, take part in a discussion, and state their opinion again, converted into one JSONL schema under `data/unified/`. Each line is one *episode* (a dialogue, debate, thread, or one-shot reading) with the proposition, the parties, their pre/post opinions, the full turn sequence, and the outcome.

On top of the unified JSONL, this repository provides a **participant-perspective dataset infrastructure** (`src/opinion_dynamics/`) that:

1. Loads all six datasets through a common interface.
2. Uses independent adapters per dataset.
3. Converts original records into participant-level episodes.
4. Preserves initial/final stances and reconstructs what each participant could observe.
5. Distinguishes SELF from OTHER dynamically while keeping original speaker IDs.
6. Formats Hugging Face-compatible chat messages for stance prediction and next-response generation.
7. Provides inspection, comparison, filtering, export, and Hugging Face Dataset conversion.

This is **data infrastructure and inspection only**. It does not implement training, SFT, GRPO, reward functions, or inference pipelines.

## Setup

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ".[dev]"
```

## Quick start (participant episodes)

```python
from opinion_dynamics import OpinionDynamicsDataLoader

loader = OpinionDynamicsDataLoader.from_config("configs/datasets.yaml")
episodes = loader.load()

print(loader.summary())

ep = loader.filter(dataset="DEBATE", participant_id="P1")[0]
print(loader.inspect(ep))
print(loader.inspect_perspective(ep.episode_id))
print(loader.inspect_visibility(ep.episode_id))
print(loader.compare_original_and_episode(ep.episode_id))

messages = ep.to_stance_prediction_messages()
messages, target = ep.to_response_generation_messages(target_turn_idx=...)

loader.export_jsonl("data/processed/episodes.jsonl")
hf_dataset = loader.to_huggingface()
```

Demo notebook: [`notebooks/01_dataset_inspection.ipynb`](notebooks/01_dataset_inspection.ipynb)

Stage-1 initial-stance examples (proposition and the participant's own opening text only; no discussion, final stance, or outcome) are built by `InitialStanceDataLoader` in `src/opinion_dynamics/tasks/initial_stance.py`. Notebook: [`notebooks/02_initial_stance_training.ipynb`](notebooks/02_initial_stance_training.ipynb).

```bash
PYTHONPATH=src .venv/bin/pytest tests/ -v
```

## Architecture

```text
data/unified/*.jsonl          # source of truth (do not rewrite)
        │
        ▼
src/opinion_dynamics/
  datasets/adapters/          # one adapter per dataset
  datasets/episode.py         # canonical Episode schema
  datasets/dataloader.py      # OpinionDynamicsDataLoader
  perspectives/               # visibility + SELF/OTHER
  formatting/                 # stance prediction / response generation
  inspection/                 # summary, inspect, comparison
```

Dataset-specific logic stays inside adapters. The DataLoader never parses dataset-specific fields. Chat formatters never mutate episodes.

### Confirmed design decisions

| Topic | Decision |
|---|---|
| DEBATE public tweets | Visible to **all** four participants |
| DEBATE private chat | Visible only to sender + recipients |
| Initial context vs turn 0 | Keep **both** (ConspiracyAI, CMV) |
| ConspiracyAI final context | `null` (no post-discussion authored statement) |
| CMV stance | Binary proxy on a 0–100 scale: `100/0/null`, not a measured rating |
| PersuasionTrace | First/last ratings only; intermediates in `extra` |
| DebateGPT | Episodes for `tracked=True` only; phase order = convention |
| `outcome.changed` | Per participant: true only if that person's stance moved |
| Missing data | Never fabricated; reported in metadata / exclusions |

## Preprocessing (unified JSONL)

Each dataset has a `scripts/preprocess_<dataset>.py` that (a) downloads the raw files into `data/<Dataset>/` if they are not there yet, (b) parses them, (c) writes `data/unified/<dataset>.jsonl`, and (d) prints summary counts. Scripts are deterministic; re-running overwrites the output.

| Dataset | Raw dir | Script | Download mechanism |
|---|---|---|---|
| ConspiracyAI | `data/ConspiracyAI/` | `preprocess_conspiracyai.py` | manual (Dryad zip, already in place) |
| DEBATE | `data/DEBATE_LLM/` | `preprocess_debate.py` | `huggingface_hub` → `seantw/DEBATE_LLM` (reuses `show_debate_conversation.py` / `export_conversations_json.py`) |
| PersuasionTrace | `data/PersuationTrace/` | `preprocess_persuasiontrace.py` | repo checkout (already in place) |
| AnthropicPersuasion | `data/AnthropicPersuasion/` | `preprocess_anthropic_persuasion.py` | `hf download Anthropic/persuasion` |
| CMV | `data/CMV/` | `preprocess_cmv.py` | `convokit.download("winning-args-corpus")` |
| DebateGPT | `data/DebateGPT/` | `preprocess_debategpt.py` | `hf download frasalvi/debategpt` |

## Unified schema (v1.0)

```jsonc
{"schema_version":"1.0","id":"<dataset_lower>:<original_id>","split":"train|val|test",
 "source":{"dataset":"...","original_id":"str","license":"str","url":"str"},
 "proposition":{"text":"declarative statement","original_text":"as in the dataset","group_key":"topic id","provenance":"dataset|rewritten|op_title|claim"},
 "structure":{"type":"multi_party_rounds|human_llm_dialogue|one_shot_argument|reddit_thread|debate_phases","n_rounds":int|null,"control":bool,"notes":"str"},
 "parties":[{"id":"P1|T|AI|OP|C1","kind":"human|llm","role":"participant|target|persuader|opponent|op|commenter","tracked":bool,"model":str|null,
             "persona":{...}|null,"assigned_stance":"support|oppose|null",
             "opinion":{"initial":{"rating_100":0-100|null,"raw":num|null,"scale":[min,max]|null,"rating_provenance":"self_reported|llm_estimated|none",
                                   "statement":str|null,"statement_provenance":"human_private|human_turn|synthesized|missing","consistency":null},
                        "final":{...same...},"change_100":final-initial|null,
                        "trace":[{"after_turn_idx":int,"rating_100":0-100,"raw":num,"provenance":"self_reported|llm_estimated"}]}}],
 "turns":[{"idx":int,"round":int|null,"channel":"public|private|dialogue|thread|debate","from":"party_id","to":["party_id"],"reply_to":idx|null,"text":"str","annotations":{}}],
 "outcome":{"changed":bool|null,"delta_100":{"party_id":float},"persuader_relative_delta_100":float|null},
 "extra":{}}
```

Conventions that hold across all six files:

- `rating_100` is the opinion mapped linearly onto 0–100 where 100 = full agreement with `proposition.text`; the untouched value and its scale are in `raw`/`scale`.
- `change_100 = final − initial` for every `tracked` party; `outcome.delta_100` repeats it per party; `outcome.changed` is `any(delta ≠ 0)` (or the dataset's binary label when no numeric rating exists).
- `persuader_relative_delta_100` is the tracked human's shift **signed toward the persuader's stance** (positive = persuaded). It is set only when there is a single, identifiable persuader with a known stance; it is `null` for symmetric human–human settings and control arms.
- `opinion.trace` holds every intermediate rating that exists (one entry = after the last turn for pre/post-only datasets; several for PersuasionTrace).
- Splits are deterministic. Where the source ships its own train/test split it is respected (DEBATE, CMV); otherwise SHA-1 of a key → 80/10/10. The key is chosen so that related episodes never straddle splits (same argument for AnthropicPersuasion, same post for CMV).
- Text is stripped of CR/LF noise and HTML entities; nothing else is altered. Empty/duplicated turns are dropped and counted in `extra`.

## Datasets

| | ConspiracyAI | DEBATE | PersuasionTrace | AnthropicPersuasion | CMV | DebateGPT |
|---|---|---|---|---|---|---|
| Paper | Costello, Pennycook & Rand, *Science* 2024 | Chuang et al., arXiv:2510.25110 | arXiv:2606.05330 | Durmus et al. 2024 (Anthropic blog) | Tan et al., WWW 2016 | Salvi et al., *Nat. Hum. Behav.* 2025 |
| License | CC0-1.0 | DEBATE research-only, non-commercial | MIT | CC-BY-NC-SA-4.0 | Reddit content / research use | CC-BY-SA-4.0 |
| Setting | human ↔ GPT-4-Turbo, 3 exchanges | 4 humans, 3 rounds of public tweet + private dyad chat | human ↔ GPT-5 chat, belief probed after every AI turn | human reads one written argument | r/ChangeMyView post + one challenger thread | human vs human / GPT-4, 3 written phases |
| `structure.type` | `human_llm_dialogue` | `multi_party_rounds` | `human_llm_dialogue` | `one_shot_argument` | `reddit_thread` | `debate_phases` |
| **Records** | **1,749** | **263** | **353** | **3,939** | **8,106** | **600** |
| Split train/val/test | 1413 / 163 / 173 | 203 / 32 / 28 | 283 / 40 / 30 | 3045 / 441 / 453 | 5798 / 792 / 1516 | 479 / 51 / 70 |
| Distinct propositions | 1,749 (1,489 topic groups) | 107 | 112 | 75 | 3,051 | 30 |
| Parties / record | 2 (P1 + AI) | 4 (P1–P4) | 2 (T + AI) | 2 (T + AI or P1) | 2 (OP + C1) | 2 (P1 + P2 or AI) |
| Tracked parties (total) | 1,749 | 1,052 | 353 | 3,939 | 8,106 | 750 |
| Turns / record (mean · median · range) | 6 · 6 · 6–6 | 42.9 · 42 · 27–67 | 7.6 · 8 · 1–21 | 1 · 1 · 1–1 | 3.4 · 3 · 2–11 | 6 · 6 · 6–6 |
| Words / turn (mean · median) | 218 · 146 | 23 · 19 | 36 · 41 | 244 · 243 | 220 · 154 | 64 · 56 |
| Words / record (mean) | 1,305 | 972 | 271 | 244 | 743 | 384 |
| Total turns / words | 10.5k / 2.28M | 11.3k / 256k | 2.7k / 96k | 3.9k / 961k | 27.4k / 6.02M | 3.6k / 231k |
| Rating scale (raw) | 0–100 | 1–6 Likert | 0–1 belief | 1–7 Likert | none (binary Δ) | 1–5 Likert |
| Intermediate ratings | no | no | yes (mean 3.9 / record) | no | no | no |
| Initial opinion statement | yes (opening message) | yes (private justification) | no | no | yes (OP post) | no |
| Persona on humans | age, gender, party, education | age, gender, education, party, views | no | no | no | age, gender, ethnicity, education, employment, party |
| LLM(s) | gpt-4-turbo | — | gpt-5 / gpt-5.1 | Claude 1.3 · Instant 1.2 · 2 · 3 Haiku · 3 Opus | — | gpt-4 |
| Control records | 0 | 0 | 68 (off-topic "dogs vs cats" dialogue) | 57 (undisputed factual claims) | 0 | 0 |
| Mean Δ (0–100, toward prop.) | −11.8 | +0.6 | −4.4 | +6.9 | — | −2.3 |
| Mean \|Δ\| (0–100) | 16.2 | 14.9 | 18.1 | 9.0 | — | 19.5 |
| Share of tracked parties who moved | 72% | 45% | 95% | 39% | — | 51% |
| Mean shift toward persuader | +11.8 (AI debunks) | n/a | +13.3 (n=285) | +7.0 (n=3,882) | n/a | +1.5 (Human-AI, n=300) |
| `outcome.changed` share | 72% | 90% (any of 4) | 95% | 39% | 52% (Δ awarded) | 58% |
| Unified file size | 21 MB | 4.4 MB | 2.3 MB | 16 MB | 81 MB | 3.4 MB |
| Raw dir size | 109 MB | 26 MB | 131 MB | 6.6 MB | 343 MB | 1.9 MB |

"Words" are whitespace tokens. Mean Δ is `change_100` averaged over tracked parties (negative = moved away from the proposition; for ConspiracyAI the proposition is the conspiracy belief, so negative means debunked).

## Per-dataset notes and exceptions

**ConspiracyAI** (`conspiracyai.jsonl`). Source `AllDataForPublication.PPI.8.28.24.csv` (5,647 rows, R-written; only `""`/`"NA"` are nulls). Record = one participant; `--filter complete` (default, 1,749) keeps rows with all 6 turns and both ratings; `valid` (1,269) and `analysis` (1,146) replicate the paper's exclusions. Proposition = GPT's restatement of the participant's conspiracy with the "The assertion is that …" wrapper stripped (`provenance: rewritten`); `group_key` buckets near-identical restatements. P1's `initial.statement` is their opening message. Follow-up ratings (10 days / 2 months) are in `extra`.

**DEBATE** (`debate.jsonl`). Only the `golden/` subset (263 of 264 files; one unusable). Parsing, usability filter, and md5-path split are taken verbatim from the reference scripts. Four tracked humans, each with a private 1–6 rating + free-text justification before and after. Turns alternate per round: one public tweet per person (`channel: public`, `reply_to: null`) then the private dyad (`channel: private`); partners rotate every round. Whether a person moved is `parties[].opinion.changed` (and `outcome.delta_100`); `outcome.changed` is null so it is not read as a group label, and `outcome.any_changed` is true when any of the four moved. `persuader_relative_delta_100` is null (no designated persuader). One corrupted record (`01JSN7VPATX8B4TZZ13D1WH07K`) has null speakers; the adapter keeps it with `speaker_role=unknown` and excludes unattributable private turns.

**PersuasionTrace** (`persuasiontrace.jsonl`). 15 result files, 450 rounds, 92 identical re-exports removed by content fingerprint, 5 empty rounds dropped → 353. Belief is 0–1; a serial question after every persuader message gives the only per-turn trace in the corpus, stored both in `opinion.trace` and as `annotations.target_belief_after` on the persuader turn. Rhetoric (logos/pathos/ethos) scores from the `annotations/` folder are joined onto persuader turns where available. 68 control dialogues discuss a different topic than the claim that was rated. `proposition.text` is that discussion topic (`proposition_during_round`); the rated claim is kept as `proposition.original_text` and `extra.measured_proposition`. `--exclude-control` → 296. The DataLoader supports `exclude_control=True` / `filter(control=False)`.

**AnthropicPersuasion** (`anthropicpersuasion.jsonl`). Record = one (reader, argument); every argument was shown to 3 readers (`extra.argument_id` links them; split is hashed on it). Persuader is `AI` with the model name, or `P1` for human-written arguments; all arguments argue *for* the claim, so `assigned_stance: support`. The published `persuasiveness_metric` disagrees with `final − initial` in 4 rows; we keep the recomputed value and flag the mismatch. 57 control rows use factual claims.

**CMV** (`cmv.jsonl`). The ConvoKit Winning-Arguments corpus labels *threads*, not posts: every one of the 3,051 posts has ≥1 successful (Δ) thread and a topic-matched unsuccessful sibling, so a per-post record would always be "changed". Record = one labelled thread (root challenger comment + the OP↔challenger exchange beneath it; all are dyadic). `proposition.text` is the title with "CMV:" stripped; `initial.statement` is the post body with the moderator footnote removed; `final.statement` is the OP's last reply in the thread (absent in 2,092 threads, mostly unsuccessful). No numeric ratings anywhere — the participant-episode layer applies an explicit binary proxy on a 0–100 scale (`stance_type=binary_proxy`): successful Δ → initial=100/final=0; unsuccessful → 100/100; unknown → 100/null. Paper's heldout posts → `test`; 12% of train posts → `val`. `extra.paired_record_ids` gives the matched counterpart(s). `--include-unlabelled` emits all ~270k unlabelled threads with `changed: null`.

**DebateGPT** (`debategpt.jsonl`). 750 participant rows → 600 debates: Human-Human debates have both participants (P1 = Pro, P2 = Con, both tracked); the three single-row treatments have one tracked P1 and an untracked opponent (`P2` human or `AI` = GPT-4). Debate *questions* are rewritten to declarative statements that the Pro side defends (dictionary in the script, `provenance: rewritten`). Phases were written in parallel; within a phase the Pro turn is listed first. `extra.participants` keeps `topic_prior`, `perceived_opponent`, and the derived agreement-with-opponent's-side scores.

## Other things worth knowing

- **Granularity differs**: ConspiracyAI/PersuasionTrace/DebateGPT records are single conversations; AnthropicPersuasion records are single readings (no reply); CMV records are sub-threads of a larger post; DEBATE records contain four tracked opinions at once. Filter on `structure.type` or `len(parties)` before pooling.
- **Only three datasets have a human-authored opinion statement** (ConspiracyAI, DEBATE, CMV); only PersuasionTrace has a within-conversation opinion trajectory.
- **Persuader identity**: LLM persuaders are `kind: llm` with `model` set; human persuaders (CMV challengers, Anthropic human writers, DebateGPT opponents) are `kind: human` and `tracked: false`.
- **Known imperfections**: CMV has 11 title-only posts (turn 0 falls back to the title, `body_empty: true`) and 2 posts with an unstripped footnote variant; 423 CMV threads branch (`thread_is_chain: false`, structure preserved via `reply_to`); DebateGPT turn order within a phase is a convention, not observed timing.
- **Non-commercial licenses** apply to DEBATE and AnthropicPersuasion; CMV is Reddit user content; check `source.license` before redistribution.
- `python.org` Python 3.13 ships without CA certificates; `preprocess_cmv.py` sets `SSL_CERT_FILE` from `certifi` before calling `convokit.download`.
