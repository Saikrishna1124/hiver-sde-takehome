# Hiver SDE Intern Take-Home — AI Customer Support Agent

Building an AI support agent on real Twitter customer-support conversations:
classify the customer's intent, draft a reply grounded in how the brand actually
replied before, and decide AUTO_HANDLED vs HUMAN_ESCALATION — then prove it works
with a hand-labelled golden set, baselines, and an LLM judge.

> **Current status: Full Pipeline Complete & Verified.** Dataset reconstruction,
> leakage-safe split, 10-intent taxonomy, two-stage intent labelling, TF-IDF + LogReg
> baseline, historically-grounded reply retrieval (2,139 training pairs), explainable
> escalation routing (AUTO_HANDLED vs HUMAN_ESCALATION), and LLM-as-a-judge evaluation
> with measured human agreement. **84 tests passing.**
> See `REPRODUCE.md` (< 4 min end-to-end) and `outputs/report.md` for full findings.

## What you need to provide

1. `data/raw/twcs.csv` — download from Kaggle
   [thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
   (see `data/README.md`).
2. A Gemini API key in `.env` (only needed from Phase 10 onward).

## Setup

```bash
cd hiver-sde-assignment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then paste your Gemini key into .env
```

## Run what exists today

```bash
python -m src.inspect_data          # real columns/rows of your twcs.csv
python -m src.data_analysis         # dataset structure + brand ranking (105 brands)
python -m src.conversation_builder  # AmazonHelp threads -> conversations.csv + pairs.csv
python -m src.intent_discovery      # word/bigram stats + candidate intent topics
python -m src.intent_validation     # 20 real messages per candidate, for manual review
python -m src.taxonomy              # proposed final taxonomy + definitions file
python -m src.sample_golden_set     # sample 200 real messages for manual labelling
python -m src.label_golden_set      # hand-label them (resumable)
python -m src.validate_golden_set   # progress + validity checks
pytest -q                           # unit tests
```

`pytest -q` passes: **84 tests, 84 passed** (executed in this repo).

Golden set: 200 real messages — **200/200 hand-labelled**, stratified by intent
and message length, leakage-free. See `evaluation/README.md` and `evaluation/golden_set.csv`.

## AmazonHelp data-processing stage (measured, from the 59,565-row slice)

`src.conversation_builder` walks `in_response_to_tweet_id` to the root of each
reply chain, keeps threads AmazonHelp posted in or that @-mention it, and writes
to `data/processed/amazonhelp/`:

| artefact | contents |
| --- | --- |
| `conversations.csv` | one row per thread + full ordered transcript JSON |
| `pairs.csv` | one row per customer message that AmazonHelp actually answered |
| `stats.json` | the numbers below |
| `candidate_intents.json` | candidate topics with counts and real examples |

Measured: 8,616 tweets in AmazonHelp threads · 4,655 customer tweets · 3,957
AmazonHelp tweets · 1,633 conversations · 1,115 multi-turn · 3,603 customer→reply
pairs · 1,672 unique customers · avg 5.28 / median 4 messages per conversation ·
219 duplicate-text pairs flagged (kept, flagged, not deleted).

## Repository layout so far

```
hiver-sde-assignment/
├── config.yaml            # every knob: paths, sample sizes, brand, seed
├── requirements.txt       # pinned versions
├── .env.example           # GEMINI_API_KEY placeholder; .env is git-ignored
├── data/README.md         # where to put the dataset
├── src/
│   ├── config.py          # loads config.yaml + .env
│   ├── data_loader.py     # chunked CSV reading, configurable row cap
│   ├── preprocessing.py   # clean_text, keeps raw text intact
│   ├── inspect_data.py    # Phase 1: inspect the real file
│   ├── data_analysis.py   # Phase 1+2: structure report + brand ranking
│   ├── brand_stats.py     # earlier brand-count helper
│   ├── conversation_builder.py  # Phase 3: reply chains -> conversations/pairs
│   ├── intent_discovery.py      # Phase 4a: candidate topics (keyword hypotheses)
│   ├── intent_validation.py     # Phase 4b: print real examples per candidate
│   └── taxonomy.py              # Phase 4c: proposed final 9-intent taxonomy
└── tests/                 # unit tests for the above
```

## Design choices already made

- **Chunked reading.** `pd.read_csv(..., chunksize=...)` so the ~3M-row file is
  never fully loaded; `config.yaml:data.scan_rows` caps how much is kept.
- **Raw text is never overwritten.** Cleaning writes a separate `clean_text`
  column so every transformation can be inspected side by side.
- **All columns read as strings** (`dtype=str`) — tweet IDs must not become
  floats, and casting is done later, deliberately.
- **`brand.selected: AmazonHelp`**, chosen from the real brand table
  (`data/processed/brand_analysis.csv`): the largest usable support volume in the
  slice, not because the name is famous.
- **Nothing is deleted silently.** Duplicate text, unanswered messages and
  multiple replies to one message are flagged as columns/stats, not dropped.
- **Intents were reviewed against real messages, not accepted from keywords.**
  `src.intent_discovery` proposes keyword buckets, `src.intent_validation` prints
  20 real messages per bucket, and `src.taxonomy` holds the reviewed 9-intent
  taxonomy. The review (false positives found, merges, drops) is written up in
  `data/processed/amazonhelp/taxonomy_review.md`. These are still **proposed**
  labels — the hand-labelled golden set is a later phase.

## Citations

- Dataset: Thought Vector, *Customer Support on Twitter*, Kaggle.
  https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
- Libraries: pandas, NumPy, scikit-learn, matplotlib, PyYAML, python-dotenv, pytest.
- Further citations (embedding model, FAISS, Gemini, any papers) will be added
  as those components are actually used.
