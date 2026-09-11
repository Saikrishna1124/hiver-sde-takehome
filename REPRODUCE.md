# Reproduction Guide

End-to-end pipeline for the Hiver SDE Intern take-home assignment.
Every command below has been verified on the project. No internet access is
required beyond the initial `pip install` and the raw dataset download.

---

## Prerequisites

**Python**: 3.10 or later  
**Dataset**: Download `twcs.csv` from
[Kaggle — Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
and place it at `data/raw/twcs.csv`.

> This repository was developed and tested on the first 59,565 rows of that
> file. All measured counts in the reports reflect this slice.

**Install dependencies** (one-time):

```
pip install -r requirements.txt
```

`requirements.txt` pins: `pandas==2.2.2`, `numpy==1.26.4`,
`scikit-learn==1.5.1`, `matplotlib==3.9.2`, `pyyaml==6.0.2`,
`python-dotenv==1.0.1`, `pytest==8.3.2`.

No Gemini API key is required to reproduce the full pipeline. The Stage 2
dataset is reproduced from a committed cache of 29 Gemini labels (see Step 5).

---

## Step-by-step pipeline

All commands are run from the **project root** (`hiver-sde-assignment/`).

---

### Step 1 — Conversation reconstruction

Walks the raw `twcs.csv` reply chains, keeps AmazonHelp threads, and writes
conversation and customer-reply-pair tables.

```
py -m src.conversation_builder
```

**Outputs**

| File | Description |
|---|---|
| `data/processed/amazonhelp/conversations.csv` | 1,633 full conversation threads |
| `data/processed/amazonhelp/pairs.csv` | 3,603 customer to AmazonHelp reply pairs |
| `data/processed/amazonhelp/stats.json` | Measured counts |

**Approximate runtime**: < 60 s

---

### Step 2 — Leakage-safe train / dev split

Partitions the 3,603 pairs into train and dev at the **conversation level** so
no conversation appears in both splits. All 200 golden-set conversations and
their siblings are excluded before splitting.

```
py -m src.split
```

**Outputs**

| File | Rows | Description |
|---|---|---|
| `data/processed/amazonhelp/train.csv` | 2,139 | Training pairs (1,115 conversations) |
| `data/processed/amazonhelp/dev.csv` | 579 | Dev pairs (278 conversations) |
| `data/processed/amazonhelp/split_report.json` | — | Leakage verification counts |

Verified leakage: `conversation_overlap = 0`, `golden_tweet_overlap_train = 0`,
`golden_conv_overlap_train = 0`.

**Approximate runtime**: < 30 s

---

### Step 3 — Stage 1 training labels

Applies the locked 10-intent taxonomy regex rules to every training row and
categorises each as `single_match` (high-confidence), `multi_match` (ambiguous),
or `zero_match` (no rule fires). Stage 1 keeps single-match rows and a
controlled random sample of 400 zero-match ("other") rows. Multi-match rows
are excluded entirely.

```
py -m src.generate_training_labels
```

**Outputs**

| File | Rows | Description |
|---|---|---|
| `data/processed/amazonhelp/train_stage1.csv` | 1,224 | High-confidence labelled training set |
| `data/processed/amazonhelp/training_label_report.json` | — | Match-category breakdown |
| `data/processed/amazonhelp/training_label_strategy.md` | — | Methodology document |

**Approximate runtime**: < 30 s

---

### Step 4 — Validate the golden evaluation set

Confirms the 200 hand-labelled golden examples are structurally valid and
contain no leakage into the training data. Exits with code 0 if valid.

```
py -m src.validate_golden_set
```

Expected output:
```
rows: 200   labelled: 200   remaining: 0
OK: golden set is valid and has enough labels.
```

**Golden set location**: `evaluation/golden_set.csv`  
**Labels**: 200 / 200 completed (all 10 intents present)  
**Approximate runtime**: < 5 s

---

### Step 5 — Stage 2 dataset (cache replay, no API required)

Stage 2 uses Gemini to re-label the 50 most ambiguous (multi_match) training
rows and then assembles a combined dataset that also contains the Stage 1
high-confidence rows. 29 of the 50 labels are already cached in
`data/processed/amazonhelp/cache/gemini_labels.json` from prior runs. The
remaining 21 are handled by the existing fallback logic: uncached multi-match
rows are skipped (not fabricated), so the Stage 2 dataset is fully reproducible
from cache alone using --dry-run.

```
py -m src.generate_stage2_labels --subset ambiguous --limit 50 --dry-run
```

**Outputs**

| File | Rows | Description |
|---|---|---|
| `data/processed/amazonhelp/train_stage2.csv` | 1,253 | Stage 2 training set |
| `data/processed/amazonhelp/stage2_label_report.json` | — | Source breakdown and label distribution |
| `data/processed/amazonhelp/stage1_vs_stage2_comparison.md` | — | Agreement / disagreement report |

**Stage 2 composition**:
- `stage1_rule`: 824 rows (high-confidence single-match labels)
- `stage1_sampled_other`: 400 rows (controlled zero-match sample)
- `llm_gemini`: 29 rows (Gemini-distilled labels from cache)

**LLM agreement with Stage 1 heuristics**: 17 / 29 = 58.6%  
**Approximate runtime**: < 10 s (no API calls made)

---

### Step 6 — Train and evaluate the baseline classifier

Trains a TF-IDF + Logistic Regression pipeline on `train_stage1.csv` and
evaluates it on two splits:

1. **Dev set** (579 rows, heuristic proposed labels) — for model development.
2. **Golden set** (200 rows, human ground-truth labels) — the sealed benchmark.

```
py -m src.train_baseline
```

**Outputs**

| File | Description |
|---|---|
| `models/baseline_tfidf_logreg.joblib` | Saved sklearn pipeline |
| `outputs/baseline_report.md` | Human-readable evaluation report |
| `data/processed/amazonhelp/baseline_report.json` | Full metrics + confusion matrix (JSON) |

**Key results (golden set)**:

| Metric | Score |
|---|---|
| Accuracy | 60.0% |
| Macro F1 | 0.2298 |
| Weighted F1 | 0.5247 |

**Approximate runtime**: < 60 s

---

### Step 7 — Run the test suite

```
py -m pytest -q
```

Expected: **67 passed** in < 15 s. Tests cover data loading, preprocessing,
conversation building, splits, golden-set validation, Stage 1 labels,
LLM labeler (mock mode, cache, leakage checks), the baseline classifier,
and the majority-class baselines.

---

## Total end-to-end runtime

| Step | Time |
|---|---|
| Step 1 - conversation builder | ~60 s |
| Step 2 - split | ~30 s |
| Step 3 - Stage 1 labels | ~30 s |
| Step 4 - golden set validation | ~5 s |
| Step 5 - Stage 2 (cache replay) | ~10 s |
| Step 6 - baseline train + eval | ~60 s |
| Step 7 - tests | ~15 s |
| **Total** | **< 4 minutes** |

---

## Key generated files (already committed)

The following files are already present in the repository and will be
**overwritten** (identically) by re-running the pipeline:

```
data/processed/amazonhelp/
  conversations.csv
  pairs.csv
  train.csv
  dev.csv
  train_stage1.csv
  train_stage2.csv          <- Stage 2; reproduced from cache without API
  baseline_report.json
  stage2_label_report.json
  stage1_vs_stage2_comparison.md
  training_label_strategy.md
  cache/
    gemini_labels.json      <- 29 Gemini labels; DO NOT DELETE
evaluation/
  golden_set.csv            <- 200/200 hand-labelled; DO NOT OVERWRITE
models/
  baseline_tfidf_logreg.joblib
outputs/
  baseline_report.md
```

> **Important**: `evaluation/golden_set.csv` contains the sealed human labels.
> Do not re-run `py -m src.sample_golden_set` -- it will refuse to overwrite
> once labels are present, but the warning is here for clarity.

---

## Scope note

This submission implements and evaluates the **intent-classification core** of
the AI support agent pipeline:

- Conversation reconstruction and leakage-safe dataset construction
- 10-intent taxonomy with locked definitions
- Two-stage training-data labelling (rule-based + Gemini distillation)
- TF-IDF + Logistic Regression baseline evaluated on a sealed golden set
- 67 unit and integration tests

Reply generation, retrieval, and escalation routing are the planned next
engineering phase, documented in `roadmap.md`.
