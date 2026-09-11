# Decision Log

Non-obvious engineering and data decisions made during this project, with
rationale and the risk each decision prevents. Listed in roughly pipeline order.

---

## D1 - AmazonHelp selected by measured support volume, not name recognition

**Decision**: The brand was chosen by running `src/data_analysis.py` against
the real dataset and sorting all 105 brands by `replies_to_customers`.
AmazonHelp ranked first. The choice is recorded in `config.yaml:brand.selected`
and the full ranking is written to `data/processed/brand_analysis.csv`.

**Why**: Picking by name could introduce selection bias and would not be
reproducible. Using a measured, sortable metric makes the decision defensible
and repeatable by anyone who re-runs the pipeline.

**Risk prevented**: Accidentally choosing a brand with too few examples to
train or evaluate a classifier (some brands have fewer than 10 reply pairs).

---

## D2 - Raw dataset read in chunks; never fully loaded into memory

**Decision**: `src/data_loader.py` uses `pd.read_csv(..., chunksize=100_000)`
(line 37) and never loads the entire file at once. `config.yaml:scan_rows`
caps how many rows are retained.

**Why**: The full Kaggle file is ~2.8 M rows. Loading it in one call would
exceed 4-8 GB of RAM and make the pipeline unusable on a laptop.

**Risk prevented**: Out-of-memory crashes on typical development hardware.
The chunked approach also keeps the pipeline honest: all counts are reported
as "on this slice", not claimed as the whole dataset.

---

## D3 - Raw tweet text is never overwritten; cleaning writes a separate column

**Decision**: `src/preprocessing.py` (docstring lines 3-4, implementation
line 49) adds a new `customer_message_clean` column and never modifies
`customer_message`. Every subsequent stage reads from the clean column but
the original is always present.

**Why**: Cleaning is lossy (URLs and @mentions are dropped). If the cleaning
logic changes, the original text must still be available for re-processing
without re-downloading anything.

**Risk prevented**: Irreversible data corruption. A cleaning bug found after
labelling can be fixed by re-running preprocessing; no labels are lost.

---

## D4 - All CSV columns read as strings (dtype=str)

**Decision**: Every `pd.read_csv` call passes `dtype=str` and
`keep_default_na=False` (e.g. `data_loader.py` line 39,
`generate_stage2_labels.py`, `split.py`).

**Why**: Tweet IDs are large integers. Pandas infers them as `float64` by
default, which silently appends `.0` and breaks all ID-based joins, leakage
checks, and cache lookups (e.g. "11078" vs "11078.0").

**Risk prevented**: Silent ID mismatches causing incorrect leakage assertions,
broken conversation reconstruction, and wrong train/dev splits.

---

## D5 - Near-duplicate messages removed before splitting and golden-set sampling

**Decision**: Both `src/split.py` (lines 56-59) and `src/sample_golden_set.py`
(lines 10-12) independently normalise message text (lowercase, strip
punctuation/digits, collapse whitespace) and drop all but the first occurrence
of each normalised string before proceeding.

**Why**: Twitter support data contains many near-identical messages. If a
near-duplicate of a golden message appeared in training, the classifier gets
an unfair preview of its test set at a near-lexical level.

**Risk prevented**: Near-duplicate leakage. The split report records 138
near-duplicates removed; the golden-set sampling report records 119.

---

## D6 - Conversation-level train/dev split (not row-level)

**Decision**: `src/split.py` (lines 64-75) groups rows by `conversation_id`,
splits the list of conversation IDs, then assigns all messages from a
conversation to the same split.

**Why**: A single customer thread can contain 2-10 messages. A row-level split
would place some messages from the same thread in train and others in dev,
giving the model contextual pre-exposure to its own test examples.

**Risk prevented**: Conversation-context leakage. `split_report.json` verifies
`conversation_overlap = 0` and `golden_conv_overlap_train = 0` explicitly.

---

## D7 - Golden evaluation set locks out entire conversations, not just golden tweets

**Decision**: `src/validate_golden_set.py:excluded_tweet_ids()` returns the 200
golden tweet IDs. `src/split.py` (lines 43-54) goes further: it identifies every
conversation containing any golden message and excludes ALL messages from that
conversation from train and dev.

**Why**: A golden message has conversation siblings (other customer and support
tweets in the same thread). If those appear in training, the model learns patterns
from the conversation context of its test examples.

**Risk prevented**: Indirect leakage through conversation siblings. 547 sibling
messages across 181 full conversations were excluded (verified in `split_report.json`).

---

## D8 - Stratified golden-set sampling by intent x message length

**Decision**: `src/sample_golden_set.py` (lines 13-23) builds strata as
`proposed_intent x length_bucket` (short/medium/long), allocates 200 slots
proportionally with a floor of 1 per stratum, then samples inside each stratum
with a fixed seed.

**Why**: Simple random sampling would under-represent rare intents (e.g.
`prime_membership` has only 21 examples in the pool). A purely proportional
allocation would squeeze rare intents to zero slots. The floor-of-1 guarantee
means every intent-length combination is represented.

**Risk prevented**: A golden set dominated by the majority class (`other`, 45%)
making evaluation useless for rare intents. All 10 intents are present.

---

## D9 - Stage 1 excludes multi-match rows entirely; no tie-breaking rule

**Decision**: `src/generate_training_labels.py` drops all `multi_match` rows
(where 2+ taxonomy rules fire) from the Stage 1 training set. No priority
ordering or random choice picks a label among the matching rules.

**Why**: A row matching multiple rules is genuinely ambiguous. Applying a
priority ordering injects the author's arbitrary preference as a training signal.
Stage 2 LLM labelling is the designed mechanism for resolving these rows.

**Risk prevented**: Contradictory supervision signals in the training set that
would hurt classifier accuracy and conceal the value of Stage 2 distillation.
163 ambiguous rows fall into this category.

---

## D10 - Zero-match ("other") rows sampled at exactly 400 with seed 42

**Decision**: `src/generate_training_labels.py` sets
`DEFAULT_OTHER_SAMPLE_SIZE = 400` and `DEFAULT_SEED = 42`. The sample is drawn
reproducibly with these fixed values.

**Why**: There are ~1,000+ zero-match rows. Including all would make ~50% of
the training set "other", causing model collapse. A fixed cap and seed make the
training set both balanced and fully deterministic.

**Risk prevented**: Class-imbalance-driven model collapse and non-reproducible
training data that differs between runs.

---

## D11 - Gemini LLM cache uses atomic write (write-then-rename)

**Decision**: `src/llm_labeler.py:DiskCache.save()` (lines 74-82) writes to a
`.tmp` file first, then renames it to the real path with
`tmp_path.replace(cache_path)`. Each successful label is persisted immediately
after API validation.

**Why**: A process crash or Ctrl-C mid-write would corrupt a partially written
JSON file and lose completed labels. The write-then-rename idiom is atomic on
all major OS filesystems.

**Risk prevented**: Cache corruption that would force re-spending API quota on
already-answered examples - critical under a free-tier daily quota limit.

---

## D12 - Gemini API called with temperature=0.0 for deterministic labels

**Decision**: `src/llm_labeler.py` (line 301) sets `"temperature": 0.0` in
the Gemini generation config payload.

**Why**: Intent classification is a deterministic categorisation task.
Temperature=0 makes the model always output its highest-probability token
sequence, so repeated calls on the same message produce the same label,
which is required for a meaningful cache.

**Risk prevented**: Non-deterministic labels where the same message is
classified differently on different API calls, making the cache unreliable
and the distillation process unreproducible.

---

## D13 - Stage 2 skips uncached multi-match rows rather than fabricating labels

**Decision**: `src/generate_stage2_labels.py:build_stage2_dataset()` (lines
93-95) uses `continue` for any multi-match row not in the Gemini cache and
with no Stage 1 fallback. Those rows are absent from `train_stage2.csv`.
No label is invented.

**Why**: 21 rows could not be labelled due to a Gemini free-tier daily quota
limit. The correct response is to exclude them, not assign a default or
majority label that would be fabricated ground truth.

**Risk prevented**: Noise in the training set through fake labels. Every row
in `train_stage2.csv` has an auditable label source: `stage1_rule`,
`stage1_sampled_other`, or `llm_gemini`.

---

## D14 - Train-fitted majority is the valid trivial baseline; golden-oracle majority is a reference only

**Decision**: `src/majority_baseline.py` computes and clearly labels two
majority-class numbers. Baseline A (train-fitted, **22.0%**) picks
`delivery_order_status` (most frequent in training) and predicts it for
every golden example. Baseline B (golden-oracle, **45.0%**) is computed
from the golden set itself and explicitly labelled as an upper-bound reference.

**Why**: A system cannot know the test-set distribution at prediction time.
Using 45.0% (which requires knowing 45% of the golden set is "other") as the
comparison bar is misleading.

**Risk prevented**: Understating the model improvement. Against the train-fitted
majority (22.0%), the TF-IDF classifier's 60.0% is a +38 pp gain - not the
weaker +15 pp it appears against the oracle majority.

---

## D15 - Heuristic suggestions kept in a separate column, never used as labels

**Decision**: `src/sample_golden_set.py` (line 51) stores the taxonomy regex
guess in `suggested_intent`, structurally distinct from `gold_intent`. The
labelling tool (`src/label_golden_set.py`) displays it marked as
"heuristic hint - NOT the answer".

**Why**: `suggested_intent` is computed by the same regex rules used to
generate Stage 1 training labels. If stored in the same column as ground truth,
there is a risk of accidentally evaluating the classifier against its own
training heuristic.

**Risk prevented**: Circular evaluation - testing the model against the same
rules it was trained on - which would produce inflated and meaningless results.
