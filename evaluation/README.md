# Golden evaluation set — AmazonHelp

> **Status: 200/200 examples hand-labelled.** `gold_intent` is complete in
> every row. Validation passes (`py -m src.validate_golden_set` exits 0).
> Results are reported in `outputs/report.md` and `outputs/baseline_report.md`.

## What is here

| file | what it is |
| --- | --- |
| `golden_set.csv` | the 200 sampled real messages; `gold_intent` complete (200/200) |
| `sampling_report.json` | the exact numbers produced by the sampler |
| `labeling_guidelines.md` | intent-by-intent rules the annotator follows |
| `label_progress.json` | skipped example ids (created by the labelling tool) |

Columns: `example_id, conversation_id, customer_tweet_id, customer_message,
gold_intent, label_notes, suggested_intent, length_bucket, is_near_duplicate_of`.

`suggested_intent` is a **regex heuristic hint**, never ground truth, and lives
in its own column so it can never be mistaken for a label.

## How the 200 were sampled

Source: `data/processed/amazonhelp/pairs.csv` — 3,603 real customer messages
AmazonHelp actually replied to (measured, from our 59,565-row slice).

1. **Near-duplicate removal.** Text is normalised (lowercase, punctuation and
   digits stripped, whitespace collapsed) and only the first occurrence of each
   normalised string is kept: **119 near-duplicates removed, 3,363 candidates**.
   Nothing is deleted from disk.
2. **Stratification.** stratum = heuristic intent × length bucket
   (short < 60 chars, medium 60–140, long > 140) → **29 non-empty strata**.
3. **Allocation.** proportional to stratum size, with a floor of 1 per stratum,
   so rare intents and unusual lengths cannot be squeezed out; spare slots are
   redistributed so the total is exactly 200.
4. **Draw.** `df.sample(n, random_state=42)` inside each stratum; seed comes
   from `config.yaml:random_seed`.
5. **Order.** the final 200 are sorted by `created_at`, so the annotator sees a
   time-ordered mix across the slice's date range rather than topic blocks, and
   the ordering is not itself a hint.

Resulting spread (hints, **not** labels): other 94 · delivery_order_status 44 ·
service_complaint 13 · refund_return 12 · device_digital_content 9 ·
seller_marketplace 6 · account_access 6 · item_quality_issue 6 ·
prime_membership 5 · payment_billing 5. Lengths: medium 108 · long 58 · short 34.

Re-running the sampler is deterministic. It **refuses to overwrite**
`golden_set.csv` once any manual label exists.

## How they are labelled

```bash
cd hiver-sde-assignment
python3 -m src.sample_golden_set     # already done; regenerates the sample
python3 -m src.label_golden_set      # label; quit and resume any time
python3 -m src.validate_golden_set   # progress + checks
```

The annotator sees one message at a time with its id, progress counter, length,
conversation id and the heuristic hint clearly marked as not the answer. Keys
`1-9` and `0` are the ten intents, `s` skip, `p` previous (clears that label),
`n` add a note, `q` save and quit. Every answer is written straight to
`golden_set.csv` via a temp-file replace, so progress survives a crash and
resuming picks the first unlabelled, unskipped row.

Ambiguity handling and the definition of `other` are in
`labeling_guidelines.md`. Short version: `other` is used when none of the nine
intents describes the real issue; unclear messages are never forced into a
category.

## Leakage prevention

`src/validate_golden_set.py:excluded_tweet_ids()` returns every
`customer_tweet_id` in the golden set.

**Rule: any future train/dev split MUST remove those tweet ids before
training.** No classifier, TF-IDF baseline, embedding model or retrieval index
may be fitted on a row whose `customer_tweet_id` is in that set. The split code
built in the next phase will call this function rather than re-deriving the list
by hand, and a test will assert the intersection is empty.

## Validation checks

`python3 -m src.validate_golden_set` fails (exit 1) on: missing columns,
duplicate `example_id`, duplicate `customer_tweet_id`, empty
`customer_message`, an intent name outside the locked ten, or fewer than 150
labelled rows. No evaluation number may be reported while it fails.

## Label distribution (actual gold_intent, 200 examples)

| Intent | Count |
| :--- | :--- |
| `other` | 90 |
| `delivery_order_status` | 44 |
| `service_complaint` | 15 |
| `refund_return` | 12 |
| `device_digital_content` | 9 |
| `account_access` | 7 |
| `payment_billing` | 6 |
| `seller_marketplace` | 6 |
| `item_quality_issue` | 6 |
| `prime_membership` | 5 |

## Completed TODOs

- [x] label all 200 examples by hand (200/200 complete)
- [x] report the real label distribution (see table above)
- [x] wire `excluded_tweet_ids()` into the train/dev split
- [x] leakage verified: 0 golden tweets in train, 0 golden convs in train
      (see `data/processed/amazonhelp/split_report.json`)

## Remaining TODOs

- [ ] second-annotator pass on a subset for human-agreement (later phase)
- [ ] LLM-as-judge evaluation (Phase 15)
