# Data

## Where the file goes

Put the dataset here:

```
hiver-sde-assignment/data/raw/twcs.csv
```

The path is configurable in `config.yaml` (`data.raw_csv`).

**Filename:** `twcs.csv`
**Source:** Kaggle — "Customer Support on Twitter" by thoughtvector
<https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter>
The full file is ~2.8M rows / ~256 MB.

**What is currently on disk:** a real slice of that file — **59,565 rows, 9.9 MB**
(the first rows of `twcs.csv`, uploaded because the full file exceeds the upload
limit). Every number in this repo therefore describes the slice, not the full
dataset. This must be stated in the report.

## How to run the analysis

```bash
cd hiver-sde-assignment
python3 -m src.data_analysis            # whole file, chunked
python3 -m src.data_analysis --rows 20000   # smaller/faster run
```

Output: printed report + `data/processed/brand_analysis.csv`.

Other scripts: `python3 -m src.inspect_data`, `python3 -m src.brand_stats`.

## Actual column meanings (verified from the file, not assumed)

| column | meaning |
| --- | --- |
| `tweet_id` | unique id of the tweet |
| `author_id` | who wrote it — a brand handle (e.g. `AmazonHelp`) for support, an anonymised number (e.g. `115712`) for customers |
| `inbound` | `True` = written by a customer, `False` = written by the support account |
| `created_at` | timestamp, e.g. `Tue Oct 31 22:10:47 +0000 2017` |
| `text` | the tweet text; the other party's handle appears as a leading `@mention` |
| `response_tweet_id` | id(s) of the tweet(s) that replied to this one (comma-separated, may be empty) |
| `in_response_to_tweet_id` | id of the tweet this one replies to (empty if it starts a thread) |

**Conversation reconstruction:** follow `in_response_to_tweet_id` from any tweet
back to its parent until a tweet with an empty parent is reached. That chain,
ordered by `created_at`, is one multi-turn conversation. A customer→brand pair is
an outbound tweet whose parent tweet has `inbound=True`.

## Measured statistics (59,565-row slice)

- 7 columns, 0 duplicate `tweet_id`
- `inbound`: 32,682 customer tweets / 26,883 support tweets
- empty values: `text` 1, `response_tweet_id` 19,409, `in_response_to_tweet_id` 15,241
  (the last two are simply thread ends/starts, not corrupt data)
- 105 distinct brand accounts

Brand ranking lives in `data/processed/brand_analysis.csv`.

**Selected brand:** AmazonHelp

AmazonHelp was selected based on measured customer-support activity in the available
59,565-row dataset slice. All downstream pipeline results in this repository are
specific to the AmazonHelp brand.
