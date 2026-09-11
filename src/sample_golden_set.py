"""Phase 12a - sample 200 real AmazonHelp customer messages for MANUAL labelling.

Nothing here produces ground truth. It only *selects* which real messages the
human annotator will see, in a way that is reproducible and defensible.

Sampling strategy (seeded, seed comes from config.yaml:random_seed)
-------------------------------------------------------------------
1. Start from `data/processed/amazonhelp/pairs.csv` - every real customer
   message that AmazonHelp actually answered (3,603 rows on our slice).
2. Drop near-duplicates: normalise text (lowercase, strip punctuation and
   digits, collapse whitespace) and keep the FIRST occurrence of each
   normalised string. Removed rows are counted, never deleted from disk.
3. Build a stratum key = proposed_intent x length bucket
   (short < 60 chars, medium 60-140, long > 140).
   The proposed_intent comes from `src.taxonomy` and is a *heuristic*, used
   here only to spread the sample across topics - it is NOT a label.
4. Allocate the 200 slots proportionally to stratum size, but give every
   non-empty stratum at least 1 slot, so rare intents and unusual lengths
   are represented. `other` therefore stays present (it is 56% of the data)
   but does not swallow the whole sample.
5. Sample inside each stratum with `df.sample(n, random_state=seed)`, then
   sort the final 200 by `created_at` so the annotator sees a time-ordered
   mix rather than topic-ordered blocks.

Run:  python3 -m src.sample_golden_set
Writes: evaluation/golden_set.csv        (gold_intent EMPTY - you fill it)
        evaluation/sampling_report.json  (the numbers below)
"""

from __future__ import annotations

import json
import re

import pandas as pd

from .config import ROOT, load_config, resolve
from .taxonomy import assign

TARGET_N = 200
GOLDEN_CSV = ROOT / "evaluation" / "golden_set.csv"
REPORT_JSON = ROOT / "evaluation" / "sampling_report.json"

COLUMNS = [
    "example_id",
    "conversation_id",
    "customer_tweet_id",
    "customer_message",
    "gold_intent",  # <- stays empty until a human labels it
    "label_notes",
    "suggested_intent",  # <- heuristic hint ONLY, never ground truth
    "length_bucket",
    "is_near_duplicate_of",
]

_NORM_RE = re.compile(r"[^a-z ]+")


def normalise(text: str) -> str:
    """Aggressive normalisation used ONLY for near-duplicate detection."""
    t = str(text).lower()
    t = _NORM_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def length_bucket(text: str) -> str:
    n = len(str(text))
    if n < 60:
        return "short"
    if n <= 140:
        return "medium"
    return "long"


def allocate(sizes: dict[str, int], target: int) -> dict[str, int]:
    """Proportional allocation with a floor of 1 per non-empty stratum."""
    strata = [k for k, v in sizes.items() if v > 0]
    alloc = {k: 1 for k in strata}
    remaining = target - len(alloc)
    if remaining <= 0:
        return alloc
    total = sum(sizes[k] for k in strata)
    # largest-remainder method, so the counts add up exactly
    exact = {k: remaining * sizes[k] / total for k in strata}
    for k in strata:
        alloc[k] += int(exact[k])
    left = target - sum(alloc.values())
    for k in sorted(strata, key=lambda k: exact[k] - int(exact[k]), reverse=True)[:left]:
        alloc[k] += 1
    # never ask for more rows than a stratum has, and hand the freed slots to
    # strata that still have spare rows, so we always reach `target`
    for k in strata:
        alloc[k] = min(alloc[k], sizes[k])
    while sum(alloc.values()) < min(target, sum(sizes[k] for k in strata)):
        spare = sorted(
            (k for k in strata if alloc[k] < sizes[k]),
            key=lambda k: sizes[k] - alloc[k],
            reverse=True,
        )
        alloc[spare[0]] += 1
    return alloc



def build_sample(pairs: pd.DataFrame, seed: int, target: int = TARGET_N):
    df = pairs.copy()
    df["_norm"] = df["customer_message_clean"].map(normalise)
    df = df[df["_norm"].str.len() > 0]

    first_seen = df.drop_duplicates("_norm", keep="first")
    n_dupes_removed = len(df) - len(first_seen)
    pool = first_seen.copy()

    pool["suggested_intent"] = pool["customer_message_clean"].map(assign)
    pool["length_bucket"] = pool["customer_message"].map(length_bucket)
    pool["_stratum"] = pool["suggested_intent"] + "|" + pool["length_bucket"]

    sizes = pool["_stratum"].value_counts().to_dict()
    alloc = allocate(sizes, target)

    picked = [
        pool[pool["_stratum"] == k].sample(n, random_state=seed)
        for k, n in sorted(alloc.items())
        if n > 0
    ]
    out = pd.concat(picked, ignore_index=True)
    out = out.sort_values("created_at", kind="mergesort").reset_index(drop=True)
    out["example_id"] = [f"g{ i + 1 :03d}" for i in range(len(out))]
    out["gold_intent"] = ""
    out["label_notes"] = ""
    out["is_near_duplicate_of"] = ""
    return out[COLUMNS], {
        "pool_rows": int(len(df)),
        "near_duplicates_removed": int(n_dupes_removed),
        "deduplicated_pool": int(len(pool)),
        "sampled": int(len(out)),
        "seed": seed,
        "strata_used": len(alloc),
        "allocation": {k: int(v) for k, v in sorted(alloc.items())},
        "suggested_intent_spread": out["suggested_intent"].value_counts().to_dict(),
        "length_spread": out["length_bucket"].value_counts().to_dict(),
    }


def main() -> None:
    cfg = load_config()
    brand = cfg["brand"]["selected"]
    seed = int(cfg.get("random_seed", 42))
    pairs_path = resolve(cfg["data"]["processed_dir"]) / brand.lower() / "pairs.csv"
    pairs = pd.read_csv(pairs_path, keep_default_na=False, dtype=str)

    if GOLDEN_CSV.exists():
        existing = pd.read_csv(GOLDEN_CSV, keep_default_na=False, dtype=str)
        if (existing.get("gold_intent", pd.Series(dtype=str)).str.len() > 0).any():
            print(
                f"REFUSING to overwrite {GOLDEN_CSV}: it already contains manual labels.\n"
                "Delete or rename it first if you really want a fresh sample."
            )
            return

    sample, report = build_sample(pairs, seed)
    GOLDEN_CSV.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(GOLDEN_CSV, index=False)
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"sampled {report['sampled']} messages from {report['deduplicated_pool']} "
          f"de-duplicated candidates (removed {report['near_duplicates_removed']} near-duplicates)")
    print(f"seed={seed}  strata={report['strata_used']}")
    print("\nsuggested_intent spread (HINTS ONLY, not labels):")
    for k, v in report["suggested_intent_spread"].items():
        print(f"  {k:>24}: {v}")
    print("\nlength spread:", report["length_spread"])
    print(f"\nwritten: {GOLDEN_CSV}  (gold_intent is EMPTY - label it yourself)")
    print(f"written: {REPORT_JSON}")


if __name__ == "__main__":
    main()
