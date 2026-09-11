"""Phase 12c - sanity checks on the golden set.

Run:  python3 -m src.validate_golden_set
Exit code 0 = usable for evaluation, 1 = problems found.
"""

from __future__ import annotations

import sys

import pandas as pd

from .config import ROOT
from .intents import INTENTS

GOLDEN_CSV = ROOT / "evaluation" / "golden_set.csv"
MIN_LABELLED = 150


def validate(df: pd.DataFrame, min_labelled: int = MIN_LABELLED) -> list[str]:
    """Return a list of problems. Empty list = the set is valid."""
    problems: list[str] = []

    required = {"example_id", "conversation_id", "customer_tweet_id",
                "customer_message", "gold_intent", "label_notes"}
    missing = required - set(df.columns)
    if missing:
        problems.append(f"missing columns: {sorted(missing)}")
        return problems

    if df["example_id"].duplicated().any():
        dup = df.loc[df["example_id"].duplicated(), "example_id"].tolist()
        problems.append(f"duplicate example_id: {dup}")

    if df["customer_tweet_id"].duplicated().any():
        dup = df.loc[df["customer_tweet_id"].duplicated(), "customer_tweet_id"].tolist()
        problems.append(f"duplicate customer_tweet_id: {dup}")

    empty_msgs = df[df["customer_message"].astype(str).str.strip() == ""]
    if len(empty_msgs):
        problems.append(f"{len(empty_msgs)} rows have an empty customer_message")

    labelled = df[df["gold_intent"].astype(str).str.strip() != ""]
    bad = sorted(set(labelled["gold_intent"]) - set(INTENTS))
    if bad:
        problems.append(f"invalid intent names: {bad}")

    if len(labelled) < min_labelled:
        problems.append(
            f"only {len(labelled)} labelled examples; the assignment needs at least "
            f"{min_labelled} before any evaluation number may be reported"
        )
    return problems


def excluded_tweet_ids() -> set[str]:
    """LEAKAGE RULE: every id returned here must be removed from any future
    training/development split. Import this, do not re-derive it by hand."""
    df = pd.read_csv(GOLDEN_CSV, keep_default_na=False, dtype=str)
    return set(df["customer_tweet_id"].astype(str))


def main() -> None:
    df = pd.read_csv(GOLDEN_CSV, keep_default_na=False, dtype=str)
    labelled = df[df["gold_intent"].astype(str).str.strip() != ""]

    print(f"rows: {len(df)}   labelled: {len(labelled)}   remaining: {len(df) - len(labelled)}")
    if len(labelled):
        print("\nlabel counts so far (YOUR labels):")
        for k, v in labelled["gold_intent"].value_counts().items():
            print(f"  {k:>24}: {v}")

    problems = validate(df)
    if problems:
        print("\nNOT READY for evaluation:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nOK: golden set is valid and has enough labels.")


if __name__ == "__main__":
    main()
