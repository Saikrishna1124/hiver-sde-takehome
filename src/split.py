"""Phase 5 - leakage-safe conversation-level train/dev split.

Ensures:
1. No tweet from the golden evaluation set enters train or dev.
2. No conversation containing a golden tweet enters train or dev (all siblings excluded).
3. Near-duplicates are identified and removed.
4. Train and dev splits have ZERO conversation overlap.
5. `proposed_intent` is assigned via regex heuristics for baseline exploration only.

Run:  python -m src.split
Writes: data/processed/amazonhelp/train.csv
        data/processed/amazonhelp/dev.csv
        data/processed/amazonhelp/split_report.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ROOT, load_config, resolve
from .sample_golden_set import normalise
from .taxonomy import assign
from .validate_golden_set import excluded_tweet_ids

DEV_CONV_COUNT = 278
SPLIT_SEED = 190


def build_split(
    pairs: pd.DataFrame,
    golden_tweet_ids: set[str],
    dev_conv_count: int = DEV_CONV_COUNT,
    seed: int = SPLIT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Partition pairs into train and dev splits with zero leakage."""
    total_pairs = len(pairs)

    # 1. Identify golden messages and conversations containing them
    is_golden = pairs["customer_tweet_id"].isin(golden_tweet_ids)
    golden_pairs = pairs[is_golden]
    n_golden = len(golden_pairs)

    golden_conv_ids = set(golden_pairs["conversation_id"])

    # 2. Identify conversation siblings (in same thread as a golden message, but not the golden message itself)
    is_sibling = pairs["conversation_id"].isin(golden_conv_ids) & ~is_golden
    n_siblings = int(is_sibling.sum())

    # 3. Filter out all golden conversations
    rem = pairs[~pairs["conversation_id"].isin(golden_conv_ids)].copy()

    # 4. Remove near-duplicates on normalised customer_message
    rem["_norm"] = rem["customer_message"].map(normalise)
    is_near_dup = rem["_norm"].str.len().gt(0) & rem.duplicated("_norm", keep="first")
    n_near_dupes = int(is_near_dup.sum())

    usable = rem[~is_near_dup].copy().drop(columns=["_norm"])
    n_usable = len(usable)

    # 5. Conversation-level split with strict disjointness
    conv_sizes = usable.groupby("conversation_id").size().to_dict()
    conv_ids = sorted(conv_sizes.keys())

    train_convs, dev_convs = train_test_split(
        conv_ids, test_size=dev_conv_count, random_state=seed
    )
    train_conv_set = set(train_convs)
    dev_conv_set = set(dev_convs)

    train_df = usable[usable["conversation_id"].isin(train_conv_set)].copy()
    dev_df = usable[usable["conversation_id"].isin(dev_conv_set)].copy()

    # 6. Assign proposed_intent (heuristic hint for baseline exploration only)
    train_df["proposed_intent"] = train_df["customer_message_clean"].map(assign)
    dev_df["proposed_intent"] = dev_df["customer_message_clean"].map(assign)

    report = {
        "original_pairs": total_pairs,
        "golden_messages_excluded": n_golden,
        "conversation_siblings_excluded": n_siblings,
        "golden_conversations_excluded": len(golden_conv_ids),
        "near_duplicates_removed": n_near_dupes,
        "usable_examples": n_usable,
        "train_examples": len(train_df),
        "dev_examples": len(dev_df),
        "train_conversations": len(train_conv_set),
        "dev_conversations": len(dev_conv_set),
        "conversation_overlap": len(train_conv_set.intersection(dev_conv_set)),
        "golden_tweet_overlap_train": len(set(train_df["customer_tweet_id"]).intersection(golden_tweet_ids)),
        "golden_tweet_overlap_dev": len(set(dev_df["customer_tweet_id"]).intersection(golden_tweet_ids)),
        "golden_conv_overlap_train": len(set(train_df["conversation_id"]).intersection(golden_conv_ids)),
        "golden_conv_overlap_dev": len(set(dev_df["conversation_id"]).intersection(golden_conv_ids)),
        "split_seed": seed,
        "train_intent_distribution": train_df["proposed_intent"].value_counts().to_dict(),
        "dev_intent_distribution": dev_df["proposed_intent"].value_counts().to_dict(),
    }

    return train_df.reset_index(drop=True), dev_df.reset_index(drop=True), report


def main() -> None:
    cfg = load_config()
    brand = cfg["brand"]["selected"]
    out_dir = resolve(cfg["data"]["processed_dir"]) / brand.lower()

    pairs_path = out_dir / "pairs.csv"
    if not pairs_path.exists():
        raise FileNotFoundError(f"{pairs_path} not found. Run `python -m src.conversation_builder` first.")

    pairs = pd.read_csv(pairs_path, keep_default_na=False, dtype=str)
    golden_ids = excluded_tweet_ids()

    train_df, dev_df, report = build_split(pairs, golden_ids)

    train_path = out_dir / "train.csv"
    dev_path = out_dir / "dev.csv"
    report_path = out_dir / "split_report.json"

    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== PHASE 5: LEAKAGE-SAFE SPLIT REPORT ===")
    print(f"Original pairs considered:           {report['original_pairs']}")
    print(f"Golden messages excluded:            {report['golden_messages_excluded']}")
    print(f"Golden conversation siblings excl.:  {report['conversation_siblings_excluded']}")
    print(f"Near-duplicates removed:             {report['near_duplicates_removed']}")
    print(f"Usable examples:                     {report['usable_examples']}")
    print(f"Train examples:                      {report['train_examples']} ({report['train_conversations']} convs)")
    print(f"Dev examples:                        {report['dev_examples']} ({report['dev_conversations']} convs)")
    print(f"Conversation overlap:                {report['conversation_overlap']}")
    print(f"Golden tweet leakage:                {report['golden_tweet_overlap_train'] + report['golden_tweet_overlap_dev']}")
    print(f"Golden conv leakage:                 {report['golden_conv_overlap_train'] + report['golden_conv_overlap_dev']}")
    print(f"\nWritten: {train_path}")
    print(f"Written: {dev_path}")
    print(f"Written: {report_path}")


if __name__ == "__main__":
    main()
