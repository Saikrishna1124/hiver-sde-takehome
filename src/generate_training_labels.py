"""Stage 1 Training-Label Generation & Rule Analysis.

Evaluates all 9 substantive taxonomy regex rules for every training row:
- zero_match   -> 0 rules matched (default 'other')
- single_match -> exactly 1 rule matched (high confidence)
- multi_match  -> 2+ rules matched (ambiguous, conflicting keywords)

Stage 1 Strategy:
- Exclude all multi_match rows (noisy tie-breaks).
- Retain all single_match rows with the matched intent as label.
- Draw a reproducible, controlled sample of zero_match ('other') rows (seed=42).
- Produce data/processed/amazonhelp/train_stage1.csv
- Produce data/processed/amazonhelp/training_label_report.json
- Produce data/processed/amazonhelp/training_label_strategy.md

Run: python -m src.generate_training_labels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import ROOT, load_config, resolve
from .intents import INTENTS, OTHER
from .taxonomy import FALLBACK, TAXONOMY, match_rules
from .validate_golden_set import GOLDEN_CSV, excluded_tweet_ids

DEFAULT_SEED = 42
DEFAULT_OTHER_SAMPLE_SIZE = 400


def categorize_matches(rules: list[str]) -> str:
    """Categorize row based on number of matching taxonomy rules."""
    n = len(rules)
    if n == 0:
        return "zero_match"
    if n == 1:
        return "single_match"
    return "multi_match"


def analyze_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze all taxonomy regex matches for every row in df."""
    out = df.copy()
    out["matched_rules"] = out["customer_message_clean"].map(match_rules)
    out["match_count"] = out["matched_rules"].map(len)
    out["match_category"] = out["matched_rules"].map(categorize_matches)

    # Assign tentative label
    def get_label(row: pd.Series) -> str:
        cat = row["match_category"]
        if cat == "single_match":
            return row["matched_rules"][0]
        if cat == "zero_match":
            return OTHER
        return ""  # multi_match has no clean single label

    out["training_label"] = out.apply(get_label, axis=1)
    return out


def build_stage1_dataset(
    analyzed_df: pd.DataFrame,
    other_sample_size: int = DEFAULT_OTHER_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, dict]:
    """Filter to single_match + controlled sample of zero_match (other)."""
    # 1. Separate by category
    single_df = analyzed_df[analyzed_df["match_category"] == "single_match"].copy()
    multi_df = analyzed_df[analyzed_df["match_category"] == "multi_match"].copy()
    zero_df = analyzed_df[analyzed_df["match_category"] == "zero_match"].copy()

    # 2. Draw controlled sample of zero_match ('other')
    n_sample_other = min(other_sample_size, len(zero_df))
    sampled_other = zero_df.sample(n=n_sample_other, random_state=seed).copy()

    # 3. Combine single_match and sampled_other (multi_match completely excluded)
    combined = pd.concat([single_df, sampled_other], ignore_index=True)

    # Sort deterministically
    combined = combined.sort_values(
        ["created_at", "conversation_id", "customer_tweet_id"]
    ).reset_index(drop=True)

    # Clean matched_rules for CSV serialization
    combined["matched_rules_str"] = combined["matched_rules"].map(lambda r: ";".join(r))

    # Metrics
    category_counts = analyzed_df["match_category"].value_counts().to_dict()
    label_distribution = combined["training_label"].value_counts().to_dict()

    report = {
        "total_train_rows_evaluated": len(analyzed_df),
        "match_breakdown": {
            "zero_match": category_counts.get("zero_match", 0),
            "single_match": category_counts.get("single_match", 0),
            "multi_match": category_counts.get("multi_match", 0),
        },
        "stage1_selection": {
            "single_match_kept": len(single_df),
            "multi_match_excluded": len(multi_df),
            "zero_match_pool": len(zero_df),
            "zero_match_sampled": n_sample_other,
            "seed": seed,
            "total_stage1_rows": len(combined),
        },
        "label_distribution": label_distribution,
    }

    return combined, report


def generate_markdown_strategy(report: dict, out_path: Path) -> None:
    """Generate Markdown summary documenting the Stage 1 training strategy."""
    lines = [
        "# Training Label Strategy: Stage 1 (Rule-Confidence Filtering)",
        "",
        "## Overview",
        "The training set contains heuristic regex proposals that exhibit noise when",
        "multiple rules fire or when no rules fire. To create a high-quality baseline",
        "dataset without manual annotation of 2,139 rows, Stage 1 uses a hybrid",
        "rule-confidence filtering strategy.",
        "",
        "## Measured Rule Matches on Train Set (2,139 rows)",
        f"- **Single Match (High Confidence)**: {report['match_breakdown']['single_match']} rows ({report['match_breakdown']['single_match'] / report['total_train_rows_evaluated']:.1%})",
        f"- **Multi Match (Ambiguous / Conflicting)**: {report['match_breakdown']['multi_match']} rows ({report['match_breakdown']['multi_match'] / report['total_train_rows_evaluated']:.1%})",
        f"- **Zero Match (Default Fallback 'other')**: {report['match_breakdown']['zero_match']} rows ({report['match_breakdown']['zero_match'] / report['total_train_rows_evaluated']:.1%})",
        "",
        "## Stage 1 Actions",
        "1. **Drop Multi-Match**: All multi-match rows are completely excluded from Stage 1 to prevent contradictory supervision signals.",
        "2. **Retain Single-Match**: All single-match rows are retained with their unambiguous rule match as ground truth.",
        f"3. **Controlled 'Other' Sampling**: From the zero-match rows, a controlled sample of {report['stage1_selection']['zero_match_sampled']} examples is drawn with seed {report['stage1_selection']['seed']} to balance the class distribution.",
        f"4. **Final Stage 1 Size**: {report['stage1_selection']['total_stage1_rows']} examples.",
        "",
        "## Stage 1 Label Distribution",
        "| Intent | Count | Proportion |",
        "| :--- | :--- | :--- |",
    ]
    total = report["stage1_selection"]["total_stage1_rows"]
    for intent, count in report["label_distribution"].items():
        lines.append(f"| `{intent}` | {count} | {count / total:.1%} |")

    lines.extend([
        "",
        "## Stage 2 Roadmap (LLM Distillation)",
        "When a Gemini API key is provided, the ambiguous and sampled zero-match rows will be",
        "labelled with Gemini 2.5/Flash and locally cached, training a Stage 2 model to compare",
        "against this Stage 1 baseline on the sealed 200-example golden set.",
    ])

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 1 training labels.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for other sampling")
    parser.add_argument(
        "--other-samples",
        type=int,
        default=DEFAULT_OTHER_SAMPLE_SIZE,
        help="Number of zero_match / other examples to sample",
    )
    args = parser.parse_args()

    cfg = load_config()
    brand = cfg["brand"]["selected"]
    out_dir = resolve(cfg["data"]["processed_dir"]) / brand.lower()

    train_path = out_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found. Run `python -m src.split` first.")

    train_df = pd.read_csv(train_path, keep_default_na=False, dtype=str)

    # 1. Leakage check before processing
    golden_ids = excluded_tweet_ids()
    if GOLDEN_CSV.exists():
        golden_df = pd.read_csv(GOLDEN_CSV, keep_default_na=False, dtype=str)
        golden_conv_ids = set(golden_df["conversation_id"])
    else:
        golden_conv_ids = set()

    assert golden_ids.isdisjoint(set(train_df["customer_tweet_id"])), "Golden tweet leakage detected in train.csv!"
    assert golden_conv_ids.isdisjoint(set(train_df["conversation_id"])), "Golden conv leakage detected in train.csv!"

    # 2. Analyze rules
    analyzed_df = analyze_training_rows(train_df)

    # 3. Build Stage 1 dataset
    stage1_df, report = build_stage1_dataset(
        analyzed_df,
        other_sample_size=args.other_samples,
        seed=args.seed,
    )

    # 4. Strict leakage assert on final Stage 1 dataset
    assert golden_ids.isdisjoint(set(stage1_df["customer_tweet_id"])), "Golden tweet leaked into train_stage1.csv!"
    assert golden_conv_ids.isdisjoint(set(stage1_df["conversation_id"])), "Golden conv leaked into train_stage1.csv!"
    assert (stage1_df["match_category"] != "multi_match").all(), "Multi-match row found in train_stage1.csv!"

    # 5. Save outputs
    out_csv = out_dir / "train_stage1.csv"
    out_json = out_dir / "training_label_report.json"
    out_md = out_dir / "training_label_strategy.md"

    stage1_df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    generate_markdown_strategy(report, out_md)

    print("\n=== STAGE 1 TRAINING-LABEL GENERATION REPORT ===")
    print(f"Total train rows evaluated: {report['total_train_rows_evaluated']}")
    print(f"  - single_match:           {report['match_breakdown']['single_match']}")
    print(f"  - multi_match:            {report['match_breakdown']['multi_match']} (EXCLUDED)")
    print(f"  - zero_match:             {report['match_breakdown']['zero_match']} (sampled {report['stage1_selection']['zero_match_sampled']})")
    print(f"Total Stage 1 rows:         {report['stage1_selection']['total_stage1_rows']}")
    print("\nStage 1 Label Distribution:")
    for intent, count in report["label_distribution"].items():
        print(f"  {intent:>24}: {count}")
    print(f"\nWritten: {out_csv}")
    print(f"Written: {out_json}")
    print(f"Written: {out_md}")


if __name__ == "__main__":
    main()
