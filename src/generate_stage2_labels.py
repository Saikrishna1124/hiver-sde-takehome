"""Phase 8 - Stage 2 Training Data Generation with Gemini LLM Distillation.

Orchestrates:
1. Candidate row selection from `train.csv` (e.g. 163 ambiguous multi-match rows, or all rows).
2. Strict golden evaluation set leakage verification.
3. Disk-cached Gemini labeling with rate limiting and retry handling.
4. Production of `train_stage2.csv` without modifying or overwriting `train_stage1.csv`.
5. Comparison report detailing agreements and disagreements between Stage 1 heuristics and Stage 2 LLM labels.

Run (dry-run / inspect only):
    python -m src.generate_stage2_labels --dry-run
Run (controlled batch of ambiguous rows):
    python -m src.generate_stage2_labels --subset ambiguous --limit 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ROOT, load_config, resolve
from .generate_training_labels import analyze_training_rows
from .intents import INTENTS, is_valid
from .llm_labeler import DEFAULT_CACHE_PATH, DEFAULT_MODEL, DiskCache, LLMLabeler, get_golden_conversation_ids
from .validate_golden_set import excluded_tweet_ids


def select_candidates(
    train_df: pd.DataFrame,
    subset: str = "ambiguous",
    limit: int | None = None,
) -> pd.DataFrame:
    """Select training rows targeted for LLM labeling."""
    analyzed = analyze_training_rows(train_df)

    if subset == "ambiguous":
        candidates = analyzed[analyzed["match_category"] == "multi_match"].copy()
    elif subset == "zero_match":
        candidates = analyzed[analyzed["match_category"] == "zero_match"].copy()
    elif subset == "all":
        candidates = analyzed.copy()
    else:
        raise ValueError(f"Unknown subset: {subset}. Choose 'ambiguous', 'zero_match', or 'all'.")

    if limit is not None and limit > 0:
        candidates = candidates.head(limit)

    return candidates.reset_index(drop=True)


def build_stage2_dataset(
    train_df: pd.DataFrame,
    stage1_df: pd.DataFrame,
    cache: DiskCache,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assemble Stage 2 training dataset combining Stage 1 high-confidence rows

    with successfully cached LLM labels for ambiguous/unlabeled rows.
    """
    analyzed = analyze_training_rows(train_df)

    rows = []
    comparison_records = []

    for _, row in analyzed.iterrows():
        tid = str(row["customer_tweet_id"])
        msg = row["customer_message"]
        clean_msg = row["customer_message_clean"]
        stage1_label = row.get("proposed_intent", "")
        cat = row["match_category"]

        cached_entry = cache.get(tid)

        if cached_entry and is_valid(cached_entry.get("assigned_intent", "")):
            final_label = cached_entry["assigned_intent"]
            label_source = "llm_gemini"
            llm_reasoning = cached_entry.get("reasoning", "")
        elif cat == "single_match":
            # Fall back to high-confidence single rule if not explicitly labeled by LLM
            final_label = row["training_label"]
            label_source = "stage1_rule"
            llm_reasoning = "High-confidence single rule match"
        elif cat == "zero_match" and tid in set(stage1_df["customer_tweet_id"]):
            # Retain Stage 1 sampled 'other' if not in LLM cache
            final_label = "other"
            label_source = "stage1_sampled_other"
            llm_reasoning = "Sampled zero-match other"
        else:
            # Unlabeled and no valid rule/sample -> omitted from Stage 2 training
            continue

        record = {
            "conversation_id": row["conversation_id"],
            "customer_tweet_id": tid,
            "customer_message": msg,
            "customer_message_clean": clean_msg,
            "match_category": cat,
            "stage1_proposed_intent": stage1_label,
            "training_label": final_label,
            "label_source": label_source,
            "llm_reasoning": llm_reasoning,
            "created_at": row.get("created_at", ""),
        }
        rows.append(record)

        if cached_entry and is_valid(cached_entry.get("assigned_intent", "")):
            comparison_records.append(
                {
                    "customer_tweet_id": tid,
                    "customer_message": msg,
                    "match_category": cat,
                    "stage1_label": stage1_label,
                    "llm_label": cached_entry["assigned_intent"],
                    "agreement": stage1_label == cached_entry["assigned_intent"],
                }
            )

    stage2_df = pd.DataFrame(rows)
    if not stage2_df.empty:
        stage2_df = stage2_df.sort_values(
            ["created_at", "conversation_id", "customer_tweet_id"]
        ).reset_index(drop=True)

    summary = {
        "total_stage2_rows": len(stage2_df),
        "source_breakdown": stage2_df["label_source"].value_counts().to_dict() if not stage2_df.empty else {},
        "label_distribution": stage2_df["training_label"].value_counts().to_dict() if not stage2_df.empty else {},
        "comparison_records": comparison_records,
    }

    return stage2_df, summary


def generate_disagreement_report(
    summary: dict[str, Any],
    out_path: Path,
) -> None:
    """Generate Markdown report comparing Stage 1 heuristic proposals vs LLM labels."""
    records = summary.get("comparison_records", [])
    total_compared = len(records)
    agreed = sum(1 for r in records if r["agreement"])
    rate = agreed / total_compared if total_compared > 0 else 0.0

    lines = [
        "# Stage 1 vs. Stage 2 (LLM Distillation) Label Comparison",
        "",
        "## Summary",
        f"- **Total Rows Evaluated by LLM**: {total_compared}",
        f"- **Agreements with Heuristic Proposal**: {agreed} ({rate:.1%})",
        f"- **Disagreements Resolved by LLM**: {total_compared - agreed} ({1 - rate:.1%})",
        "",
        "## Stage 2 Dataset Composition",
        "| Label Source | Count |",
        "| :--- | :--- |",
    ]
    for source, count in summary.get("source_breakdown", {}).items():
        lines.append(f"| `{source}` | {count} |")

    lines.extend([
        "",
        "## Final Stage 2 Label Distribution",
        "| Intent | Count |",
        "| :--- | :--- |",
    ])
    for intent, count in summary.get("label_distribution", {}).items():
        lines.append(f"| `{intent}` | {count} |")

    if total_compared > agreed:
        lines.extend([
            "",
            "## Sample Disagreements (LLM Correction vs. Priority Heuristic)",
            "| Tweet ID | Category | Stage 1 Heuristic | Stage 2 LLM Label | Customer Message |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])
        for r in [rec for rec in records if not rec["agreement"]][:15]:
            snippet = r["customer_message"].replace("\n", " ").replace("|", "\\|")[:80]
            lines.append(
                f"| `{r['customer_tweet_id']}` | {r['match_category']} | `{r['stage1_label']}` | **`{r['llm_label']}`** | {snippet}... |"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 Gemini LLM label generation.")
    parser.add_argument(
        "--subset",
        type=str,
        default="ambiguous",
        choices=["ambiguous", "zero_match", "all"],
        help="Subset of training rows to label with Gemini",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on number of rows to label in this run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect candidate rows and cache status without calling the Gemini API",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Gemini model to use",
    )
    args = parser.parse_args()

    cfg = load_config()
    brand = cfg["brand"]["selected"]
    proc_dir = resolve(cfg["data"]["processed_dir"]) / brand.lower()

    train_path = proc_dir / "train.csv"
    stage1_path = proc_dir / "train_stage1.csv"
    cache_path = proc_dir / "cache" / "gemini_labels.json"

    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found.")
    if not stage1_path.exists():
        raise FileNotFoundError(f"{stage1_path} not found. Run Stage 1 first.")

    train_df = pd.read_csv(train_path, keep_default_na=False, dtype=str)
    stage1_df = pd.read_csv(stage1_path, keep_default_na=False, dtype=str)

    # 1. Leakage checks
    golden_ids = excluded_tweet_ids()
    golden_conv_ids = get_golden_conversation_ids()
    assert golden_ids.isdisjoint(set(train_df["customer_tweet_id"])), "Golden tweet leakage in train.csv!"
    assert golden_conv_ids.isdisjoint(set(train_df["conversation_id"])), "Golden conv leakage in train.csv!"

    # 2. Select candidates
    candidates = select_candidates(train_df, subset=args.subset, limit=args.limit)
    print(f"\nTarget subset: '{args.subset}' | Candidates selected: {len(candidates)}")

    # 3. Cache inspection
    cache = DiskCache(cache_path)
    cached_ids = set(cache.keys())
    candidate_ids = set(candidates["customer_tweet_id"])
    already_cached = candidate_ids.intersection(cached_ids)
    pending = [cid for cid in candidate_ids if cid not in already_cached]

    print(f"Already cached: {len(already_cached)} | Pending API labeling: {len(pending)}")

    if args.dry_run:
        print("\n[DRY RUN]: No API calls will be made. Compiling dataset from existing cache...")
        stage2_df, summary = build_stage2_dataset(train_df, stage1_df, cache)
        stage2_csv_path = proc_dir / "train_stage2.csv"
        stage2_report_path = proc_dir / "stage2_label_report.json"
        comparison_md_path = proc_dir / "stage1_vs_stage2_comparison.md"
        stage2_df.to_csv(stage2_csv_path, index=False)
        stage2_report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        generate_disagreement_report(summary, comparison_md_path)
        print(f"Stage 2 rows compiled from cache: {len(stage2_df)}")
        for src, cnt in summary.get("source_breakdown", {}).items():
            print(f"  - {src}: {cnt}")
        print(f"Written: {stage2_csv_path}")
        print(f"Written: {stage2_report_path}")
        print(f"Written: {comparison_md_path}")
        return

    # 4. Run Labeler
    labeler = LLMLabeler(model=args.model, cache_path=cache_path)
    print(f"\nLabeling {len(pending)} pending messages with {args.model}...")

    success_count = 0
    fail_count = 0

    for idx, row in candidates.iterrows():
        tid = str(row["customer_tweet_id"])
        cid = str(row["conversation_id"])
        msg = row["customer_message"]

        res = labeler.label_message(
            tweet_id=tid,
            conversation_id=cid,
            customer_message=msg,
        )
        if res["status"] in ("success", "cached"):
            success_count += 1
            tag = "cached" if res["status"] == "cached" else "api"
            print(f"  [{idx + 1}/{len(candidates)}] tweet {tid} ({tag}) -> {res['assigned_intent']}")
            if res["status"] == "success":
                time.sleep(0.5)
        else:
            fail_count += 1
            print(f"  [{idx + 1}/{len(candidates)}] [WARN] tweet {tid}: {res['status']} ({res['reasoning']})")

    # 5. Build and save Stage 2 dataset
    stage2_df, summary = build_stage2_dataset(train_df, stage1_df, cache)

    stage2_csv_path = proc_dir / "train_stage2.csv"
    stage2_report_path = proc_dir / "stage2_label_report.json"
    comparison_md_path = proc_dir / "stage1_vs_stage2_comparison.md"

    stage2_df.to_csv(stage2_csv_path, index=False)
    stage2_report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    generate_disagreement_report(summary, comparison_md_path)

    print("\n=== STAGE 2 LLM DISTILLATION REPORT ===")
    print(f"Stage 2 total rows: {len(stage2_df)}")
    for src, cnt in summary.get("source_breakdown", {}).items():
        print(f"  - {src}: {cnt}")
    print(f"\nWritten: {stage2_csv_path}")
    print(f"Written: {stage2_report_path}")
    print(f"Written: {comparison_md_path}")


if __name__ == "__main__":
    main()
