"""Phase 7b - Majority-Class Baseline (Trivial Lower Bound).

Computes two honest majority-class baselines evaluated against the sealed
200-example golden evaluation set:

1. **Train-fitted majority**: fit sklearn DummyClassifier(strategy='most_frequent')
   on `train_stage1.csv` labels, then predict the single most-frequent training
   class for every golden-set example.  This is the standard ML "dumb baseline"
   that any real model must beat.

2. **Golden-oracle majority**: the most frequent intent *in the golden set itself*.
   Represents an upper bound on what a majority predictor can achieve if it could
   see the test distribution.  Included for context; it is NOT an honest baseline
   (the test distribution is not visible at training time).

Both baselines predict the same label for every example, so their accuracy
equals the frequency of that label in the golden set.

Run:
    python -m src.majority_baseline

Writes:
    data/processed/amazonhelp/majority_baseline_report.json
    outputs/majority_baseline_report.md
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report

from .config import ROOT, load_config, resolve
from .intents import INTENTS
from .validate_golden_set import GOLDEN_CSV, excluded_tweet_ids

OUTPUT_DIR = ROOT / "outputs"


def compute_train_fitted(
    train_df: pd.DataFrame,
    golden_df: pd.DataFrame,
) -> dict:
    """Fit DummyClassifier on training labels; evaluate on golden set.

    The dummy classifier is fitted on the training label distribution and
    always predicts the most-frequent training class.  Features are ignored.
    """
    y_train = train_df["training_label"].astype(str)
    y_golden = golden_df["gold_intent"].astype(str)

    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    # DummyClassifier needs a 2-D X; we pass a constant column since it is unused.
    dummy.fit([[0]] * len(y_train), y_train)
    preds = dummy.predict([[0]] * len(y_golden))

    majority_class = str(preds[0])
    correct = int((y_golden == majority_class).sum())
    total = len(y_golden)
    accuracy = correct / total

    # Per-intent breakdown from the golden set perspective
    per_intent: dict[str, dict] = {}
    golden_counts = y_golden.value_counts().to_dict()
    for intent in INTENTS:
        support = golden_counts.get(intent, 0)
        # Only the majority class can have any correct predictions
        tp = support if intent == majority_class else 0
        precision = 1.0 if intent == majority_class else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        per_intent[intent] = {
            "support": support,
            "predicted": int((preds == intent).sum()),
            "tp": tp,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return {
        "strategy": "most_frequent (train-fitted)",
        "majority_class": majority_class,
        "correct_predictions": correct,
        "total_examples": total,
        "accuracy": round(accuracy, 4),
        "per_intent": per_intent,
    }


def compute_golden_oracle(golden_df: pd.DataFrame) -> dict:
    """Majority class computed from the golden set distribution.

    NOTE: This is NOT a valid standalone baseline — the test distribution is
    not visible at training time.  It is included only as an upper-bound
    reference for what a majority predictor *could* achieve with test knowledge.
    """
    y_golden = golden_df["gold_intent"].astype(str)
    vc = y_golden.value_counts()
    majority_class = str(vc.index[0])
    correct = int(vc.iloc[0])
    total = len(y_golden)
    accuracy = correct / total

    per_intent: dict[str, dict] = {}
    for intent in INTENTS:
        support = int(vc.get(intent, 0))
        tp = support if intent == majority_class else 0
        precision = 1.0 if intent == majority_class else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        per_intent[intent] = {
            "support": support,
            "tp": tp,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return {
        "strategy": "golden-oracle majority (test distribution — NOT a blind baseline)",
        "majority_class": majority_class,
        "correct_predictions": correct,
        "total_examples": total,
        "accuracy": round(accuracy, 4),
        "per_intent": per_intent,
    }


def generate_markdown_report(
    train_fitted: dict,
    golden_oracle: dict,
    out_path: Path,
) -> None:
    """Write a concise Markdown report for the majority-class baselines."""
    tf = train_fitted
    go = golden_oracle

    lines = [
        "# Majority-Class Baseline Report",
        "",
        "Two majority-class baselines are reported. Both predict a single constant",
        "label for every example.",
        "",
        "## Baseline A — Train-Fitted Majority (Standard Lower Bound)",
        "",
        f"- **Strategy**: `DummyClassifier(strategy='most_frequent')` fitted on `train_stage1.csv`",
        f"- **Majority class (from training data)**: `{tf['majority_class']}`",
        f"- **Golden set**: {tf['total_examples']} examples",
        f"- **Correct predictions**: {tf['correct_predictions']}",
        f"- **Accuracy**: **{tf['accuracy']:.1%}**",
        "",
        "> This is the standard trivial baseline. It predicts the most common training",
        "> label for every test example. Any useful classifier must outperform this.",
        "",
        "## Baseline B — Golden-Oracle Majority (Upper-Bound Reference Only)",
        "",
        f"- **Strategy**: most frequent intent in the golden set itself",
        f"- **Majority class (from golden set)**: `{go['majority_class']}`",
        f"- **Correct predictions**: {go['correct_predictions']} / {go['total_examples']}",
        f"- **Accuracy**: **{go['accuracy']:.1%}**",
        "",
        "> **Not a blind baseline**: this uses knowledge of the test distribution.",
        "> It is the highest accuracy any constant predictor can achieve on this",
        "> golden set. Reported for reference only.",
        "",
        "## Summary",
        "",
        "| Baseline | Majority Class | Accuracy | Note |",
        "| :--- | :--- | :--- | :--- |",
        f"| Train-fitted majority | `{tf['majority_class']}` | **{tf['accuracy']:.1%}** | Standard lower bound |",
        f"| Golden-oracle majority | `{go['majority_class']}` | **{go['accuracy']:.1%}** | Upper bound reference |",
        f"| TF-IDF + LogReg (from `baseline_report.md`) | — | **60.0%** | Main baseline |",
        "",
        "The TF-IDF + LogReg model substantially outperforms both trivial baselines",
        f"(+{0.600 - tf['accuracy']:.1%} over train-fitted, +{0.600 - go['accuracy']:.1%} over oracle).",
        "However, its Macro F1 of 0.2298 reveals that 6 of 10 intents are effectively",
        "invisible to the model (see `baseline_report.md` for the confusion matrix).",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = load_config()
    brand = cfg["brand"]["selected"]
    proc_dir = resolve(cfg["data"]["processed_dir"]) / brand.lower()

    train_path = proc_dir / "train_stage1.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found. Run `py -m src.generate_training_labels` first."
        )
    if not GOLDEN_CSV.exists():
        raise FileNotFoundError(f"{GOLDEN_CSV} not found.")

    train_df = pd.read_csv(train_path, keep_default_na=False, dtype=str)
    golden_df = pd.read_csv(GOLDEN_CSV, keep_default_na=False, dtype=str)

    # Leakage assertion
    golden_ids = excluded_tweet_ids()
    train_ids = set(train_df["customer_tweet_id"].astype(str))
    assert golden_ids.isdisjoint(train_ids), "Golden tweet leakage into training data!"

    labelled = golden_df[golden_df["gold_intent"].str.strip() != ""]
    if len(labelled) < 150:
        raise ValueError(
            f"Golden set has only {len(labelled)} labels. "
            "Run `py -m src.validate_golden_set` to check status."
        )
    golden_df = labelled.copy()

    # Compute both baselines
    train_fitted = compute_train_fitted(train_df, golden_df)
    golden_oracle = compute_golden_oracle(golden_df)

    # Save JSON report
    report = {
        "train_fitted_majority": train_fitted,
        "golden_oracle_majority": golden_oracle,
    }
    report_json_path = proc_dir / "majority_baseline_report.json"
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Save Markdown report
    out_md_path = OUTPUT_DIR / "majority_baseline_report.md"
    generate_markdown_report(train_fitted, golden_oracle, out_md_path)

    print("\n=== MAJORITY-CLASS BASELINE REPORT ===")
    print(f"\nBaseline A — Train-Fitted Majority (standard lower bound):")
    print(f"  Majority class (from training):  {train_fitted['majority_class']}")
    print(f"  Correct predictions:             {train_fitted['correct_predictions']} / {train_fitted['total_examples']}")
    print(f"  Accuracy on golden set:          {train_fitted['accuracy']:.1%}")

    print(f"\nBaseline B — Golden-Oracle Majority (upper-bound reference only):")
    print(f"  Majority class (in golden set):  {golden_oracle['majority_class']}")
    print(f"  Correct predictions:             {golden_oracle['correct_predictions']} / {golden_oracle['total_examples']}")
    print(f"  Accuracy on golden set:          {golden_oracle['accuracy']:.1%}")

    print(f"\nWritten: {report_json_path}")
    print(f"Written: {out_md_path}")


if __name__ == "__main__":
    main()
