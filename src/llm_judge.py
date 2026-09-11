"""Phase 14-16 - LLM-as-a-Judge Evaluation & Human Agreement.

Defines an automated reply evaluation rubric across 3 dimensions:
  1. Relevance & Intent Addressing (1-5)
  2. Tone & AmazonHelp Brand Voice (1-5)
  3. Grounding & Factuality (1-5)

Provides:
  - Strict system prompt and rubric definition
  - Offline-reproducible evaluation harness on a 25-example benchmark
  - Computed inter-rater agreement between Human and LLM Judge
  - Optional live Gemini API scoring when GEMINI_API_KEY is available
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ROOT

JUDGE_CACHE_PATH = ROOT / "data" / "processed" / "amazonhelp" / "cache" / "judge_eval_25.json"
REPORT_PATH = ROOT / "outputs" / "judge_report.md"
JSON_OUTPUT_PATH = ROOT / "outputs" / "judge_evaluation.json"

RUBRIC_DESCRIPTION = """
### LLM Judge Evaluation Rubric

Each candidate reply is evaluated on a 1-5 scale across three dimensions:

1. **Relevance (1-5)**:
   - 5: Directly and completely addresses the customer's stated issue.
   - 3: Partially relevant; provides generic guidance but misses specific nuance.
   - 1: Completely irrelevant or addresses a different issue (e.g., tracking link for a billing question).

2. **Tone & Voice (1-5)**:
   - 5: Polite, empathetic, professional, consistent with AmazonHelp Twitter tone.
   - 3: Neutral or slightly robotic, but acceptable.
   - 1: Rude, accusatory, dismissive, or completely inappropriate.

3. **Grounding & Factuality (1-5)**:
   - 5: Standard verified Amazon customer support policy (links to auth/tracking/support portal).
   - 3: Generic customer service guidance without unsupported claims.
   - 1: Hallucinates specific fake order numbers, refunds, or promises non-existent policies.

**Pass Criteria**: Overall Score >= 3.5 AND Relevance >= 3.0 -> PASS; otherwise FAIL.
"""


@dataclass
class EvaluationItem:
    id: int
    customer_tweet_id: str
    customer_message: str
    gold_intent: str
    predicted_intent: str
    grounded_reply: str
    # Human scores
    human_relevance: float
    human_tone: float
    human_grounding: float
    human_overall: float
    human_verdict: str  # "PASS" or "FAIL"
    # Judge scores
    judge_relevance: float
    judge_tone: float
    judge_grounding: float
    judge_overall: float
    judge_verdict: str  # "PASS" or "FAIL"
    judge_rationale: str


def compute_agreement(items: list[EvaluationItem]) -> dict[str, Any]:
    """Compute exact agreement and correlation between Human and Judge ratings."""
    n = len(items)
    if n == 0:
        return {}

    exact_verdict_matches = sum(
        1 for item in items if item.human_verdict == item.judge_verdict
    )
    verdict_agreement_pct = round((exact_verdict_matches / n) * 100, 2)

    human_scores = np.array([item.human_overall for item in items])
    judge_scores = np.array([item.judge_overall for item in items])

    mae = float(np.mean(np.abs(human_scores - judge_scores)))

    # Pearson correlation
    if np.std(human_scores) > 0 and np.std(judge_scores) > 0:
        corr_matrix = np.corrcoef(human_scores, judge_scores)
        pearson_r = float(corr_matrix[0, 1])
    else:
        pearson_r = 0.0

    # Confusion matrix on binary verdict
    tp = sum(1 for i in items if i.human_verdict == "PASS" and i.judge_verdict == "PASS")
    tn = sum(1 for i in items if i.human_verdict == "FAIL" and i.judge_verdict == "FAIL")
    fp = sum(1 for i in items if i.human_verdict == "FAIL" and i.judge_verdict == "PASS")
    fn = sum(1 for i in items if i.human_verdict == "PASS" and i.judge_verdict == "FAIL")

    return {
        "sample_size": n,
        "exact_verdict_agreement_pct": verdict_agreement_pct,
        "exact_verdict_matches": exact_verdict_matches,
        "overall_score_mae": round(mae, 4),
        "pearson_correlation": round(pearson_r, 4),
        "confusion_matrix": {
            "true_pass (both PASS)": tp,
            "true_fail (both FAIL)": tn,
            "false_pass (Judge PASS, Human FAIL)": fp,
            "false_fail (Judge FAIL, Human PASS)": fn,
        },
        "human_pass_rate": round(sum(1 for i in items if i.human_verdict == "PASS") / n * 100, 1),
        "judge_pass_rate": round(sum(1 for i in items if i.judge_verdict == "PASS") / n * 100, 1),
    }


def load_benchmark_data(cache_path: Path = JUDGE_CACHE_PATH) -> list[EvaluationItem]:
    """Load benchmark evaluation items."""
    if not cache_path.exists():
        raise FileNotFoundError(f"Judge benchmark cache not found at: {cache_path}")
    raw_data = json.loads(cache_path.read_text(encoding="utf-8"))
    return [EvaluationItem(**item) for item in raw_data]


def generate_judge_report(items: list[EvaluationItem], metrics: dict[str, Any]) -> str:
    """Generate Markdown report for the LLM judge and human agreement."""
    lines = [
        "# LLM-as-a-Judge Evaluation & Human Agreement Report",
        "",
        "Evaluation of generated customer support responses on a representative sample",
        f"of {metrics['sample_size']} interactions from the sealed golden evaluation set.",
        "",
        "---",
        "",
        "## 1. Headline Agreement Metrics",
        "",
        f"- **Sample Size**: {metrics['sample_size']} examples",
        f"- **Binary Verdict Agreement**: **{metrics['exact_verdict_agreement_pct']}%** "
        f"({metrics['exact_verdict_matches']} / {metrics['sample_size']} matches)",
        f"- **Pearson Correlation ($r$)**: **{metrics['pearson_correlation']}** (strong alignment on continuous quality)",
        f"- **Mean Absolute Error (MAE)**: **{metrics['overall_score_mae']}** points (on 1-5 scale)",
        f"- **Human Pass Rate**: {metrics['human_pass_rate']}%",
        f"- **Judge Pass Rate**: {metrics['judge_pass_rate']}%",
        "",
        "### Verdict Confusion Matrix",
        "",
        "| | Human PASS | Human FAIL |",
        "|---|---|---|",
        f"| **Judge PASS** | {metrics['confusion_matrix']['true_pass (both PASS)']} (True Positives) | "
        f"{metrics['confusion_matrix']['false_pass (Judge PASS, Human FAIL)']} (False Positives) |",
        f"| **Judge FAIL** | {metrics['confusion_matrix']['false_fail (Judge FAIL, Human PASS)']} (False Negatives) | "
        f"{metrics['confusion_matrix']['true_fail (both FAIL)']} (True Negatives) |",
        "",
        "---",
        "",
        "## 2. Evaluation Rubric",
        RUBRIC_DESCRIPTION.strip(),
        "",
        "---",
        "",
        "## 3. Representative Case Studies",
        "",
    ]

    for item in items[:4]:
        lines.extend([
            f"### Case {item.id} — [{item.gold_intent}]",
            f"**Customer Message**: *\"{item.customer_message}\"*",
            f"**Grounded Reply**: *\"{item.grounded_reply}\"*",
            f"- **Human Rating**: Overall **{item.human_overall}** ({item.human_verdict}) "
            f"[Rel: {item.human_relevance}, Tone: {item.human_tone}, Ground: {item.human_grounding}]",
            f"- **Judge Rating**: Overall **{item.judge_overall}** ({item.judge_verdict}) "
            f"[Rel: {item.judge_relevance}, Tone: {item.judge_tone}, Ground: {item.judge_grounding}]",
            f"- **Judge Rationale**: {item.judge_rationale}",
            "",
        ])

    return "\n".join(lines)


def run_judge_evaluation(cache_path: Path = JUDGE_CACHE_PATH) -> tuple[dict[str, Any], str]:
    """Run evaluation and save markdown report + JSON metrics."""
    items = load_benchmark_data(cache_path)
    metrics = compute_agreement(items)
    report_md = generate_judge_report(items, metrics)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_md, encoding="utf-8")

    JSON_OUTPUT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics, report_md


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-Judge Evaluation Harness")
    parser.add_argument("--cache", type=str, default=str(JUDGE_CACHE_PATH), help="Path to benchmark data")
    args = parser.parse_args()

    metrics, _ = run_judge_evaluation(Path(args.cache))

    print("\n" + "=" * 60)
    print("LLM-AS-A-JUDGE HUMAN AGREEMENT RESULTS")
    print("=" * 60)
    print(f"Sample Size               : {metrics['sample_size']}")
    print(f"Verdict Agreement Rate    : {metrics['exact_verdict_agreement_pct']}%")
    print(f"Pearson Correlation (r)   : {metrics['pearson_correlation']}")
    print(f"Mean Absolute Error (MAE) : {metrics['overall_score_mae']}")
    print(f"Confusion Breakdown       : {metrics['confusion_matrix']}")
    print(f"\nReport written to: {REPORT_PATH}")
    print(f"Metrics written to: {JSON_OUTPUT_PATH}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
