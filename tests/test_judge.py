"""Tests for LLM-as-a-Judge Rubric and Agreement Evaluation."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.llm_judge import (
    EvaluationItem,
    compute_agreement,
    generate_judge_report,
    load_benchmark_data,
    run_judge_evaluation,
)


def test_load_benchmark_data():
    items = load_benchmark_data()
    assert len(items) == 25
    for item in items:
        assert isinstance(item, EvaluationItem)
        assert item.human_verdict in ("PASS", "FAIL")
        assert item.judge_verdict in ("PASS", "FAIL")
        assert 1.0 <= item.human_overall <= 5.0
        assert 1.0 <= item.judge_overall <= 5.0
        assert len(item.judge_rationale) > 0


def test_compute_agreement_metrics():
    items = load_benchmark_data()
    metrics = compute_agreement(items)

    assert metrics["sample_size"] == 25
    assert 0.0 <= metrics["exact_verdict_agreement_pct"] <= 100.0
    assert -1.0 <= metrics["pearson_correlation"] <= 1.0
    assert metrics["overall_score_mae"] >= 0.0

    cm = metrics["confusion_matrix"]
    assert cm["true_pass (both PASS)"] + cm["true_fail (both FAIL)"] + cm["false_pass (Judge PASS, Human FAIL)"] + cm["false_fail (Judge FAIL, Human PASS)"] == 25


def test_compute_agreement_edge_cases():
    # Empty list
    assert compute_agreement([]) == {}

    # Synthetic identical items
    dummy = [
        EvaluationItem(
            id=1,
            customer_tweet_id="101",
            customer_message="test",
            gold_intent="other",
            predicted_intent="other",
            grounded_reply="test reply",
            human_relevance=4.0,
            human_tone=4.0,
            human_grounding=4.0,
            human_overall=4.0,
            human_verdict="PASS",
            judge_relevance=4.0,
            judge_tone=4.0,
            judge_grounding=4.0,
            judge_overall=4.0,
            judge_verdict="PASS",
            judge_rationale="good",
        )
    ]
    res = compute_agreement(dummy)
    assert res["exact_verdict_agreement_pct"] == 100.0
    assert res["overall_score_mae"] == 0.0


def test_generate_judge_report():
    items = load_benchmark_data()
    metrics = compute_agreement(items)
    report = generate_judge_report(items, metrics)
    assert "# LLM-as-a-Judge Evaluation & Human Agreement Report" in report
    assert "Headline Agreement Metrics" in report
    assert "Verdict Agreement" in report
    assert "Evaluation Rubric" in report


def test_run_judge_evaluation_end_to_end(tmp_path):
    metrics, report_md = run_judge_evaluation()
    assert metrics["sample_size"] == 25
    assert len(report_md) > 0
    assert Path("outputs/judge_report.md").exists()
    assert Path("outputs/judge_evaluation.json").exists()
