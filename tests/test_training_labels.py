"""Tests for Stage 1 training-label generation and rule analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import ROOT
from src.generate_training_labels import (
    analyze_training_rows,
    build_stage1_dataset,
    categorize_matches,
)
from src.intents import INTENTS
from src.taxonomy import match_rules
from src.validate_golden_set import excluded_tweet_ids

DATA_DIR = ROOT / "data" / "processed" / "amazonhelp"
TRAIN_PATH = DATA_DIR / "train.csv"
STAGE1_PATH = DATA_DIR / "train_stage1.csv"
REPORT_PATH = DATA_DIR / "training_label_report.json"
GOLDEN_PATH = ROOT / "evaluation" / "golden_set.csv"


def test_rule_matching_and_categorization():
    # 0 matches -> zero_match
    rules_0 = match_rules("hello good morning")
    assert len(rules_0) == 0
    assert categorize_matches(rules_0) == "zero_match"

    # 1 match -> single_match
    rules_1 = match_rules("where is my order tracking status")
    assert rules_1 == ["delivery_order_status"]
    assert categorize_matches(rules_1) == "single_match"

    # 2+ matches -> multi_match
    # "my account is locked and where is my delivery parcel"
    rules_2 = match_rules("my password was hacked and delivery tracking is delayed")
    assert len(rules_2) >= 2
    assert categorize_matches(rules_2) == "multi_match"


def test_analyze_training_rows_on_sample():
    sample_df = pd.DataFrame(
        {
            "customer_tweet_id": ["t1", "t2", "t3"],
            "conversation_id": ["c1", "c2", "c3"],
            "customer_message": [
                "just saying hello",
                "where is my package",
                "kindle screen broken and refund my money back",
            ],
            "customer_message_clean": [
                "just saying hello",
                "where is my package",
                "kindle screen broken and refund my money back",
            ],
            "created_at": ["2017-10-01", "2017-10-02", "2017-10-03"],
        }
    )
    analyzed = analyze_training_rows(sample_df)
    assert analyzed.loc[0, "match_category"] == "zero_match"
    assert analyzed.loc[0, "training_label"] == "other"

    assert analyzed.loc[1, "match_category"] == "single_match"
    assert analyzed.loc[1, "training_label"] == "delivery_order_status"

    assert analyzed.loc[2, "match_category"] == "multi_match"
    assert analyzed.loc[2, "training_label"] == ""


def test_build_stage1_dataset_excludes_multi_match():
    sample_df = pd.DataFrame(
        {
            "customer_tweet_id": ["t1", "t2", "t3", "t4"],
            "conversation_id": ["c1", "c2", "c3", "c4"],
            "customer_message": ["a", "b", "c", "d"],
            "customer_message_clean": [
                "hello",
                "where is my package",
                "kindle screen broken refund my money",
                "can I cancel my prime membership",
            ],
            "created_at": ["2017-10-01", "2017-10-02", "2017-10-03", "2017-10-04"],
        }
    )
    analyzed = analyze_training_rows(sample_df)
    stage1, report = build_stage1_dataset(analyzed, other_sample_size=1, seed=42)

    # Multi-match row must be dropped
    assert "t3" not in set(stage1["customer_tweet_id"])
    assert (stage1["match_category"] != "multi_match").all()
    # Single match rows must be kept
    assert "t2" in set(stage1["customer_tweet_id"])
    assert "t4" in set(stage1["customer_tweet_id"])


def test_build_stage1_reproducibility():
    train = pd.read_csv(TRAIN_PATH, keep_default_na=False, dtype=str)
    analyzed = analyze_training_rows(train)

    df1, rep1 = build_stage1_dataset(analyzed, other_sample_size=200, seed=42)
    df2, rep2 = build_stage1_dataset(analyzed, other_sample_size=200, seed=42)

    pd.testing.assert_frame_equal(df1, df2)
    assert rep1 == rep2


def test_generated_stage1_file_integrity():
    assert STAGE1_PATH.exists(), f"Missing {STAGE1_PATH}"
    assert REPORT_PATH.exists(), f"Missing {REPORT_PATH}"

    stage1 = pd.read_csv(STAGE1_PATH, keep_default_na=False, dtype=str)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    # Zero multi-match
    assert (stage1["match_category"] != "multi_match").all()
    assert report["stage1_selection"]["multi_match_excluded"] > 0

    # Labels must be valid taxonomy intents
    invalid_labels = set(stage1["training_label"]) - set(INTENTS)
    assert not invalid_labels, f"Found invalid training labels: {invalid_labels}"

    # Counts must match report
    assert len(stage1) == report["stage1_selection"]["total_stage1_rows"]


def test_stage1_golden_set_leakage_safety():
    stage1 = pd.read_csv(STAGE1_PATH, keep_default_na=False, dtype=str)
    golden = pd.read_csv(GOLDEN_PATH, keep_default_na=False, dtype=str)

    golden_tweets = set(golden["customer_tweet_id"].astype(str))
    golden_convs = set(golden["conversation_id"].astype(str))

    stage1_tweets = set(stage1["customer_tweet_id"].astype(str))
    stage1_convs = set(stage1["conversation_id"].astype(str))

    assert golden_tweets.isdisjoint(stage1_tweets), (
        f"Found {len(golden_tweets.intersection(stage1_tweets))} golden tweets in train_stage1.csv!"
    )
    assert golden_convs.isdisjoint(stage1_convs), (
        f"Found {len(golden_convs.intersection(stage1_convs))} golden conversations in train_stage1.csv!"
    )
