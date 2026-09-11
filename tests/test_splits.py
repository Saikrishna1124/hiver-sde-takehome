"""Tests for Phase 5 leakage-safe conversation-level splits."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import ROOT
from src.validate_golden_set import excluded_tweet_ids

DATA_DIR = ROOT / "data" / "processed" / "amazonhelp"
TRAIN_PATH = DATA_DIR / "train.csv"
DEV_PATH = DATA_DIR / "dev.csv"
SPLIT_REPORT_PATH = DATA_DIR / "split_report.json"
GOLDEN_PATH = ROOT / "evaluation" / "golden_set.csv"


@pytest.fixture(scope="module")
def split_data():
    assert TRAIN_PATH.exists(), f"Missing {TRAIN_PATH}"
    assert DEV_PATH.exists(), f"Missing {DEV_PATH}"
    assert SPLIT_REPORT_PATH.exists(), f"Missing {SPLIT_REPORT_PATH}"
    assert GOLDEN_PATH.exists(), f"Missing {GOLDEN_PATH}"

    train = pd.read_csv(TRAIN_PATH, keep_default_na=False, dtype=str)
    dev = pd.read_csv(DEV_PATH, keep_default_na=False, dtype=str)
    golden = pd.read_csv(GOLDEN_PATH, keep_default_na=False, dtype=str)
    report = json.loads(SPLIT_REPORT_PATH.read_text(encoding="utf-8"))

    return {
        "train": train,
        "dev": dev,
        "golden": golden,
        "report": report,
    }


def test_split_counts_match_specification(split_data):
    train = split_data["train"]
    dev = split_data["dev"]
    report = split_data["report"]

    assert len(train) == 2139, f"Expected 2,139 train rows, got {len(train)}"
    assert len(dev) == 579, f"Expected 579 dev rows, got {len(dev)}"
    assert len(train) + len(dev) == 2718, f"Expected 2,718 total usable rows, got {len(train) + len(dev)}"

    assert report["original_pairs"] == 3603
    assert report["golden_messages_excluded"] == 200
    assert report["conversation_siblings_excluded"] == 547
    assert report["near_duplicates_removed"] == 138
    assert report["usable_examples"] == 2718
    assert report["train_examples"] == 2139
    assert report["dev_examples"] == 579


def test_zero_golden_tweet_leakage_in_train_or_dev(split_data):
    golden_tweet_ids = set(split_data["golden"]["customer_tweet_id"].astype(str))
    assert len(golden_tweet_ids) == 200, "Golden set must have 200 unique tweet IDs"

    train_tweet_ids = set(split_data["train"]["customer_tweet_id"].astype(str))
    dev_tweet_ids = set(split_data["dev"]["customer_tweet_id"].astype(str))

    assert golden_tweet_ids.isdisjoint(train_tweet_ids), (
        f"Found {len(golden_tweet_ids.intersection(train_tweet_ids))} golden tweets in train!"
    )
    assert golden_tweet_ids.isdisjoint(dev_tweet_ids), (
        f"Found {len(golden_tweet_ids.intersection(dev_tweet_ids))} golden tweets in dev!"
    )


def test_zero_golden_conversation_leakage_in_train_or_dev(split_data):
    golden_conv_ids = set(split_data["golden"]["conversation_id"].astype(str))
    train_conv_ids = set(split_data["train"]["conversation_id"].astype(str))
    dev_conv_ids = set(split_data["dev"]["conversation_id"].astype(str))

    assert golden_conv_ids.isdisjoint(train_conv_ids), (
        f"Found {len(golden_conv_ids.intersection(train_conv_ids))} golden conversations in train!"
    )
    assert golden_conv_ids.isdisjoint(dev_conv_ids), (
        f"Found {len(golden_conv_ids.intersection(dev_conv_ids))} golden conversations in dev!"
    )


def test_zero_conversation_overlap_between_train_and_dev(split_data):
    train_convs = set(split_data["train"]["conversation_id"].astype(str))
    dev_convs = set(split_data["dev"]["conversation_id"].astype(str))

    overlap = train_convs.intersection(dev_convs)
    assert len(overlap) == 0, f"Train and dev share {len(overlap)} conversations!"


def test_no_empty_customer_messages(split_data):
    train = split_data["train"]
    dev = split_data["dev"]

    assert not (train["customer_message"].str.strip() == "").any(), "Train contains empty customer_message"
    assert not (dev["customer_message"].str.strip() == "").any(), "Dev contains empty customer_message"


def test_proposed_intent_column_is_valid(split_data):
    from src.intents import INTENTS

    train = split_data["train"]
    dev = split_data["dev"]

    assert "proposed_intent" in train.columns
    assert "proposed_intent" in dev.columns

    invalid_train = set(train["proposed_intent"]) - set(INTENTS)
    invalid_dev = set(dev["proposed_intent"]) - set(INTENTS)

    assert not invalid_train, f"Invalid intents in train: {invalid_train}"
    assert not invalid_dev, f"Invalid intents in dev: {invalid_dev}"
