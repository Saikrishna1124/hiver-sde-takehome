"""Tests for Phase 7 Baseline Intent Classifier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import ROOT
from src.intents import INTENTS
from src.train_baseline import (
    MODEL_DIR,
    build_pipeline,
    evaluate_dataset,
    load_model,
    save_model,
    train_model,
)

DATA_DIR = ROOT / "data" / "processed" / "amazonhelp"
TRAIN_STAGE1_PATH = DATA_DIR / "train_stage1.csv"
DEV_PATH = DATA_DIR / "dev.csv"
GOLDEN_PATH = ROOT / "evaluation" / "golden_set.csv"
REPORT_JSON_PATH = DATA_DIR / "baseline_report.json"


@pytest.fixture(scope="module")
def train_data():
    assert TRAIN_STAGE1_PATH.exists(), f"Missing {TRAIN_STAGE1_PATH}"
    df = pd.read_csv(TRAIN_STAGE1_PATH, keep_default_na=False, dtype=str)
    return df


def test_training_data_integrity(train_data):
    # Non-empty dataset
    assert len(train_data) > 0
    # Must have clean message and training label
    assert "customer_message_clean" in train_data.columns
    assert "training_label" in train_data.columns

    # All labels must be in the locked 10-intent taxonomy
    invalid_labels = set(train_data["training_label"]) - set(INTENTS)
    assert not invalid_labels, f"Invalid intent labels: {invalid_labels}"


def test_golden_set_isolation(train_data):
    golden = pd.read_csv(GOLDEN_PATH, keep_default_na=False, dtype=str)
    golden_tweets = set(golden["customer_tweet_id"].astype(str))
    golden_convs = set(golden["conversation_id"].astype(str))

    train_tweets = set(train_data["customer_tweet_id"].astype(str))
    train_convs = set(train_data["conversation_id"].astype(str))

    assert golden_tweets.isdisjoint(train_tweets), "Golden tweet found in training set!"
    assert golden_convs.isdisjoint(train_convs), "Golden conversation found in training set!"


def test_training_reproducibility(train_data):
    pipe1 = train_model(train_data, random_state=42)
    pipe2 = train_model(train_data, random_state=42)

    sample_texts = ["where is my parcel", "cancel prime", "wrong item delivered"]
    preds1 = pipe1.predict(sample_texts)
    preds2 = pipe2.predict(sample_texts)

    assert list(preds1) == list(preds2)


def test_model_save_and_load(tmp_path, train_data):
    pipe = train_model(train_data, random_state=42)
    model_file = tmp_path / "test_model.joblib"

    save_model(pipe, model_file)
    assert model_file.exists()

    loaded = load_model(model_file)
    sample_texts = ["my order is late", "refund please"]
    assert list(pipe.predict(sample_texts)) == list(loaded.predict(sample_texts))


def test_evaluation_output_integrity(train_data):
    golden = pd.read_csv(GOLDEN_PATH, keep_default_na=False, dtype=str)
    pipe = train_model(train_data, random_state=42)

    res = evaluate_dataset(
        pipe,
        golden,
        text_col="customer_message",
        label_col="gold_intent",
    )

    overall = res["overall"]
    assert 0.0 <= overall["accuracy"] <= 1.0
    assert 0.0 <= overall["macro_f1"] <= 1.0
    assert overall["total_support"] == 200

    cm = res["confusion_matrix"]
    assert len(cm["labels"]) == len(INTENTS)
    assert len(cm["matrix"]) == len(INTENTS)
    # Sum of confusion matrix elements must equal total examples
    assert sum(sum(row) for row in cm["matrix"]) == 200

    # Predictions must belong to locked taxonomy
    for pred in res["raw_predictions"]:
        assert pred in INTENTS
