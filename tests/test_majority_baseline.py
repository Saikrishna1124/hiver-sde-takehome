"""Tests for the majority-class baseline (Phase 7b)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import ROOT
from src.intents import INTENTS
from src.majority_baseline import compute_golden_oracle, compute_train_fitted

TRAIN_PATH = ROOT / "data" / "processed" / "amazonhelp" / "train_stage1.csv"
GOLDEN_PATH = ROOT / "evaluation" / "golden_set.csv"


@pytest.fixture(scope="module")
def train_df():
    assert TRAIN_PATH.exists(), f"Missing {TRAIN_PATH}"
    return pd.read_csv(TRAIN_PATH, keep_default_na=False, dtype=str)


@pytest.fixture(scope="module")
def golden_df():
    assert GOLDEN_PATH.exists(), f"Missing {GOLDEN_PATH}"
    df = pd.read_csv(GOLDEN_PATH, keep_default_na=False, dtype=str)
    return df[df["gold_intent"].str.strip() != ""].copy()


class TestTrainFittedMajority:
    def test_majority_class_is_valid_intent(self, train_df, golden_df):
        result = compute_train_fitted(train_df, golden_df)
        assert result["majority_class"] in INTENTS

    def test_majority_class_from_training_data(self, train_df, golden_df):
        """Majority class must be the most frequent label in the TRAINING set."""
        result = compute_train_fitted(train_df, golden_df)
        train_majority = train_df["training_label"].value_counts().index[0]
        assert result["majority_class"] == train_majority

    def test_accuracy_range(self, train_df, golden_df):
        result = compute_train_fitted(train_df, golden_df)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_correct_prediction_count_consistent(self, train_df, golden_df):
        result = compute_train_fitted(train_df, golden_df)
        assert result["correct_predictions"] == int(
            (golden_df["gold_intent"] == result["majority_class"]).sum()
        )

    def test_total_examples_is_200(self, train_df, golden_df):
        result = compute_train_fitted(train_df, golden_df)
        assert result["total_examples"] == 200

    def test_per_intent_keys_match_taxonomy(self, train_df, golden_df):
        result = compute_train_fitted(train_df, golden_df)
        assert set(result["per_intent"].keys()) == set(INTENTS)

    def test_non_majority_intents_have_zero_precision_and_tp(self, train_df, golden_df):
        result = compute_train_fitted(train_df, golden_df)
        for intent, m in result["per_intent"].items():
            if intent != result["majority_class"]:
                assert m["tp"] == 0
                assert m["precision"] == 0.0

    def test_majority_class_precision_is_one(self, train_df, golden_df):
        result = compute_train_fitted(train_df, golden_df)
        maj = result["majority_class"]
        assert result["per_intent"][maj]["precision"] == 1.0

    def test_computed_accuracy_matches_known_values(self, train_df, golden_df):
        """Regression: delivery_order_status is training majority -> 22.0% on golden."""
        result = compute_train_fitted(train_df, golden_df)
        assert result["majority_class"] == "delivery_order_status"
        assert result["accuracy"] == pytest.approx(0.22, abs=0.01)


class TestGoldenOracleMajority:
    def test_majority_class_is_valid_intent(self, golden_df):
        result = compute_golden_oracle(golden_df)
        assert result["majority_class"] in INTENTS

    def test_majority_class_is_most_frequent_in_golden(self, golden_df):
        result = compute_golden_oracle(golden_df)
        expected = golden_df["gold_intent"].value_counts().index[0]
        assert result["majority_class"] == expected

    def test_computed_accuracy_matches_known_values(self, golden_df):
        """Regression: 'other' is golden majority -> 45.0%."""
        result = compute_golden_oracle(golden_df)
        assert result["majority_class"] == "other"
        assert result["accuracy"] == pytest.approx(0.45, abs=0.01)

    def test_total_examples_is_200(self, golden_df):
        result = compute_golden_oracle(golden_df)
        assert result["total_examples"] == 200

    def test_per_intent_keys_match_taxonomy(self, golden_df):
        result = compute_golden_oracle(golden_df)
        assert set(result["per_intent"].keys()) == set(INTENTS)
