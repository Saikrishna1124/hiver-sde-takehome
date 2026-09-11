"""Tests for Phase 8 Gemini LLM Labelling Infrastructure & Caching."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import ROOT
from src.generate_stage2_labels import build_stage2_dataset, select_candidates
from src.intents import INTENTS, is_valid
from src.llm_labeler import (
    DiskCache,
    LLMLabeler,
    build_system_prompt,
    build_user_prompt,
    extract_json_payload,
    get_golden_conversation_ids,
)
from src.validate_golden_set import GOLDEN_CSV, excluded_tweet_ids


@pytest.fixture()
def temp_cache(tmp_path):
    return tmp_path / "test_gemini_labels.json"


def test_prompt_contains_all_locked_intents():
    system_prompt = build_system_prompt()
    for intent in INTENTS:
        assert intent in system_prompt, f"Locked intent '{intent}' missing from system prompt!"

    user_prompt = build_user_prompt("Where is my order #123?")
    assert "Where is my order #123?" in user_prompt
    assert "JSON" in user_prompt


def test_extract_json_payload_valid_and_fenced():
    # Plain JSON
    raw_1 = '{"intent": "delivery_order_status", "reasoning": "tracking inquiry"}'
    parsed_1 = extract_json_payload(raw_1)
    assert parsed_1 == {"intent": "delivery_order_status", "reasoning": "tracking inquiry"}

    # Markdown fenced JSON
    raw_2 = '```json\n{"intent": "refund_return", "reasoning": "wants money back"}\n```'
    parsed_2 = extract_json_payload(raw_2)
    assert parsed_2 == {"intent": "refund_return", "reasoning": "wants money back"}

    # Extra text around JSON
    raw_3 = 'Here is the label:\n{"intent": "account_access", "reasoning": "password reset"}\nThank you!'
    parsed_3 = extract_json_payload(raw_3)
    assert parsed_3["intent"] == "account_access"


def test_extract_json_payload_malformed():
    raw_bad = "This is not json at all"
    parsed_bad = extract_json_payload(raw_bad)
    assert parsed_bad is None


def test_disk_cache_lifecycle(temp_cache):
    cache = DiskCache(temp_cache)
    assert len(cache) == 0

    # Set item
    cache.set(
        tweet_id="1001",
        customer_message="cancel prime",
        assigned_intent="prime_membership",
        reasoning="cancel requested",
        model="test-model",
    )
    assert len(cache) == 1
    assert temp_cache.exists()

    # Get valid item
    item = cache.get("1001")
    assert item is not None
    assert item["assigned_intent"] == "prime_membership"

    # Reload from disk
    cache2 = DiskCache(temp_cache)
    assert len(cache2) == 1
    assert cache2.get("1001")["assigned_intent"] == "prime_membership"

    # Invalid intent should not be returned by get()
    cache2.set(
        tweet_id="1002",
        customer_message="random",
        assigned_intent="non_existent_intent",
        reasoning="bad",
        model="test-model",
        status="invalid_intent",
    )
    assert cache2.get("1002") is None


def test_mock_labeler_cache_hit_and_miss(temp_cache):
    call_counts = {"calls": 0}

    def mock_fn(message: str) -> dict[str, str]:
        call_counts["calls"] += 1
        if "refund" in message.lower():
            return {"intent": "refund_return", "reasoning": "wants refund"}
        return {"intent": "other", "reasoning": "general"}

    labeler = LLMLabeler(
        cache_path=temp_cache,
        mock_handler=mock_fn,
    )

    # 1. First call -> Cache MISS, executes mock
    res1 = labeler.label_message("tweet_1", "conv_1", "I need a refund")
    assert res1["status"] == "success"
    assert res1["assigned_intent"] == "refund_return"
    assert call_counts["calls"] == 1

    # 2. Second call with same tweet ID -> Cache HIT, 0 new mock calls
    res2 = labeler.label_message("tweet_1", "conv_1", "I need a refund")
    assert res2["status"] == "cached"
    assert res2["assigned_intent"] == "refund_return"
    assert call_counts["calls"] == 1, "Cache hit should not trigger mock_handler!"


def test_mock_labeler_rejects_invalid_intent(temp_cache):
    def bad_mock_fn(message: str) -> dict[str, str]:
        return {"intent": "completely_fake_category", "reasoning": "hallucinated"}

    labeler = LLMLabeler(
        cache_path=temp_cache,
        mock_handler=bad_mock_fn,
    )
    res = labeler.label_message("tweet_fake", "conv_fake", "some text")
    assert res["status"] == "invalid_intent"
    assert res["assigned_intent"] == ""  # Never assigned as a valid label


def test_golden_set_tweet_exclusion_assertion(temp_cache):
    golden_ids = excluded_tweet_ids()
    assert len(golden_ids) == 200
    sample_golden_id = next(iter(golden_ids))

    labeler = LLMLabeler(
        cache_path=temp_cache,
        mock_handler=lambda msg: {"intent": "other", "reasoning": ""},
    )

    # Attempting to label any golden tweet must raise ValueError immediately
    with pytest.raises(ValueError, match="LEAKAGE VIOLATION: Tweet ID"):
        labeler.label_message(
            tweet_id=sample_golden_id,
            conversation_id="non_golden_conv",
            customer_message="golden message",
        )


def test_golden_set_conversation_exclusion_assertion(temp_cache):
    golden_convs = get_golden_conversation_ids()
    assert len(golden_convs) > 0
    sample_golden_conv = next(iter(golden_convs))

    labeler = LLMLabeler(
        cache_path=temp_cache,
        mock_handler=lambda msg: {"intent": "other", "reasoning": ""},
    )

    # Attempting to label any message in a golden conversation must raise ValueError
    with pytest.raises(ValueError, match="LEAKAGE VIOLATION: Conversation ID"):
        labeler.label_message(
            tweet_id="new_non_golden_tweet",
            conversation_id=sample_golden_conv,
            customer_message="sibling message in golden thread",
        )


def test_select_candidates_integrity():
    train_path = ROOT / "data" / "processed" / "amazonhelp" / "train.csv"
    assert train_path.exists()
    train_df = pd.read_csv(train_path, keep_default_na=False, dtype=str)

    # Ambiguous subset
    ambiguous = select_candidates(train_df, subset="ambiguous")
    assert len(ambiguous) == 163, f"Expected 163 ambiguous multi-match rows, got {len(ambiguous)}"

    # Limit parameter
    limited = select_candidates(train_df, subset="ambiguous", limit=20)
    assert len(limited) == 20


def test_build_stage2_dataset_integration(temp_cache):
    train_path = ROOT / "data" / "processed" / "amazonhelp" / "train.csv"
    stage1_path = ROOT / "data" / "processed" / "amazonhelp" / "train_stage1.csv"
    train_df = pd.read_csv(train_path, keep_default_na=False, dtype=str)
    stage1_df = pd.read_csv(stage1_path, keep_default_na=False, dtype=str)

    # Pre-populate cache with a mock label for one ambiguous tweet
    ambiguous_tids = select_candidates(train_df, subset="ambiguous")
    first_ambiguous_tid = ambiguous_tids.iloc[0]["customer_tweet_id"]

    cache = DiskCache(temp_cache)
    cache.set(
        tweet_id=first_ambiguous_tid,
        customer_message="test ambiguous message",
        assigned_intent="refund_return",
        reasoning="resolved by llm",
        model="mock",
    )

    stage2_df, summary = build_stage2_dataset(train_df, stage1_df, cache)

    # Assert Stage 2 incorporates the newly labeled ambiguous row
    assert first_ambiguous_tid in set(stage2_df["customer_tweet_id"])
    labeled_row = stage2_df[stage2_df["customer_tweet_id"] == first_ambiguous_tid].iloc[0]
    assert labeled_row["training_label"] == "refund_return"
    assert labeled_row["label_source"] == "llm_gemini"

    # All labels must be non-empty and valid
    assert not (stage2_df["training_label"].str.strip() == "").any()
    invalid = set(stage2_df["training_label"]) - set(INTENTS)
    assert not invalid, f"Invalid intents in Stage 2: {invalid}"
