"""Tests for sampling, labelling save/resume, and golden-set validation."""

from __future__ import annotations

import pandas as pd
import pytest

from src import label_golden_set as lab
from src.intents import INTENTS
from src.sample_golden_set import allocate, build_sample, length_bucket, normalise
from src.validate_golden_set import validate


def make_pairs(n: int = 300) -> pd.DataFrame:
    texts = []
    for i in range(n):
        if i % 3 == 0:
            texts.append(f"where is my order {i} it is late")
        elif i % 3 == 1:
            texts.append(f"i want a refund for order {i} please " + "x" * (i % 200))
        else:
            texts.append(f"hello {i}")
    return pd.DataFrame(
        {
            "conversation_id": [f"c{i}" for i in range(n)],
            "customer_tweet_id": [str(1000 + i) for i in range(n)],
            "customer_message": texts,
            "customer_message_clean": texts,
            "created_at": [f"2017-10-{(i % 28) + 1:02d}" for i in range(n)],
        }
    )


def test_normalise_and_bucket():
    assert normalise("Where IS my order #123!!") == "where is my order"
    assert length_bucket("hi") == "short"
    assert length_bucket("a" * 100) == "medium"
    assert length_bucket("a" * 200) == "long"


def test_allocate_sums_to_target_and_floors_at_one():
    alloc = allocate({"a": 500, "b": 10, "c": 2}, 100)
    assert sum(alloc.values()) == 100
    assert all(v >= 1 for v in alloc.values())


def test_sampling_is_reproducible_and_unlabelled():
    pairs = make_pairs()
    a, report = build_sample(pairs, seed=42, target=50)
    b, _ = build_sample(pairs, seed=42, target=50)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 50
    assert (a["gold_intent"] == "").all()  # never pre-filled
    assert a["example_id"].is_unique
    assert a["customer_tweet_id"].is_unique
    assert report["near_duplicates_removed"] >= 0


def test_different_seed_gives_a_different_sample():
    pairs = make_pairs()
    a, _ = build_sample(pairs, seed=42, target=50)
    b, _ = build_sample(pairs, seed=7, target=50)
    assert list(a["customer_tweet_id"]) != list(b["customer_tweet_id"])


def test_intent_list_is_the_locked_ten():
    assert len(INTENTS) == 10
    assert INTENTS[-1] == "other"
    assert "delivery_order_status" in INTENTS


def test_validate_flags_problems():
    df = pd.DataFrame(
        {
            "example_id": ["g001", "g001"],
            "conversation_id": ["c1", "c2"],
            "customer_tweet_id": ["1", "1"],
            "customer_message": ["hi", ""],
            "gold_intent": ["refund_return", "not_a_real_intent"],
            "label_notes": ["", ""],
        }
    )
    problems = validate(df, min_labelled=2)
    joined = " ".join(problems)
    assert "duplicate example_id" in joined
    assert "duplicate customer_tweet_id" in joined
    assert "empty customer_message" in joined
    assert "invalid intent names" in joined


def test_validate_accepts_a_good_set():
    df = pd.DataFrame(
        {
            "example_id": [f"g{i:03d}" for i in range(3)],
            "conversation_id": [f"c{i}" for i in range(3)],
            "customer_tweet_id": [str(i) for i in range(3)],
            "customer_message": ["a", "b", "c"],
            "gold_intent": ["other", "refund_return", "payment_billing"],
            "label_notes": ["", "", ""],
        }
    )
    assert validate(df, min_labelled=3) == []


def test_save_and_resume(tmp_path):
    pairs = make_pairs(60)
    sample, _ = build_sample(pairs, seed=42, target=20)
    path = tmp_path / "golden_set.csv"
    sample.to_csv(path, index=False)

    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    assert lab.next_index(df, []) == 0
    df.at[0, "gold_intent"] = "refund_return"
    lab.save(df, path)

    reloaded = pd.read_csv(path, keep_default_na=False, dtype=str)
    assert reloaded.at[0, "gold_intent"] == "refund_return"
    assert lab.next_index(reloaded, []) == 1  # resumes at the next unlabelled row
    assert lab.next_index(reloaded, [reloaded.at[1, "example_id"]]) == 2  # skip respected
    assert lab.counts(reloaded) == {"refund_return": 1}


def test_key_map_covers_every_intent():
    assert sorted(lab.KEYS.values()) == sorted(INTENTS)


@pytest.mark.parametrize("key", ["1", "0"])
def test_keys_are_valid_intents(key):
    assert lab.KEYS[key] in INTENTS
