"""Tests for the conversation reconstruction logic.

These use a tiny hand-written frame that imitates the real column layout, so
the logic is checked without needing the 10 MB CSV.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.conversation_builder import (
    build_conversations,
    build_pairs,
    find_roots,
    summarise,
)

BRAND = "AmazonHelp"


def _row(tid, author, inbound, parent, text, minute):
    return {
        "tweet_id": tid,
        "author_id": author,
        "inbound": inbound,
        "created_at": f"Tue Oct 31 22:{minute:02d}:00 +0000 2017",
        "text": text,
        "response_tweet_id": "",
        "in_response_to_tweet_id": parent,
        "brand": BRAND,
    }


@pytest.fixture()
def frame() -> pd.DataFrame:
    rows = [
        # thread A: customer -> brand -> customer -> brand (multi-turn)
        _row("1", "c1", "True", "", "where is my order", 1),
        _row("2", BRAND, "False", "1", "sorry, please DM us", 2),
        _row("3", "c1", "True", "2", "sent a DM", 3),
        _row("4", BRAND, "False", "3", "thanks, checking now", 4),
        # thread B: customer only, never answered
        _row("5", "c2", "True", "", "@AmazonHelp refund please", 5),
        # thread C: brand replies twice to the same customer message
        _row("6", "c3", "True", "", "parcel damaged", 6),
        _row("7", BRAND, "False", "6", "we are sorry to hear that", 7),
        _row("8", BRAND, "False", "6", "any update for us?", 8),
        # thread D: empty customer text -> must be skipped in pairs
        _row("9", "c4", "True", "", "   ", 9),
        _row("10", BRAND, "False", "9", "hello there", 10),
    ]
    df = pd.DataFrame(rows)
    df["created_dt"] = pd.to_datetime(
        df["created_at"], format="%a %b %d %H:%M:%S %z %Y"
    )
    df["root_id"] = find_roots(df)
    return df


def test_roots_follow_reply_chain(frame):
    roots = dict(zip(frame["tweet_id"], frame["root_id"]))
    assert roots["4"] == "1"          # deep chain resolves to the first tweet
    assert roots["5"] == "5"          # no parent -> own root
    assert roots["8"] == "6"


def test_conversations_group_by_thread(frame):
    convs = build_conversations(frame).set_index("conversation_id")
    assert convs.loc["1", "n_messages"] == 4
    assert bool(convs.loc["1", "is_multi_turn"]) is True
    assert convs.loc["5", "n_messages"] == 1
    assert bool(convs.loc["5", "has_brand_reply"]) is False


def test_pairs_are_customer_to_brand_only(frame):
    pairs = build_pairs(frame)
    ids = set(pairs["customer_tweet_id"])
    assert "1" in ids and "3" in ids and "6" in ids
    assert "9" not in ids            # empty customer text dropped
    assert len(pairs) == 3           # thread C collapses to one pair


def test_multiple_replies_are_counted_not_duplicated(frame):
    pairs = build_pairs(frame).set_index("customer_tweet_id")
    assert pairs.loc["6", "n_brand_replies_to_message"] == 2
    assert pairs.loc["6", "brand_reply_id"] == "7"   # earliest reply kept


def test_summary_counts_are_consistent(frame):
    convs = build_conversations(frame)
    pairs = build_pairs(frame)
    s = summarise(frame, convs, pairs)
    assert s["brand"] == BRAND
    assert s["customer_tweets"] == 5
    assert s["brand_tweets"] == 5
    assert s["conversations_without_brand_reply"] == 1
    assert s["customer_brand_pairs"] == 3
    assert s["unique_customers"] == 4
