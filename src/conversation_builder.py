"""Phase 3 - rebuild AmazonHelp conversations from the raw reply links.

Two artefacts are produced (both real, both derived from the CSV on disk):

1. `conversations.csv` - one row per reply chain (thread), with the full
   ordered transcript and how many turns each side contributed.
2. `pairs.csv`         - one row per (customer message -> AmazonHelp reply)
   pair. This is the unit the retrieval + reply parts of the assignment
   will later use.

Run:  python3 -m src.conversation_builder
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

from .config import load_config, resolve
from .data_loader import load_rows
from .preprocessing import clean_text

log = logging.getLogger(__name__)

CUSTOMER = "True"  # `inbound` is stored as the string "True"/"False"


def _to_datetime(series: pd.Series) -> pd.Series:
    """Twitter format: 'Tue Oct 31 22:10:47 +0000 2017'."""
    return pd.to_datetime(series, format="%a %b %d %H:%M:%S %z %Y", errors="coerce")


def load_brand_frame(cfg: dict | None = None) -> pd.DataFrame:
    """Load the raw rows and keep only tweets belonging to brand threads."""
    cfg = cfg or load_config()
    brand = cfg["brand"]["selected"]
    if not brand:
        raise SystemExit("config.yaml -> brand.selected is null. Pick a brand first.")

    df = load_rows(
        cfg["data"]["raw_csv"],
        max_rows=cfg["data"].get("scan_rows"),
        chunk_size=cfg["data"].get("chunk_size", 100_000),
    )
    df["created_dt"] = _to_datetime(df["created_at"])
    df["root_id"] = find_roots(df)

    # A thread belongs to the brand if the brand posted in it, or if a customer
    # tweet in it @-mentions the brand handle.
    mention = df["text"].str.contains("@" + brand, case=False, regex=False)
    brand_authored = df["author_id"] == brand
    brand_roots = set(df.loc[brand_authored | mention, "root_id"])

    sub = df[df["root_id"].isin(brand_roots)].copy()
    sub["brand"] = brand
    log.info("brand=%s raw rows=%d brand-thread rows=%d", brand, len(df), len(sub))
    return sub


def find_roots(df: pd.DataFrame) -> pd.Series:
    """Walk `in_response_to_tweet_id` upwards until a tweet has no parent.

    Returns the root tweet id for every row. Rows whose parent is not present
    in the loaded slice become their own root (the thread is truncated by the
    slice, which we keep rather than throw away).
    """
    parent = dict(zip(df["tweet_id"], df["in_response_to_tweet_id"]))
    known = set(df["tweet_id"])
    cache: dict[str, str] = {}

    def root_of(tid: str) -> str:
        seen: list[str] = []
        cur = tid
        while True:
            if cur in cache:
                r = cache[cur]
                break
            p = parent.get(cur, "")
            if not p or p not in known or p in seen:
                r = cur
                break
            seen.append(cur)
            cur = p
        for s in seen:
            cache[s] = r
        cache[tid] = r
        return r

    return df["tweet_id"].map(root_of)


def build_conversations(sub: pd.DataFrame) -> pd.DataFrame:
    """One row per thread, messages ordered by time."""
    sub = sub.sort_values(["root_id", "created_dt"])
    rows = []
    for root, g in sub.groupby("root_id", sort=False):
        cust = g[g["inbound"] == CUSTOMER]
        brnd = g[g["inbound"] != CUSTOMER]
        transcript = [
            {
                "tweet_id": t.tweet_id,
                "role": "customer" if t.inbound == CUSTOMER else "brand",
                "author_id": t.author_id,
                "created_at": t.created_at,
                "text": t.text,
            }
            for t in g.itertuples()
        ]
        rows.append(
            {
                "conversation_id": root,
                "brand": g["brand"].iloc[0],
                "n_messages": len(g),
                "n_customer_messages": len(cust),
                "n_brand_messages": len(brnd),
                "has_brand_reply": len(brnd) > 0,
                "is_multi_turn": len(g) > 2,
                "customer_ids": ",".join(sorted(set(cust["author_id"]))),
                "started_at": g["created_at"].iloc[0],
                "transcript_json": json.dumps(transcript, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def build_pairs(sub: pd.DataFrame) -> pd.DataFrame:
    """One row per customer message that the brand actually answered."""
    brand = sub["brand"].iloc[0]
    by_id = sub.set_index("tweet_id")
    replies = sub[(sub["author_id"] == brand) & (sub["in_response_to_tweet_id"] != "")]

    rows = []
    for r in replies.itertuples():
        pid = r.in_response_to_tweet_id
        if pid not in by_id.index:
            continue  # parent outside the slice
        parent = by_id.loc[pid]
        if isinstance(parent, pd.DataFrame):  # defensive: duplicate ids
            parent = parent.iloc[0]
        if parent["inbound"] != CUSTOMER:
            continue  # brand answering itself, not a customer
        if not str(parent["text"]).strip() or not str(r.text).strip():
            continue  # empty text
        rows.append(
            {
                "conversation_id": r.root_id,
                "customer_tweet_id": pid,
                "customer_message": parent["text"],
                "customer_message_clean": clean_text(parent["text"]),
                "brand_reply_id": r.tweet_id,
                "brand_reply": r.text,
                "created_at": parent["created_at"],
                "customer_id": parent["author_id"],
                "brand": brand,
            }
        )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs

    # Multiple brand replies to the same customer tweet: keep the earliest one
    # (the first response is the substantive one; later ones are follow-ups).
    pairs["_dt"] = _to_datetime(pairs["created_at"])
    pairs = pairs.sort_values(["customer_tweet_id", "brand_reply_id"])
    pairs["n_brand_replies_to_message"] = pairs.groupby("customer_tweet_id")[
        "brand_reply_id"
    ].transform("size")
    pairs = pairs.drop_duplicates("customer_tweet_id", keep="first")

    # Near-duplicate flag: same normalised customer text seen before.
    norm = (
        pairs["customer_message_clean"]
        .str.lower()
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    pairs["is_duplicate_text"] = norm.duplicated(keep="first")
    return pairs.drop(columns=["_dt"]).reset_index(drop=True)


def summarise(sub: pd.DataFrame, convs: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    brand = sub["brand"].iloc[0]
    cust = sub[sub["inbound"] == CUSTOMER]
    lengths = convs["n_messages"]
    return {
        "brand": brand,
        "brand_thread_tweets": len(sub),
        "customer_tweets": len(cust),
        "brand_tweets": int((sub["author_id"] == brand).sum()),
        "other_account_tweets": int(
            ((sub["author_id"] != brand) & (sub["inbound"] != CUSTOMER)).sum()
        ),
        "conversations": len(convs),
        "conversations_with_brand_reply": int(convs["has_brand_reply"].sum()),
        "conversations_without_brand_reply": int((~convs["has_brand_reply"]).sum()),
        # a customer message can sit inside an answered thread and still never
        # get a direct reply of its own - counted separately, not hidden
        "customer_messages_without_direct_reply": int(
            (~cust["tweet_id"].isin(set(pairs["customer_tweet_id"]))).sum()
        )
        if len(pairs)
        else len(cust),
        "multi_turn_conversations": int(convs["is_multi_turn"].sum()),
        "customer_brand_pairs": len(pairs),
        "duplicate_text_pairs": int(pairs["is_duplicate_text"].sum()) if len(pairs) else 0,
        "unique_customers": int(cust["author_id"].nunique()),
        "empty_text_rows": int((sub["text"].str.strip() == "").sum()),
        "avg_conversation_length": round(float(lengths.mean()), 2),
        "median_conversation_length": float(lengths.median()),
        "max_conversation_length": int(lengths.max()),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config()
    brand = cfg["brand"]["selected"]
    sub = load_brand_frame(cfg)
    convs = build_conversations(sub)
    pairs = build_pairs(sub)
    stats = summarise(sub, convs, pairs)

    out = resolve(cfg["data"]["processed_dir"]) / brand.lower()
    out.mkdir(parents=True, exist_ok=True)
    convs.to_csv(out / "conversations.csv", index=False)
    pairs.to_csv(out / "pairs.csv", index=False)
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\n=== A/B. AmazonHelp DATA + CONVERSATION STATISTICS ===")
    for k, v in stats.items():
        print(f"{k:>34}: {v}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
