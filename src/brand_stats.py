"""Phase 2 - which company (brand) should we build the assistant for?

Run:
    python -m src.brand_stats

Idea in one line: a brand is any author that posts NON-inbound tweets
(inbound == "False" means the tweet came FROM the support account).
We count, per brand:
  * how many support replies it wrote,
  * how many distinct customers it talked to,
  * how many inbound customer tweets are addressed to it.

Nothing is hardcoded: every number comes from the CSV on disk.
"""

from __future__ import annotations

import logging
from collections import Counter

import pandas as pd

from .config import load_config, resolve
from .data_loader import iter_chunks

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

pd.set_option("display.width", 200)


def build_brand_table(cfg: dict | None = None) -> pd.DataFrame:
    """Scan the CSV in chunks and return one row per brand."""
    cfg = cfg or load_config()
    csv_path = cfg["data"]["raw_csv"]
    chunk_size = cfg["data"].get("chunk_size", 100_000)

    replies: Counter[str] = Counter()          # support tweets written
    customers: dict[str, set[str]] = {}        # distinct customers replied to
    mentions: Counter[str] = Counter()         # inbound tweets mentioning brand

    for chunk in iter_chunks(csv_path, chunk_size=chunk_size):
        outbound = chunk[chunk["inbound"] == "False"]
        for brand, sub in outbound.groupby("author_id"):
            replies[brand] += len(sub)

        # link each support reply back to the customer tweet it answers
        by_id = dict(zip(chunk["tweet_id"], chunk["author_id"]))
        for brand, parent in zip(outbound["author_id"], outbound["in_response_to_tweet_id"]):
            cust = by_id.get(parent)
            if cust:
                customers.setdefault(brand, set()).add(cust)

        inbound = chunk[chunk["inbound"] == "True"]
        for text in inbound["text"]:
            # a customer tweet normally starts by @-mentioning the brand
            if text.startswith("@"):
                handle = text[1:].split()[0].strip(":,")
                if handle and not handle.isdigit():
                    mentions[handle] += 1

    rows = []
    for brand, n_replies in replies.items():
        rows.append(
            {
                "brand": brand,
                "support_replies": n_replies,
                "distinct_customers": len(customers.get(brand, ())),
                "inbound_mentions": mentions.get(brand, 0),
            }
        )
    df = pd.DataFrame(rows).sort_values("support_replies", ascending=False)
    return df.reset_index(drop=True)


def main() -> None:
    cfg = load_config()
    df = build_brand_table(cfg)

    out_dir = resolve(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "brand_stats.csv"
    df.to_csv(out_path, index=False)

    print("\n=== TOP 20 BRANDS IN THIS FILE ===")
    print(df.head(20).to_string(index=False))
    print(f"\nTotal brands found: {len(df)}")
    print(f"Written to: {out_path}")
    print(
        "\nNOTE: if you loaded only a slice of the Kaggle file, these counts "
        "describe THAT slice, not the full dataset. Say so in the report."
    )


if __name__ == "__main__":
    main()
