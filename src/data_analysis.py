"""Phase 1 + Phase 2 (analysis only) - inspect the raw dataset and rank brands.

Everything printed by this script is computed from the CSV that is actually on
disk. Nothing here is hardcoded or assumed.

Run:  python3 -m src.data_analysis
Optional: python3 -m src.data_analysis --rows 100000

The file is read in chunks so we never need the whole dataset in RAM.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .config import load_config, resolve
from .data_loader import iter_chunks

CUSTOMER_FLAG = "True"  # the `inbound` column is the string "True"/"False"


def structure_report(path: Path, sample_rows: int = 2000) -> pd.DataFrame:
    """Print columns, dtypes, missing values and a few real rows."""
    sample = pd.read_csv(path, nrows=sample_rows, keep_default_na=False)
    print("\n=== A. DATASET STRUCTURE ===")
    print(f"file            : {path}")
    print(f"size on disk    : {path.stat().st_size / 1_048_576:.1f} MB")
    print(f"columns ({len(sample.columns)}) : {list(sample.columns)}")
    print("\ninferred dtypes (from the first %d rows):" % sample_rows)
    print(sample.dtypes.to_string())
    print("\nfirst 5 rows (text truncated to 90 chars for readability):")
    preview = sample.head(5).copy()
    if "text" in preview.columns:
        preview["text"] = preview["text"].str.slice(0, 90)
    print(preview.to_string(index=False))
    return sample


def scan(path: Path, chunk_size: int, max_rows: int | None):
    """Single chunked pass that collects every statistic we need."""
    total_rows = 0
    missing = defaultdict(int)          # column -> count of empty strings
    seen_tweet_ids: set[str] = set()
    duplicate_tweet_ids = 0
    inbound_counts = defaultdict(int)   # value of `inbound` -> rows

    # brand = author_id of an outbound (support) tweet
    brand_outbound = defaultdict(int)          # brand -> tweets it sent
    brand_replies_to = defaultdict(int)        # brand -> outbound tweets that answer a customer
    brand_customers: dict[str, set] = defaultdict(set)   # brand -> distinct customers replied to
    brand_threads: dict[str, set] = defaultdict(set)     # brand -> distinct source tweets answered

    # first pass needs customer authorship, so we remember who wrote each tweet
    author_of: dict[str, str] = {}
    inbound_of: dict[str, str] = {}
    pending: list[tuple[str, str, str]] = []  # (brand, parent_id, tweet_id)

    columns: list[str] = []
    for chunk in iter_chunks(path, chunk_size=chunk_size):
        if max_rows is not None and total_rows >= max_rows:
            break
        if max_rows is not None and total_rows + len(chunk) > max_rows:
            chunk = chunk.iloc[: max_rows - total_rows]
        columns = list(chunk.columns)
        total_rows += len(chunk)

        for col in chunk.columns:
            missing[col] += int((chunk[col].astype(str).str.strip() == "").sum())

        for tid in chunk["tweet_id"]:
            if tid in seen_tweet_ids:
                duplicate_tweet_ids += 1
            else:
                seen_tweet_ids.add(tid)

        for v, c in chunk["inbound"].value_counts().items():
            inbound_counts[str(v)] += int(c)

        author_of.update(dict(zip(chunk["tweet_id"], chunk["author_id"])))
        inbound_of.update(dict(zip(chunk["tweet_id"], chunk["inbound"])))

        out = chunk[chunk["inbound"] != CUSTOMER_FLAG]
        for brand, parent, tid in zip(
            out["author_id"], out["in_response_to_tweet_id"], out["tweet_id"]
        ):
            brand_outbound[brand] += 1
            if parent:
                pending.append((brand, parent, tid))

    # resolve which outbound tweets actually answered a customer tweet
    for brand, parent, _tid in pending:
        if inbound_of.get(parent) == CUSTOMER_FLAG:
            brand_replies_to[brand] += 1
            brand_customers[brand].add(author_of.get(parent, parent))
            brand_threads[brand].add(parent)

    # customer tweets that mention a brand handle (@BrandName in the text)
    brand_inbound = defaultdict(int)
    brands = set(brand_outbound)
    for chunk in iter_chunks(path, chunk_size=chunk_size):
        inb = chunk[chunk["inbound"] == CUSTOMER_FLAG]
        for text in inb["text"]:
            for token in text.split():
                if token.startswith("@"):
                    handle = token[1:].strip(",.:;!?")
                    if handle in brands:
                        brand_inbound[handle] += 1
        if max_rows is not None:
            break_after = sum(brand_inbound.values())  # noqa: F841

    stats = {
        "total_rows": total_rows,
        "columns": columns,
        "missing": dict(missing),
        "duplicate_tweet_ids": duplicate_tweet_ids,
        "inbound_counts": dict(inbound_counts),
    }
    table = pd.DataFrame(
        [
            {
                "brand": b,
                "outbound_tweets": brand_outbound[b],
                "replies_to_customers": brand_replies_to.get(b, 0),
                "distinct_customers": len(brand_customers.get(b, ())),
                "reply_chains": len(brand_threads.get(b, ())),
                "inbound_mentions": brand_inbound.get(b, 0),
            }
            for b in brands
        ]
    ).sort_values("replies_to_customers", ascending=False, ignore_index=True)
    return stats, table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=None, help="limit rows scanned")
    args = ap.parse_args()

    cfg = load_config()
    path = resolve(cfg["data"]["raw_csv"])
    if not path.exists():
        raise SystemExit(f"Dataset not found at {path}. See data/README.md.")

    structure_report(path)
    stats, table = scan(path, cfg["data"].get("chunk_size", 100_000), args.rows)

    print("\n=== B. ACTUAL DATASET STATISTICS ===")
    print(f"rows scanned          : {stats['total_rows']:,}")
    print(f"columns               : {len(stats['columns'])}")
    print(f"duplicate tweet_id    : {stats['duplicate_tweet_ids']:,}")
    print(f"inbound value counts  : {stats['inbound_counts']}")
    print("empty values per column:")
    for col, n in stats["missing"].items():
        print(f"  {col:>24}: {n:,}")

    out_dir = resolve(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "brand_analysis.csv"
    table.to_csv(out_csv, index=False)

    print("\n=== C. TOP 20 BRANDS BY REPLIES TO CUSTOMERS ===")
    print(table.head(20).to_string(index=False))
    print(f"\nfull table written to {out_csv} ({len(table)} brands)")


if __name__ == "__main__":
    main()
