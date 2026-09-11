"""Phase 1 entry point: look at the REAL file before assuming anything.

Run:
    python -m src.inspect_data

Prints: header, dtypes, null counts, a few sample rows, and a first guess at
which columns hold text / author / reply links. Nothing here is hardcoded from
memory - it all comes from your copy of the file.
"""

from __future__ import annotations

import logging

import pandas as pd

from .config import load_config, resolve
from .data_loader import iter_chunks

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 90)


def main() -> None:
    cfg = load_config()
    path = resolve(cfg["data"]["raw_csv"])
    print(f"File: {path}")
    print(f"Size: {path.stat().st_size / 1e6:.1f} MB" if path.exists() else "MISSING")

    first = next(iter_chunks(cfg["data"]["raw_csv"], chunk_size=2000))

    print("\n=== COLUMNS ===")
    print(list(first.columns))

    print("\n=== FIRST 5 ROWS ===")
    print(first.head(5).to_string())

    print("\n=== EMPTY-VALUE COUNTS (first chunk) ===")
    print((first == "").sum())

    print("\n=== VALUE SAMPLES PER COLUMN ===")
    for col in first.columns:
        vals = first[col].head(3).tolist()
        print(f"{col:>22}: {vals}")

    print("\n=== CARDINALITY (first chunk) ===")
    for col in first.columns:
        print(f"{col:>22}: {first[col].nunique()} distinct")

    print(
        "\nNext step: use these real column names to fill in Phase 1 notes in "
        "data/README.md, then run `python -m src.brand_stats` for Phase 2."
    )


if __name__ == "__main__":
    main()
