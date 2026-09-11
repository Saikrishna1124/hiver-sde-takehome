"""Phase 4b - manual/example-based validation of the CANDIDATE intents.

Purpose: print 15-20 REAL customer messages per candidate bucket so a human can
read them, spot false positives and decide the final taxonomy. Nothing here
invents labels; every line printed is copied verbatim from pairs.csv.

Run:  python3 -m src.intent_validation            (all candidates)
      python3 -m src.intent_validation delivery_problem order_status
"""

from __future__ import annotations

import re
import sys

import pandas as pd

from .config import load_config, resolve
from .intent_discovery import CANDIDATES

SAMPLE_N = 20
SEED = 42


def bucket_mask(low: pd.Series, keywords: list[str]) -> pd.Series:
    mask = pd.Series(False, index=low.index)
    for kw in keywords:
        mask |= low.str.contains(re.escape(kw), regex=True)
    return mask


def main() -> None:
    cfg = load_config()
    brand = cfg["brand"]["selected"]
    out_dir = resolve(cfg["data"]["processed_dir"]) / brand.lower()
    pairs = pd.read_csv(out_dir / "pairs.csv", keep_default_na=False)
    low = pairs["customer_message_clean"].str.lower()

    wanted = sys.argv[1:] or list(CANDIDATES)
    for name in wanted:
        spec = CANDIDATES[name]
        matched = pairs[bucket_mask(low, spec["keywords"])]
        n = len(matched)
        sample = matched.sample(min(SAMPLE_N, n), random_state=SEED)
        print(f"\n{'='*78}\n{name}  (keyword matches: {n})\n{spec['description']}\n{'='*78}")
        for i, txt in enumerate(sample["customer_message_clean"], 1):
            print(f"{i:>3}. {txt[:300]}")


if __name__ == "__main__":
    main()
