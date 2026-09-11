"""Phase 1 - text cleaning.

The raw text column is always preserved; cleaning writes a NEW column so
nothing is lost and every step can be inspected side by side.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

log = logging.getLogger(__name__)

MENTION_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_RE = re.compile(r"\s+")
# Kaggle's twcs.csv replaces some links with a literal placeholder token.
PLACEHOLDER_RE = re.compile(r"\bhttps://t\.co/\w+\b")


def clean_text(text: str) -> str:
    """Light, reversible-in-spirit cleaning. No stemming, no lowercasing tricks.

    Steps (each one is easy to justify in an interview):
      1. drop URLs      - they carry no intent signal
      2. drop @mentions - they are mostly the brand handle / ticket routing
      3. collapse whitespace and strip
    """
    if not isinstance(text, str):
        return ""
    t = URL_RE.sub(" ", text)
    t = PLACEHOLDER_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = WHITESPACE_RE.sub(" ", t)
    return t.strip()


def add_clean_column(
    df: pd.DataFrame, text_col: str, out_col: str = "clean_text"
) -> pd.DataFrame:
    """Return a copy of df with an extra cleaned-text column."""
    if text_col not in df.columns:
        raise KeyError(
            f"Column {text_col!r} not in dataframe. Columns: {list(df.columns)}"
        )
    out = df.copy()
    out[out_col] = out[text_col].map(clean_text)
    log.info("cleaned %d rows from %r into %r", len(out), text_col, out_col)
    return out


def drop_unusable(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """Remove rows that cannot be used: empty text after cleaning.

    Logged so the size at every stage is visible (Phase 1 requirement).
    """
    before = len(df)
    out = df[df[text_col].str.len() > 0].copy()
    log.info("drop_unusable: %d -> %d rows (removed %d empty)", before, len(out), before - len(out))
    return out
