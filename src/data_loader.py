"""Phase 1 - memory-conscious loading of the raw Twitter support CSV.

Design rules:
* never read the whole ~3M-row file into memory unless explicitly asked;
* everything goes through pandas' chunked reader;
* the caller decides how many rows to keep.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import pandas as pd

from .config import load_config, resolve

log = logging.getLogger(__name__)


def iter_chunks(
    csv_path: str | Path,
    chunk_size: int = 100_000,
    usecols: list[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield the CSV one chunk at a time."""
    path = resolve(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}.\n"
            "Download twcs.csv from Kaggle and place it there "
            "(see data/README.md)."
        )
    reader = pd.read_csv(
        path,
        chunksize=chunk_size,
        usecols=usecols,
        dtype=str,  # keep raw text exactly as-is; cast later, explicitly
        keep_default_na=False,
    )
    for i, chunk in enumerate(reader):
        log.info("read chunk %d: %d rows", i, len(chunk))
        yield chunk


def load_rows(
    csv_path: str | Path,
    max_rows: int | None = None,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Load at most `max_rows` rows (None = the whole file)."""
    frames: list[pd.DataFrame] = []
    total = 0
    for chunk in iter_chunks(csv_path, chunk_size=chunk_size):
        if max_rows is not None and total + len(chunk) >= max_rows:
            frames.append(chunk.iloc[: max_rows - total])
            total = max_rows
            break
        frames.append(chunk)
        total += len(chunk)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log.info("loaded %d rows from %s", len(df), csv_path)
    return df


def load_from_config(cfg: dict | None = None) -> pd.DataFrame:
    """Convenience wrapper that reads paths/sizes from config.yaml."""
    cfg = cfg or load_config()
    return load_rows(
        cfg["data"]["raw_csv"],
        max_rows=cfg["data"].get("scan_rows"),
        chunk_size=cfg["data"].get("chunk_size", 100_000),
    )
