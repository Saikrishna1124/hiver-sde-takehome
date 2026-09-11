"""Loads config.yaml into a plain dictionary.

Kept deliberately tiny: no config framework, no classes to memorise.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Repository root = the folder that contains config.yaml
ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read config.yaml and return it as a dict."""
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve(relative_path: str | Path) -> Path:
    """Turn a config path like 'data/raw/twcs.csv' into an absolute path."""
    p = Path(relative_path)
    return p if p.is_absolute() else ROOT / p


def gemini_api_key() -> str | None:
    """Read the Gemini key from the environment. Never hardcode it."""
    return os.environ.get("GEMINI_API_KEY")
