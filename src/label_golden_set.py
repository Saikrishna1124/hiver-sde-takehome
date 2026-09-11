"""Phase 12b - the local hand-labelling tool.

No web server, no database: it prints one real customer message at a time and
writes your answer straight back into evaluation/golden_set.csv, so you can quit
any time (Ctrl-C or `q`) and resume where you left off.

Run:  python3 -m src.label_golden_set
Keys: 1-9,0 = intent   s = skip   p = previous   n = add a note
      q = save and quit
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import ROOT
from .intents import INTENTS

GOLDEN_CSV = ROOT / "evaluation" / "golden_set.csv"
PROGRESS_JSON = ROOT / "evaluation" / "label_progress.json"

# 1..9 for the nine specific intents, 0 for `other`
KEYS = {str(i + 1): name for i, name in enumerate(INTENTS[:9])}
KEYS["0"] = INTENTS[9]


def load() -> pd.DataFrame:
    if not GOLDEN_CSV.exists():
        raise FileNotFoundError(
            f"{GOLDEN_CSV} not found. Run `python3 -m src.sample_golden_set` first."
        )
    return pd.read_csv(GOLDEN_CSV, keep_default_na=False, dtype=str)


def save(df: pd.DataFrame, path: Path = GOLDEN_CSV) -> None:
    """Atomic-ish save: write a temp file, then replace, so a crash mid-write
    cannot destroy labels you already did."""
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def load_progress() -> dict:
    if PROGRESS_JSON.exists():
        return json.loads(PROGRESS_JSON.read_text(encoding="utf-8"))
    return {"skipped": []}


def save_progress(progress: dict) -> None:
    PROGRESS_JSON.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def next_index(df: pd.DataFrame, skipped: list[str], start: int = 0) -> int | None:
    """First row that has no gold_intent and was not skipped this run."""
    for i in range(start, len(df)):
        if df.at[i, "gold_intent"].strip() == "" and df.at[i, "example_id"] not in skipped:
            return i
    return None


def counts(df: pd.DataFrame) -> dict[str, int]:
    labelled = df[df["gold_intent"].str.strip() != ""]
    return labelled["gold_intent"].value_counts().to_dict()


def render(df: pd.DataFrame, i: int, skipped: list[str]) -> str:
    row = df.iloc[i]
    done = int((df["gold_intent"].str.strip() != "").sum())
    lines = [
        "=" * 72,
        f"[{row['example_id']}]  labelled {done}/{len(df)}   "
        f"remaining {len(df) - done}   skipped {len(skipped)}",
        "=" * 72,
        "",
        row["customer_message"],
        "",
        f"(length: {len(row['customer_message'])} chars | conversation {row['conversation_id']})",
        f"(heuristic hint - NOT the answer: {row.get('suggested_intent', '')})",
        "-" * 72,
    ]
    for key, name in KEYS.items():
        lines.append(f"  {key} = {name}")
    lines.append("  s = skip    p = previous    n = note    q = save & quit")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - interactive loop
    df = load()
    progress = load_progress()
    skipped: list[str] = progress.get("skipped", [])
    history: list[int] = []

    i = next_index(df, skipped)
    while i is not None:
        print("\n" + render(df, i, skipped))
        choice = input("> ").strip().lower()

        if choice == "q":
            break
        if choice == "p":
            if history:
                i = history.pop()
                df.at[i, "gold_intent"] = ""
            continue
        if choice == "s":
            skipped.append(df.at[i, "example_id"])
            save_progress({"skipped": skipped})
            history.append(i)
            i = next_index(df, skipped, i + 1)
            continue
        if choice == "n":
            df.at[i, "label_notes"] = input("note> ").strip()
            save(df)
            continue
        if choice in KEYS:
            df.at[i, "gold_intent"] = KEYS[choice]
            save(df)
            history.append(i)
            i = next_index(df, skipped, i + 1)
            continue
        print("unknown key - try again")

    save(df)
    save_progress({"skipped": skipped})
    done = int((df["gold_intent"].str.strip() != "").sum())
    print(f"\nsaved. labelled {done}/{len(df)}, skipped {len(skipped)}")
    for k, v in counts(df).items():
        print(f"  {k:>24}: {v}")
    print(f"\nfile: {GOLDEN_CSV}")
    print("check it any time with: python3 -m src.validate_golden_set")


if __name__ == "__main__":
    main()
