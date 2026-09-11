"""Phase 9 - Historical support response retrieval.

Grounds drafted replies in real AmazonHelp responses by finding top-k nearest
customer inquiries from the leakage-safe training split (train.csv).
Uses TF-IDF + cosine similarity (scikit-learn) for fast, deterministic, offline execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import ROOT, load_config, resolve
from src.preprocessing import clean_text

DEFAULT_TRAIN_PATH = ROOT / "data" / "processed" / "amazonhelp" / "train.csv"
LEADING_MENTION_RE = re.compile(r"^(\s*@\w+[\s,:]*)+")


def clean_brand_reply(reply: str) -> str:
    """Format a historical brand reply into a general grounded template.

    Strips leading customer handles (@115850 etc.) while preserving the actual
    brand response content and links.
    """
    if not isinstance(reply, str):
        return ""
    cleaned = LEADING_MENTION_RE.sub("", reply).strip()
    return cleaned if cleaned else reply.strip()


@dataclass
class RetrievalResult:
    similarity: float
    brand_reply_grounded: str
    brand_reply_raw: str
    historical_customer_message: str
    customer_tweet_id: str
    brand_reply_id: str
    intent: str


class HistoricalReplyRetriever:
    """Retrieves historically grounded AmazonHelp replies from training data."""

    def __init__(
        self,
        data_source: str | Path | pd.DataFrame | None = None,
        max_features: int = 10000,
    ) -> None:
        if data_source is None:
            data_source = DEFAULT_TRAIN_PATH

        if isinstance(data_source, pd.DataFrame):
            self.df = data_source.copy()
        else:
            path = Path(data_source)
            if not path.exists():
                raise FileNotFoundError(f"Training data not found at: {path}")
            self.df = pd.read_csv(path, keep_default_na=False, dtype=str)

        if "customer_message_clean" not in self.df.columns:
            self.df["customer_message_clean"] = self.df["customer_message"].map(clean_text)

        # Drop any row with empty customer text or empty brand reply
        valid_mask = (
            (self.df["customer_message_clean"].str.strip().str.len() > 0)
            & (self.df["brand_reply"].str.strip().str.len() > 0)
        )
        self.df = self.df[valid_mask].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError("No valid training pairs found to index.")

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            stop_words="english",
            sublinear_tf=True,
        )
        self.doc_vectors = self.vectorizer.fit_transform(self.df["customer_message_clean"])

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        intent_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve top-k historically similar customer interactions and brand replies."""
        cleaned_query = clean_text(query)
        if not cleaned_query:
            return []

        query_vec = self.vectorizer.transform([cleaned_query])
        sims = cosine_similarity(query_vec, self.doc_vectors).ravel()

        if intent_filter and "proposed_intent" in self.df.columns:
            intent_mask = (self.df["proposed_intent"] == intent_filter).to_numpy()
            # If there are matches within the intent, apply filter; otherwise fallback to full set
            if np.any(intent_mask):
                sims = np.where(intent_mask, sims, -1.0)

        top_indices = np.argsort(-sims)
        results: list[RetrievalResult] = []

        for idx in top_indices:
            score = float(sims[idx])
            if score < 0 or len(results) >= top_k:
                break

            row = self.df.iloc[idx]
            raw_reply = str(row.get("brand_reply", ""))
            grounded_reply = clean_brand_reply(raw_reply)
            intent = str(row.get("training_label", row.get("proposed_intent", "other")))

            results.append(
                RetrievalResult(
                    similarity=round(score, 4),
                    brand_reply_grounded=grounded_reply,
                    brand_reply_raw=raw_reply,
                    historical_customer_message=str(row.get("customer_message_clean", "")),
                    customer_tweet_id=str(row.get("customer_tweet_id", "")),
                    brand_reply_id=str(row.get("brand_reply_id", "")),
                    intent=intent,
                )
            )

        return results


if __name__ == "__main__":
    retriever = HistoricalReplyRetriever()
    test_query = "Where is my order? The tracking number says delivered but I received nothing."
    matches = retriever.retrieve(test_query, top_k=2)
    print(f"Query: {test_query}\n")
    for i, m in enumerate(matches, 1):
        print(f"[{i}] Sim: {m.similarity} | Intent: {m.intent}")
        print(f"    Hist Inquiry: {m.historical_customer_message[:80]}...")
        print(f"    Grounded Reply: {m.brand_reply_grounded[:100]}...\n")
