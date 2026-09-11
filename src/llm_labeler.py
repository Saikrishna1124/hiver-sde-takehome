"""Phase 8 - Gemini LLM Labelling Infrastructure & Disk Cache.

Provides:
- DiskCache for persistent, resumable LLM responses.
- Structured prompt formatting based on the locked 10-intent taxonomy.
- Strict golden evaluation set leakage prevention assertions.
- Safe response validation against `src.intents.INTENTS`.
- Exponential backoff retry logic for transient API issues.
- Deterministic offline/mock capabilities for testing without API keys or network.

Never sends any golden set example to Gemini.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

from .config import ROOT, gemini_api_key, load_config
from .intents import INTENTS, is_valid
from .taxonomy import TAXONOMY
from .validate_golden_set import GOLDEN_CSV, excluded_tweet_ids

log = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = (
    ROOT / "data" / "processed" / "amazonhelp" / "cache" / "gemini_labels.json"
)
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def get_golden_conversation_ids() -> set[str]:
    """Return all conversation IDs present in the sealed golden evaluation set."""
    if not GOLDEN_CSV.exists():
        return set()
    import pandas as pd

    df = pd.read_csv(GOLDEN_CSV, keep_default_na=False, dtype=str)
    return set(df["conversation_id"].astype(str))


class DiskCache:
    """Thread-safe and atomic-write JSON disk cache for LLM labelling responses."""

    def __init__(self, cache_path: Path = DEFAULT_CACHE_PATH) -> None:
        self.cache_path = Path(cache_path)
        self._entries: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """Load cache from disk if it exists."""
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._entries = data
                else:
                    self._entries = {}
            except Exception as e:
                log.warning("Could not load cache at %s: %s", self.cache_path, e)
                self._entries = {}
        else:
            self._entries = {}

    def save(self) -> None:
        """Atomically persist cache to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.cache_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(self._entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_path.replace(self.cache_path)

    def get(self, tweet_id: str) -> dict[str, Any] | None:
        """Retrieve cached result by tweet_id if valid."""
        entry = self._entries.get(str(tweet_id))
        if entry and entry.get("status") == "success" and is_valid(entry.get("assigned_intent", "")):
            return entry
        return None

    def set(
        self,
        tweet_id: str,
        customer_message: str,
        assigned_intent: str,
        reasoning: str,
        model: str,
        status: str = "success",
        raw_response: str = "",
    ) -> dict[str, Any]:
        """Store a labelling response in cache and persist to disk."""
        record = {
            "customer_tweet_id": str(tweet_id),
            "customer_message": customer_message,
            "assigned_intent": assigned_intent,
            "reasoning": reasoning,
            "model": model,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_response": raw_response,
        }
        self._entries[str(tweet_id)] = record
        self.save()
        return record

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self) -> set[str]:
        return set(self._entries.keys())


def build_system_prompt() -> str:
    """Build taxonomy instructions prompt incorporating locked intent definitions."""
    lines = [
        "You are an expert customer-support intent classification assistant for AmazonHelp.",
        "Your task is to classify a single customer message into EXACTLY ONE primary intent.",
        "Choose only from the following 10 locked intents:\n",
    ]
    for intent in INTENTS:
        if intent in TAXONOMY:
            spec = TAXONOMY[intent]
            lines.append(f"Intent: {intent}")
            lines.append(f"Description: {spec['description']}")
            lines.append("Includes: " + "; ".join(spec["belongs"]))
            lines.append("Excludes: " + "; ".join(spec["not_belongs"]))
            lines.append("")
        else:
            lines.append(f"Intent: {intent}")
            lines.append(
                "Description: Message does not fit any other intent (e.g. general chit-chat, ambiguous fragments, questions without context, foreign languages)."
            )
            lines.append("Excludes: Concrete problems that match one of the 9 specific intents.")
            lines.append("")

    lines.extend(
        [
            "Guidelines:",
            "1. Focus on the customer's PRIMARY ask.",
            "2. If multiple issues are mentioned, identify what the customer most wants resolved.",
            "3. If the message is unclear or lacks necessary context, use 'other'. Do not force an unclear message into a category.",
            "4. Return strictly a JSON object with keys 'intent' and 'reasoning'.",
            "5. The 'intent' value MUST be one of the exact 10 intent names.",
        ]
    )
    return "\n".join(lines)


def build_user_prompt(message: str) -> str:
    """Format user message for classification."""
    return (
        f"Customer Support Message:\n\"\"\"{message}\"\"\"\n\n"
        "Classify this message into exactly one of the 10 allowed intents. "
        "Output ONLY a JSON object:\n"
        "{\n"
        '  "intent": "<intent_name>",\n'
        '  "reasoning": "<concise rationale>"\n'
        "}"
    )


def extract_json_payload(response_text: str) -> dict[str, Any] | None:
    """Safely extract JSON object from raw response text."""
    text = response_text.strip()
    if text.startswith("```"):
        # Strip markdown code fencing if present
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text).strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Fallback: search for first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return None


class LLMLabeler:
    """Gemini LLM Labeller with disk caching, leakage assertion, and mock mode."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        cache_path: Path = DEFAULT_CACHE_PATH,
        mock_handler: Callable[[str], dict[str, str]] | None = None,
    ) -> None:
        self.api_key = api_key or gemini_api_key()
        if model == "gemini-2.0-flash":
            model = "gemini-3.6-flash"
        self.model = model
        self.cache = DiskCache(cache_path)
        self.mock_handler = mock_handler
        self._golden_tweet_ids = excluded_tweet_ids()
        self._golden_conv_ids = get_golden_conversation_ids()

    def label_message(
        self,
        tweet_id: str,
        conversation_id: str,
        customer_message: str,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Label a customer message with strict golden set leakage prevention and disk caching."""
        tid = str(tweet_id)
        cid = str(conversation_id)

        # 1. Hard Leakage Assertion
        if tid in self._golden_tweet_ids:
            raise ValueError(
                f"LEAKAGE VIOLATION: Tweet ID {tid} belongs to the sealed golden evaluation set!"
            )
        if cid in self._golden_conv_ids:
            raise ValueError(
                f"LEAKAGE VIOLATION: Conversation ID {cid} contains golden evaluation set messages!"
            )

        # 2. Check Disk Cache
        if not force_refresh:
            cached = self.cache.get(tid)
            if cached is not None:
                return {
                    "customer_tweet_id": tid,
                    "assigned_intent": cached["assigned_intent"],
                    "reasoning": cached.get("reasoning", ""),
                    "status": "cached",
                    "model": cached.get("model", self.model),
                }

        # 3. If in mock mode, execute mock handler
        if self.mock_handler is not None:
            mock_res = self.mock_handler(customer_message)
            raw_intent = mock_res.get("intent", "")
            reasoning = mock_res.get("reasoning", "Mock response")
            if is_valid(raw_intent):
                self.cache.set(
                    tweet_id=tid,
                    customer_message=customer_message,
                    assigned_intent=raw_intent,
                    reasoning=reasoning,
                    model="mock",
                    status="success",
                )
                return {
                    "customer_tweet_id": tid,
                    "assigned_intent": raw_intent,
                    "reasoning": reasoning,
                    "status": "success",
                    "model": "mock",
                }
            else:
                return {
                    "customer_tweet_id": tid,
                    "assigned_intent": "",
                    "reasoning": f"Invalid mock intent: {raw_intent}",
                    "status": "invalid_intent",
                    "model": "mock",
                }

        # 4. Check for API key
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Add it to .env or pass it to LLMLabeler."
            )

        # 5. Call Gemini API with exponential backoff
        system_inst = build_system_prompt()
        user_prompt = build_user_prompt(customer_message)

        url = f"{API_BASE_URL}/{self.model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        payload = {
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "systemInstruction": {"parts": [{"text": system_inst}]},
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }

        max_retries = 3
        backoff_seconds = 2.0
        response_data = None
        last_error = ""

        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=30)
                if resp.status_code == 200:
                    response_data = resp.json()
                    break
                elif resp.status_code in (429, 500, 503):
                    last_error = f"HTTP {resp.status_code}: {resp.text}"
                    time.sleep(backoff_seconds * (2**attempt))
                else:
                    last_error = f"HTTP {resp.status_code}: {resp.text}"
                    break
            except requests.RequestException as e:
                last_error = str(e)
                time.sleep(backoff_seconds * (2**attempt))

        if not response_data:
            return {
                "customer_tweet_id": tid,
                "assigned_intent": "",
                "reasoning": f"API request failed: {last_error}",
                "status": "error",
                "model": self.model,
            }

        # 6. Parse and validate response
        try:
            candidates = response_data.get("candidates", [])
            if not candidates:
                return {
                    "customer_tweet_id": tid,
                    "assigned_intent": "",
                    "reasoning": "No candidates returned by Gemini",
                    "status": "error",
                    "model": self.model,
                }
            parts = candidates[0].get("content", {}).get("parts", [])
            text_content = parts[0].get("text", "") if parts else ""
            parsed = extract_json_payload(text_content)
        except Exception as e:
            parsed = None
            last_error = str(e)

        if not parsed or "intent" not in parsed:
            return {
                "customer_tweet_id": tid,
                "assigned_intent": "",
                "reasoning": f"Malformed JSON from LLM: {last_error}",
                "status": "malformed_response",
                "model": self.model,
            }

        predicted_intent = str(parsed["intent"]).strip().lower()
        reasoning = str(parsed.get("reasoning", "")).strip()

        if not is_valid(predicted_intent):
            # Record invalid intent for audit, but do NOT treat as success
            self.cache.set(
                tweet_id=tid,
                customer_message=customer_message,
                assigned_intent=predicted_intent,
                reasoning=f"Invalid intent name: {predicted_intent}",
                model=self.model,
                status="invalid_intent",
                raw_response=text_content,
            )
            return {
                "customer_tweet_id": tid,
                "assigned_intent": "",
                "reasoning": f"Model returned invalid intent: {predicted_intent}",
                "status": "invalid_intent",
                "model": self.model,
            }

        # 7. Cache valid successful result
        self.cache.set(
            tweet_id=tid,
            customer_message=customer_message,
            assigned_intent=predicted_intent,
            reasoning=reasoning,
            model=self.model,
            status="success",
            raw_response=text_content,
        )

        return {
            "customer_tweet_id": tid,
            "assigned_intent": predicted_intent,
            "reasoning": reasoning,
            "status": "success",
            "model": self.model,
        }
