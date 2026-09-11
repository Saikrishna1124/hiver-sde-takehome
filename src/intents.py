"""The locked AmazonHelp intent list, in one place.

Every other module imports from here so the names can never drift apart.
"""

from __future__ import annotations

from .taxonomy import FALLBACK, TAXONOMY

# 9 substantive intents + `other`
INTENTS: list[str] = list(TAXONOMY.keys()) + [FALLBACK]

OTHER = FALLBACK


def is_valid(intent: str) -> bool:
    return intent in INTENTS
