"""Phase 4 (exploration only) - find CANDIDATE support topics in real data.

This script does NOT decide the intent taxonomy. It surfaces evidence:

* the most frequent words / bigrams in real AmazonHelp customer messages;
* a set of keyword-defined candidate topics, each with its real match count
  and real example messages copied verbatim from the dataset.

A human (you) reads `data/processed/amazonhelp/candidate_intents.json` and
decides the final small taxonomy afterwards.

Run:  python3 -m src.intent_discovery
"""

from __future__ import annotations

import json
import re
from collections import Counter

import pandas as pd

from .config import load_config, resolve

# Very common English words removed so the frequency table shows topic words.
STOPWORDS = set(
    """a an the and or but if so to of in on for with at by from as is are was were be been
being it its it's this that these those i i'm ive i've im me my mine you your yours we us our
they them their he she his her do does did doing done have has had not no yes can could will
would should just now get got getting go going out up down about into over then than there here
what when where why how who which all any some more most other same too very s t don't dont
cant can't didn't didnt still please thanks thank hi hello hey amp u ur pls plz ok okay one two
back also again let know need want said say says like even much many via re ve ll m d""".split()
)

# Candidate buckets. Keywords come from reading the top-frequency table and
# real messages - they are hypotheses to review, NOT the final taxonomy.
CANDIDATES: dict[str, dict] = {
    "order_status": {
        "description": "Where is my order / has it shipped / tracking shows nothing.",
        "keywords": ["order status", "where is my order", "tracking", "not shipped",
                     "hasn't shipped", "has not shipped", "dispatch", "shipment"],
    },
    "delivery_problem": {
        "description": "Parcel late, missing, marked delivered but not received, driver issue.",
        "keywords": ["delivery", "delivered", "deliver", "late", "arrived", "not arrive",
                     "courier", "driver", "parcel", "package"],
    },
    "refund_return": {
        "description": "Customer wants money back, a return, or a replacement.",
        "keywords": ["refund", "return", "replacement", "money back", "reimburse",
                     "send it back", "returned"],
    },
    "prime_membership": {
        "description": "Prime subscription: signup, benefits, renewal, cancellation, charges.",
        "keywords": ["prime", "membership", "subscription", "renew", "free trial"],
    },
    "payment_billing": {
        "description": "Charges, card declined, double billing, gift cards, pricing disputes.",
        "keywords": ["charge", "charged", "payment", "card", "billing", "invoice",
                     "gift card", "price", "overcharged", "debit"],
    },
    "account_login": {
        "description": "Cannot log in, account locked/closed, password or email problems.",
        "keywords": ["account", "log in", "login", "sign in", "password", "locked",
                     "suspended", "verify", "verification"],
    },
    "cancellation": {
        "description": "Customer wants to cancel an order, subscription or service.",
        "keywords": ["cancel", "cancelled", "cancellation", "canceling"],
    },
    "item_quality_wrong_item": {
        "description": "Damaged, faulty, counterfeit, or the wrong item was received.",
        "keywords": ["damaged", "broken", "faulty", "defective", "wrong item",
                     "wrong product", "fake", "used", "missing item"],
    },
    "customer_service_complaint": {
        "description": "Complaint about the support experience itself (rude, no answer, going in circles).",
        "keywords": ["customer service", "customer support", "no one", "nobody",
                     "worst", "terrible", "unacceptable", "complaint", "hours on the phone"],
    },
    "digital_content_devices": {
        "description": "Kindle, Echo/Alexa, Fire TV, app, video/music streaming problems.",
        "keywords": ["kindle", "alexa", "echo", "fire tv", "firestick", "app",
                     "video", "music", "stream", "download"],
    },
    "seller_marketplace": {
        "description": "Third-party seller behaviour, listings, reviews, A-to-Z claims.",
        "keywords": ["seller", "marketplace", "listing", "review", "third party",
                     "a-to-z", "vendor"],
    },
    "contact_channel_request": {
        "description": "Customer asking to be contacted / for DM / for a phone number or email.",
        "keywords": ["dm", "direct message", "phone number", "email me", "call me",
                     "contact", "escalate", "supervisor", "manager"],
    },
    "praise_thanks": {
        "description": "Positive feedback or thanks - usually needs no action.",
        "keywords": ["thank you so much", "great service", "awesome", "love amazon",
                     "appreciate", "brilliant"],
    },
    "delivery_slot_address": {
        "description": "Change delivery address, time slot, safe place, locker or pickup point.",
        "keywords": ["address", "time slot", "reschedule", "locker", "pickup",
                     "safe place", "neighbour", "redeliver"],
    },
}


def tokenise(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z']{3,}", text.lower()) if w not in STOPWORDS]


def frequency_tables(messages: pd.Series, top: int = 30):
    unigrams: Counter = Counter()
    bigrams: Counter = Counter()
    for msg in messages:
        toks = tokenise(msg)
        unigrams.update(toks)
        bigrams.update(f"{a} {b}" for a, b in zip(toks, toks[1:]))
    return unigrams.most_common(top), bigrams.most_common(top)


def match_candidates(df: pd.DataFrame, text_col: str, n_examples: int = 5) -> list[dict]:
    low = df[text_col].str.lower()
    out = []
    for name, spec in CANDIDATES.items():
        mask = pd.Series(False, index=df.index)
        for kw in spec["keywords"]:
            mask |= low.str.contains(re.escape(kw), regex=True)
        matched = df[mask]
        examples = matched[text_col].head(n_examples).tolist()
        out.append(
            {
                "candidate_intent": name,
                "description": spec["description"],
                "keywords": spec["keywords"],
                "n_matched_messages": int(mask.sum()),
                "share_of_messages": round(float(mask.mean()), 4),
                "examples": examples,
            }
        )
    return sorted(out, key=lambda d: -d["n_matched_messages"])


def main() -> None:
    cfg = load_config()
    brand = cfg["brand"]["selected"]
    out_dir = resolve(cfg["data"]["processed_dir"]) / brand.lower()
    pairs_path = out_dir / "pairs.csv"
    if not pairs_path.exists():
        raise SystemExit(
            f"{pairs_path} missing. Run `python3 -m src.conversation_builder` first."
        )

    pairs = pd.read_csv(pairs_path, keep_default_na=False)
    # keyword matching is done on the CLEANED text (no @handles, no URLs)
    msgs = pairs["customer_message_clean"]

    uni, bi = frequency_tables(msgs)
    print(f"\n=== C1. TOP WORDS in {len(msgs)} real {brand} customer messages ===")
    for w, c in uni:
        print(f"  {w:>18}: {c}")
    print("\n=== C2. TOP BIGRAMS ===")
    for w, c in bi:
        print(f"  {w:>26}: {c}")

    cands = match_candidates(pairs, "customer_message_clean")
    print("\n=== C3. CANDIDATE TOPICS (keyword hypotheses - NOT final) ===")
    print(f"{'candidate_intent':>28} {'matched':>8} {'share':>7}")
    for c in cands:
        print(f"{c['candidate_intent']:>28} {c['n_matched_messages']:>8} {c['share_of_messages']:>7}")

    overlap = sum(c["n_matched_messages"] for c in cands) / max(len(msgs), 1)
    print(f"\nnote: buckets overlap - total matches / messages = {overlap:.2f}")

    payload = {
        "brand": brand,
        "n_messages_analysed": int(len(msgs)),
        "top_unigrams": uni,
        "top_bigrams": bi,
        "candidates": cands,
        "status": "CANDIDATES ONLY - final taxonomy not decided",
    }
    path = out_dir / "candidate_intents.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
