"""Phase 4c - the PROPOSED final AmazonHelp intent taxonomy.

Important honesty note
----------------------
The rules below are *not* ground truth. They are a priority-ordered,
single-label heuristic that turns the overlapping keyword buckets from
`intent_discovery.py` into one proposed label per message, so that we can:

  * count roughly how many real messages each proposed intent has, and
  * pull real example messages for each intent.

The hand-labelled golden set (150-250 examples) comes later and is what we
will actually measure against.

Design rules that came out of reading real messages:
  * one label per message = the customer's PRIMARY ask;
  * rules are ordered, first match wins (see PRIORITY below);
  * anything that matches nothing goes to `other`.

Run:  python3 -m src.taxonomy
Writes: data/processed/amazonhelp/intent_definitions.json
        data/processed/amazonhelp/taxonomy_review.md (counts table only)
"""

from __future__ import annotations

import json
import re

import pandas as pd

from .config import load_config, resolve

# Each intent: description, what belongs / what does not, and the regex used
# only to PROPOSE a label. Order in this dict IS the matching priority.
TAXONOMY: dict[str, dict] = {
    "account_access": {
        "description": "Customer cannot get into or control their Amazon account.",
        "belongs": [
            "cannot log in / wrong password / two-step verification",
            "account locked, suspended, closed or hacked",
            "email or phone number on the account is wrong",
        ],
        "not_belongs": [
            "'money not credited to my bank account' -> payment_billing",
            "'cancel my Prime account' -> prime_membership",
        ],
        "pattern": r"(hack(ed)?|log ?in|logged in|sign ?in|password|account (is )?(locked|blocked|suspended|closed|on hold)|locked (out )?of my account|verify my (account|identity)|two.step)",
    },
    "prime_membership": {
        "description": "Anything about the Prime subscription itself: trial, renewal, benefits, membership charge, membership cancellation.",
        "belongs": [
            "charged for Prime / trial auto-renewed",
            "cancel my Prime membership",
            "what does Prime include, Prime student, complimentary Prime",
        ],
        "not_belongs": [
            "'my Prime order is late' -> delivery_order_status (Prime is only context)",
            "Prime Video / Music playback -> device_digital_content",
        ],
        "pattern": r"(prime (membership|subscription|trial|student|renew)|membership (fee|charge|cancel|renew)|cancel (my )?prime|free trial|renewal of prime)",
    },
    "device_digital_content": {
        "description": "Amazon-made devices and Amazon digital services misbehaving: Kindle, Echo/Alexa, Fire TV, Prime Video, Amazon Music, Audible, and the Amazon app/website itself.",
        "belongs": [
            "Kindle screen / sync / Kindle Unlimited problems",
            "Echo, Alexa, Fire TV setup or faults",
            "Prime Video, Amazon Music, Audible playback",
            "Amazon app or website not working",
        ],
        "not_belongs": [
            "'my Echo Dot has not been delivered' -> delivery_order_status",
            "the words 'app' or 'video' alone are NOT enough (that is what inflated the old bucket)",
        ],
        "pattern": r"(kindle|alexa|echo dot|echo plus|amazon echo|fire ?tv|fire ?stick|paperwhite|audible|prime video|amazon music|music unlimited|kindle unlimited|cloud reader|(amazon|the|your) (app|website|site) (is )?(down|not working|crash|broken|won'?t)|streaming|stream (music|video))",
    },
    "seller_marketplace": {
        "description": "The problem is with a third-party seller or vendor, not with Amazon's own operation.",
        "belongs": [
            "seller not responding, seller sold a fake/wrong item",
            "A-to-Z claim against a seller, vendor central issues",
        ],
        "not_belongs": [
            "damaged item with no mention of a seller -> item_quality_issue",
        ],
        "pattern": r"(seller|third[ -]party|vendor central|marketplace|a[- ]to[- ]z)",
    },
    "refund_return": {
        "description": "Customer wants money back, wants to return an item, or a promised refund/replacement has not arrived.",
        "belongs": [
            "refund not received / refund delayed",
            "return pickup not arranged, self-return cost disputes",
            "replacement requested or rejected",
        ],
        "not_belongs": [
            "'I was charged twice' with no refund request -> payment_billing",
            "Prime membership refund -> prime_membership",
        ],
        "pattern": r"(refund|money back|return (my|the|this|it|pickup|request)|self[ -]?return|replacement|reimburse|returned the)",
    },
    "payment_billing": {
        "description": "Charges, card and gift-card problems, pricing and invoices - where no refund/return is being asked for.",
        "belongs": [
            "card declined, double charge, pending transactions",
            "gift card / balance / EMI / price shown differs from checkout",
        ],
        "not_belongs": ["'please refund me' -> refund_return"],
        "pattern": r"(charg(ed|ing|es)|double charg|payment (declined|failed|method)|debit|credit card|gift card|invoice|billing|price (is|shown|higher|lower)|deducted|transaction)",
    },
    "item_quality_issue": {
        "description": "What arrived is not what it should be: damaged, faulty, counterfeit, used, wrong item, missing item, empty box.",
        "belongs": ["broken/damaged on arrival", "fake or used product", "wrong or missing item, empty box"],
        "not_belongs": ["nothing arrived at all -> delivery_order_status"],
        "pattern": r"(damaged|broken|faulty|defective|fake product|counterfeit|wrong (item|product|size|colour|color)|missing item|empty box|not as described|warranty)",
    },
    "delivery_order_status": {
        "description": "Where is my order: not dispatched, late, stuck in tracking, marked delivered but not received, courier/driver problems, delivery address or slot changes.",
        "belongs": [
            "order not shipped / no tracking update",
            "late or failed delivery, driver did not attempt",
            "marked delivered but not received, left with neighbour",
            "change address, slot, safe place, locker, redelivery",
        ],
        "not_belongs": [
            "item arrived but is broken -> item_quality_issue",
            "asking only for a refund for a late order -> refund_return",
        ],
        "pattern": r"(deliver|delivery|delivered|dlvr|dispatch|shipp|shipment|tracking|courier|driver|parcel|package|my order|the order|order (id|no|number|status|still|not|hasn'?t|has been)|where is my|out for delivery|locker|safe place|neighbour|neighbor|redeliver|time slot|reschedul|pickup|pick ?up|arrive|delay)",
    },
    "service_complaint": {
        "description": "The complaint is about the support experience itself, or the customer is demanding a call/DM/manager rather than describing a new problem.",
        "belongs": [
            "no one replied, going in circles, agent was rude",
            "'call me on <number>', 'I want a manager/supervisor', 'escalate this'",
        ],
        "not_belongs": [
            "message that also states a concrete problem -> that problem's intent (this rule runs last)",
        ],
        "pattern": r"(customer (service|care|support)|cust\.? serv|support (team|person|member)|no ?one (has )?(called|replied|responded|contact)|nobody (has )?(called|replied)|no (reply|response)|not (yet )?(resolved|solved)|worst|pathetic|terrible|unacceptable|escalate|supervisor|manager|call me|dm me|contact me|phone number|complaint|emailed|chat with)",
    },
}

FALLBACK = "other"


def match_rules(text: str) -> list[str]:
    """Return all taxonomy rule names whose regex matches the text."""
    low = str(text).lower()
    return [name for name, spec in TAXONOMY.items() if re.search(spec["pattern"], low)]


def assign(text: str) -> str:
    low = str(text).lower()
    for name, spec in TAXONOMY.items():
        if re.search(spec["pattern"], low):
            return name
    return FALLBACK


def main() -> None:
    cfg = load_config()
    brand = cfg["brand"]["selected"]
    out_dir = resolve(cfg["data"]["processed_dir"]) / brand.lower()
    pairs = pd.read_csv(out_dir / "pairs.csv", keep_default_na=False)
    col = "customer_message_clean"
    pairs["proposed_intent"] = pairs[col].map(assign)

    counts = pairs["proposed_intent"].value_counts()
    print(f"proposed single-label distribution over {len(pairs)} real messages")
    for name, n in counts.items():
        print(f"  {name:>28}: {n:>5}  ({n / len(pairs):.1%})")

    definitions = {}
    for name, spec in TAXONOMY.items():
        sub = pairs[pairs["proposed_intent"] == name]
        examples = sub[col].sample(min(5, len(sub)), random_state=7).tolist()
        definitions[name] = {
            "description": spec["description"],
            "positive_examples": examples,
            "boundary_notes": ["INCLUDES: " + b for b in spec["belongs"]]
            + ["EXCLUDES: " + b for b in spec["not_belongs"]],
            "example_count": int(len(sub)),
        }
    sub = pairs[pairs["proposed_intent"] == FALLBACK]
    definitions[FALLBACK] = {
        "description": "Message does not fit any intent above (chit-chat, unclear, non-English fragments, general questions).",
        "positive_examples": sub[col].sample(min(5, len(sub)), random_state=7).tolist(),
        "boundary_notes": [
            "Kept deliberately so the classifier is not forced to guess.",
            "If this bucket grows past ~15% of the golden set, the taxonomy needs another intent.",
        ],
        "example_count": int(len(sub)),
    }

    (out_dir / "intent_definitions.json").write_text(
        json.dumps(definitions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pairs[["conversation_id", "customer_tweet_id", col, "proposed_intent"]].to_csv(
        out_dir / "proposed_labels.csv", index=False
    )
    print(f"\nwritten: {out_dir/'intent_definitions.json'}")
    print(f"written: {out_dir/'proposed_labels.csv'}  (PROPOSED labels, not ground truth)")


if __name__ == "__main__":
    main()
