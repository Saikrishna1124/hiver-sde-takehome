# Labelling guidelines — AmazonHelp golden evaluation set

You are labelling **the customer's primary ask** in a single message. One label
per message. If a message contains two problems, pick the one the customer most
wants solved; if they are truly equal, pick the one an agent would act on first
and write why in the note (`n` key).

The ten intents are locked (`src/intents.py`, definitions in
`data/processed/amazonhelp/intent_definitions.json`).

---

## 1. delivery_order_status  (key 1)
**Definition.** Where is my order: not dispatched, late, no tracking update,
marked delivered but not received, courier/driver problems, address / slot /
safe-place / redelivery changes.

- **Belongs:** "order not shipped", "no tracking update", "driver never came",
  "left with a neighbour", "please change the delivery slot".
- **Does NOT belong:** the item arrived but is broken → `item_quality_issue`;
  the customer only wants money back for a late order → `refund_return`.
- **Boundary:** the *only* ask is "when will it arrive?" → still this intent.
  We merged the old `order_status` and `delivery_slot_address` into it, because
  the agent action is the same.

## 2. service_complaint  (key 2)
**Definition.** The complaint is about the *support experience itself*, or the
customer only demands a call/DM/manager instead of describing a new problem.

- **Belongs:** "nobody replied for 5 days", "your agent was rude", "call me on
  0123", "escalate to a supervisor", "worst service ever" with no concrete issue.
- **Does NOT belong:** angry wording attached to a concrete problem — label the
  problem instead. This intent is the *last resort* among the nine.
- **Boundary:** "I've emailed 3 times about my missing parcel" → the parcel is
  the ask → `delivery_order_status`.

## 3. refund_return  (key 3)
**Definition.** Wants money back, wants to return an item, or a promised
refund/replacement has not arrived.

- **Belongs:** "refund not received", "return pickup not arranged", "replacement
  rejected", "self-return cost".
- **Does NOT belong:** "I was charged twice" with no refund request →
  `payment_billing`; refund of a Prime fee → `prime_membership`.

## 4. device_digital_content  (key 4)
**Definition.** Amazon-made devices and Amazon digital services misbehaving:
Kindle, Echo/Alexa, Fire TV/Stick, Prime Video, Amazon Music, Audible, and the
Amazon app/website itself.

- **Belongs:** "Kindle won't sync", "Alexa stopped responding", "Prime Video
  keeps buffering", "your app crashes at checkout".
- **Does NOT belong:** "my Echo Dot hasn't been delivered" → delivery. The bare
  words "app" or "video" are **not** enough — this is exactly the false-positive
  trap that inflated the old bucket.

## 5. seller_marketplace  (key 5)
**Definition.** The problem is with a third-party seller/vendor, not Amazon's
own operation.
- **Belongs:** "seller won't respond", "seller sent a fake", "A-to-Z claim",
  Vendor Central issues.
- **Does NOT belong:** damaged item with no seller mentioned →
  `item_quality_issue`.

## 6. account_access  (key 6)
**Definition.** Cannot get into or control the Amazon account.
- **Belongs:** can't log in, password/2-step, account locked/suspended/hacked,
  wrong email or phone on the account.
- **Does NOT belong:** "money not credited to my **bank** account" →
  `payment_billing`; "cancel my Prime account" → `prime_membership`.

## 7. payment_billing  (key 7)
**Definition.** Charges, cards, gift cards, pricing, invoices — where no
refund/return is being requested.
- **Belongs:** card declined, double charge, pending transaction, gift-card
  balance, EMI, price differs at checkout.
- **Does NOT belong:** "please refund me" → `refund_return`.

## 8. item_quality_issue  (key 8)
**Definition.** What arrived is not what it should be.
- **Belongs:** damaged/broken on arrival, faulty, counterfeit, used, wrong item
  or size, missing item, empty box.
- **Does NOT belong:** nothing arrived at all → `delivery_order_status`.

## 9. prime_membership  (key 9)
**Definition.** The Prime subscription itself: trial, renewal, benefits,
membership charge, cancellation.
- **Does NOT belong:** "my Prime order is late" (Prime is only context) →
  delivery; Prime Video playback → `device_digital_content`.

## 0. other  (key 0)
Use `other` when **none of the nine** accurately describes the customer's
underlying issue. Do not force an unclear message into a category.

Typical real cases in this data:
- short follow-up turns inside a thread: "Yes, I've sent an email through your
  link.", "DM sent", "thanks";
- messages in a non-Latin script we cannot read confidently;
- general questions, jokes, sarcasm, praise;
- messages where the problem is genuinely unclear without the earlier turns.

`other` being large is an honest result, not a failure. If you find yourself
guessing, the answer is `other` plus a short note.

---

## Ambiguity procedure
1. Read only the message shown — that is what the classifier will see.
2. If two intents fit, apply the exclusion notes above.
3. If still unsure after ~15 seconds: press `s` (skip), come back at the end.
4. If it is unsure *by nature*: label `other` and press `n` to record why.

## The hint column
`suggested_intent` in the CSV is a **regex heuristic**, shown only to speed you
up. It is wrong often (that is measured in `taxonomy_review.md`). Disagreeing
with it is expected and valuable — those disagreements are the most interesting
rows in the whole set.
