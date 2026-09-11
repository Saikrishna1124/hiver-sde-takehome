# Project Report — AI Customer Support Agent (AmazonHelp Intent Classifier)

> **Dataset**: AmazonHelp slice from *Customer Support on Twitter* (Kaggle),
> 59,565 rows. All counts and metrics are measured on this slice.
> See `REPRODUCE.md` for the full pipeline in < 4 minutes.

---

## 1. Problem Framing

### What we are building
A customer-support intent classifier for AmazonHelp Twitter conversations.
Given a single customer message, the system must assign it to exactly one of
ten locked intents so that a downstream agent (human or automated) can respond
appropriately.

The ten intents, defined in `src/intents.py` and `evaluation/labeling_guidelines.md`:

| Intent | Meaning |
| :--- | :--- |
| `delivery_order_status` | Where is my order, tracking, courier issues |
| `refund_return` | Wants money back or to return an item |
| `service_complaint` | Complaint about the support experience itself |
| `device_digital_content` | Kindle, Echo, Prime Video, Amazon app |
| `account_access` | Can't log in, account locked/hacked |
| `payment_billing` | Charges, cards, gift cards — no refund requested |
| `item_quality_issue` | Item arrived damaged, wrong, or missing |
| `seller_marketplace` | Third-party seller problems |
| `prime_membership` | Prime subscription, trial, renewal, benefits |
| `other` | None of the above; unclear or ambiguous messages |

### What "good" means for this task
A useful classifier must:
1. Achieve accuracy **well above the train-fitted majority-class baseline** (22.0%).
2. Show non-trivial **recall on minority intents** — not just classify everything
   as `delivery_order_status` or `other`.
3. Be evaluated on a **sealed golden set** that was never used in training,
   hyperparameter selection, or thresholding.

The primary evaluation metric is **Macro F1**, because it weights each intent
equally regardless of frequency. Accuracy alone is misleading on an imbalanced
label set (see Section 5).

### What we chose not to build in this submission
This submission implements and evaluates the intent-classification foundation.
The following components are designed and roadmapped but not yet implemented:

- **Reply generation**: drafting a historically-grounded response using retrieved
  past AmazonHelp replies (Phase 9-10 in `roadmap.md`).
- **Escalation decision**: AUTO_HANDLED vs HUMAN_ESCALATION routing (Phase 11).
- **LLM-as-judge**: automated reply quality scoring with human-agreement
  validation (Phase 15).
- **Embedding-based classifier**: Phase 8; would replace TF-IDF with
  sentence-transformer features.

These are documented in `roadmap.md` and referenced in Section 6 (one-more-week plan).

---

## 2. Data Pipeline & Training Data

### Source
3,603 customer-to-AmazonHelp reply pairs extracted from 1,633 conversations
(`src/conversation_builder.py`). The golden evaluation set (200 messages) and
its entire conversation threads (547 sibling messages, 181 conversations) are
excluded before any modelling.

### Stage 1 — Rule-based labels
Taxonomy regex rules are applied to all 2,139 training messages:

| Category | Count | Treatment |
| :--- | :--- | :--- |
| `single_match` (1 rule fires) | 824 | Retained with that rule's label |
| `multi_match` (2+ rules fire) | 163 | **Excluded** — genuinely ambiguous |
| `zero_match` (no rule fires) | 1,152 | Sample of 400 taken as `other` |

**Stage 1 training set**: 1,224 rows.

### Stage 2 — LLM distillation
50 most ambiguous (`multi_match`) rows were sent to the Gemini API for
re-labelling. 29 responses were successfully cached; 21 hit the free-tier daily
quota and are excluded by the pipeline's skip-not-fabricate policy.

**Stage 2 training set**: 1,253 rows
(`stage1_rule`: 824 · `stage1_sampled_other`: 400 · `llm_gemini`: 29).

The LLM agreed with the Stage 1 heuristic on 17/29 reviewed examples (58.6%),
and corrected 12 (41.4%) — confirming the value of distillation for ambiguous cases.

---

## 3. Results vs. Baselines

Evaluated on the sealed golden set: **200 hand-labelled examples**, all 10
intents present, never seen during training.

### Baseline comparison

| Model | Majority Class Predicted | Accuracy | Macro F1 | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Train-fitted majority** | `delivery_order_status` | **22.0%** | ~0.04 | Valid trivial lower bound |
| *Golden-oracle majority* | `other` | *45.0%* | *~0.07* | *Upper-bound reference only — uses test distribution* |
| **TF-IDF + LogReg** | (varies) | **60.0%** | **0.2298** | Main baseline; `train_stage1.csv` |

> The golden-oracle majority (45.0%) is **not a blind baseline**: it requires
> knowing that 45% of the test set is `other`, which a real system cannot know.
> The correct comparison is against the train-fitted majority (**22.0%**).
> See `src/majority_baseline.py` and `outputs/majority_baseline_report.md`.

The TF-IDF + LogReg classifier improves over the train-fitted majority by
**+38 percentage points** in accuracy. However, the Macro F1 of 0.2298 reveals
a severe class imbalance problem (see Section 5).

### Per-intent results (TF-IDF + LogReg on golden set)

| Intent | Support | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- | :--- |
| `delivery_order_status` | 44 | 0.48 | 0.89 | 0.62 |
| `other` | 90 | 0.67 | 0.80 | 0.73 |
| `service_complaint` | 15 | 0.78 | 0.47 | 0.58 |
| `device_digital_content` | 9 | 1.00 | 0.22 | 0.36 |
| `refund_return` | 12 | 0.00 | 0.00 | 0.00 |
| `account_access` | 7 | 0.00 | 0.00 | 0.00 |
| `payment_billing` | 6 | 0.00 | 0.00 | 0.00 |
| `item_quality_issue` | 6 | 0.00 | 0.00 | 0.00 |
| `seller_marketplace` | 6 | 0.00 | 0.00 | 0.00 |
| `prime_membership` | 5 | 0.00 | 0.00 | 0.00 |

6 of 10 intents have F1 = 0.00 on the golden set.

---

## 4. Failure Analysis — Top 5 Failure Modes

Evidence comes from the confusion matrix in `data/processed/amazonhelp/baseline_report.json`
and real message examples from `evaluation/golden_set.csv`.

---

### Failure Mode 1: Low-resource intents are invisible to the model

**Evidence**: 6 intents (`refund_return`, `account_access`, `payment_billing`,
`item_quality_issue`, `seller_marketplace`, `prime_membership`) all have F1 = 0.00.
Combined, they account for 42 of the 200 golden examples (21%) — and the model
predicts zero of them correctly.

**Root cause**: These intents have low training-set representation (5–39 examples
each in Stage 1). TF-IDF n-gram features overlap heavily with the dominant
`delivery_order_status` and `other` classes.

**Representative real message (from `evaluation/golden_set.csv`)**:
> *"@AmazonHelp Im not receiving 2step code by text or call in my smartphone
> can't log in into my account already got help b..."*
> — gold label: `account_access`, model prediction: `other`

---

### Failure Mode 2: `refund_return` systematically misclassified as `delivery_order_status`

**Evidence**: 6 of 12 `refund_return` golden examples are predicted as
`delivery_order_status`; 6 more as `other`. F1 = 0.00.

**Root cause**: Refund messages frequently mention the order, the item, and
delivery as context (e.g. "my replacement hasn't arrived and I want a refund").
TF-IDF cannot distinguish the primary ask (refund) from contextual delivery
language.

**Representative real message**:
> *"More like @AmazonHelp Sub-Prime. Late deliveries, no deliveries, unable to
> arrange replacements. No pro-active comms from..."*
> — gold label: `refund_return`, model prediction: `delivery_order_status`

---

### Failure Mode 3: `other` mass-prediction absorbs minority intent errors

**Evidence**: 18 examples truly labelled as `other` are predicted as
`delivery_order_status` (reverse error), while the model predicts `other`
for 6 `account_access`, 7 `service_complaint`, 6 `refund_return`, 5
`device_digital_content`, and 4 `seller_marketplace` examples.

**Root cause**: `other` (90 examples = 45% of golden set) and
`delivery_order_status` (44 examples = 22%) dominate the training distribution.
The model learns a strong prior toward these two classes and uses them as
catch-all categories.

---

### Failure Mode 4: `device_digital_content` has high precision but near-zero recall

**Evidence**: Precision = 1.00 but Recall = 0.22 (only 2 of 9 examples
correctly identified). 5 `device_digital_content` examples are predicted as
`other`, 2 as `delivery_order_status`.

**Root cause**: The intent is characterised by product names (Kindle, Echo,
Alexa, Fire TV). When a customer message mentions a device only in passing or
uses a non-standard term, the TF-IDF features don't fire. The model only
predicts this intent when device keywords appear exactly as in training.

**Representative real message**:
> *"@AmazonHelp Just got an email that it was lost or damaged"*
> — gold label: `item_quality_issue` (no device keyword visible; misclassified)

---

### Failure Mode 5: `service_complaint` recall limited by surface-feature overlap

**Evidence**: Recall = 0.47 (7 of 15 correct); 7 predicted as `other`,
1 as `delivery_order_status`.

**Root cause**: Service complaints use the same emotional vocabulary
("horrible", "no response", "been waiting") that also appears in delivery
and other complaint types. Without understanding that the complaint targets
the support *experience itself* rather than a concrete problem, a bag-of-words
model cannot separate the two.

---

## 5. What Is Misleading About the Headline Number?

**The headline number is 60.0% accuracy on 200 sealed golden examples.**

This is misleading in two specific ways:

**Reason 1 — Accuracy hides per-class collapse.**
Macro F1 = 0.2298 (weighted F1 = 0.5247). Six of ten intents have F1 = 0.00,
meaning the model is completely useless for `refund_return`, `account_access`,
`payment_billing`, `item_quality_issue`, `seller_marketplace`, and
`prime_membership`. A support tool that cannot identify refund requests or
account issues is not a useful support tool, regardless of its overall accuracy.

**Reason 2 — The improvement bar feels low but isn't.**
60.0% vs 45.0% (the golden-oracle majority) sounds like only a +15 pp gain.
But the 45.0% oracle figure requires knowing the test distribution in advance.
Against the honest train-fitted majority (22.0%), the gain is +38 pp. The
framing of the comparison matters significantly.

**Implication**: The accuracy score alone would pass a superficial review.
The Macro F1 = 0.23 is the honest number to report and the primary target for
improvement in the next engineering phase.

---

## 6. What We Would Do With One More Week

In priority order:

**1. Embedding-based intent classifier (Phase 8)**
Replace TF-IDF with sentence-transformer embeddings (e.g.
`all-MiniLM-L6-v2`). Embeddings capture semantic meaning rather than surface
n-grams, which directly addresses Failure Modes 1–5. Expected: significant
improvement in Macro F1 and non-zero F1 for minority intents.

**2. Train on Stage 2 dataset**
Retrain the baseline on `train_stage2.csv` (1,253 rows with 29 Gemini-distilled
labels for the hardest ambiguous examples). The Stage 2 labels resolve cases
where Stage 1 heuristics disagreed — expected to reduce the dominant-class bias.

**3. Retrieval for historically-grounded replies (Phase 9)**
Build a FAISS index over past AmazonHelp responses. For a classified intent,
retrieve the 3–5 most semantically similar past replies as grounding context
for a reply generator.

**4. Reply generation (Phase 10)**
Use the Gemini API to draft a reply conditioned on the classified intent and
retrieved past responses. The reply should be factual, on-brand, and escalate
if the classifier is uncertain.

**5. LLM-as-judge + human agreement evaluation (Phase 15)**
Sample 30–50 generated replies, have the LLM score them on a rubric
(relevance, accuracy, tone, resolution), and have a human annotator score the
same sample. Report inter-rater agreement to establish judge reliability.

---

## 7. Conclusion

The intent-classification foundation is complete and evaluated:

- **200-example sealed golden set** — hand-labelled, stratified, leakage-free.
- **Two honest baselines** — train-fitted majority (22.0%) and TF-IDF + LogReg
  (60.0% accuracy, Macro F1 = 0.23).
- **Two-stage training pipeline** — rule-based Stage 1 (1,224 rows) plus
  Gemini-distilled Stage 2 (1,253 rows, 29 LLM labels verified).
- **67 tests passing**, fully reproducible pipeline (< 4 minutes from raw data).
- **15 documented non-obvious decisions** in `decision_log.md`.

The 60.0% accuracy headline is honest but the Macro F1 = 0.23 reveals the
real challenge: minority-intent discrimination. The next phase (embedding
classifier, reply generation, LLM judge) is designed, roadmapped, and
unblocked. All evaluation numbers come from the sealed golden set and no
result has been fabricated or post-hoc adjusted.
