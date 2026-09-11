# Hiver SDE Intern Take-Home — AI Customer Support Agent

Building an AI support agent on real Twitter customer-support conversations:
classify the customer's intent, draft a reply grounded in how the brand actually
replied before, and decide `AUTO_HANDLED` vs `HUMAN_ESCALATION` — then prove it works
with a hand-labelled golden set, baselines, and an LLM judge.

> **Current status: Full Pipeline Complete & Verified.** Dataset reconstruction,
> leakage-safe split, locked 10-intent taxonomy, two-stage intent labelling, TF-IDF + LogReg
> baseline, historically-grounded reply retrieval (2,139 training pairs), explainable
> escalation routing (AUTO_HANDLED vs HUMAN_ESCALATION), and LLM-as-a-judge evaluation
> with measured human agreement. **84 tests passing.**
> See `REPRODUCE.md` (< 4 min end-to-end, 100% offline) and `outputs/report.md` for full findings.

---

## End-to-End Pipeline

```text
Customer message
        │
        ▼
[Intent Classification] (TF-IDF + LogReg, 10-intent taxonomy)
        │
        ▼
[Historical Knowledge Retrieval] (k-NN over 2,139 real AmazonHelp support pairs)
        │
        ▼
[Grounded Reply Selection] (Authentic historical AmazonHelp response template)
        │
        ▼
[Escalation Policy Router] (Deterministic safety rules + explicit reasons)
        │
        ├── AUTO_HANDLED (High-confidence, grounded transactional inquiries)
        └── HUMAN_ESCALATION (Sensitive categories, low confidence, urgent keywords)
```

The system uses historical AmazonHelp support conversations to ground responses and applies deterministic escalation rules when confidence, grounding, intent sensitivity, or urgency require human review.

---

## Demo — Working CLI Inference

### Example 1: Automated Resolution (`AUTO_HANDLED`)

```bash
python -m src.agent --message "Where is my parcel?"
```

```text
============================================================
AI SUPPORT AGENT INFERENCE RESULT
============================================================
Customer Message : Where is my parcel?
Predicted Intent : delivery_order_status (Confidence: 69.9%)
Top Hist. Sim    : 0.40
Routing Decision : [AUTO_HANDLED]
Routing Reason   : High-confidence delivery_order_status inquiry (0.70 >= 0.45) with verified historical response template (similarity: 0.40).
Reply Source     : historical_retrieval
------------------------------------------------------------
Grounded Reply   :
"I'm sorry about the delay! What information is provided on your tracking? You can see that here: https://t.co/aaDyEz1VgE ^TR"
============================================================
```

### Example 2: Safety Escalation (`HUMAN_ESCALATION`)

```bash
python -m src.agent --message "Someone made a fraud charge on my card connect me to a manager"
```

```text
============================================================
AI SUPPORT AGENT INFERENCE RESULT
============================================================
Customer Message : Someone made a fraud charge on my card connect me to a manager
Predicted Intent : other (Confidence: 39.9%)
Top Hist. Sim    : 0.17
Routing Decision : [HUMAN_ESCALATION]
Routing Reason   : Urgent escalation keyword detected in message: 'fraud'
Reply Source     : historical_retrieval
============================================================
```

---

## What you need to provide

1. `data/raw/twcs.csv` — download from Kaggle [thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) if you wish to re-run the raw conversation builder from scratch.
2. **No API key is required** for the final agent CLI, batch evaluation, or judge reproduction path.
3. Gemini is optional for the earlier LLM-assisted labelling workflow (`--use-llm`).

---

## Setup

```bash
cd hiver-sde-assignment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # optional: only needed if making live Gemini calls
```

---

## Quick Start — Run the Pipeline

```bash
# 1. Run a single customer message through the agent
python -m src.agent --message "Where is my parcel?"

# 2. Evaluate the agent across the 200-example sealed golden set
python -m src.agent --batch evaluation/golden_set.csv

# 3. Run the LLM-as-a-judge evaluation harness (88.0% human agreement)
python -m src.llm_judge

# 4. Train and evaluate the baseline classifier (60.0% golden accuracy)
python -m src.train_baseline

# 5. Evaluate the real majority-class baseline (22.0% train-fitted)
python -m src.majority_baseline

# 6. Run all unit and integration tests (84 passed in < 4 seconds)
pytest -q
```

---

## Key Measured Results

| Deliverable | Metric | Score / Status | Evidence |
|---|---|---|---|
| **Golden Evaluation Set** | Size & Labels | **200 / 200 hand-labelled** | `evaluation/golden_set.csv`, all 10 intents |
| **Train-Fitted Majority** | Golden Accuracy | **22.0%** (`delivery_order_status`) | `outputs/majority_baseline_report.md` |
| **Golden-Oracle Majority** | Golden Accuracy | **45.0%** (`other` - reference only) | `outputs/majority_baseline_report.md` |
| **TF-IDF + LogReg Baseline**| Golden Accuracy | **60.0%** (Macro F1 = 0.2298) | `outputs/baseline_report.md` |
| **Support Agent Routing** | Golden Set Batch | **6.0% Auto-Handled / 94.0% Escalated** | `outputs/agent_evaluation.json` |
| **LLM-as-a-Judge** | Human Agreement | **88.0%** (Pearson $r = 0.927$, MAE = 0.22) | `outputs/judge_report.md` |
| **Test Suite** | Automated Tests | **84 passed in 3.5s** | `tests/` (11 test modules) |

---

## AmazonHelp Dataset Slice

Reconstructed from the available 59,565-row slice of `twcs.csv`:
- **8,615** tweets in AmazonHelp threads
- **1,632** reconstructed multi-turn conversations
- **3,603** customer-to-brand reply pairs
- **2,139** training pairs (1,115 conversations)
- **579** dev pairs (278 conversations)
- **200** sealed golden evaluation examples (zero conversation/tweet overlap with train)

---

## Repository Layout

```
hiver-sde-assignment/
├── README.md                                      # Project overview & quickstart
├── REPRODUCE.md                                   # End-to-end 9-step reproduction guide (< 4 min)
├── decision_log.md                                # 15 documented engineering & data decisions
├── requirements.txt                               # Pinned dependencies
├── config.yaml                                    # Pipeline configuration & thresholds
├── .env.example                                   # Placeholder for optional Gemini API key
│
├── src/
│   ├── agent.py                                   # SupportAgent class (inference & batch eval)
│   ├── retrieval.py                               # TF-IDF k-NN historical reply retriever
│   ├── escalation.py                              # Deterministic safety routing policy
│   ├── llm_judge.py                               # 3-dimension rubric judge & agreement harness
│   ├── train_baseline.py                          # TF-IDF + LogReg classifier
│   ├── majority_baseline.py                       # Train-fitted vs golden-oracle majority
│   ├── conversation_builder.py                    # Thread reconstruction
│   ├── split.py                                   # Leakage-safe conversation partitioning
│   ├── taxonomy.py                                # Locked 10-intent taxonomy definitions
│   ├── intents.py                                 # Intent constants (9 domain + other)
│   ├── generate_training_labels.py                # Stage 1 rule-based labeling
│   ├── generate_stage2_labels.py                  # Stage 2 LLM distillation (cache replay)
│   ├── validate_golden_set.py                     # Golden set validator
│   └── preprocessing.py                           # Reversible text cleaning
│
├── evaluation/
│   ├── golden_set.csv                             # 200/200 sealed human labels
│   ├── sampling_report.json                       # Stratified sampling breakdown
│   ├── labeling_guidelines.md                     # Annotation rubric
│   └── README.md                                  # Evaluation documentation
│
├── outputs/
│   ├── report.md                                  # Main 6-page submission report
│   ├── baseline_report.md                         # Classifier metrics & confusion matrix
│   ├── majority_baseline_report.md                # Majority baseline comparison
│   ├── agent_evaluation.json                      # Operational batch metrics
│   ├── judge_report.md                            # Judge vs human agreement report
│   └── judge_evaluation.json                      # Machine-readable judge metrics
│
├── models/
│   └── baseline_tfidf_logreg.joblib               # Fitted sklearn pipeline
│
└── tests/                                         # 84 automated tests across 11 modules
```

---

## Citations

- Dataset: Thought Vector, *Customer Support on Twitter*, Kaggle. https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
- Libraries: scikit-learn, pandas, NumPy, matplotlib, PyYAML, python-dotenv, pytest.
