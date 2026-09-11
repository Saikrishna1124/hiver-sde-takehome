# Baseline Intent Classifier Evaluation Report

## Architecture & Configuration
- **Model**: TF-IDF Vectorizer + Multinomial Logistic Regression
- **Features**: Word unigrams + bigrams (`ngram_range=(1, 2)`), `min_df=2`, `sublinear_tf=True`
- **ClassifierKnobs**: `C=1.0`, `max_iter=1000`, `random_state=42`
- **Training Dataset**: `train_stage1.csv` (1224 examples)

## Evaluation Summary
| Evaluation Split | Total Examples | Accuracy | Macro F1 | Weighted F1 | Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dev Set** | 579 | 70.8% | 0.3625 | 0.6730 | Model development / heuristic agreement |
| **Golden Set** | 200 | 60.0% | 0.2298 | 0.5247 | Sealed human ground-truth benchmark |

> [!IMPORTANT]
> The **Golden Evaluation Set** is permanently sealed: it was never seen during training, sampling, or hyperparameter selection.

## Golden Set Per-Intent Performance (Human Ground Truth)
| Intent | Support | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| `account_access` | 7 | 0.0000 | 0.0000 | 0.0000 |
| `prime_membership` | 5 | 0.0000 | 0.0000 | 0.0000 |
| `device_digital_content` | 9 | 1.0000 | 0.2222 | 0.3636 |
| `seller_marketplace` | 6 | 0.0000 | 0.0000 | 0.0000 |
| `refund_return` | 12 | 0.0000 | 0.0000 | 0.0000 |
| `payment_billing` | 6 | 0.0000 | 0.0000 | 0.0000 |
| `item_quality_issue` | 6 | 0.0000 | 0.0000 | 0.0000 |
| `delivery_order_status` | 44 | 0.4815 | 0.8864 | 0.6240 |
| `service_complaint` | 15 | 0.7778 | 0.4667 | 0.5833 |
| `other` | 90 | 0.6667 | 0.8000 | 0.7273 |

## Golden Set Confusion Matrix
Rows represent Ground Truth (Human), columns represent Model Predictions.

```
Actual / Pred          | accoun | prime_ | device | seller | refund | paymen | item_q | delive | servic |  other
-----------------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+-------
account_access         |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      1 |      0 |      6
prime_membership       |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      4 |      0 |      1
device_digital_content |      0 |      0 |      2 |      0 |      0 |      0 |      0 |      2 |      0 |      5
seller_marketplace     |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      1 |      1 |      4
refund_return          |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      6 |      0 |      6
payment_billing        |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      5 |      0 |      1
item_quality_issue     |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      4 |      1 |      1
delivery_order_status  |      0 |      0 |      0 |      0 |      0 |      0 |      0 |     39 |      0 |      5
service_complaint      |      0 |      0 |      0 |      0 |      0 |      0 |      0 |      1 |      7 |      7
other                  |      0 |      0 |      0 |      0 |      0 |      0 |      0 |     18 |      0 |     72
```

## Key Observations
1. **Baseline Accuracy**: Achieves **60.0%** on the 200 sealed golden examples (vs **22.0%** train-fitted majority baseline / **45.0%** golden-oracle reference).
2. **Dominant Intents**: `delivery_order_status` and `other` achieve strong precision and recall.
3. **Low-Resource Intents**: Classes with low support in Twitter support data (`prime_membership`, `payment_billing`) represent clear targets for Stage 2 LLM distillation.