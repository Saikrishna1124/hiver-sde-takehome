# Training Label Strategy: Stage 1 (Rule-Confidence Filtering)

## Overview
The training set contains heuristic regex proposals that exhibit noise when
multiple rules fire or when no rules fire. To create a high-quality baseline
dataset without manual annotation of 2,139 rows, Stage 1 uses a hybrid
rule-confidence filtering strategy.

## Measured Rule Matches on Train Set (2,139 rows)
- **Single Match (High Confidence)**: 824 rows (38.5%)
- **Multi Match (Ambiguous / Conflicting)**: 163 rows (7.6%)
- **Zero Match (Default Fallback 'other')**: 1152 rows (53.9%)

## Stage 1 Actions
1. **Drop Multi-Match**: All multi-match rows are completely excluded from Stage 1 to prevent contradictory supervision signals.
2. **Retain Single-Match**: All single-match rows are retained with their unambiguous rule match as ground truth.
3. **Controlled 'Other' Sampling**: From the zero-match rows, a controlled sample of 400 examples is drawn with seed 42 to balance the class distribution.
4. **Final Stage 1 Size**: 1224 examples.

## Stage 1 Label Distribution
| Intent | Count | Proportion |
| :--- | :--- | :--- |
| `delivery_order_status` | 470 | 38.4% |
| `other` | 400 | 32.7% |
| `service_complaint` | 113 | 9.2% |
| `device_digital_content` | 73 | 6.0% |
| `refund_return` | 59 | 4.8% |
| `account_access` | 33 | 2.7% |
| `seller_marketplace` | 26 | 2.1% |
| `payment_billing` | 23 | 1.9% |
| `item_quality_issue` | 20 | 1.6% |
| `prime_membership` | 7 | 0.6% |

## Stage 2 Roadmap (LLM Distillation)
When a Gemini API key is provided, the ambiguous and sampled zero-match rows will be
labelled with Gemini 2.5/Flash and locally cached, training a Stage 2 model to compare
against this Stage 1 baseline on the sealed 200-example golden set.