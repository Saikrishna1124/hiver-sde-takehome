# Majority-Class Baseline Report

Two majority-class baselines are reported. Both predict a single constant
label for every example.

## Baseline A — Train-Fitted Majority (Standard Lower Bound)

- **Strategy**: `DummyClassifier(strategy='most_frequent')` fitted on `train_stage1.csv`
- **Majority class (from training data)**: `delivery_order_status`
- **Golden set**: 200 examples
- **Correct predictions**: 44
- **Accuracy**: **22.0%**

> This is the standard trivial baseline. It predicts the most common training
> label for every test example. Any useful classifier must outperform this.

## Baseline B — Golden-Oracle Majority (Upper-Bound Reference Only)

- **Strategy**: most frequent intent in the golden set itself
- **Majority class (from golden set)**: `other`
- **Correct predictions**: 90 / 200
- **Accuracy**: **45.0%**

> **Not a blind baseline**: this uses knowledge of the test distribution.
> It is the highest accuracy any constant predictor can achieve on this
> golden set. Reported for reference only.

## Summary

| Baseline | Majority Class | Accuracy | Note |
| :--- | :--- | :--- | :--- |
| Train-fitted majority | `delivery_order_status` | **22.0%** | Standard lower bound |
| Golden-oracle majority | `other` | **45.0%** | Upper bound reference |
| TF-IDF + LogReg (from `baseline_report.md`) | — | **60.0%** | Main baseline |

The TF-IDF + LogReg model substantially outperforms both trivial baselines
(+38.0% over train-fitted, +15.0% over oracle).
However, its Macro F1 of 0.2298 reveals that 6 of 10 intents are effectively
invisible to the model (see `baseline_report.md` for the confusion matrix).