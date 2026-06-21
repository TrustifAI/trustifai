# TrustifAI Benchmark Report

**Generated on:** 2026-06-21 15:25:36

## Dataset Details

This benchmark uses the [FactCHD benchmark](https://github.com/zjunlp/FactCHD) for fact-conflicting hallucination detection in LLMs. It contains a large collection of query-answer instances paired with fact-based evidence chains and labels indicating whether the answer is factual or non-factual. The benchmark spans multiple domains, including health, medicine, climate, and science, and covers several reasoning patterns such as vanilla facts, multi-hop reasoning, comparison, and set-operation questions.

Dataset used in this testing includes a sample of 50 QA pairs evaluated.

The goal is to evaluate whether a system can detect hallucinations by grounding its judgment in supporting evidence.


## What Is Being Evaluated?

TrustifAI assigns a **trust score between 0 and 1** to each answer.

- **High score** → Reliable Answer
- **Moderate Score** → Acceptable answer (with caution)
- **Low score** → Unreliable (Likely Hallucinated) Answer

We evaluate TrustifAI on: **LLM-generated answers against Ground-truth answers**

**Expected behavior:** TrustifAI should assign higher trust scores answers which have the label 'Factual' in the Ground-truth.


## Hallucination Detection (Binary Classification)

Labels are mapped as:
- **Trustworthy (1)** → RELIABLE, ACCEPTABLE (WITH CAUTION)
- **Untrustworthy (0)** → UNRELIABLE

**Interpretation:**
- ROC-AUC → separability between trustworthy vs untrustworthy answers
- PR-AUC → robustness under class imbalance

**Results:**
```text
ROC-AUC  : 0.864
PR-AUC   : 0.780
```

## Confusion Matrix

**Results:**

|                   |   Predicted Reliable |   Predicted Unreliable |
|:------------------|---------------------:|-----------------------:|
| Actual Reliable   |                   20 |                      1 |
| Actual Unreliable |                    6 |                     23 |

## Accuracy

**Results:** 86.00%


## Verdict

TrustifAI demonstrates **meaningful separation** between grounded and hallucinated answers. Ground-truth responses consistently receive higher trust scores, indicating:

- Effective hallucination detection
- Reasonable score calibration
- Practical usefulness in RAG evaluation pipelines
