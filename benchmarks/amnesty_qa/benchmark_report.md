# TrustifAI Benchmark Report

**Generated on:** 2026-06-21 17:59:51

## Dataset Details

This benchmark is conducted using the [vibrantlabsai/amnesty_qa dataset](https://huggingface.co/datasets/vibrantlabsai/amnesty_qa) from huggingface which contains question-answer pairs related to human rights and Amnesty International reports. The dataset includes:

- 20 ground-truth answers sourced directly from verified Amnesty International documents
- 20 LLM-generated answers produced by querying language models
- Total of 40 QA pairs evaluated

The ground truth answers serve as a reliable baseline, while the LLM answers help assess TrustifAI's ability to detect potential hallucinations and inaccuracies in model-generated content.


## What Is Being Evaluated?

TrustifAI assigns a **trust score between 0 and 1** to each answer.

- **High score** → 🟩 Reliable Answer
- **Moderate Score** → 🟨 Acceptable answer (with caution)
- **Low score** → 🟥 Unreliable (Likely Hallucinated) Answer

We evaluate TrustifAI on:
1. **LLM-generated answers**
2. **Ground-truth answers** (known to be correct)

**Expected behavior:** Ground-truth answers should consistently receive higher trust scores than LLM answers.


## Reliability Distribution Comparison

If TrustifAI works correctly, it should assign mostly **RELIABLE**/**ACCEPTABLE** labels to **Ground Truth**

**Results:**

| Type         |   ACCEPTABLE (WITH CAUTION) |   RELIABLE |   UNRELIABLE |
|:-------------|----------------------------:|-----------:|-------------:|
| Ground Truth |                           0 |         19 |            1 |
| LLM          |                           8 |         10 |            2 |

## Verdict

TrustifAI demonstrates **meaningful separation** between grounded and hallucinated answers. Ground-truth responses consistently receive higher trust scores, indicating:

- Effective hallucination detection
- Reasonable score calibration
- Practical usefulness in RAG evaluation pipelines
