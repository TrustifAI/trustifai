"""
example_script.py
=================
"""

# ── A: Plain Python script ────────────────────────────────────────────────────
import asyncio
from langchain_core.documents import Document

from trustifai.async_pipeline import AsyncTrustifai, evaluate_dataset
from trustifai import MetricContext


# ── Shared setup ──────────────────────────────────────────────────────────────

engine = AsyncTrustifai("config_file.yaml")

# Example dataset — replace with your actual data source (CSV, DB, etc.)
RAW_DATASET = [
    {
        "query": "What is the capital of France?",
        "answer": "The capital of France is Paris.",
        "docs": ["Paris is the capital and most populous city of France."],
    },
    {
        "query": "Who wrote Romeo and Juliet?",
        "answer": "Romeo and Juliet was written by William Shakespeare.",
        "docs": ["Romeo and Juliet is a tragedy written by William Shakespeare."],
    },
    {
        "query": "What is photosynthesis?",
        "answer": "Photosynthesis converts sunlight into glucose using CO2 and water.",
        "docs": [
            "Photosynthesis is a process used by plants to convert light energy into chemical energy.",
            "The process uses carbon dioxide and water, releasing oxygen as a by-product.",
        ],
    },
    # Add as many rows as you need — the semaphore handles rate limiting.
]


def build_contexts(dataset: list[dict]) -> list[MetricContext]:
    return [
        MetricContext(
            query=row["query"],
            answer=row["answer"],
            documents=[
                Document(page_content=text, metadata={"source": f"doc_{i}"})
                for i, text in enumerate(row["docs"])
            ],
        )
        for row in dataset
    ]


# ── A: Single async evaluation ────────────────────────────────────────────────

async def run_single():
    context = build_contexts(RAW_DATASET)[0]
    result = await engine.get_trust_score(context)
    print(f"\n[Single] Trust Score: {result['score']}  Label: {result['label']}")
    return result


# ── B: Full batch evaluation ──────────────────────────────────────────────────

async def run_batch():
    contexts = build_contexts(RAW_DATASET)

    batch = await evaluate_dataset(
        engine,
        contexts,
        concurrency=5,        # max simultaneous LLM calls
        show_progress=True,   # tqdm bar in terminal or Jupyter
    )

    print("\n" + batch.summary())

    # Access individual results in original order
    for i, result in enumerate(batch.results):
        q = RAW_DATASET[i]["query"]
        print(f"  [{i}] Q: {q!r}  →  score={result['score']}  label={result['label']}")

    # Handle any failures gracefully
    if batch.failed:
        print(f"\n  ⚠ {len(batch.failed)} evaluation(s) failed:")
        for f in batch.failed:
            print(f"    index={f['index']}  error={f['error']}")

    return batch


# ── Entry point (script mode) ─────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(run_single())
    asyncio.run(run_batch())


# ─────────────────────────────────────────────────────────────────────────────
# JUPYTER / IPYTHON USAGE
# ─────────────────────────────────────────────────────────────────────────────
# Jupyter already runs an event loop, so `asyncio.run()` will raise.
# Use nest_asyncio to patch it, then just `await` directly:
#
#   pip install nest_asyncio
#
#   import nest_asyncio
#   nest_asyncio.apply()           # patch the running loop — call once per kernel
#
#   engine = AsyncTrustifai("config_file.yaml")
#   contexts = build_contexts(RAW_DATASET)
#
#   # Single
#   result = await engine.get_trust_score(contexts[0])
#
#   # Batch
#   batch = await evaluate_dataset(engine, contexts, concurrency=5)
#   print(batch.summary())
#
#   # Plug into pandas
#   import pandas as pd
#   df = pd.DataFrame(batch.results)
#   df[["score", "label"]].describe()
# ─────────────────────────────────────────────────────────────────────────────