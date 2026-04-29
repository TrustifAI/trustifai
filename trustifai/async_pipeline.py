"""
async_pipeline.py
==================
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from tqdm.asyncio import tqdm_asyncio as async_tqdm

from trustifai import MetricContext, Trustifai

logger = logging.getLogger(__name__)


# ── Rate Limiter (Token Bucket) ───────────────────────────────────────────

class RateLimiter:
    """
    Async token-bucket rate limiter.

    Proactively spaces out requests BEFORE they hit the API, to prevent 429 trigger.
    Works alongside the concurrency semaphore:
    semaphore caps *how many* run at once; this caps *how fast* they start.

    Parameters
    ----------
    requests_per_minute   Target RPM. Set this to ~80% of your actual quota
                          to leave headroom for retries.
    burst                 Max tokens that can accumulate while idle. Default 1
                          (no burst) — safe for free-tier keys. Raise for
                          enterprise keys that allow short bursts.

    Usage
    -----
    # 10 RPM — typical for free-tier LLMs
    rate_limiter = RateLimiter(requests_per_minute=10)

    async with rate_limiter:
        result = await engine.get_trust_score(ctx)
    """

    def __init__(self, requests_per_minute: float, burst: int = 1) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be > 0")
        self._interval = 60.0 / requests_per_minute   # seconds per token
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "RateLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass

    async def acquire(self) -> None:
        """Block until a token is available, then consume it."""
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                # Refill tokens based on time passed
                self._tokens = min(
                    self._burst,
                    self._tokens + elapsed / self._interval,
                )
                self._last_refill = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

                # Calculate exact wait needed for next token
                wait = (1.0 - self._tokens) * self._interval

            # Sleep outside the lock so other coroutines can check
            await asyncio.sleep(wait)


# ── Retry with Exponential Backoff ───────────────────────────────────────

# Exceptions that signal a rate limit hit — extend if your LLM SDK
# raises something more specific (e.g. openai.RateLimitError).
_RATE_LIMIT_SIGNALS = ("429", "rate limit", "rate_limit", "too many requests", "quota")


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(signal in msg for signal in _RATE_LIMIT_SIGNALS)


async def _with_retry(
    coro_fn,
    *args,
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 120.0,
    jitter: bool = True,
    **kwargs,
) -> Any:
    """
    Call `coro_fn(*args, **kwargs)` with exponential backoff on rate-limit
    errors. Non-rate-limit exceptions are re-raised immediately (fail fast).

    Backoff schedule (base_delay=2, no jitter):
        Attempt 1 → fail → wait 2s
        Attempt 2 → fail → wait 4s
        Attempt 3 → fail → wait 8s
        Attempt 4 → fail → wait 16s
        Attempt 5 → fail → raise

    With jitter=True (default), each wait is randomised ±25% so a burst of
    rate-limited calls doesn't all retry at the same instant (thundering herd).
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as exc:
            if not _is_rate_limit_error(exc):
                raise  # not a rate limit — bubble up immediately

            last_exc = exc
            if attempt == max_retries - 1:
                break  # exhausted retries

            delay = min(base_delay * (2 ** attempt), max_delay)
            if jitter:
                delay *= random.uniform(0.75, 1.25)

            logger.warning(
                "Rate limit hit (attempt %d/%d). Retrying in %.1fs. Error: %s",
                attempt + 1, max_retries, delay, exc,
            )
            await asyncio.sleep(delay)

    raise RuntimeError(
        f"Rate limit exceeded after {max_retries} retries."
    ) from last_exc


# ── Native Async Wrapper ───────────────────────────────────────────────────

class AsyncTrustifai:
    """
    Native async wrapper around the synchronous Trustifai engine.

    Offloads each call to a thread pool via `asyncio.to_thread`.

    Thread-safety
    -------------
    Trustifai.get_trust_score mutates instance state (metrics initialisation,
    embeddings). A single shared engine across threads would race under
    concurrent evaluate_dataset calls. We use threading.local() so each worker
    thread gets its own Trustifai instance, constructed lazily on first use.
    Config loading is cheap (YAML read), so this adds negligible overhead while
    giving full thread isolation with no serialisation penalty.

    Usage
    -----
    engine = AsyncTrustifai("config_file.yaml")

    # FastAPI route
    result = await engine.get_trust_score(context)

    # Batch
    batch  = await evaluate_dataset(engine, contexts, concurrency=8)
    """

    def __init__(self, config_path: str) -> None:
        self._config_path = config_path
        self._local = threading.local()

    def _get_engine(self) -> Trustifai:
        """Return this thread's own Trustifai instance, creating if needed."""
        if not hasattr(self._local, "engine"):
            self._local.engine = Trustifai(self._config_path)
        return self._local.engine

    # ── Public async API ──────────────────────────────────────────────────────

    async def get_trust_score(self, context: MetricContext) -> dict[str, Any]:
        """
        Evaluate a single RAG context. Non-blocking and thread-safe — each
        worker thread uses its own Trustifai instance via threading.local().
        """
        return await asyncio.to_thread(
            lambda: self._get_engine().get_trust_score(context)
        )

    async def build_reasoning_graph(self, result: dict[str, Any]) -> Any:
        """Async-safe reasoning graph builder."""
        return await asyncio.to_thread(
            lambda: self._get_engine().build_reasoning_graph(result)
        )

    async def visualize(self, graph: Any, graph_type: str = "pyvis") -> Any:
        """Async-safe visualizer."""
        return await asyncio.to_thread(
            lambda: self._get_engine().visualize(graph, graph_type)
        )

    # ── sync engine for callers already in a thread ─────
    @property
    def sync(self) -> Trustifai:
        """Returns this thread's Trustifai engine (creates one if needed)."""
        return self._get_engine()


# ── Batch Result Container ─────────────────────────────────────────────────

@dataclass
class BatchResult:
    """
    Holds all outputs from `evaluate_dataset`.

    Attributes
    ----------
    results         Successful TrustScore dicts, in original dataset order.
    failed          List of {index, context, error} for any evaluation that
                    raised an exception — failures are isolated and never crash
                    the whole batch.
    total           Number of input contexts.
    succeeded       Number that completed without error.
    elapsed_seconds Wall-clock time for the full batch run.
    """

    results: List[dict[str, Any]] = field(default_factory=list)
    failed: List[dict[str, Any]] = field(default_factory=list)
    total: int = 0
    succeeded: int = 0
    elapsed_seconds: float = 0.0

    # ── Aggregate helpers ─────────────────────────────────────────────────────

    @property
    def mean_score(self) -> float:
        """Mean trust score across all successful evaluations."""
        scores = [r["score"] for r in self.results if "score" in r]
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    @property
    def score_distribution(self) -> dict[str, float]:
        """Min / median / max trust scores."""
        scores = sorted(r["score"] for r in self.results if "score" in r)
        if not scores:
            return {}
        mid = len(scores) // 2
        median = (scores[mid - 1] + scores[mid]) / 2 if len(scores) % 2 == 0 else scores[mid]
        return {"min": scores[0], "median": round(median, 4), "max": scores[-1]}

    @property
    def label_distribution(self) -> dict[str, int]:
        """Count of each trust label (RELIABLE / ACCEPTABLE / UNRELIABLE)."""
        labels = [r.get("label", "UNKNOWN") for r in self.results]
        return {label: labels.count(label) for label in sorted(set(labels))}

    @property
    def failure_rate(self) -> float:
        return round(len(self.failed) / self.total, 4) if self.total else 0.0

    def summary(self) -> str:
        lines = [
            f"{'─' * 48}",
            f"  Batch Evaluation Summary",
            f"{'─' * 48}",
            f"  Total          : {self.total}",
            f"  Succeeded      : {self.succeeded}",
            f"  Failed         : {len(self.failed)} ({self.failure_rate:.1%})",
            f"  Elapsed        : {self.elapsed_seconds:.1f}s",
            f"  Throughput     : {self.succeeded / max(self.elapsed_seconds, 0.001):.1f} eval/s",
            f"{'─' * 48}",
            f"  Mean Score     : {self.mean_score}",
            f"  Score Range    : {self.score_distribution}",
            f"  Labels         : {self.label_distribution}",
            f"{'─' * 48}",
        ]
        return "\n".join(lines)


# ── Batch Evaluator ────────────────────────────────────────────────────────

async def evaluate_dataset(
    engine: AsyncTrustifai,
    contexts: List[MetricContext],
    *,
    concurrency: int = 5,
    show_progress: bool = True,
    fail_fast: bool = False,
    requests_per_minute: Optional[float] = None,
    rate_limit_burst: int = 1,
    max_retries: int = 5,
    retry_base_delay: float = 2.0,
    retry_max_delay: float = 120.0,
) -> BatchResult:
    """
    Evaluate a list of MetricContexts concurrently.

    Parameters
    ----------
    engine                AsyncTrustifai instance.
    contexts              Your dataset — one MetricContext per row.
    concurrency           Max simultaneous evaluations.
    show_progress         Show a tqdm progress bar.
    fail_fast             Cancel remaining tasks on first error.

    Rate-limiting (for tight-quota LLMs)
    -------------
    requests_per_minute   Set this to stay within your LLM's RPM quota.
                          Recommended: ~80% of your actual limit to leave
                          headroom for retries.
                            - Free-tier Gemini/Mistral : 10–15 RPM → set 8–12
                            - OpenAI Tier-1            : 500 RPM  → set 400
                            - Local model              : None (no cap)
                          If None (default), no rate limiting is applied.
    rate_limit_burst      Token-bucket burst size. Keep at 1 for free-tier
                          keys. Raise for APIs that allow short bursts.
    max_retries           Max retry attempts on a 429 / rate-limit error.
    retry_base_delay      Initial backoff in seconds (doubles each attempt).
    retry_max_delay       Cap on backoff delay in seconds.

    Returns
    -------
    BatchResult with .results (ordered), .failed, and aggregate stats.

    Examples
    --------
    # Standard — no rate limiting needed
    batch = await evaluate_dataset(engine, contexts, concurrency=10)

    # Free-tier key — 10 RPM, concurrency 1, automatic retry
    batch = await evaluate_dataset(
        engine, contexts,
        concurrency=1,
        requests_per_minute=10,
    )

    # Low-quota enterprise key — 60 RPM, small burst allowed
    batch = await evaluate_dataset(
        engine, contexts,
        concurrency=3,
        requests_per_minute=50,
        rate_limit_burst=3,
        max_retries=6,
        retry_max_delay=60.0,
    )
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")

    if requests_per_minute is not None and requests_per_minute <= 0:
        raise ValueError(f"requests_per_minute must be > 0, got {requests_per_minute}")

    semaphore = asyncio.Semaphore(concurrency)
    rate_limiter = (
        RateLimiter(requests_per_minute, burst=rate_limit_burst)
        if requests_per_minute is not None
        else None
    )

    batch = BatchResult(total=len(contexts))
    start = time.perf_counter()

    cancel_event = asyncio.Event()
    running_tasks: list[asyncio.Task] = []

    async def _eval_one(index: int, ctx: MetricContext) -> None:
        if cancel_event.is_set():
            return

        async with semaphore:
            if cancel_event.is_set():
                return

            if rate_limiter is not None:
                await rate_limiter.acquire()

            if cancel_event.is_set():
                return

            try:
                result = await _with_retry(
                    engine.get_trust_score,
                    ctx,
                    max_retries=max_retries,
                    base_delay=retry_base_delay,
                    max_delay=retry_max_delay,
                )
                result["_index"] = index
                batch.results.append(result)
                batch.succeeded += 1

            except Exception as exc:
                batch.failed.append(
                    {"index": index, "context": ctx, "error": str(exc)}
                )
                if fail_fast:
                    cancel_event.set()
                    # Cancel all sibling tasks that are still waiting
                    for t in running_tasks:
                        if not t.done():
                            t.cancel()

    tasks = [asyncio.ensure_future(_eval_one(i, ctx)) for i, ctx in enumerate(contexts)]
    running_tasks.extend(tasks)

    if show_progress:
        await async_tqdm.gather(*tasks, desc="Evaluating", unit="sample")
    else:
        await asyncio.gather(*tasks, return_exceptions=True)  # absorb CancelledErrors

    # Restore original dataset order (gather is unordered due to semaphore)
    batch.results.sort(key=lambda r: r.pop("_index", 0))
    batch.elapsed_seconds = round(time.perf_counter() - start, 3)
    return batch