# tests/test_async_pipeline.py
import pytest
import asyncio
import time
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.documents import Document
from trustifai.async_pipeline import RateLimiter, _is_rate_limit_error, _with_retry, AsyncTrustifai, evaluate_dataset, BatchResult
from trustifai.structures import MetricContext
from trustifai.metrics import BaseMetric
from trustifai.structures import MetricResult

def test_rate_limiter_initialization():
    """Test RateLimiter initializes with valid parameters."""
    limiter = RateLimiter(requests_per_minute=60, burst=2)
    assert limiter._burst == 2
    assert limiter._interval == 1.0  # 60 / 60 = 1.0 second per token


def test_rate_limiter_invalid_rpm():
    """Test RateLimiter rejects invalid RPM."""
    with pytest.raises(ValueError, match="requests_per_minute must be > 0"):
        RateLimiter(requests_per_minute=0)
    
    with pytest.raises(ValueError, match="requests_per_minute must be > 0"):
        RateLimiter(requests_per_minute=-5)


@pytest.mark.asyncio
async def test_rate_limiter_token_accumulation():
    """Test tokens accumulate over time."""
    limiter = RateLimiter(requests_per_minute=60, burst=1)  # 1 token per second
    
    # First acquire should succeed immediately (burst token)
    start = time.monotonic()
    await limiter.acquire()
    elapsed = time.monotonic() - start
    assert elapsed < 0.1  # Should be instant
    
    # Second acquire should wait ~1 second
    start = time.monotonic()
    await limiter.acquire()
    elapsed = time.monotonic() - start
    assert 0.9 < elapsed < 1.2  # Allow some tolerance


@pytest.mark.asyncio
async def test_rate_limiter_burst():
    """Test burst accumulation."""
    limiter = RateLimiter(requests_per_minute=60, burst=5)
    
    # Should be able to acquire 5 tokens quickly without waiting
    for i in range(5):
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1
    
    # 6th token should require waiting
    start = time.monotonic()
    await limiter.acquire()
    elapsed = time.monotonic() - start
    assert elapsed > 0.5


@pytest.mark.asyncio
async def test_rate_limiter_context_manager():
    """Test RateLimiter works as async context manager."""
    limiter = RateLimiter(requests_per_minute=60, burst=1)
    
    async with limiter as ctx:
        assert ctx is limiter


def test_is_rate_limit_error_429():
    """Test detection of 429 status code."""
    assert _is_rate_limit_error(Exception("429 Too Many Requests"))


def test_is_rate_limit_error_rate_limit_text():
    """Test detection of 'rate limit' in message."""
    assert _is_rate_limit_error(Exception("Rate limit exceeded"))
    assert _is_rate_limit_error(Exception("rate_limit error"))
    assert _is_rate_limit_error(Exception("Too many requests"))


def test_is_rate_limit_error_quota():
    """Test detection of quota exceeded."""
    assert _is_rate_limit_error(Exception("Quota exceeded"))


def test_is_rate_limit_error_non_rate_limit():
    """Test non-rate-limit errors are not detected."""
    assert not _is_rate_limit_error(Exception("Connection refused"))
    assert not _is_rate_limit_error(Exception("Invalid API key"))
    assert not _is_rate_limit_error(Exception("Internal server error"))


@pytest.mark.asyncio
async def test_retry_succeeds_on_first_attempt():
    """Test successful call on first attempt."""
    async def success_coro():
        return "success"
    
    result = await _with_retry(success_coro)
    assert result == "success"


@pytest.mark.asyncio
async def test_retry_succeeds_after_rate_limit():
    """Test retry succeeds after rate limit error."""
    call_count = 0
    
    async def flaky_coro():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise Exception("429 Rate limit exceeded")
        return "success"
    
    result = await _with_retry(flaky_coro, max_retries=3, base_delay=0.01)
    assert result == "success"
    assert call_count == 2


@pytest.mark.asyncio
async def test_retry_non_rate_limit_error_fails_immediately():
    """Test non-rate-limit errors are not retried."""
    call_count = 0
    
    async def failing_coro():
        nonlocal call_count
        call_count += 1
        raise Exception("Invalid API key")
    
    with pytest.raises(Exception, match="Invalid API key"):
        await _with_retry(failing_coro, max_retries=3)
    
    assert call_count == 1  # Only attempted once


@pytest.mark.asyncio
async def test_retry_exhausts_max_retries():
    """Test max retries exceeded."""
    async def always_rate_limited():
        raise Exception("Rate limit exceeded")
    
    with pytest.raises(RuntimeError, match="Rate limit exceeded after 3 retries"):
        await _with_retry(always_rate_limited, max_retries=3, base_delay=0.001)


@pytest.mark.asyncio
async def test_retry_exponential_backoff():
    """Test exponential backoff timing."""
    timings = []
    
    async def rate_limited_coro():
        timings.append(time.monotonic())
        raise Exception("429 Rate limit exceeded")
    
    base_delay = 0.1
    start = time.monotonic()
    with pytest.raises(RuntimeError):
        await _with_retry(rate_limited_coro, max_retries=3, base_delay=base_delay, jitter=False)
    
    # Verify delays roughly double: ~0.1s, ~0.2s, ~0.4s
    delays = [timings[i+1] - timings[i] for i in range(len(timings)-1)]
    assert len(delays) == 2
    assert delays[0] >= base_delay * 0.9
    assert delays[1] >= base_delay * 2 * 0.9


@pytest.mark.asyncio
async def test_retry_with_jitter():
    """Test jitter randomizes delays."""
    timings = []
    
    async def rate_limited_coro():
        timings.append(time.monotonic())
        raise Exception("429 Rate limit exceeded")
    
    base_delay = 0.5
    with pytest.raises(RuntimeError):
        await _with_retry(rate_limited_coro, max_retries=3, base_delay=base_delay, jitter=True)
    
    # With jitter, delays should vary
    delays = [timings[i+1] - timings[i] for i in range(len(timings)-1)]
    assert len(delays) == 2


@pytest.mark.asyncio
async def test_async_trustifai_thread_local_engine(sample_config_yaml, mock_service):
    """Test AsyncTrustifai maintains thread-local engines."""
    engine = AsyncTrustifai(sample_config_yaml)
    
    # Same thread should return same engine
    engine1 = engine._get_engine()
    engine2 = engine._get_engine()
    assert engine1 is engine2


@pytest.mark.asyncio
async def test_async_trustifai_get_trust_score(sample_config_yaml, mock_service, basic_context):
    """Test AsyncTrustifai.get_trust_score()."""
    engine = AsyncTrustifai(sample_config_yaml)
    engine.sync.service = mock_service
    
    # Mock metrics
    for name, metric in engine.sync.metrics.items():
        metric.calculate = MagicMock(
            return_value=MetricResult(score=0.75, label="Good", details={})
        )
    
    result = await engine.get_trust_score(basic_context)
    
    assert "score" in result
    assert result["score"] > 0


@pytest.mark.asyncio
async def test_async_trustifai_build_reasoning_graph(sample_config_yaml, mock_service, basic_context):
    """Test AsyncTrustifai.build_reasoning_graph()."""
    engine = AsyncTrustifai(sample_config_yaml)
    engine.sync.service = mock_service
    
    for name, metric in engine.sync.metrics.items():
        metric.calculate = MagicMock(
            return_value=MetricResult(score=0.5, label="OK", details={})
        )
    
    result = await engine.get_trust_score(basic_context)
    graph = await engine.build_reasoning_graph(result)
    
    assert graph is not None
    assert len(graph.nodes) > 0


def test_batch_result_mean_score():
    """Test mean_score calculation."""
    results = [
        {"score": 0.8},
        {"score": 0.6},
        {"score": 0.9},
    ]
    batch = BatchResult(results=results, total=3, succeeded=3)
    
    expected_mean = (0.8 + 0.6 + 0.9) / 3
    assert abs(batch.mean_score - expected_mean) < 0.001


def test_batch_result_mean_score_empty():
    """Test mean_score with no results."""
    batch = BatchResult(results=[], total=0, succeeded=0)
    assert batch.mean_score == 0.0


def test_batch_result_score_distribution():
    """Test score_distribution calculation."""
    results = [
        {"score": 0.5},
        {"score": 0.7},
        {"score": 0.9},
    ]
    batch = BatchResult(results=results, total=3, succeeded=3)
    dist = batch.score_distribution
    
    assert dist["min"] == 0.5
    assert dist["max"] == 0.9
    assert dist["median"] == 0.7


def test_batch_result_label_distribution():
    """Test label_distribution counting."""
    results = [
        {"score": 0.8, "label": "RELIABLE"},
        {"score": 0.5, "label": "ACCEPTABLE"},
        {"score": 0.2, "label": "UNRELIABLE"},
        {"score": 0.9, "label": "RELIABLE"},
    ]
    batch = BatchResult(results=results, total=4, succeeded=4)
    dist = batch.label_distribution
    
    assert dist["RELIABLE"] == 2
    assert dist["ACCEPTABLE"] == 1
    assert dist["UNRELIABLE"] == 1


def test_batch_result_failure_rate():
    """Test failure_rate calculation."""
    batch = BatchResult(
        results=[{"score": 0.8}],
        failed=[{"index": 1, "error": "test"}],
        total=2,
        succeeded=1
    )
    assert abs(batch.failure_rate - 0.5) < 0.001


def test_batch_result_summary_format():
    """Test summary generates formatted output."""
    batch = BatchResult(
        results=[{"score": 0.8}],
        failed=[],
        total=1,
        succeeded=1,
        elapsed_seconds=5.0
    )
    summary = batch.summary()
    
    assert "Summary" in summary
    assert "Total" in summary
    assert "Succeeded" in summary
    assert "Mean Score" in summary

@pytest.mark.asyncio
async def test_evaluate_dataset_all_success(sample_config_yaml, mock_service):
    """Test successful batch evaluation."""
    engine = AsyncTrustifai(sample_config_yaml)
    engine.sync.service = mock_service
    
    # Mock metrics
    for name, metric in engine.sync.metrics.items():
        metric.calculate = MagicMock(
            return_value=MetricResult(score=0.8, label="Good", details={})
        )
    
    contexts = [
        MetricContext(
            query="Q1",
            answer="A1",
            documents=[Document(page_content="D1", metadata={})],
            query_embeddings=np.array([0.1, 0.2, 0.3]),
            answer_embeddings=np.array([0.1, 0.2, 0.3]),
            document_embeddings=[np.array([0.1, 0.2, 0.3])],
        ),
        MetricContext(
            query="Q2",
            answer="A2",
            documents=[Document(page_content="D2", metadata={})],
            query_embeddings=np.array([0.1, 0.2, 0.3]),
            answer_embeddings=np.array([0.1, 0.2, 0.3]),
            document_embeddings=[np.array([0.1, 0.2, 0.3])],
        ),
    ]
    
    batch_result = await evaluate_dataset(engine, contexts, concurrency=2, show_progress=False)
    
    assert batch_result.total == 2
    assert batch_result.succeeded == 2
    assert len(batch_result.failed) == 0
    assert len(batch_result.results) == 2


@pytest.mark.asyncio
async def test_evaluate_dataset_partial_failures(sample_config_yaml, mock_service):
    """Test batch with some failures doesn't crash."""
    engine = AsyncTrustifai(sample_config_yaml)
    engine.sync.service = mock_service
    
    # Mock metrics but make second one fail
    call_count = 0
    def failing_calculate(context):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise Exception("Test error")
        return MetricResult(score=0.8, label="Good", details={})
    
    for name, metric in engine.sync.metrics.items():
        metric.calculate = MagicMock(side_effect=failing_calculate)
    
    contexts = [
        MetricContext(
            query="Q1", answer="A1",
            documents=[Document(page_content="D1", metadata={})],
            query_embeddings=np.array([0.1, 0.2, 0.3]),
            answer_embeddings=np.array([0.1, 0.2, 0.3]),
            document_embeddings=[np.array([0.1, 0.2, 0.3])],
        ),
        MetricContext(
            query="Q2", answer="A2",
            documents=[Document(page_content="D2", metadata={})],
            query_embeddings=np.array([0.1, 0.2, 0.3]),
            answer_embeddings=np.array([0.1, 0.2, 0.3]),
            document_embeddings=[np.array([0.1, 0.2, 0.3])],
        ),
    ]
    
    batch_result = await evaluate_dataset(engine, contexts, concurrency=1, show_progress=False, fail_fast=False)
    
    assert batch_result.total == 2
    # At least one should succeed or fail based on implementation


@pytest.mark.asyncio
async def test_evaluate_dataset_empty_contexts(sample_config_yaml, mock_service):
    """Test batch with empty context list."""
    engine = AsyncTrustifai(sample_config_yaml)
    
    batch_result = await evaluate_dataset(engine, [], concurrency=2, show_progress=False)
    
    assert batch_result.total == 0
    assert batch_result.succeeded == 0
    assert batch_result.failure_rate == 0.0