import numpy as np
from unittest.mock import MagicMock, AsyncMock
from trustifai.metrics import (
    EvidenceCoverageMetric,
    SemanticDriftMetric,
    EpistemicConsistencyMetric,
    SourceDiversityMetric,
    ConfidenceMetric,
)


def test_semantic_drift(basic_context, mock_service):
    # Setup embeddings to be identical
    basic_context.answer_embeddings = np.array([0.1, 0.2, 0.3])
    basic_context.document_embeddings = [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.1, 0.2, 0.3]),
    ]
    mock_service.embedding_call_batch.return_value = {
        "embedding": [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
        "cost": 0.0,
    }

    # Patch config with real thresholds
    config = MagicMock()
    config.thresholds = MagicMock()
    config.thresholds.STRONG_ALIGNMENT = 0.9
    config.thresholds.PARTIAL_ALIGNMENT = 0.5

    metric = SemanticDriftMetric(mock_service, config)
    result = metric.calculate(basic_context)

    assert result.score > 0.99
    assert result.label == "Strong Alignment"


def test_consistency(basic_context, mock_service):
    config = MagicMock()
    config.k_samples = 2
    config.thresholds = MagicMock()
    config.thresholds.STABLE_CONSISTENCY = 0.9
    config.thresholds.FRAGILE_CONSISTENCY = 0.7

    # 1. Mock the samples generation
    # mock_service.llm_call_async.return_value = {"response": basic_context.answer, "logprobs": [], "cost": 0.0}
    mock_service.llm_call_async = AsyncMock(
        return_value={"response": basic_context.answer, "logprobs": [], "cost": 0.0}
    )

    # 2. Mock embedding for the MAIN answer (still uses single call)
    mock_service.embedding_call.return_value = {
        "embedding": [0.1, 0.2, 0.3],
        "cost": 0.0,
    }

    # 3. Mock embedding for the SAMPLES (uses BATCH call)
    mock_service.embedding_call_batch.return_value = {
        "embedding": [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
        "cost": 0.0,
    }

    metric = EpistemicConsistencyMetric(mock_service, config)
    result = metric.calculate(basic_context)

    assert result.score > 0.99
    assert "Stable" in result.label


def test_source_diversity(basic_context, mock_service):
    # Context has 2 docs with different sources (wiki, geo_db)
    config = MagicMock()
    config.thresholds = MagicMock()
    config.thresholds.HIGH_DIVERSITY = 0.7
    config.thresholds.MODERATE_DIVERSITY = 0.4
    metric = SourceDiversityMetric(mock_service, config)
    result = metric.calculate(basic_context)

    assert result.details["unique_sources"] == 2
    assert result.score > 0.0


def test_evidence_coverage_llm(basic_context, mock_service):
    config = MagicMock()
    config.metrics = [MagicMock(type="evidence_coverage", params={"strategy": "llm"})]
    config.thresholds = MagicMock()
    config.thresholds.STRONG_GROUNDING = 0.9
    config.thresholds.PARTIAL_GROUNDING = 0.7

    # Mock LLM verification response for BATCH call
    # The 'response' key must hold a LIST of strings
    mock_service.llm_call.return_value = {
        "response": '{"spans": [{"index": 0, "supported": true, "answer": "span"}]}',
        "cost": 0.0,
    }

    metric = EvidenceCoverageMetric(mock_service, config)
    result = metric.calculate(basic_context)

    assert result.score == 1.0

def test_evidence_coverage_malformed_json(basic_context, mock_service):
    """Test that the metric handles invalid JSON from the LLM gracefully."""
    config = MagicMock()
    config.metrics = [MagicMock(type="evidence_coverage", params={"strategy": "llm"})]
    config.reranker = None

    config.thresholds = MagicMock()
    config.thresholds.STRONG_GROUNDING = 0.85
    config.thresholds.PARTIAL_GROUNDING = 0.60

    # Mock LLM returning malformed JSON in a BATCH response list
    mock_service.llm_call_batch.return_value = {"response": ["I cannot verify this."]}

    metric = EvidenceCoverageMetric(mock_service, config)
    result = metric.calculate(basic_context)

    # Should result in 0 score and recorded failed checks
    assert result.score == 0.0
    assert result.details["failed_checks"] > 0
    assert "Parse error" in str(result.details) or "unsupported_sentences" in str(
        result.details
    )


def test_confidence_metric():
    evaluator = MagicMock()
    evaluator.evaluate_confidence.return_value = ("High", "Explanation")

    # Perfect confidence (logprob 0 = prob 1)
    logprobs = [0.0, 0.0]
    result = ConfidenceMetric.calculate(logprobs, evaluator)

    # Use np.isclose for floating point comparison
    assert np.isclose(result["score"], 1.0)
    assert result["label"] == "High"

    # Empty logprobs
    result = ConfidenceMetric.calculate([], evaluator)
    assert result["label"] == "N/A"


# def test_evidence_coverage_reranker_strategy(basic_context, mock_service):
#     """Test the Reranker strategy specifically checking trust thresholds."""
#     config = MagicMock()
#     # Force strategy to reranker
#     config.metrics = [MagicMock(type="evidence_coverage", params={"strategy": "reranker"})]
#     config.reranker = MagicMock(type="cohere")

#     # Mock Thresholds
#     config.thresholds = MagicMock()
#     config.thresholds.STRONG_GROUNDING = 0.85
#     config.thresholds.PARTIAL_GROUNDING = 0.6

#     # Mock Reranker Response
#     # Scenario: 3 spans.
#     # 1. High score (>0.85) -> Trusted
#     # 2. Mid score (>0.49) -> Low Risk
#     # 3. Low score (<0.49) -> High Risk
#     mock_service.reranker_call.return_value = [
#         {"index": 0, "relevance_score": 0.90},
#         {"index": 1, "relevance_score": 0.55},
#         {"index": 2, "relevance_score": 0.10}
#     ]

#     # Override context answer to ensure we have 3 sentences to match our mock
#     basic_context.answer = "Sentence one. Sentence two. Sentence three."

#     metric = EvidenceCoverageMetric(mock_service, config)
#     result = metric.calculate(basic_context)

#     assert result.details['strategy'] == "Reranker"

#     # Check if the logic inside RerankerBasedEvidenceStrategy correctly classified them
#     detailed_scores = result.details['detailed_scores']
#     assert detailed_scores[0]['label'] == "Trusted"
#     assert detailed_scores[1]['label'] == "Low Risk"
#     assert detailed_scores[2]['label'] == "High Risk"

#     # Mean score check: (0.90 + 0.55 + 0.10) / 3 = 0.516
#     assert abs(result.score - 0.516) < 0.01
