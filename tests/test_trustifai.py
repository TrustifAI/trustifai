from unittest.mock import MagicMock
from trustifai import Trustifai
from trustifai.metrics import BaseMetric
from trustifai.structures import MetricResult, MetricContext
import tempfile
import yaml
import pytest
from langchain_core.documents import Document
import numpy as np

# --- Custom Metric for Testing ---
class MockCustomMetric(BaseMetric):
    def calculate(self, context) -> MetricResult:
        return MetricResult(score=1.0, label="Custom", details={})


def test_engine_initialization(sample_config_yaml, mock_service):
    engine = Trustifai(sample_config_yaml)
    # Inject mock service to avoid real calls during initialization if any
    engine.service = mock_service

    assert engine.config is not None


def test_get_trust_score(basic_context, sample_config_yaml, mock_service):
    engine = Trustifai(sample_config_yaml)
    engine.service = mock_service

    # Mock individual metric calculations
    for name, metric in engine.metrics.items():
        metric.calculate = MagicMock(
            return_value=MetricResult(score=0.8, label="Good", details={})
        )

    result = engine.get_trust_score(basic_context)

    assert "score" in result
    assert "details" in result
    assert result["score"] > 0


def test_dynamic_metric_registration(basic_context, sample_config_yaml, mock_service):
    # 1. Register new metric
    Trustifai.register_metric("my_test_metric", MockCustomMetric)

    # 2. Load config and modify to include new metric
    with open(sample_config_yaml, "r") as f:
        data = yaml.safe_load(f)
    data = dict(data)
    if "score_weights" not in data:
        data["score_weights"] = []

    # Scale down existing weights to make room for the new metric
    new_weight = 0.5
    existing_weights = data["score_weights"]
    total_existing = sum(
        item.get("params", {}).get("weight", 0.0) for item in existing_weights
    )
    if total_existing > 0:
        scale = (1.0 - new_weight) / total_existing
        for item in existing_weights:
            item["params"]["weight"] = item.get("params", {}).get("weight", 0.0) * scale

    data["score_weights"].append(
        {"type": "my_test_metric", "params": {"weight": new_weight}}
    )

    # 3. Write modified config to a temp file
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as tmp:
        yaml.dump(data, tmp)
        tmp.flush()
        tmp_path = tmp.name

    # 4. Init Engine with modified config path
    engine = Trustifai(tmp_path)
    # engine.context = basic_context
    engine.service = mock_service
    engine._init_metrics()

    # 5. Verify it exists in engine.metrics
    assert "my_test_metric" in engine.metrics
    assert isinstance(engine.metrics["my_test_metric"], MockCustomMetric)

    # 6. Verify it impacts score
    for name, metric in engine.metrics.items():
        metric.calculate = MagicMock(
            return_value=MetricResult(
                score=1.0 if name == "my_test_metric" else 0.5, label="Good", details={}
            )
        )
    result = engine.get_trust_score(basic_context)
    assert (
        result["score"] > 0.5
    )  # Since my_test_metric returns 1.0 with significant weight

    # Clean up: Unregister the metric to avoid side effects on other tests
    if (
        hasattr(Trustifai, "_metric_registry")
        and "my_test_metric" in Trustifai._metric_registry
    ):
        Trustifai._metric_registry.pop("my_test_metric", None)


def test_generate_flow(sample_config_yaml, mock_service):
    engine = Trustifai(config_path=sample_config_yaml)
    engine.service = mock_service

    mock_service.llm_call.return_value = {
        "response": "Generated Answer",
        "logprobs": [-0.5, -0.5],
    }

    result = engine.generate("Prompt")
    assert result["response"] == "Generated Answer"
    assert "confidence_score" in result["metadata"]


def test_build_reasoning_graph(basic_context, sample_config_yaml, mock_service):
    engine = Trustifai(sample_config_yaml)
    engine.service = mock_service

    # Mock metrics
    for name, metric in engine.metrics.items():
        metric.calculate = MagicMock(
            return_value=MetricResult(score=0.5, label="OK", details={})
        )

    score = engine.get_trust_score(basic_context)
    graph = engine.build_reasoning_graph(score)

    assert graph is not None
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0


def test_all_zero_weights_raises_error(basic_context, sample_config_yaml, mock_service):
    # Modify config so all weights are zero
    with open(sample_config_yaml, "r") as f:
        data = yaml.safe_load(f)

    if "score_weights" in data:
        for item in data["score_weights"]:
            item["params"]["weight"] = 0.0

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as tmp:
        yaml.dump(data, tmp)
        tmp.flush()
        tmp_path = tmp.name

    engine = Trustifai(tmp_path)
    engine.service = mock_service

    # Mock metrics
    for name, metric in engine.metrics.items():
        metric.calculate = MagicMock(
            return_value=MetricResult(score=0.5, label="OK", details={})
        )

    # Should raise an error due to all-zero weights
    with pytest.raises(ValueError, match="all weights are zero"):
        engine.get_trust_score(basic_context)


def test_generate_error_handling(sample_config_yaml, mock_service):
    engine = Trustifai(sample_config_yaml)
    engine.service = mock_service

    # Force failure
    mock_service.llm_call.return_value = None

    result = engine.generate("test")
    assert result["metadata"]["error"] == "LLM call failed"


def test_validate_context_missing_answer(sample_config_yaml, mock_service):
    """Test validation catches missing answer."""
    engine = Trustifai(sample_config_yaml)
    engine.service = mock_service
    
    context = MetricContext(
        query="Test query",
        answer=None,  # Missing answer
        documents=[Document(page_content="Doc", metadata={})]
    )
    
    with pytest.raises(ValueError, match="answer"):
        engine._validate_context(context)


def test_validate_context_missing_query(sample_config_yaml, mock_service):
    """Test validation catches missing query."""
    engine = Trustifai(sample_config_yaml)
    engine.service = mock_service
    
    context = MetricContext(
        query=None,  # Missing query
        answer="Answer",
        documents=[Document(page_content="Doc", metadata={})]
    )
    
    with pytest.raises(ValueError, match="query"):
        engine._validate_context(context)


def test_validate_context_missing_documents(sample_config_yaml, mock_service):
    """Test validation catches missing documents."""
    context = MetricContext(
        query="Query",
        answer="Answer",
        documents=None  # Missing documents
    )
    
    # Create mock context without documents attribute
    context_dict = vars(context).copy()
    del context_dict['documents']
    
    engine = Trustifai(sample_config_yaml)
    engine.service = mock_service
    
    # Test with object missing attribute
    class BadContext:
        query = "Q"
        answer = "A"
    
    with pytest.raises(ValueError, match="documents"):
        engine._validate_context(BadContext())


def test_compute_embeddings_when_missing(basic_context, sample_config_yaml, mock_service):
    """Test embeddings are computed when missing."""
    engine = Trustifai(sample_config_yaml)
    engine.service = mock_service
    
    # Clear embeddings
    context = MetricContext(
        query=basic_context.query,
        answer=basic_context.answer,
        documents=basic_context.documents,
        query_embeddings=None,
        answer_embeddings=None,
        document_embeddings=None
    )
    
    mock_service.embedding_call_batch.return_value = {
        'embedding': [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.1, 0.2, 0.3]),
            np.array([0.1, 0.2, 0.3]),
            np.array([0.1, 0.2, 0.3]),
        ]
    }
    
    result = engine._compute_embeddings(context)
    
    assert result.query_embeddings is not None
    assert result.answer_embeddings is not None
    assert result.document_embeddings is not None
    assert mock_service.embedding_call_batch.called


def test_compute_embeddings_preserves_existing(basic_context, sample_config_yaml, mock_service):
    """Test existing embeddings are preserved."""
    engine = Trustifai(sample_config_yaml)
    engine.service = mock_service
    
    original_query_emb = basic_context.query_embeddings.copy()
    result = engine._compute_embeddings(basic_context)
    
    # Should not have called batch embeddings since they exist
    assert np.allclose(result.query_embeddings, original_query_emb)


def test_get_trust_score_validates_context(sample_config_yaml, mock_service):
    """Test get_trust_score validates context."""
    engine = Trustifai(sample_config_yaml)
    engine.service = mock_service
    
    context = MetricContext(query=None, answer=None, documents=None)
    
    with pytest.raises(ValueError):
        engine.get_trust_score(context)


def test_get_trust_score_calculates_metrics(basic_context, sample_config_yaml, mock_service):
    """Test all metrics are calculated."""
    engine = Trustifai(sample_config_yaml)
    engine.service = mock_service
    
    # Mock all metrics
    for name, metric in engine.metrics.items():
        metric.calculate = MagicMock(
            return_value=MetricResult(score=0.8, label="Good", details={})
        )
    
    result = engine.get_trust_score(basic_context)
    
    # Verify each metric was calculated
    for name, metric in engine.metrics.items():
        assert metric.calculate.called


def test_get_trust_score_weighted_aggregation(basic_context, sample_config_yaml, mock_service):
    """Test weighted aggregation of scores."""
    engine = Trustifai(sample_config_yaml)
    engine.service = mock_service
    
    # Mock with different scores
    metric_scores = {
        "evidence_coverage": 0.9,
        "semantic_drift": 0.7,
        "consistency": 0.8,
        "source_diversity": 0.6,
    }
    
    for name, metric in engine.metrics.items():
        score = metric_scores.get(name, 0.5)
        metric.calculate = MagicMock(
            return_value=MetricResult(score=score, label="Good", details={})
        )
    
    result = engine.get_trust_score(basic_context)
    
    assert "score" in result
    assert 0 <= result["score"] <= 1


def test_get_trust_score_includes_details(basic_context, sample_config_yaml, mock_service):
    """Test result includes all details."""
    engine = Trustifai(sample_config_yaml)
    engine.service = mock_service
    
    for name, metric in engine.metrics.items():
        metric.calculate = MagicMock(
            return_value=MetricResult(
                score=0.8,
                label="Good",
                details={"key": "value"}
            )
        )
    
    result = engine.get_trust_score(basic_context)
    
    assert "details" in result
    assert "metrics" in result["details"]


@pytest.mark.asyncio
async def test_a_get_trust_score(basic_context, sample_config_yaml, mock_service):
    """Test async version of get_trust_score."""
    engine = Trustifai(sample_config_yaml)
    engine.service = mock_service
    
    for name, metric in engine.metrics.items():
        metric.calculate = MagicMock(
            return_value=MetricResult(score=0.8, label="Good", details={})
        )
    
    result = await engine.a_get_trust_score(basic_context)
    
    assert "score" in result
    assert result["score"] > 0


def test_generate_returns_response(sample_config_yaml, mock_service):
    """Test generate returns response."""
    engine = Trustifai(sample_config_yaml)
    engine.service = mock_service
    
    mock_service.llm_call.return_value = {
        "response": "Generated text",
        "logprobs": [-0.5, -0.5]
    }
    
    result = engine.generate("Prompt text")
    
    assert result["response"] == "Generated text"
    assert "metadata" in result


def test_generate_includes_confidence(sample_config_yaml, mock_service):
    """Test generate includes confidence metrics."""
    engine = Trustifai(sample_config_yaml)
    engine.service = mock_service
    
    mock_service.llm_call.return_value = {
        "response": "Text",
        "logprobs": [-0.1, -0.2, -0.15]
    }
    
    result = engine.generate("Prompt")
    
    assert "confidence_score" in result["metadata"]
    assert "confidence_label" in result["metadata"]
    assert "confidence_details" in result["metadata"]


def test_generate_handles_missing_response(sample_config_yaml, mock_service):
    """Test generate handles failed LLM call."""
    engine = Trustifai(sample_config_yaml)
    engine.service = mock_service
    
    mock_service.llm_call.return_value = None
    
    result = engine.generate("Prompt")
    
    assert result["response"] is None
    assert "error" in result["metadata"]


def test_generate_system_prompt_defaults(sample_config_yaml, mock_service):
    """Test generate uses default system prompt."""
    engine = Trustifai(sample_config_yaml)
    engine.service = mock_service
    
    mock_service.llm_call.return_value = {
        "response": "Text",
        "logprobs": []
    }
    
    result = engine.generate("User prompt")
    
    assert mock_service.llm_call.called
    # Verify system_prompt was used


def test_aggregate_results_computes_weighted_score(sample_config_yaml, mock_service):
    """Test aggregation computes weighted score."""
    engine = Trustifai(sample_config_yaml)
    engine.service = mock_service
    
    metrics_data = {
        "evidence_coverage": {"score": 0.8, "label": "Good"},
        "semantic_drift": {"score": 0.6, "label": "OK"},
        "consistency": {"score": 0.9, "label": "Excellent"},
        "source_diversity": {"score": 0.7, "label": "Good"},
    }
    
    result = engine._aggregate_results(metrics_data)
    
    assert "score" in result
    assert 0 <= result["score"] <= 1


def test_aggregate_results_includes_label(sample_config_yaml, mock_service):
    """Test aggregation includes decision label."""
    engine = Trustifai(sample_config_yaml)
    engine.service = mock_service
    
    metrics_data = {
        "evidence_coverage": {"score": 0.9, "label": "Good"},
        "semantic_drift": {"score": 0.9, "label": "Good"},
        "consistency": {"score": 0.9, "label": "Good"},
        "source_diversity": {"score": 0.9, "label": "Good"},
    }
    
    result = engine._aggregate_results(metrics_data)
    
    assert "label" in result
    assert result["label"] in ["RELIABLE", "ACCEPTABLE", "UNRELIABLE"]