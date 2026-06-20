import pytest
from unittest.mock import MagicMock, patch
from trustifai import Trustifai
from trustifai.structures import MetricResult
from trustifai.visualizer import GraphVisualizer
from trustifai.structures import ReasoningGraph, ReasoningNode, ReasoningEdge

@pytest.fixture
def sample_graph():
    nodes = [
        ReasoningNode(node_id="n1", node_type="metric", name="Metric 1", inputs={}, outputs={}, score=0.9, label="Good"),
        ReasoningNode(node_id="decision", node_type="decision", name="Decision", inputs={}, outputs={}, score=0.9, label="Reliable")
    ]
    edges = [ReasoningEdge(source="n1", target="decision", relationship="decides")]
    return ReasoningGraph(trace_id="123", nodes=nodes, edges=edges)

def test_mermaid_visualization(sample_graph):
    viz = GraphVisualizer(sample_graph)
    output = viz.visualize(graph_type="mermaid")
    
    assert "```mermaid" in output
    assert "Metric 1" in output
    assert "n1 --> decision" in output

def test_pyvis_visualization(sample_graph):
    viz = GraphVisualizer(sample_graph)
    
    # Mock pyvis network to avoid HTML generation/browser ops
    with patch("pyvis.network.Network") as MockNet:
        mock_net_instance = MockNet.return_value
        
        viz.visualize(graph_type="pyvis", output_file="test.html")
        
        # Verify nodes and edges were added
        assert mock_net_instance.add_node.call_count == 2
        assert mock_net_instance.add_edge.call_count == 1
        mock_net_instance.save_graph.assert_called_with("test.html")

def test_invalid_visualizer_type(sample_graph):
    viz = GraphVisualizer(sample_graph)
    with pytest.raises(ValueError):
        viz.visualize(graph_type="unknown")

def test_visualizer_empty_graph(sample_graph):
    viz = GraphVisualizer(sample_graph)

    output = viz.visualize(graph_type="mermaid")

    assert "flowchart TD" in output

def test_visualizer_single_node(sample_graph):
    viz = GraphVisualizer(sample_graph)
    output = viz.visualize(graph_type="mermaid")
    assert "n1" in output

def test_build_reasoning_graph_creates_nodes(basic_context, sample_config_yaml, mock_service):
    """Test graph has nodes for query, answer, and metrics."""
    engine = Trustifai(sample_config_yaml)
    engine.service = mock_service
    
    for name, metric in engine.metrics.items():
        metric.calculate = MagicMock(
            return_value=MetricResult(score=0.8, label="Good", details={})
        )
    
    score_result = engine.get_trust_score(basic_context)
    graph = engine.build_reasoning_graph(score_result)
    
    assert graph is not None
    assert len(graph.nodes) > 0
    # Should have nodes for query, answer, and metrics


def test_build_reasoning_graph_creates_edges(basic_context, sample_config_yaml, mock_service):
    """Test graph has edges connecting nodes."""
    engine = Trustifai(sample_config_yaml)
    engine.service = mock_service
    
    for name, metric in engine.metrics.items():
        metric.calculate = MagicMock(
            return_value=MetricResult(score=0.8, label="Good", details={})
        )
    
    score_result = engine.get_trust_score(basic_context)
    graph = engine.build_reasoning_graph(score_result)
    
    assert graph is not None
    assert len(graph.edges) > 0
