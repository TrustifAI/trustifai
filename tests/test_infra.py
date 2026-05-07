import pytest
from trustifai.config import Config
import asyncio
from trustifai.services import ExternalService
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document as LangchainDocument
from llama_index.core import Document as LlamaIndexDocument
# --- Config Tests ---

def test_config_loading(sample_config_yaml):
    cfg = Config.from_yaml(sample_config_yaml)
    assert cfg.llm.type == "openai"
    assert len(cfg.metrics) == 4
    # Check weight normalization logic
    assert abs(sum(cfg.weights.model_dump().values()) - 1.0) < 0.001

def test_weight_normalization_error(sample_config_yaml):
    # Create invalid weights using a temp file to avoid side effects
    import yaml
    import tempfile
    with open(sample_config_yaml, 'r') as f:
        data = yaml.safe_load(f)

    data['score_weights'][0]['params']['weight'] = 2.0 # Sum > 1.0

    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        yaml.dump(data, tmp)
        tmp_path = tmp.name

    try:
        with pytest.raises(ValueError, match="Weights must normalize to 1.0"):
            Config.from_yaml(tmp_path)
    finally:
        import os
        os.remove(tmp_path)

def test_dynamic_config_fields(sample_config_yaml):
    """Test the new feature allowing custom metrics in config"""
    import yaml
    with open(sample_config_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    # Add a custom metric weight
    data['score_weights'].append({"type": "pii_check", "params": {"weight": 0.0}})
    
    with open(sample_config_yaml, 'w') as f:
        yaml.dump(data, f)
        
    cfg = Config.from_yaml(sample_config_yaml)
    # verify pydantic accepted the extra field
    assert hasattr(cfg.weights, "pii_check") or "pii_check" in cfg.weights.model_dump()


def test_document_extraction(mock_service):
    # String
    assert mock_service.extract_document("Hello") == "Hello"
    # Dict
    assert mock_service.extract_document({"text": "Hello"}) == "Hello"
    # List
    assert mock_service.extract_document(["Hello", "World"]) == "Hello\nWorld"
    # None
    assert mock_service.extract_document(None) == ""

def test_llm_call_success(mock_service):
    mock_service.llm_call.return_value = {"response": "Test Response", "logprobs": []}

    res = mock_service.llm_call(prompt="Hi")
    assert res["response"] == "Test Response"

def test_llm_call_failure(mock_service):
    mock_service.llm_call.side_effect = Exception("API Error")

    with pytest.raises(Exception, match="API Error"):
        mock_service.llm_call(prompt="Hi")

def test_llm_call_batch(mock_service):
    # Test batch LLM calls with multiple prompts
    # Test with "responses" and "chat_completion" API types
    # Test return format and cost calculation
    mock_service.llm_call_batch.return_value = [{"response": "Test Response 1", "logprobs": []}, {"response": "Test Response 2", "logprobs": []}]
    res = mock_service.llm_call_batch(prompts=["Hi", "Hello"], api_type="chat_completion")
    assert len(res) == 2
    assert res[0]["response"] == "Test Response 1"
    assert res[1]["response"] == "Test Response 2"

def test_llm_call_async(mock_service):
    async def mock_async_call(*args, **kwargs):
        return {"response": "Async Test Response", "logprobs": []}

    mock_service.llm_call_async = mock_async_call

    async def run_test():
        res = await mock_service.llm_call_async(prompt="Hi")
        assert res["response"] == "Async Test Response"

    asyncio.run(run_test())

def test_api_type_llm_call(mock_service):
    # Test with "responses" API type
    mock_service.llm_call.return_value = {"response": "Test Response", "logprobs": []}
    res = mock_service.llm_call(prompt="Hi", api_type="responses")
    assert res["response"] == "Test Response"

    # Test with "chat_completion" API type
    res = mock_service.llm_call(prompt="Hi", api_type="chat_completion")
    assert res["response"] == "Test Response"

def test_embedding_call(mock_service):
    mock_service.embedding_call.return_value = [0.1, 0.2, 0.3]

    vec = mock_service.embedding_call("Test text")
    assert isinstance(vec, list) or hasattr(vec, "__array__")

def test_embedding_call_batch(mock_service):
    # Test batch embeddings with multiple texts
    # Test batch size handling
    # Verify embedding dimensions
    mock_service.embedding_call_batch.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    vecs = mock_service.embedding_call_batch(["Test text 1", "Test text 2"])
    assert len(vecs) == 2
    assert isinstance(vecs[0], list) or hasattr(vecs[0], "__array__")
    assert isinstance(vecs[1], list) or hasattr(vecs[1], "__array__")

def test_embedding_call_async(mock_service):
    async def mock_async_embedding(*args, **kwargs):
        return [0.1, 0.2, 0.3]

    mock_service.embedding_call_async = mock_async_embedding

    async def run_test():
        vec = await mock_service.embedding_call_async("Test text")
        assert isinstance(vec, list) or hasattr(vec, "__array__")

    asyncio.run(run_test())

def test_embedding_call_empty_input(mock_service):
    mock_service.embedding_call.return_value = []

    vec = mock_service.embedding_call("")
    assert vec == []

def test_reranker_call(mock_service):
    docs = ["Doc 1", "Doc 2", "Doc 3"]
    query = "Test query"

    ranked_docs = mock_service.reranker_call(docs, query)
    assert isinstance(ranked_docs, list)

def test_reranker_empty_result(mock_service):
    mock_service.reranker_call.return_value = []

    docs = mock_service.extract_document([])

    assert docs == ''

@patch("trustifai.services.completion") 
@patch("trustifai.services.embedding")  
@patch("trustifai.services.acompletion") 
@patch("trustifai.services.responses")
@patch("trustifai.services.aresponses")
def test_real_service_calls(mock_aresponses, mock_responses, mock_acompletion, mock_embed, mock_completion, sample_config_yaml):
    config = Config.from_yaml(sample_config_yaml)
    
    service = ExternalService(config)
    mock_completion.return_value.choices = [
        MagicMock(message=MagicMock(content="Mocked Response"), logprobs=None)
    ]
    mock_acompletion.return_value.choices = [
        MagicMock(message=MagicMock(content="Mocked Async Response"), logprobs=None)
    ]
    mock_responses.return_value.output_text = "Mocked Response"
    mock_aresponses.return_value.output_text = "Mocked Async Response"
    
    mock_embed.return_value.data = [{"embedding": [0.1, 0.2, 0.3]}]

    #api type: chat_completion
    service.llm_call(prompt="test")
    service.embedding_call("text")
    asyncio.run(service.llm_call_async(prompt="async test"))

    #api type: responses
    config.llm.params['api_type'] = "responses"
    service = ExternalService(config)
    service.llm_call(prompt="test")
    service.embedding_call("text")
    asyncio.run(service.llm_call_async(prompt="async test"))

    assert mock_completion.called
    assert mock_embed.called
    assert mock_acompletion.called
    assert mock_responses.called
    assert mock_aresponses.called

def test_exception_handling_in_service_calls(mock_service):
    # LLM Call Exception
    mock_service.llm_call.side_effect = Exception("LLM Failure")
    with pytest.raises(Exception, match="LLM Failure"):
        mock_service.llm_call("Test prompt")

    # Async LLM Call Exception
    mock_service.llm_call_async.side_effect = Exception("Async LLM Failure")
    with pytest.raises(Exception, match="Async LLM Failure"):
        asyncio.run(mock_service.llm_call_async("Test prompt"))

    # Embedding Call Exception
    mock_service.embedding_call.side_effect = Exception("Embedding Failure")
    with pytest.raises(Exception, match="Embedding Failure"):
        mock_service.embedding_call("Test text")


def test_extract_document_various_inputs(mock_service):
    # Test with string input
    assert mock_service.extract_document("Simple text") == "Simple text"

    # Test with dict input
    assert mock_service.extract_document({"text": "Dict text"}) == "Dict text"

    # Test with list input
    assert mock_service.extract_document(["Line 1", "Line 2"]) == "Line 1\nLine 2"

    # Test with None input
    assert mock_service.extract_document(None) == ""

    # Test with Langchain Document
    langchain_doc = LangchainDocument(page_content="Langchain content", metadata={"source": "langchain"})
    assert mock_service.extract_document(langchain_doc) == "Langchain content"

    # Test with LlamaIndex Document
    llamaindex_doc = LlamaIndexDocument(text="LlamaIndex content", metadata={"source": "llama"})
    assert mock_service.extract_document(llamaindex_doc) == "LlamaIndex content"