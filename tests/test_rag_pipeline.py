import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.loggers import get_logger
from src.rag_helper import retrieve_chunks, rag_pipeline

# Mock logger to avoid actual logging during tests
logger = get_logger(__name__)

@pytest.fixture
def mock_embedding_model():
    model = Mock()
    model.encode.return_value = np.array([0.1, 0.2, 0.3])
    return model

@pytest.fixture
def mock_index():
    index = Mock()
    index.search.return_value = (np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]), np.array([[0, 1, 2, 3, 4]]))
    return index

@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.return_value = "This is a test response"
    return llm

@pytest.fixture
def sample_chunks():
    return ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]

@pytest.fixture
def sample_metadata():
    return [
        {"id": 1, "source": "doc1"},
        {"id": 2, "source": "doc2"},
        {"id": 3, "source": "doc3"},
        {"id": 4, "source": "doc4"},
        {"id": 5, "source": "doc5"}
    ]

@pytest.fixture
def prompt_template():
    return "Context: {context}\nQuestion: {question}\nAnswer:"

def test_retrieve_chunks_valid_input(mock_embedding_model, mock_index, sample_chunks, sample_metadata):
    query = "test query"
    result = retrieve_chunks(query, k=5, embedding_model=mock_embedding_model, index=mock_index, chunks=sample_chunks, metadata=sample_metadata)
    
    assert len(result) == 5
    assert all(isinstance(chunk, dict) for chunk in result)
    assert all('text' in chunk and 'metadata' in chunk for chunk in result)
    assert result[0]['text'] == "chunk1"
    assert result[0]['metadata'] == {"id": 1, "source": "doc1"}

def test_retrieve_chunks_invalid_query(mock_embedding_model, mock_index, sample_chunks, sample_metadata):
    with pytest.raises(ValueError) as exc_info:
        retrieve_chunks("", k=5, embedding_model=mock_embedding_model, index=mock_index, chunks=sample_chunks, metadata=sample_metadata)
    assert "Invalid query" in str(exc_info.value)

def test_retrieve_chunks_invalid_index(mock_embedding_model, mock_index, sample_chunks, sample_metadata):
    mock_index.search.return_value = (np.array([[0.1]]), np.array([[10]]))  # Invalid index
    query = "test query"
    result = retrieve_chunks(query, k=1, embedding_model=mock_embedding_model, index=mock_index, chunks=sample_chunks, metadata=sample_metadata)
    
    assert len(result) == 0  # Should skip invalid index

def test_rag_pipeline_success(mock_embedding_model, mock_index, mock_llm, sample_chunks, sample_metadata, prompt_template):
    query = "test query"
    result = rag_pipeline(query, mock_llm, mock_embedding_model, mock_index, sample_chunks, sample_metadata, prompt_template)
    
    assert 'answer' in result
    assert 'retrieved_chunks' in result
    assert result['answer'] == "This is a test response"
    assert len(result['retrieved_chunks']) == 5
    assert mock_llm.called
    assert mock_embedding_model.encode.called

def test_rag_pipeline_invalid_query(mock_embedding_model, mock_index, mock_llm, sample_chunks, sample_metadata, prompt_template):
    with pytest.raises(ValueError) as exc_info:
        rag_pipeline("", mock_llm, mock_embedding_model, mock_index, sample_chunks, sample_metadata, prompt_template)
    assert "Invalid query" in str(exc_info.value)

def test_rag_pipeline_llm_failure(mock_embedding_model, mock_index, mock_llm, sample_chunks, sample_metadata, prompt_template):
    mock_llm.side_effect = Exception("LLM error")
    query = "test query"
    
    with pytest.raises(Exception) as exc_info:
        rag_pipeline(query, mock_llm, mock_embedding_model, mock_index, sample_chunks, sample_metadata, prompt_template)
    assert "LLM error" in str(exc_info.value)