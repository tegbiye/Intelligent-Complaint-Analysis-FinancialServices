import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.rag_helper import retrieve_chunks, rag_pipeline  # Correct module import

# Mock logger to replace the module-level logger in rag_helper
@pytest.fixture
def mock_logger():
    with patch('src.rag_helper.logger') as mock_logger:  # Patch the logger instance directly
        yield mock_logger

# Fixture for sample data
@pytest.fixture
def sample_data():
    chunks = ["Chunk 1 text", "Chunk 2 text", "Chunk 3 text"]
    metadata = [
        {'complaint_id': 0, 'product': 'Credit Card', 'chunk_id': '0_0'},
        {'complaint_id': 1, 'product': 'Mortgage', 'chunk_id': '1_0'},
        {'complaint_id': 2, 'product': 'Loan', 'chunk_id': '2_0'}
    ]
    return chunks, metadata

# Fixture for mock embedding model and FAISS index
@pytest.fixture
def mock_embedding_and_index():
    embedding_model = Mock()
    embedding_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    index = Mock()
    index.search.return_value = (np.array([[0.1, 0.2, 0.3]]), np.array([[0, 1, 2]]))
    return embedding_model, index

# Fixture for mock LLM
@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.return_value = "Generated answer"
    return llm

# Tests for retrieve_chunks
def test_retrieve_chunks_valid_query(mock_logger, sample_data, mock_embedding_and_index):
    """Test retrieve_chunks with valid query and inputs."""
    chunks, metadata = sample_data
    embedding_model, index = mock_embedding_and_index
    query = "test query"
    
    result = retrieve_chunks(query, k=3, embedding_model=embedding_model, index=index, chunks=chunks, metadata=metadata)
    
    assert len(result) == 3, "Should retrieve 3 chunks"
    assert result[0]['text'] == "Chunk 1 text", "Incorrect chunk text"
    assert result[0]['metadata'] == metadata[0], "Incorrect metadata"
    mock_logger.info.assert_called_with(f"Retrieved 3 chunks for query: {query}")

def test_retrieve_chunks_invalid_query(mock_logger, sample_data, mock_embedding_and_index):
    """Test retrieve_chunks with invalid query."""
    chunks, metadata = sample_data
    embedding_model, index = mock_embedding_and_index
    
    with pytest.raises(ValueError, match="Invalid query: "):
        retrieve_chunks("", k=3, embedding_model=embedding_model, index=index, chunks=chunks, metadata=metadata)
    mock_logger.error.assert_called_once()

def test_retrieve_chunks_invalid_index(mock_logger, sample_data, mock_embedding_and_index):
    """Test retrieve_chunks with out-of-bounds index."""
    chunks, metadata = sample_data
    embedding_model, index = mock_embedding_and_index
    index.search.return_value = (np.array([[0.1]]), np.array([[999]]))  # Invalid index
    
    result = retrieve_chunks("test query", k=1, embedding_model=embedding_model, index=index, chunks=chunks, metadata=metadata)
    
    assert len(result) == 0, "Should skip invalid index and return empty list"
    mock_logger.warning.assert_called_with("Invalid index 999 retrieved, skipping.")

def test_retrieve_chunks_empty_inputs(mock_logger, mock_embedding_and_index):
    """Test retrieve_chunks with empty chunks and metadata."""
    embedding_model, index = mock_embedding_and_index
    index.search.return_value = (np.array([[0.1, 0.2, 0.3]]), np.array([[0, 1, 2]]))  # Reset to avoid invalid index
    result = retrieve_chunks("test query", k=3, embedding_model=embedding_model, index=index, chunks=[], metadata=[])
    
    assert len(result) == 0, "Should return empty list for empty inputs"
    mock_logger.info.assert_called_with("Retrieved 0 chunks for query: test query")

# Tests for rag_pipeline
def test_rag_pipeline_valid_query(mock_logger, sample_data, mock_embedding_and_index, mock_llm):
    """Test rag_pipeline with valid inputs."""
    chunks, metadata = sample_data
    embedding_model, index = mock_embedding_and_index
    llm = mock_llm
    prompt_template = "Context: {context}\nQuestion: {question}"
    query = "test query"
    
    result = rag_pipeline(query, llm, embedding_model, index, chunks, metadata, prompt_template)
    
    assert result['answer'] == "Generated answer", "Incorrect answer"
    assert len(result['retrieved_chunks']) == 3, "Incorrect number of retrieved chunks"
    assert result['retrieved_chunks'][0]['text'] == "Chunk 1 text", "Incorrect chunk text"
    expected_context = "\n".join([chunk['text'] for chunk in result['retrieved_chunks']])
    llm.assert_called_with(prompt_template.format(context=expected_context, question=query))
    mock_logger.info.assert_called_with(f"Generated answer for query: {query}")

def test_rag_pipeline_invalid_query(mock_logger, sample_data, mock_embedding_and_index, mock_llm):
    """Test rag_pipeline with invalid query."""
    chunks, metadata = sample_data
    embedding_model, index = mock_embedding_and_index
    llm = mock_llm
    prompt_template = "Context: {context}\nQuestion: {question}"
    
    with pytest.raises(ValueError, match="Invalid query: "):
        rag_pipeline("", llm, embedding_model, index, chunks, metadata, prompt_template)
    assert mock_logger.error.call_count == 2, "Expected two error logs"
    mock_logger.error.assert_any_call("Error in retrieving chunks for query '': Invalid query: ")
    mock_logger.error.assert_any_call("Error in RAG pipeline for query '': Invalid query: ")

def test_rag_pipeline_empty_retrieved_chunks(mock_logger, sample_data, mock_embedding_and_index, mock_llm):
    """Test rag_pipeline when no chunks are retrieved."""
    chunks, metadata = sample_data
    embedding_model, index = mock_embedding_and_index
    llm = mock_llm
    prompt_template = "Context: {context}\nQuestion: {question}"
    index.search.return_value = (np.array([[0.1]]), np.array([[999]]))  # Invalid index to return no chunks
    
    result = rag_pipeline("test query", llm, embedding_model, index, chunks, metadata, prompt_template)
    
    assert result['answer'] == "Generated answer", "Incorrect answer"
    assert len(result['retrieved_chunks']) == 0, "Should have no retrieved chunks"
    llm.assert_called_with(prompt_template.format(context="", question="test query"))
    mock_logger.info.assert_called_with("Generated answer for query: test query")