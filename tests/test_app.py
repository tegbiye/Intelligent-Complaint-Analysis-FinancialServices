import pytest
import faiss
import pickle
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
import streamlit as st
import sys
import importlib
from src import app  # Correct import for app.py in src folder
from src.loggers import get_logger

# Reload src.app to ensure fresh module state
@pytest.fixture(autouse=True)
def reload_app_module():
    importlib.reload(sys.modules['src.app'])
    yield

# Mock Streamlit to avoid runtime issues
@pytest.fixture(autouse=True)
def mock_streamlit():
    with patch("streamlit.runtime.exists", return_value=False):
        yield

# Mock logger to handle src.loggers.get_logger
@pytest.fixture(autouse=True)
def mock_logger():
    with patch("src.loggers.get_logger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        yield mock_logger

# Fixture for vector store directory
@pytest.fixture
def vector_store_dir(tmp_path):
    return tmp_path / "vector_store"

# Fixture for mock FAISS index
@pytest.fixture
def mock_faiss_index():
    index = MagicMock(spec=faiss.IndexFlatL2)
    index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[0, 1]]))
    return index

# Fixture for mock metadata and chunks
@pytest.fixture
def mock_metadata():
    chunks = ["Complaint about delayed payment", "Issue with credit card", "Customer service complaint"]
    metadata = [
        {"chunk_id": 1, "product": "Credit Card"},
        {"chunk_id": 2, "product": "Loan"},
        {"chunk_id": 3, "product": "Customer Service"}
    ]
    return chunks, metadata

# Clear Streamlit cache before each test
@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    st.cache_resource.clear()
    yield

# Test load_vector_store function
@patch("src.app.faiss.read_index")  # Correct patch path
@patch("builtins.open", new_callable=mock_open)
def test_load_vector_store_success(mock_file, mock_read_index, vector_store_dir, mock_metadata):
    mock_read_index.return_value = MagicMock(spec=faiss.IndexFlatL2)
    mock_file.return_value.__enter__.return_value.read.return_value = pickle.dumps({
        'chunks': mock_metadata[0],
        'metadata': mock_metadata[1]
    })
    original_vector_store_dir = app.VECTOR_STORE_DIR
    app.VECTOR_STORE_DIR = vector_store_dir
    try:
        index, chunks, metadata = app.load_vector_store()
        mock_read_index.assert_called_once_with(str(vector_store_dir / 'faiss_index.faiss'))
        mock_file.assert_called_once_with(vector_store_dir / 'metadata.pkl', 'rb')
        assert len(chunks) == 3
        assert len(metadata) == 3
        assert chunks[0] == "Complaint about delayed payment"
        assert metadata[0]["chunk_id"] == 1
    finally:
        app.VECTOR_STORE_DIR = original_vector_store_dir

@patch("src.app.faiss.read_index")  # Correct patch path
def test_load_vector_store_file_not_found(mock_read_index, vector_store_dir):
    mock_read_index.side_effect = FileNotFoundError("File not found")
    original_vector_store_dir = app.VECTOR_STORE_DIR
    app.VECTOR_STORE_DIR = vector_store_dir
    try:
        with pytest.raises(FileNotFoundError, match="File not found"):
            app.load_vector_store()
        mock_read_index.assert_called_once_with(str(vector_store_dir / 'faiss_index.faiss'))
    finally:
        app.VECTOR_STORE_DIR = original_vector_store_dir

@patch("src.app.SentenceTransformer")  # Correct patch path
def test_load_embedding_model_success(mock_sentence_transformer):
    mock_model = MagicMock(spec=SentenceTransformer)
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    mock_sentence_transformer.return_value = mock_model
    embedding_model = app.load_embedding_model()
    mock_sentence_transformer.assert_called_once_with('sentence-transformers/all-MiniLM-L6-v2')
    assert embedding_model == mock_model
    mock_model.encode.assert_called_once_with(["test sentence"])

@patch("src.app.SentenceTransformer")  # Correct patch path
def test_load_embedding_model_failure(mock_sentence_transformer):
    mock_sentence_transformer.side_effect = Exception("Model loading failed")
    with pytest.raises(Exception, match="Model loading failed"):
        app.load_embedding_model()
    mock_sentence_transformer.assert_called_once_with('sentence-transformers/all-MiniLM-L6-v2')

@patch("src.app.torch.cuda.is_available")  # Correct patch path
@patch("src.app.pipeline")  # Correct patch path
def test_load_llm_success(mock_pipeline, mock_cuda_available):
    mock_cuda_available.return_value = False  # Simulate CPU
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.return_value = [{"generated_text": "Test output"}]
    mock_pipeline.return_value = mock_pipeline_instance
    llm = app.load_llm()
    assert isinstance(llm, HuggingFacePipeline)
    mock_pipeline.assert_called_once_with(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1,
        max_length=200,
        do_sample=True,
        temperature=0.7
    )
    mock_pipeline_instance.assert_called_once_with("Test input")

@patch("src.app.pipeline")  # Correct patch path
def test_load_llm_failure(mock_pipeline):
    mock_pipeline.side_effect = Exception("LLM loading failed")
    with pytest.raises(Exception, match="LLM loading failed"):
        app.load_llm()
    mock_pipeline.assert_called_once_with(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1,
        max_length=200,
        do_sample=True,
        temperature=0.7
    )

def test_retrieve_chunks_success(mock_faiss_index, mock_metadata):
    mock_embedding_model = MagicMock()
    mock_embedding_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    query = "Test query"
    chunks, metadata = mock_metadata
    k = 2
    retrieved = app.retrieve_chunks(query, mock_embedding_model, mock_faiss_index, chunks, metadata, k=k)
    assert len(retrieved) == 2
    assert retrieved[0]['text'] == chunks[0]
    assert retrieved[0]['metadata'] == metadata[0]
    assert retrieved[1]['text'] == chunks[1]
    assert retrieved[1]['metadata'] == metadata[1]
    mock_embedding_model.encode.assert_called_once_with([query])
    mock_faiss_index.search.assert_called_once()

def test_retrieve_chunks_invalid_query(mock_faiss_index, mock_metadata):
    mock_embedding_model = MagicMock()
    chunks, metadata = mock_metadata
    with pytest.raises(ValueError, match="Invalid query"):
        app.retrieve_chunks("", mock_embedding_model, mock_faiss_index, chunks, metadata)

@patch("src.app.retrieve_chunks")  # Correct patch path
def test_rag_pipeline_success(mock_retrieve_chunks, mock_metadata, mock_logger):
    query = "What are common complaints?"
    mock_embedding_model = MagicMock()
    mock_index = MagicMock()
    chunks, metadata = mock_metadata
    mock_llm = MagicMock()
    mock_llm.return_value = "Common complaints include delayed payments and poor customer service."
    mock_retrieve_chunks.return_value = [
        {'text': chunks[0], 'metadata': metadata[0]},
        {'text': chunks[1], 'metadata': metadata[1]}
    ]
    print(f"Patching retrieve_chunks: {mock_retrieve_chunks}")  # Debug
    result = app.rag_pipeline(query, mock_embedding_model, mock_index, chunks, metadata, mock_llm)
    print(f"Result: {result}")  # Debug
    mock_retrieve_chunks.assert_called_once_with(query, mock_embedding_model, mock_index, chunks, metadata, k=5)
    assert result['answer'] == "Common complaints include delayed payments and poor customer service."
    assert len(result['retrieved_chunks']) == 2
    assert result['retrieved_chunks'][0]['text'] == chunks[0]
    mock_llm.assert_called_once()

@patch("src.app.retrieve_chunks")  # Correct patch path
def test_rag_pipeline_retrieval_failure(mock_retrieve_chunks, mock_metadata, mock_logger):
    query = "What are common complaints?"
    mock_embedding_model = MagicMock()
    mock_index = MagicMock()
    chunks, metadata = mock_metadata
    mock_llm = MagicMock()
    mock_retrieve_chunks.side_effect = Exception("Retrieval failed")
    print(f"Patching retrieve_chunks: {mock_retrieve_chunks}")  # Debug
    with pytest.raises(Exception, match="Retrieval failed"):
        app.rag_pipeline(query, mock_embedding_model, mock_index, chunks, metadata, mock_llm)
    print("After rag_pipeline call in failure test")  # Debug
    mock_retrieve_chunks.assert_called_once_with(query, mock_embedding_model, mock_index, chunks, metadata, k=5)

if __name__ == "__main__":
    pytest.main()