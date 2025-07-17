import pytest
import numpy as np
from unittest.mock import Mock, patch, mock_open
from src.app import load_vector_store, load_embedding_model, load_llm, retrieve_chunks, rag_pipeline, prompt_template
import faiss
import pickle
from pathlib import Path
from langchain_community.llms import HuggingFacePipeline

# Mock logger fixture
@pytest.fixture
def mock_logger():
    with patch('src.app.logger') as mock_logger:
        yield mock_logger

# Mock Streamlit to prevent UI-related errors
@pytest.fixture(autouse=True)
def mock_streamlit():
    with patch('streamlit.error') as mock_st_error:
        with patch('streamlit.cache_resource.clear', return_value=None):  # Clear cache to avoid st.cache_resource issues
            yield mock_st_error

# Mock LLM fixture
@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.return_value = "Generated answer"
    return llm

# Fixture for sample vector store data
@pytest.fixture
def sample_vector_store_data():
    chunks = [
        "two transactions were done through citi bank and we have proven that two transactions were done as fraud. we were buying into xxxx xxxx for them to take over our existing timeshare. first amount of 1200.00 to xxxx xxxx partial payment in lue of taking over our xxxx xxxx times share, which is located in xxxx xxxx nevada. case id xxxx .the second payment being 6300.00 case id xxxx",
        "Issue with mortgage payment processing.",
        "Loan application was denied without clear reason."
    ]
    metadata = [
        {'complaint_id': 0, 'product': 'Credit Card', 'chunk_id': '0_0'},
        {'complaint_id': 1, 'product': 'Mortgage', 'chunk_id': '1_0'},
        {'complaint_id': 2, 'product': 'Loan', 'chunk_id': '2_0'}
    ]
    index = faiss.IndexFlatL2(384)  # Match expected dimension
    # Add dummy vectors to the index to allow searching
    dummy_vectors = np.random.random((3, 384)).astype('float32')
    index.add(dummy_vectors)
    return index, chunks, metadata

# Test load_vector_store
def test_load_vector_store_success(mock_logger, sample_vector_store_data):
    """Test load_vector_store with valid files."""
    index, chunks, metadata = sample_vector_store_data
    with patch('src.app.faiss.read_index', return_value=index) as mock_read_index:
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('pickle.load', return_value={'chunks': chunks, 'metadata': metadata}) as mock_pickle:
                result = load_vector_store()
    
    assert result[1] == chunks and result[2] == metadata, "Incorrect chunks or metadata returned"
    assert isinstance(result[0], faiss.IndexFlatL2), "Incorrect index type returned"
    mock_read_index.assert_called_with(str(Path('vector_store') / 'faiss_index.faiss'))
    mock_file.assert_called_with(Path('vector_store') / 'metadata.pkl', 'rb')
    mock_logger.info.assert_called_with("Vector store and metadata loaded successfully.")

def test_load_vector_store_file_not_found(mock_logger, mock_streamlit):
    """Test load_vector_store when files are missing."""
    with patch('src.app.faiss.read_index', side_effect=FileNotFoundError("File not found")):
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_vector_store()
    mock_logger.error.assert_called_with("Vector store files not found: File not found")
    mock_streamlit.assert_called_with("Vector store files not found. Please ensure 'vector_store/faiss_index.bin' and 'vector_store/metadata.pkl' exist.")

def test_load_vector_store_general_error(mock_logger, mock_streamlit):
    """Test load_vector_store with general error."""
    with patch('src.app.faiss.read_index', side_effect=Exception("General error")):
        with pytest.raises(Exception, match="General error"):
            load_vector_store()
    mock_logger.error.assert_called_with("Error loading vector store or metadata: General error")
    mock_streamlit.assert_called_with("Error loading vector store: General error")

# Test load_embedding_model
def test_load_embedding_model_success(mock_logger):
    """Test load_embedding_model with successful loading."""
    mock_model = Mock()
    mock_model.encode.return_value = np.array([[0.1] * 384])  # Match expected dimension
    with patch('src.app.SentenceTransformer', return_value=mock_model):
        result = load_embedding_model()
    
    assert result == mock_model, "Incorrect embedding model returned"
    mock_logger.info.assert_any_call("Loading embedding model 'sentence-transformers/all-MiniLM-L6-v2'...")
    mock_logger.info.assert_called_with("Embedding model loaded successfully. Test embedding shape: (1, 384)")

def test_load_embedding_model_none(mock_logger, mock_streamlit):
    """Test load_embedding_model when model is None."""
    with patch('src.app.SentenceTransformer', return_value=None):
        with pytest.raises(ValueError, match="Embedding model is None after initialization."):
            load_embedding_model()
    mock_logger.error.assert_called_with("Failed to load embedding model: Embedding model is None after initialization.")
    mock_streamlit.assert_called_with("Failed to load embedding model: Embedding model is None after initialization.")

def test_load_embedding_model_general_error(mock_logger, mock_streamlit):
    """Test load_embedding_model with general error."""
    with patch('src.app.SentenceTransformer', side_effect=Exception("Model error")):
        with pytest.raises(Exception, match="Model error"):
            load_embedding_model()
    mock_logger.error.assert_called_with("Failed to load embedding model: Model error")
    mock_streamlit.assert_called_with("Failed to load embedding model: Model error")

# Test load_llm
def test_load_llm_success(mock_logger):
    """Test load_llm with successful loading."""
    mock_pipeline = Mock()
    mock_pipeline.return_value = [{'generated_text': "Test output"}]
    with patch('src.app.pipeline', return_value=mock_pipeline) as mock_pipeline_patch:
        with patch('src.app.torch.cuda.is_available', return_value=False):
            result = load_llm()
    
    assert isinstance(result, HuggingFacePipeline), "Incorrect LLM type returned"
    mock_pipeline_patch.assert_called_once_with(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1,
        max_length=200,
        do_sample=True,
        temperature=0.7
    )
    mock_logger.info.assert_any_call("Loading LLM 'flan-t5-base'...")
    mock_logger.info.assert_called_with("LLM (flan-t5-base) loaded successfully. Test output: Test output...")

def test_load_llm_general_error(mock_logger, mock_streamlit):
    """Test load_llm with general error."""
    with patch('src.app.pipeline', side_effect=Exception("LLM error")):
        with pytest.raises(Exception, match="LLM error"):
            load_llm()
    mock_logger.error.assert_called_with("Failed to load LLM 't5-small': LLM error")
    mock_streamlit.assert_called_with("Failed to load LLM: LLM error")

# Test retrieve_chunks
def test_retrieve_chunks_valid_query(mock_logger, sample_vector_store_data):
    """Test retrieve_chunks with valid query."""
    index, chunks, metadata = sample_vector_store_data
    embedding_model = Mock()
    embedding_model.encode.return_value = np.array([[0.1] * 384])  # Match FAISS index dimension
    query = "test query"
    
    result = retrieve_chunks(query, embedding_model, index, chunks, metadata, k=3)
    
    assert len(result) == 3, "Should retrieve 3 chunks"
    assert result[0]['text'] == chunks[0], "Incorrect chunk text"
    assert result[0]['metadata'] == metadata[0], "Incorrect metadata"
    mock_logger.info.assert_called_with(f"Retrieved 3 chunks for query: {query}")

def test_retrieve_chunks_invalid_query(mock_logger, sample_vector_store_data):
    """Test retrieve_chunks with invalid query."""
    index, chunks, metadata = sample_vector_store_data
    embedding_model = Mock()
    
    with pytest.raises(ValueError, match="Invalid query: "):
        retrieve_chunks("", embedding_model, index, chunks, metadata, k=3)
    mock_logger.error.assert_called_with("Error in retrieving chunks for query '': Invalid query: ")

def test_retrieve_chunks_none_embedding_model(mock_logger, sample_vector_store_data):
    """Test retrieve_chunks with None embedding model."""
    index, chunks, metadata = sample_vector_store_data
    with pytest.raises(ValueError, match="Embedding model is None in retrieve_chunks."):
        retrieve_chunks("test query", None, index, chunks, metadata, k=3)
    mock_logger.error.assert_called_with("Error in retrieving chunks for query 'test query': Embedding model is None in retrieve_chunks.")

# Test rag_pipeline
def test_rag_pipeline_valid_query(mock_logger, sample_vector_store_data, mock_llm):
    """Test rag_pipeline with valid inputs."""
    index, chunks, metadata = sample_vector_store_data
    embedding_model = Mock()
    embedding_model.encode.return_value = np.array([[0.1] * 384])  # Match FAISS index dimension
    llm = mock_llm
    query = "test query"
    
    result = rag_pipeline(query, embedding_model, index, chunks, metadata, llm)
    
    assert result['answer'] == "Generated answer", "Incorrect answer"
    assert len(result['retrieved_chunks']) == 3, "Incorrect number of retrieved chunks"
    assert result['retrieved_chunks'][0]['text'] == chunks[0], "Incorrect chunk text"
    expected_context = "\n".join([chunk['text'] for chunk in result['retrieved_chunks']])
    llm.assert_called_with(prompt_template.format(context=expected_context, question=query))
    mock_logger.info.assert_called_with(f"Generated answer for query: {query}")

def test_rag_pipeline_invalid_query(mock_logger, sample_vector_store_data, mock_llm):
    """Test rag_pipeline with invalid query."""
    index, chunks, metadata = sample_vector_store_data
    embedding_model = Mock()
    llm = mock_llm
    
    with pytest.raises(ValueError, match="Invalid query: "):
        rag_pipeline("", embedding_model, index, chunks, metadata, llm)
    assert mock_logger.error.call_count == 2, "Expected two error logs"
    mock_logger.error.assert_any_call("Error in retrieving chunks for query '': Invalid query: ")
    mock_logger.error.assert_any_call("Error in RAG pipeline for query '': Invalid query: ")