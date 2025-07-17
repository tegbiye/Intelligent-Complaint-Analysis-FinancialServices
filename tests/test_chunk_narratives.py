import pandas as pd
import pytest
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.text_chunker import chunk_narratives  # Replace with actual module name

def create_test_dataframe():
    """Helper function to create a sample DataFrame for testing."""
    data = {
        'Product': ['Credit Card', 'Mortgage', 'Loan'],
        'cleaned_narrative': [
            "This is a long complaint about a credit card issue. It has multiple sentences. The issue was not resolved.",
            "Short complaint.",
            ""  # Empty narrative
        ]
    }
    return pd.DataFrame(data)

def test_chunk_narratives_basic_functionality():
    """Test basic functionality of chunk_narratives with valid input."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    df = create_test_dataframe()
    
    chunks, metadata = chunk_narratives(df, text_splitter)
    
    # Check that chunks are created
    assert len(chunks) > 0, "No chunks were created"
    assert len(chunks) == len(metadata), "Mismatch between chunks and metadata length"
    
    # Verify chunk content for the first narrative
    assert chunks[0].startswith("This is a long complaint"), "First chunk content is incorrect"
    
    # Verify metadata structure
    assert metadata[0]['complaint_id'] == 0, "Complaint ID incorrect"
    assert metadata[0]['product'] == 'Credit Card', "Product metadata incorrect"
    assert metadata[0]['chunk_id'] == '0_0', "Chunk ID format incorrect"

def test_chunk_narratives_empty_narrative():
    """Test handling of empty narrative."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    df = create_test_dataframe()
    
    chunks, metadata = chunk_narratives(df, text_splitter)
    
    # Check metadata for empty narrative (index 2)
    empty_narrative_metadata = [m for m in metadata if m['complaint_id'] == 2]
    assert len(empty_narrative_metadata) == 1, "Empty narrative should produce one chunk"
    assert chunks[metadata.index(empty_narrative_metadata[0])] == "", "Empty narrative chunk should be empty string"

def test_chunk_narratives_chunk_id_format():
    """Test that chunk IDs are correctly formatted."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=5)
    df = create_test_dataframe()
    
    chunks, metadata = chunk_narratives(df, text_splitter)
    
    # Check chunk_id format for first complaint
    first_complaint_metadata = [m for m in metadata if m['complaint_id'] == 0]
    for i, meta in enumerate(first_complaint_metadata):
        assert meta['chunk_id'] == f"0_{i}", f"Chunk ID {meta['chunk_id']} incorrectly formatted"

def test_chunk_narratives_metadata_preservation():
    """Test that metadata is correctly preserved for each chunk."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    df = create_test_dataframe()
    
    chunks, metadata = chunk_narratives(df, text_splitter)
    
    # Verify product metadata is preserved
    for meta in metadata:
        assert meta['product'] == df.loc[meta['complaint_id'], 'Product'], f"Product metadata mismatch for complaint {meta['complaint_id']}"

def test_chunk_narratives_empty_dataframe():
    """Test handling of empty DataFrame."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    df = pd.DataFrame(columns=['Product', 'cleaned_narrative'])
    
    chunks, metadata = chunk_narratives(df, text_splitter)
    
    assert len(chunks) == 0, "Chunks should be empty for empty DataFrame"
    assert len(metadata) == 0, "Metadata should be empty for empty DataFrame"