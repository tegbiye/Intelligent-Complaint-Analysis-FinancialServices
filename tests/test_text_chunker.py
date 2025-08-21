# test_chunker.py

import pytest
import pandas as pd
from unittest.mock import MagicMock
import sys
import os


# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.text_chuncker as text_c  # import the text_chunker


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Product": ["Credit card", "Loan"],
        "cleaned_narrative": [
            "This is a test narrative. It should be split into chunks.",
            "Another narrative for testing."
        ]
    })


def test_chunk_narratives_basic(sample_df):
    # Mock text splitter
    mock_splitter = MagicMock()
    mock_splitter.split_text.side_effect = [
        ["chunk1", "chunk2"],
        ["chunkA"]
    ]

    chunks, metadata = text_c.chunk_narratives(sample_df, mock_splitter)

    assert chunks == ["chunk1", "chunk2", "chunkA"]
    assert len(metadata) == 3

    # First row metadata
    assert metadata[0]["complaint_id"] == 0
    assert metadata[0]["product"] == "Credit card"
    assert metadata[0]["chunk_id"] == "0_0"

    # Ensure the splitter was called correctly
    assert mock_splitter.split_text.call_count == 2
    assert all(isinstance(c, str) for c in chunks)


def test_chunk_narratives_empty_text(sample_df):
    df = sample_df.copy()
    df.loc[0, "cleaned_narrative"] = ""  # empty narrative

    mock_splitter = MagicMock()
    mock_splitter.split_text.side_effect = [[""]]  # return one empty chunk, second row handled normally
    mock_splitter.split_text.return_value = [""]

    chunks, metadata = text_c.chunk_narratives(df.iloc[:1], mock_splitter)

    assert chunks == [""]
    assert metadata[0]["complaint_id"] == 0
    assert metadata[0]["chunk_id"] == "0_0"


def test_chunk_narratives_multiple_chunks(sample_df):
    # Force multiple chunks for one row
    mock_splitter = MagicMock()
    mock_splitter.split_text.side_effect = [
        ["part1", "part2", "part3"],
        ["only1"]
    ]

    chunks, metadata = text_c.chunk_narratives(sample_df, mock_splitter)

    assert len(chunks) == 4
    assert metadata[0]["chunk_id"] == "0_0"
    assert metadata[1]["chunk_id"] == "0_1"
    assert metadata[2]["chunk_id"] == "0_2"
    assert metadata[3]["chunk_id"] == "1_0"


