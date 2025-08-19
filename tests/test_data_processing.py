import pytest
import pandas as pd
import numpy as np
import re
from unittest.mock import patch, MagicMock
from src.data_process import data_loader, clean_text, clean_narrative, process_chunk, random_sample_large_csv

# Fixture for sample DataFrame
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Consumer complaint narrative": ["Test complaint 1!", "Test complaint 2 with special chars @#$", None],
        "Product": ["Credit card", "Mortgage", "Loan"],
        "Issue": ["Billing issue", "Payment issue", "Other"],
        "Company": ["Company A", "Company B", "Company C"],
        "Date received": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "Other column": [1, 2, 3]
    })

# Test data_loader
@patch("pandas.read_csv")
@patch("src.data_process.logger")
def test_data_loader_success(mock_logger, mock_read_csv):
    # Arrange
    mock_df = pd.DataFrame({"col1": [1, 2, 3]})
    mock_read_csv.return_value = mock_df
    file_path = "test.csv"

    # Act
    result = data_loader(file_path)

    # Assert
    mock_read_csv.assert_called_once_with(file_path, low_memory=True)
    mock_logger.info.assert_called_once_with(f"Data loaded successfully from {file_path}")
    pd.testing.assert_frame_equal(result, mock_df)

@patch("pandas.read_csv")
@patch("src.data_process.logger")
def test_data_loader_file_not_found(mock_logger, mock_read_csv):
    # Arrange
    mock_read_csv.side_effect = FileNotFoundError("File not found")
    file_path = "nonexistent.csv"

    # Act/Assert
    with pytest.raises(FileNotFoundError):
        data_loader(file_path)
    mock_logger.error.assert_called_once_with(f"Error loading data from {file_path}: File not found")

# Test clean_text
def test_clean_text():
    # Arrange
    input_text = "  Hello, World! @#$  Extra   Spaces  "
    expected = "hello world extra spaces"

    # Act
    result = clean_text(input_text)

    # Assert
    assert result == expected

def test_clean_text_empty():
    # Arrange
    input_text = ""
    expected = ""

    # Act
    result = clean_text(input_text)

    # Assert
    assert result == expected

def test_clean_text_non_string():
    # Arrange
    input_text = None
    expected = ""

    # Act
    result = clean_text(input_text)

    # Assert
    assert result == expected

# Test clean_narrative
def test_clean_narrative():
    # Arrange
    input_text = "I am writing to file a complaint. This is a TEST complaint! @#$ Extra  spaces."
    expected = "this is a test complaint extra spaces"

    # Act
    result = clean_narrative(input_text)

    # Assert
    assert result == expected

def test_clean_narrative_no_boilerplate():
    # Arrange
    input_text = "This is a simple complaint without boilerplate."
    expected = "this is a simple complaint without boilerplate"

    # Act
    result = clean_narrative(input_text)

    # Assert
    assert result == expected

def test_clean_narrative_empty():
    # Arrange
    input_text = ""
    expected = ""

    # Act
    result = clean_narrative(input_text)

    # Assert
    assert result == expected

def test_clean_narrative_non_string():
    # Arrange
    input_text = None
    expected = ""

    # Act
    result = clean_narrative(input_text)

    # Assert
    assert result == expected

# Test process_chunk
def test_process_chunk(sample_df):
    # Arrange
    expected_columns = ["Consumer complaint narrative", "Product", "Issue", "Company", "Date received"]

    # Act
    result = process_chunk(sample_df)

    # Assert
    assert list(result.columns) == expected_columns
    pd.testing.assert_frame_equal(result, sample_df[expected_columns])

def test_process_chunk_missing_columns():
    # Arrange
    df = pd.DataFrame({"Product": ["Credit card"], "Issue": ["Billing issue"]})

    # Act/Assert
    with pytest.raises(KeyError, match=r"Missing required columns:.*"):
        process_chunk(df)

# Test random_sample_large_csv
@patch("pandas.read_csv")
@patch("pandas.DataFrame.to_csv")
@patch("src.data_process.logger")
@patch("src.data_process.process_chunk")
def test_random_sample_large_csv(mock_process_chunk, mock_logger, mock_to_csv, mock_read_csv):
    # Arrange
    input_path = "input.csv"
    output_path = "output.csv"
    chunk_size = 2
    target_rows = 4
    mock_chunk1 = pd.DataFrame({
        "Consumer complaint narrative": ["Complaint 1", "Complaint 2"],
        "Product": ["Product A", "Product B"],
        "Issue": ["Issue A", "Issue B"],
        "Company": ["Company A", "Company B"],
        "Date received": ["2023-01-01", "2023-01-02"]
    })
    mock_chunk2 = pd.DataFrame({
        "Consumer complaint narrative": ["Complaint 3", "Complaint 4"],
        "Product": ["Product C", "Product D"],
        "Issue": ["Issue C", "Issue D"],
        "Company": ["Company C", "Company D"],
        "Date received": ["2023-01-03", "2023-01-04"]
    })
    mock_read_csv.return_value = iter([mock_chunk1, mock_chunk2])
    mock_process_chunk.side_effect = [mock_chunk1, mock_chunk2]

    # Act
    random_sample_large_csv(input_path, output_path, chunk_size=chunk_size, target_rows=target_rows)

    # Assert
    mock_read_csv.assert_called_once_with(input_path, chunksize=chunk_size, encoding='utf-8', on_bad_lines='skip')
    assert mock_process_chunk.call_count == 2
    mock_to_csv.assert_called_once_with(output_path, index=False, encoding='utf-8')
    mock_logger.info.assert_any_call(f"✅ Final sampled dataset saved to {output_path} with 4 rows")

@patch("pandas.read_csv")
@patch("pandas.DataFrame.to_csv")
@patch("src.data_process.logger")
def test_random_sample_large_csv_empty_file(mock_logger, mock_to_csv, mock_read_csv):
    # Arrange
    input_path = "input.csv"
    output_path = "output.csv"
    mock_read_csv.return_value = iter([])
    expected_columns = ["Consumer complaint narrative", "Product", "Issue", "Company", "Date received"]
    expected_df = pd.DataFrame(columns=expected_columns)

    # Act
    random_sample_large_csv(input_path, output_path)

    # Assert
    mock_read_csv.assert_called_once_with(input_path, chunksize=50_000, encoding='utf-8', on_bad_lines='skip')
    mock_to_csv.assert_called_once_with(output_path, index=False, encoding='utf-8')
    mock_logger.warning.assert_called_once_with("No data sampled from input file.")
    mock_logger.info.assert_called_once_with(f"✅ Final sampled dataset saved to {output_path} with 0 rows")