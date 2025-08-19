# test_data_processing.py

import pytest
import pandas as pd
import os
import re
from unittest.mock import MagicMock, patch
from io import StringIO

import sys


# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.data_process as dp


@pytest.fixture(autouse=True)
def mock_logger(monkeypatch):
    """Automatically mock logger in all tests."""
    fake_logger = MagicMock()
    monkeypatch.setattr(dp, "logger", fake_logger)
    return fake_logger


def test_data_loader_success(tmp_path, mock_logger):
    # Create a temporary CSV file
    csv_content = "col1,col2\n1,hello\n2,world\n"
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)

    df = dp.data_loader(csv_file)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    mock_logger.info.assert_called_once()


def test_data_loader_failure(mock_logger):
    with pytest.raises(Exception):
        dp.data_loader("non_existent.csv")
    mock_logger.error.assert_called_once()


@pytest.mark.parametrize("raw,expected", [
    ("Hello WORLD!!", "hello world"),   # no !!
    ("   Multiple    spaces   ", "multiple spaces"),
    ("Symbols *&^%$#", "symbols"),
    ("Mixed 123 text.", "mixed 123 text"),  # no final period
])
def test_clean_text(raw, expected):
    assert dp.clean_text(raw) == expected


def test_clean_narrative_removes_boilerplate():
    raw = "I am writing to file a complaint. Please help me with this issue!"
    cleaned = dp.clean_narrative(raw)
    assert "complaint" not in cleaned
    assert "please help" not in cleaned
    assert cleaned.startswith("i am") is False


def test_process_chunk_filters_columns():
    df = pd.DataFrame({
        "Consumer complaint narrative": ["text"],
        "Product": ["prod"],
        "Issue": ["issue"],
        "Company": ["company"],
        "Date received": ["2020-01-01"],
        "Extra": ["dropme"]
    })
    result = dp.process_chunk(df)
    assert list(result.columns) == [
        "Consumer complaint narrative", "Product", "Issue", "Company", "Date received"
    ]


def test_random_sample_large_csv(tmp_path, mock_logger):
    # Create a CSV with 100 rows
    df = pd.DataFrame({
        "Consumer complaint narrative": [f"text{i}" for i in range(100)],
        "Product": ["prod"] * 100,
        "Issue": ["issue"] * 100,
        "Company": ["company"] * 100,
        "Date received": ["2020-01-01"] * 100,
    })

    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    df.to_csv(input_file, index=False)

    # Run sampler with small target
    dp.random_sample_large_csv(
        input_path=input_file,
        output_path=output_file,
        chunk_size=50,
        target_rows=20
    )

    assert output_file.exists()
    sampled = pd.read_csv(output_file)
    assert sampled.shape[0] == 20
    mock_logger.info.assert_called()