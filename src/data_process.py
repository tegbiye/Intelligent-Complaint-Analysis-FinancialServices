import pandas as pd
import os
from src.loggers import get_logger
import re
import numpy as np

logger = get_logger(__name__)

# data_process.py

# This module provides functions to load and process data for analysis.

def data_loader(file):
    """
    Load data from a CSV file and return a DataFrame.

    Args:
        file (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    try:
        df = pd.read_csv(file, low_memory=True)
        logger.info(f"Data loaded successfully from {file}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file}: {e}")
        raise

def clean_text(text: str) -> str:
    """
    Clean text by converting to lowercase, removing special characters, and normalizing spaces.

    Args:
        text (str): Input text to clean.

    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove all punctuation except spaces
    text = re.sub(r"\s+", " ", text)  # Normalize spaces
    return text.strip()

def clean_narrative(text):
    """
    Clean narrative text by removing boilerplate phrases and special characters.

    Args:
        text (str): Input narrative text.

    Returns:
        str: Cleaned narrative text.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    boilerplate_patterns = [
        r'i am writing to file a complaint',
        r'please help me with this issue',
        r'this is regarding a complaint'
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove all punctuation
    text = re.sub(r"\s+", " ", text)  # Normalize spaces
    return text.strip()

def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and clean necessary columns.

    Args:
        chunk (pd.DataFrame): Input DataFrame chunk.

    Returns:
        pd.DataFrame: Processed DataFrame with required columns.
    """
    required_columns = [
        "Consumer complaint narrative",
        "Product",
        "Issue",
        "Company",
        "Date received"
    ]
    missing_columns = [col for col in required_columns if col not in chunk.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    chunk = chunk[required_columns]
    return chunk

def random_sample_large_csv(input_path: str, output_path: str, chunk_size: int = 50_000, target_rows: int = 1_000_000):
    """
    Randomly sample rows from a large CSV using memory-safe chunking.

    Args:
        input_path (str): Path to input CSV.
        output_path (str): Path to save sampled CSV.
        chunk_size (int): Number of rows per chunk.
        target_rows (int): Target number of rows to sample.
    """
    sampled_rows = []
    total_sampled = 0

    try:
        for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size, encoding='utf-8', on_bad_lines='skip')):
            logger.info(f"🔄 Reading chunk {i + 1}")
            processed = process_chunk(chunk)

            remaining = target_rows - total_sampled
            if remaining <= 0:
                break

            sample_n = min(len(processed), remaining)
            if sample_n > 0:
                sampled = processed.sample(n=sample_n, random_state=42)
                sampled_rows.append(sampled)
                total_sampled += len(sampled)
                logger.info(f"✅ Sampled {len(sampled)} rows from chunk {i + 1} (Total: {total_sampled})")

            if total_sampled >= target_rows:
                logger.info(f"🎯 Reached {target_rows:,} sampled rows. Stopping.")
                break

        if not sampled_rows:
            logger.warning("No data sampled from input file.")
            final_df = pd.DataFrame(columns=[
                "Consumer complaint narrative", "Product", "Issue", "Company", "Date received"
            ])
        else:
            final_df = pd.concat(sampled_rows, ignore_index=True)

        final_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"✅ Final sampled dataset saved to {output_path} with {len(final_df)} rows")
    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")
        raise