import pandas as pd
import os
from loggers import get_logger
import re
import numpy as np

logger = get_logger(__name__)

# data_processing.py
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
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,!? ]", "", text)
    return text.strip()


def clean_narrative(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, keep alphanumeric and basic punctuation
    text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
    # Remove boilerplate phrases (example patterns)
    boilerplate_patterns = [
        r'i am writing to file a complaint',
        r'please help me with this issue',
        r'this is regarding a complaint'
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and clean necessary columns.
    """
    required_columns = [
        "Consumer complaint narrative",
        "Product",
        "Issue",
        "Company",
        "Date received"
    ]
    chunk = chunk[required_columns]
    # chunk = chunk.dropna(subset=["Consumer complaint narrative"])
    # chunk["Cleaned_Narrative"] = chunk["Consumer complaint narrative"].apply(clean_text)
    return chunk


def random_sample_large_csv(input_path: str, output_path: str, chunk_size: int = 50_000, target_rows: int = 1_000_000):
    """
    Randomly sample 1000,000 rows from a large CSV using memory-safe chunking.
    """
    sampled_rows = []
    total_sampled = 0

    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size, encoding='utf-8', on_bad_lines='skip')):
        logger.info(f"🔄 Reading chunk {i + 1}")

        processed = process_chunk(chunk)

        # Determine how many rows to sample from this chunk
        remaining = target_rows - total_sampled
        if remaining <= 0:
            break

        # Sample proportionally based on available rows
        sample_n = min(len(processed), remaining)
        sampled = processed.sample(n=sample_n, random_state=42)

        sampled_rows.append(sampled)
        total_sampled += len(sampled)

        logger.info(
            f"✅ Sampled {len(sampled)} rows from chunk {i + 1} (Total: {total_sampled})")

        if total_sampled >= target_rows:
            logger.info(f"🎯 Reached 1,0000,000 sampled rows. Stopping.")
            break

    # Combine all sampled data
    final_df = pd.concat(sampled_rows, ignore_index=True)
    final_df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(
        f"✅ Final sampled dataset saved to {output_path} with {len(final_df)} rows")
