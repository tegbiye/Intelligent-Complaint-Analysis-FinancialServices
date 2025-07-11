import pandas as pd
import numpy as np
from langchain import text_splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2
import pickle
from pathlib import Path
import uuid

# Function to chunk narratives and preserve metadata


def chunk_narratives(df, text_splitter):
    chunks = []
    metadata = []
    for idx, row in df.iterrows():
        complaint_id = idx  # Using index as complaint ID; adjust if dataset has specific ID column
        product = row['Product']
        narrative = row['cleaned_narrative']
        chunked_texts = text_splitter.split_text(narrative)
        for i, chunk in enumerate(chunked_texts):
            chunks.append(chunk)
            metadata.append({
                'complaint_id': complaint_id,
                'product': product,
                'chunk_id': f"{complaint_id}_{i}"
            })
    return chunks, metadata
