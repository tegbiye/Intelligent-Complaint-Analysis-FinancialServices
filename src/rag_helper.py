import numpy as np
import os
import sys

from loggers import get_logger

logger = get_logger(__name__)


# Retriever function
def retrieve_chunks(query, k=5, embedding_model=None, index=None, chunks=None, metadata=None):
    try:
        if not query or not isinstance(query, str):
            raise ValueError(f"Invalid query: {query}")
        query_embedding = embedding_model.encode([query])[0]
        distances, indices = index.search(np.array([query_embedding]), k)
        retrieved_chunks = []
        for idx in indices[0]:
            if idx < len(chunks):
                chunk_info = {
                    'text': chunks[idx],
                    'metadata': metadata[idx]
                }
                retrieved_chunks.append(chunk_info)
            else:
                logger.warning(f"Invalid index {idx} retrieved, skipping.")
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query: {query}")
        return retrieved_chunks
    except Exception as e:
        logger.error(f"Error in retrieving chunks for query '{query}': {e}")
        raise

# RAG pipeline
def rag_pipeline(query, llm, embedding_model, index, chunks, metadata, prompt_template):
    try:
        retrieved_chunks = retrieve_chunks(
            query=query,
            k=5,
            embedding_model=embedding_model,
            index=index,
            chunks=chunks,
            metadata=metadata
        )
        context = "\n".join([chunk['text'] for chunk in retrieved_chunks])
        prompt = prompt_template.format(context=context, question=query)
        response = llm(prompt)
        answer = response.strip()
        logger.info(f"Generated answer for query: {query}")
        return {
            'answer': answer,
            'retrieved_chunks': retrieved_chunks
        }
    except Exception as e:
        logger.error(f"Error in RAG pipeline for query '{query}': {e}")
        raise
