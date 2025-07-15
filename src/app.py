import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2
import faiss
import pickle
from pathlib import Path
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
import torch
from loggers import get_logger



# Set up logging
logger = get_logger(__name__)

# Set up directories
VECTOR_STORE_DIR = Path('vector_store')

# Streamlit app configuration
st.set_page_config(page_title="CrediTrust Complaint Analysis", layout="wide")
st.title("CrediTrust Complaint Analysis Chatbot")
st.markdown("Ask questions about customer complaints, and I'll provide answers based on our complaint database. Sources are shown for transparency.")

# Initialize session state for conversation history and input
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'query' not in st.session_state:
    st.session_state.query = ""

# Load vector store and metadata
@st.cache_resource
def load_vector_store():
    try:
        index = faiss.read_index(str(VECTOR_STORE_DIR / 'faiss_index.faiss'))
        with open(VECTOR_STORE_DIR / 'metadata.pkl', 'rb') as f:
            store_data = pickle.load(f)
            chunks = store_data['chunks']
            metadata = store_data['metadata']
        logger.info("Vector store and metadata loaded successfully.")
        return index, chunks, metadata
    except FileNotFoundError as e:
        logger.error(f"Vector store files not found: {e}")
        st.error("Vector store files not found. Please ensure 'vector_store/faiss_index.bin' and 'vector_store/metadata.pkl' exist.")
        raise
    except Exception as e:
        logger.error(f"Error loading vector store or metadata: {e}")
        st.error(f"Error loading vector store: {e}")
        raise

# Initialize embedding model
@st.cache_resource
def load_embedding_model():
    try:
        logger.info("Loading embedding model 'sentence-transformers/all-MiniLM-L6-v2'...")
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        if embedding_model is None:
            raise ValueError("Embedding model is None after initialization.")
        test_embedding = embedding_model.encode(["test sentence"])
        logger.info(f"Embedding model loaded successfully. Test embedding shape: {test_embedding.shape}")
        return embedding_model
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        st.error(f"Failed to load embedding model: {e}")
        raise

# Initialize LLM
@st.cache_resource
def load_llm():
    try:
        logger.info("Loading LLM 'flan-t5-base'...")
        llm_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=0 if torch.cuda.is_available() else -1,
            max_length=200,
            do_sample=True,
            temperature=0.7,
        )
        test_output = llm_pipeline("Test input")[0]['generated_text']
        logger.info(f"LLM (flan-t5-base) loaded successfully. Test output: {test_output[:50]}...")
        return HuggingFacePipeline(pipeline=llm_pipeline)
    except Exception as e:
        logger.error(f"Failed to load LLM 't5-small': {e}")
        st.error(f"Failed to load LLM: {e}")
        raise

# Define prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints based solely on the provided context. If the context doesn't contain enough information to answer the question, state clearly: "Insufficient information in the provided context to fully answer the question." Provide a concise and accurate response.

Context: {context}

Question: {question}

Answer:
"""
)

# Retriever function
def retrieve_chunks(query, embedding_model, index, chunks, metadata, k=5):
    try:
        if not query or not isinstance(query, str):
            raise ValueError(f"Invalid query: {query}")
        if embedding_model is None:
            raise ValueError("Embedding model is None in retrieve_chunks.")
        logger.info(f"Encoding query: {query}")
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
def rag_pipeline(query, embedding_model, index, chunks, metadata, llm):
    try:
        retrieved_chunks = retrieve_chunks(query, embedding_model, index, chunks, metadata, k=5)
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

# Load resources
try:
    index, chunks, metadata = load_vector_store()
    embedding_model = load_embedding_model()
    llm = load_llm()
except Exception as e:
    st.error("Failed to initialize the chatbot. Please check logs and ensure all dependencies are installed.")
    st.stop()

# Streamlit UI
with st.form(key="query_form"):
    query = st.text_input("Enter your question about customer complaints:", value=st.session_state.query)
    col1, col2 = st.columns([1, 1])
    with col1:
        submit_button = st.form_submit_button("Submit")
    with col2:
        clear_button = st.form_submit_button("Clear")

# Handle submit
if submit_button and query:
    with st.spinner("Generating answer..."):
        try:
            result = rag_pipeline(query, embedding_model, index, chunks, metadata, llm)
            st.session_state.conversation.append({
                'question': query,
                'answer': result['answer'],
                'sources': result['retrieved_chunks']
            })
            st.session_state.query = ""  # Clear input after submission
        except Exception as e:
            st.error(f"Error processing query: {e}")
            logger.error(f"Error processing query '{query}': {e}")

# Handle clear
if clear_button:
    st.session_state.conversation = []
    st.session_state.query = ""
    st.experimental_rerun()

# Display conversation history
if st.session_state.conversation:
    st.markdown("### Conversation History")
    for i, entry in enumerate(reversed(st.session_state.conversation)):  # Show newest first
        with st.expander(f"Q: {entry['question']}", expanded=(i == 0)):
            st.markdown(f"**Answer**: {entry['answer']}")
            st.markdown("**Sources**:")
            source_data = [
                {
                    "Chunk ID": chunk['metadata']['chunk_id'],
                    "Product": chunk['metadata']['product'],
                    "Text": chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text']
                }
                for chunk in entry['sources']
            ]
            st.table(source_data)
else:
    st.info("Ask a question to start the conversation!")