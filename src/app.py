import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2
import faiss
import pickle
from pathlib import Path
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
import torch
import sys
import os
from functools import lru_cache
import io

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.loggers import get_logger

# Set up logging
logger = get_logger(__name__)
logger.info("Starting CrediTrust Complaint Analysis Chatbot")

# Constants
VECTOR_STORE_DIR = Path('vector_store')
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL_NAME = 'google/flan-t5-base'
RETRIEVAL_K = 5
DISTANCE_THRESHOLD = 1.0  # Maximum FAISS distance for relevant chunks
MAX_CONTEXT_LENGTH = 1000  # Maximum characters for context

# Streamlit app configuration
st.set_page_config(
    page_title="CrediTrust Complaint Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Custom CSS for a fancy look with improved text visibility
st.markdown("""
<style>
    body {
        color: #2d3748 !important; /* Dark gray text for high contrast */
        font-family: 'Arial', sans-serif;
    }
    .main {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        color: #2d3748; /* Ensure text is visible on light background */
    }
    .welcome-message {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        color: #1a3c6e !important; /* Darker blue for welcome text */
        font-size: 16px;
        line-height: 1.5;
        margin-bottom: 20px;
    }
    .stTextInput > div > input {
        border: 2px solid #007bff;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
        color: #2d3748;
        background-color: #ffffff;
    }
    .stButton > button {
        background-color: #007bff;
        color: #ffffff;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    .stSpinner > div {
        color: #007bff;
    }
    .stExpander {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-bottom: 10px;
        color: #2d3748; /* Ensure text in expanders is visible */
    }
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        color: #2d3748;
    }
    .sidebar .stButton > button {
        width: 100%;
        margin-bottom: 10px;
    }
    .stProgress > div > div {
        background-color: #007bff;
    }
    h1, h2, h3 {
        color: #1a3c6e;
        font-family: 'Arial', sans-serif;
    }
    .stMarkdown {
        font-family: 'Arial', sans-serif;
        color: #2d3748 !important;
    }
    /* Ensure text in all Streamlit elements is visible */
    .stText, .stMarkdown, .stDataFrame, .stExpander, .stAlert {
        color: #2d3748 !important;
    }
    /* Fix for sidebar text visibility */
    .sidebar .sidebar-content {
        color: #2d3748 !important;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables for conversation, query, and form."""
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'query' not in st.session_state:
        st.session_state.query = ""
    if 'form_key' not in st.session_state:
        st.session_state.form_key = "query_form_0"
    if 'feedback' not in st.session_state:
        st.session_state.feedback = {}
    if 'progress' not in st.session_state:
        st.session_state.progress = 0
    if 'perform_clear' not in st.session_state:
        st.session_state.perform_clear = False

@st.cache_resource
def load_vector_store():
    """
    Load FAISS index and metadata from disk.

    Returns:
        tuple: (index, chunks, metadata) containing FAISS index, text chunks, and metadata.
    
    Raises:
        FileNotFoundError: If vector store files are missing.
        Exception: For other loading errors.
    """
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
        st.error("Vector store files not found. Ensure 'vector_store/faiss_index.faiss' and 'vector_store/metadata.pkl' exist.")
        raise
    except Exception as e:
        logger.error(f"Error loading vector store or metadata: {e}")
        st.error(f"Error loading vector store: {e}")
        raise

@st.cache_resource
def load_embedding_model():
    """
    Load the sentence transformer model for embeddings.

    Returns:
        SentenceTransformer: Loaded embedding model.
    
    Raises:
        Exception: If model loading fails.
    """
    try:
        logger.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        if embedding_model is None:
            raise ValueError("Embedding model is None after initialization.")
        test_embedding = embedding_model.encode(["test sentence"])
        logger.info(f"Embedding model loaded. Test embedding shape: {test_embedding.shape}")
        return embedding_model
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        st.error(f"Failed to load embedding model: {e}")
        raise

@st.cache_resource
def load_llm():
    """
    Load the language model pipeline.

    Returns:
        HuggingFacePipeline: Loaded LLM pipeline.
    
    Raises:
        Exception: If LLM loading fails.
    """
    try:
        logger.info(f"Loading LLM '{LLM_MODEL_NAME}'...")
        llm_pipeline = pipeline(
            "text2text-generation",
            model=LLM_MODEL_NAME,
            device=0 if torch.cuda.is_available() else -1,
            max_length=200,
            do_sample=True,
            temperature=0.7,
        )
        test_output = llm_pipeline("Test input")[0]['generated_text']
        logger.info(f"LLM loaded successfully. Test output: {test_output[:50]}...")
        return HuggingFacePipeline(pipeline=llm_pipeline)
    except Exception as e:
        logger.error(f"Failed to load LLM: {e}")
        st.error(f"Failed to load LLM: {e}")
        raise

def get_prompt_template():
    """
    Create and return the prompt template for the RAG pipeline.

    Returns:
        PromptTemplate: Configured prompt template.
    """
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a financial analyst assistant for CrediTrust. Provide a detailed and accurate answer to the question about customer complaints, using only the provided context. If the context is insufficient, state: "Insufficient information in the provided context to fully answer the question." Ensure the response is professional, concise, and directly addresses the question.

Context: {context}

Question: {question}

Answer:
"""
    )

@lru_cache(maxsize=100)
def encode_query(query: str, embedding_model):
    """
    Encode a query into an embedding vector with caching.

    Args:
        query (str): The query to encode.
        embedding_model: The SentenceTransformer model.

    Returns:
        np.ndarray: The query embedding.

    Raises:
        ValueError: If query is invalid or embedding model is None.
    """
    if not query or not isinstance(query, str):
        raise ValueError(f"Invalid query: {query}")
    if embedding_model is None:
        raise ValueError("Embedding model is None.")
    logger.info(f"Encoding query: {query}")
    return embedding_model.encode([query])[0]

def retrieve_chunks(query, embedding_model, index, chunks, metadata, k=RETRIEVAL_K):
    """
    Retrieve relevant chunks from the vector store based on the query.

    Args:
        query (str): The user query.
        embedding_model: The SentenceTransformer model.
        index: The FAISS index.
        chunks (list): List of text chunks.
        metadata (list): List of metadata dictionaries.
        k (int): Number of chunks to retrieve.

    Returns:
        list: List of dictionaries containing retrieved chunks and metadata.

    Raises:
        Exception: If retrieval fails.
    """
    try:
        query_embedding = encode_query(query, embedding_model)
        distances, indices = index.search(np.array([query_embedding]), k)
        retrieved_chunks = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(chunks) and dist < DISTANCE_THRESHOLD:
                retrieved_chunks.append({
                    'text': chunks[idx],
                    'metadata': metadata[idx],
                    'distance': float(dist)
                })
            else:
                logger.warning(f"Skipping chunk {idx} with distance {dist:.4f} (threshold: {DISTANCE_THRESHOLD})")
        if not retrieved_chunks:
            logger.warning(f"No relevant chunks retrieved for query: {query}")
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query: {query}")
        return retrieved_chunks
    except Exception as e:
        logger.error(f"Error retrieving chunks for query '{query}': {e}")
        raise

def rag_pipeline(query, embedding_model, index, chunks, metadata, llm):
    """
    Run the Retrieval-Augmented Generation pipeline.

    Args:
        query (str): The user query.
        embedding_model: The SentenceTransformer model.
        index: The FAISS index.
        chunks (list): List of text chunks.
        metadata (list): List of metadata dictionaries.
        llm: The language model pipeline.

    Returns:
        dict: Dictionary with answer and retrieved chunks.

    Raises:
        Exception: If the pipeline fails.
    """
    try:
        retrieved_chunks = retrieve_chunks(query, embedding_model, index, chunks, metadata)
        context = "\n".join([chunk['text'] for chunk in retrieved_chunks])
        if len(context) > MAX_CONTEXT_LENGTH:
            context = context[:MAX_CONTEXT_LENGTH]
            logger.info(f"Context trimmed to {MAX_CONTEXT_LENGTH} characters")
        prompt = get_prompt_template().format(context=context, question=query)
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

def export_conversation():
    """
    Export conversation history to a CSV file.

    Returns:
        bytes: CSV file content.
    """
    if not st.session_state.conversation:
        return None
    data = [
        {
            "Question": entry['question'],
            "Answer": entry['answer'],
            "Sources": "; ".join([f"Chunk {chunk['metadata']['chunk_id']} ({chunk['metadata']['product']}): {chunk['text'][:50]}..." for chunk in entry['sources']]),
            "Feedback": st.session_state.feedback.get(str(i), "None")
        }
        for i, entry in enumerate(st.session_state.conversation)
    ]
    df = pd.DataFrame(data)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode('utf-8')

def display_conversation_history():
    """Display the conversation history with feedback options."""
    if st.session_state.conversation:
        st.markdown("### 📜 Conversation History")
        for i, entry in enumerate(reversed(st.session_state.conversation)):
            with st.expander(f"Q: {entry['question']}", expanded=(i == 0)):
                st.markdown(f"**Answer**: {entry['answer']}")
                st.markdown("**Sources**:")
                source_data = [
                    {
                        "Chunk ID": chunk['metadata']['chunk_id'],
                        "Product": chunk['metadata']['product'],
                        "Text": chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text'],
                        "Distance": f"{chunk['distance']:.4f}"
                    }
                    for chunk in entry['sources']
                ]
                st.dataframe(pd.DataFrame(source_data), use_container_width=True)
                st.markdown("**Feedback**:")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Helpful", key=f"thumbs_up_{i}"):
                        st.session_state.feedback[str(i)] = "Helpful"
                        st.success("Thank you for your feedback!")
                        st.rerun()
                with col2:
                    if st.button("👎 Not Helpful", key=f"thumbs_down_{i}"):
                        st.session_state.feedback[str(i)] = "Not Helpful"
                        st.success("Thank you for your feedback!")
                        st.rerun()
                if str(i) in st.session_state.feedback:
                    st.markdown(f"**Your Feedback**: {st.session_state.feedback[str(i)]}")
    else:
        st.info("Ask a question to start the conversation! 💬")

def main():
    """Main function to run the Streamlit app."""
    # Initialize session state
    initialize_session_state()

    # Load resources
    try:
        index, chunks, metadata = load_vector_store()
        embedding_model = load_embedding_model()
        llm = load_llm()
    except Exception as e:
        st.error("Failed to initialize the chatbot. Please check logs and ensure all dependencies are installed. 🚨")
        st.stop()

    # Sidebar with Query Suggestions and FAQ
    with st.sidebar:
        st.header("🔍 Query Suggestions")
        suggestions = [
            "What are common issues with credit card complaints?",
            "How many complaints involve loan delays?",
            "What are customer sentiments about payment disputes?",
            "Are there trends in mortgage-related complaints?"
        ]
        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggestion_{suggestion}"):
                st.session_state.query = suggestion
                st.rerun()

        st.header("📋 FAQ")
        with st.expander("How does this chatbot work?"):
            st.markdown("This chatbot uses a Retrieval-Augmented Generation (RAG) pipeline to analyze customer complaints. It retrieves relevant data from a vector store and generates answers using a language model.")
        with st.expander("What data is used?"):
            st.markdown("The chatbot uses a pre-indexed dataset of customer complaints stored in a FAISS vector store, with embeddings generated by a sentence transformer model.")
        with st.expander("Can I export my conversation?"):
            st.markdown("Yes! Use the 'Download Conversation as CSV' button below to export your chat history.")

        st.header("💾 Export Conversation")
        csv_data = export_conversation()
        if csv_data:
            st.download_button(
                label="Download Conversation as CSV",
                data=csv_data,
                file_name="conversation_history.csv",
                mime="text/csv",
                help="Download your conversation history as a CSV file."
            )

    # UI Header
    st.title("📊 CrediTrust Complaint Analysis Chatbot")
    st.markdown("""
        <div class="welcome-message">
            Welcome to the CrediTrust Complaint Analysis Chatbot! Ask questions about customer complaints to gain insights from our database. 
            Sources are provided for transparency, and you can use the sidebar for query suggestions or to export your conversation history.
        </div>
    """, unsafe_allow_html=True)

    # Query Input Form
    form_key = st.session_state.form_key
    with st.form(key=form_key):
        query = st.text_input(
            "Enter your question about customer complaints:",
            value=st.session_state.query,
            key="query_text_input",
            placeholder="e.g., What are common issues with credit card complaints?",
            help="Type your question here and click Submit to get insights."
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            submit_button = st.form_submit_button("Submit 🚀")
        with col2:
            clear_button = st.form_submit_button("Clear 🗑️", help="Clears the input and conversation history")

    # Handle Clear Action from Form Submission
    if clear_button:
        # Instead of directly clearing, set a state flag to trigger confirmation
        st.session_state.perform_clear = True
        st.rerun()

    # Handle the clear action based on the state flag
    if st.session_state.perform_clear:
        # Ask for confirmation outside the form
        with st.container():
            st.warning("Are you sure you want to clear the conversation? This action cannot be undone.")
            col_conf1, col_conf2 = st.columns([1, 1])
            with col_conf1:
                if st.button("Yes, Clear", key="confirm_clear_yes"):
                    st.session_state.conversation = []
                    st.session_state.query = ""
                    st.session_state.feedback = {}
                    st.session_state.form_key = "query_form_0"
                    st.session_state.perform_clear = False
                    st.success("Conversation and input cleared! 🧹")
                    st.rerun()
            with col_conf2:
                if st.button("No, Cancel", key="confirm_clear_no"):
                    st.session_state.perform_clear = False
                    st.rerun()

    # Handle Submit
    if submit_button and query.strip():
        # Reset the clear flag if a new query is submitted
        st.session_state.perform_clear = False
        progress_bar = st.progress(0)
        with st.spinner("Analyzing complaint data..."):
            try:
                # Simulate progress
                for i in range(10, 100, 10):
                    st.session_state.progress = i / 100
                    progress_bar.progress(st.session_state.progress)
                    import time
                    time.sleep(0.1)  # Simulate processing time
                result = rag_pipeline(query.strip(), embedding_model, index, chunks, metadata, llm)
                st.session_state.conversation.append({
                    'question': query.strip(),
                    'answer': result['answer'],
                    'sources': result['retrieved_chunks']
                })
                st.session_state.query = ""
                st.session_state.form_key = f"query_form_{len(st.session_state.conversation)}"
                progress_bar.progress(1.0)
                st.success("Query processed successfully! 🎉")
                st.rerun()
            except Exception as e:
                progress_bar.progress(0)
                st.error(f"Error processing query: {e} 🚨")
                logger.error(f"Error processing query '{query}': {e}")
                
    # Display Conversation History
    display_conversation_history()

if __name__ == "__main__":
    main()