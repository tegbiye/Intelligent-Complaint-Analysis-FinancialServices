# Intelligent Complaint Analysis for Financial Services

## Building a RAG-Powered Chatbot to Turn Customer Feedback into Actionable Insights

## Task-1: Exploratory Data Analysis and Data Preprocessing

#### EDA and Preprocessing Findings

✅ The exploratory data analysis of the CFPB complaint dataset reveals key insights into its structure and content.The dataset contains a diverse set of complaints across multiple financial products, with the initial analysis showing the distribution of complaints across products, highlighting which financial products (e.g., Credit card, Personal loan) receive the most complaints.

✅ The narrative length analysis indicates a wide range of complaint lengths, with some narratives being very short (<10 words) and others excessively long (>500 words), suggesting variability in consumer detail.

✅ Complaints without narratives were identified, and these were excluded from the final dataset to ensure quality for the RAG pipeline.

✅ After filtering for the specified products (Credit card, Personal loan, Buy Now, Pay Later, Savings account, Money transfers) and removing records with empty narratives, the dataset was significantly reduced in size, ensuring relevance and usability.

✅ Text cleaning involved lowercasing, removing special characters, and eliminating common boilerplate phrases to improve embedding quality.

✅ The cleaned dataset, saved as **filtered_complaints.csv**, retains essential metadata and cleaned narratives, making it suitable for downstream tasks like embedding and retrieval.

✅ The preprocessing steps ensure that the narratives are standardized and free of noise, enhancing the performance of the chatbot in answering queries based on real-world feedback.

## Task-2 Text Chunking, Embedding, and Vector Store Indexing

#### Report Section: Chunking Strategy and Embedding Model Choice

✅ For the text chunking strategy, I utilized LangChain's RecursiveCharacterTextSplitter with a chunk_size of 500 characters and a chunk_overlap of 50 characters.

✅ The chunk size was chosen to balance capturing coherent segments of complaint narratives (approximately 100-150 words) while ensuring embeddings remain semantically meaningful. Complaints often contain distinct issues (e.g., billing disputes, customer service issues), and smaller chunks help isolate these for better retrieval precision.

✅ The overlap of 50 characters maintains context across chunk boundaries, especially for narratives split mid-sentence. I experimented with larger chunk sizes (e.g., 1000 characters), but they risked diluting specific issues in longer narratives, while smaller chunks (e.g., 200 characters) fragmented context excessively. The chosen parameters were validated by inspecting sample chunks, ensuring they retained meaningful complaint details.

✅ The sentence-transformers/all-MiniLM-L6-v2 model was selected for embedding due to its efficiency and performance in semantic similarity tasks. This lightweight model (22M parameters, 384-dimensional embeddings) is optimized for short text, making it ideal for complaint narratives, which are typically concise yet descriptive. It provides a good balance between embedding quality and computational efficiency, suitable for indexing large datasets like the CFPB complaints. The model’s pre-training on diverse datasets ensures robust handling of financial terminology and consumer language.

✅ FAISS was chosen for the vector store due to its speed and scalability for similarity search, with metadata (complaint ID, product, chunk ID) stored alongside each embedding to enable traceability to the original complaint. The vector store and metadata are persisted in the vector_store/ directory for downstream retrieval tasks.

## Task 3: Building the RAG Core Logic and Evaluation

#### Deliverables

    1. Python Module (rag_pipeline.py): 
     - The script successfully produced the evaluation table
    2. Evaluation Table (evaluation_table.md)
     - The table (provided in the document) contains answers and sources for five questions, with a quality score  
       and a comment to “Review answer and sources for accuracy and relevance.”
## Project Structure

<pre>
Intelligent-Complaint-Analysis-Financial Services/
├── .github/workflows/ci.yml   # For CI/CD
├── data/                       # add this folder to .gitignore
|   ├── evaluation_table.md     # evaluation table generated 
│   ├── raw/                   # Raw data goes here 
│   └── processed/             # Processed data for training
├── vectore_store/
|   ├── sample_chunks.csv    # Sample chunks for verification
|   ├── metadata.pkl         # chunks metadata
|   ├── faiss_index.faiss         #  FAISS index
|   └── faiss_index.bin      # FAISS index
├── models/                  # Saved embedding model
├── notebooks/
|   ├── README.md
|   ├── RAG-pipeline.ipynb           # Pipeline notebook
|   ├── text_chunk.ipynb             # Text chunk, embedding notebook
│   └── complaints-EDA.ipynb          # Exploratory, one-off analysis
├── src/
│   ├── __init__.py
|   ├── rag_helper.py     # helper script gopt the rag
│   ├── data_process.py     # Script for Data Processing (EDA)
|   ├── text_chunker.py     # Helper function for the text chunking
│   └── loggers.py    # logging to the files and output
├── tests/
|   ├── __init__.py
|   ├── test_chunk_narratives.py
|   ├── test_retrieval_rag.py
│   └── test_sample.py         # Unit tests
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
</pre>

## Getting Started

Clone the repository

`git clone http://github.com/tegbiye/Intelligent-Complaint-Analysis-FinancialServices.git`

`cd Intelligent-Complaint-Analysis-FinancialServices`

Create environment using venv

`python -m venv .venv`

Activate the environment

`.venv\Scripts\activate` (Windows)

`source .venv\bin\activate` (Linux / Mac)

Install Dependencies

`pip install -r requirements.txt`

📜 License This project is licensed under the MIT License. Feel free to use, modify, and distribute with proper attribution.
