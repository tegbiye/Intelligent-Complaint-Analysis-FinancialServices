# Intelligent Complaint Analysis for Financial Services

## Building a RAG-Powered Chatbot to Turn Customer Feedback into Actionable Insights

## Task-1: Exploratory Data Analysis and Data Preprocessing

**EDA and Preprocessing Findings**
✅ The exploratory data analysis of the CFPB complaint dataset reveals key insights into its structure and content.
The dataset contains a diverse set of complaints across multiple financial products, with the initial analysis showing the distribution of complaints across products, highlighting which financial products (e.g., Credit card, Personal loan) receive the most complaints.
✅ The narrative length analysis indicates a wide range of complaint lengths, with some narratives being very short (<10 words) and others excessively long (>500 words), suggesting variability in consumer detail.
✅ Complaints without narratives were identified, and these were excluded from the final dataset to ensure quality for the RAG pipeline.

✅ After filtering for the specified products (Credit card, Personal loan, Buy Now, Pay Later, Savings account, Money transfers) and removing records with empty narratives, the dataset was significantly reduced in size, ensuring relevance and usability.
✅ Text cleaning involved lowercasing, removing special characters, and eliminating common boilerplate phrases to improve embedding quality.
✅ The cleaned dataset, saved as **filtered_complaints.csv**, retains essential metadata and cleaned narratives, making it suitable for downstream tasks like embedding and retrieval.
✅ The preprocessing steps ensure that the narratives are standardized and free of noise, enhancing the performance of the chatbot in answering queries based on real-world feedback.

## Project Structure

<pre>
Credit-Risk-Model-Automation/
├── .github/workflows/ci.yml   # For CI/CD
├── data/                       # add this folder to .gitignore
│   ├── raw/                   # Raw data goes here 
│   └── processed/             # Processed data for training
├── notebooks/
|   ├── README.md
│   └── complaints-EDA.ipynb          # Exploratory, one-off analysis
├── src/
│   ├── __init__.py
│   ├── data_process.py     # Script for Data Processing (EDA)
│   └── loggers.py    # logging to the files and output
├── tests/
|   ├── __init__.py
│   └── test_sample.py         # Unit tests
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
</pre>

Getting Started
Clone the repository
`git clone http://github.com/tegbiye/Intelligent-Complaint-Analysis-FinancialServices.git`

`cd Intelligent-Complaint-Analysis-FinancialServices`

Create environment using venv
`python -m venv .venv`

Activate the environment

`.venv\Scripts\activate`

`source .venv\bin\activate`

Install Dependencies
`pip install -r requirements.txt`

📜 License This project is licensed under the MIT License. Feel free to use, modify, and distribute with proper attribution.
