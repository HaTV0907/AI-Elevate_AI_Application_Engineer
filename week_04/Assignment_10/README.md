# 🧠 Semantic Product Search with Pinecone + Azure OpenAI

This project demonstrates how to build a semantic product search engine using vector embeddings and Pinecone’s similarity search.

## 🎯 Objective

- Store product embeddings using Azure OpenAI
- Perform similarity queries using Pinecone
- Retrieve top 3 most relevant products for a given query

## 🚀 How to Run
### 1. Install requirements
    pip install -r requirements.txt
### 2, Create .env file
ensure your .env file contain below information with your real key
AZURE_OPENAI_ENDPOINT=https://aiportalapi.stu-platform.live/jpe
AZURE_OPENAI_API_KEY=sk-
AZURE_DEPLOYMENT_NAME=text-embedding-3-small
PINECONE_API_KEY=pcsk_7MzeDz_3u4XbMQs3k3
### 3. Run scripts
    python main.py