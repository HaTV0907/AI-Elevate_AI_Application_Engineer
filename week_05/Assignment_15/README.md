# 🛍️ Walmart RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain and Azure OpenAI to answer Walmart product and policy questions.

## 🚀 Features

- Retrieves relevant policy documents using semantic search
- Generates human-friendly answers using GPT-4o-mini
- Uses FAISS vector store and in-memory docstore
- Auto Q&A demo with realistic questions

## 📦 Setup

### 1. create .env file and add below content
AZURE_OPENAI_EMBEDDING_API_KEY=sk-
AZURE_OPENAI_EMBEDDING_ENDPOINT=https://aiportalapi.stu-platform.live/jpe
AZURE_OPENAI_EMBED_MODEL=text-embedding-3-small

AZURE_OPENAI_LLM_API_KEY=sk-
AZURE_OPENAI_LLM_ENDPOINT=https://aiportalapi.stu-platform.live/jpe
AZURE_OPENAI_LLM_MODEL=GPT-4o-mini

* remember to use your real keys
### 2. install required packages
    pip install -r requirements.txt
### 3. run script
    python walmart_rag_chatbot.py