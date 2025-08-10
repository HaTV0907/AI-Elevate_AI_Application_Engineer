# 🏥 Healthcare RAG Chatbot (Pinecone Edition)

This project builds a Retrieval-Augmented Generation (RAG) chatbot for healthcare using Pinecone, Langchain, and Azure/OpenAI function calling.

## 🚀 Features
- Vector search with Pinecone
- Conversational retrieval with Langchain
- Function calling for dynamic responses
- Streamlit UI for interaction

## 🛠️ Setup

1. create .env file and fill these info into it
OPENAI_API_KEY=sk-
OPENAI_API_BASE=https://aiportalapi.stu-platform.live/jpe
OPENAI_API_VERSION=2024-07-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=GPT-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
PINECONE_API_KEY=
PINECONE_ENV=us-east-1
PINECONE_INDEX=healthcare-chatbot-index

don't forget to use your own key

2. install required packages
   pip install -r requirements.txt

3. run app
   streamlit run app.py