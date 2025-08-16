# 🩺 Patient Information Collection & Advisory Chatbot

This project builds an AI-powered chatbot agent that interactively collects patient information and provides preliminary health advice using AzureChatOpenAI and LangGraph.

## 🚀 Features

- Collects patient details (e.g., symptoms)
- Provides health advice from internal knowledge base
- Optionally fetches real-time info via Tavily
- Uses LangGraph to manage conversation flow

## 🛠️ Setup

1. create .env file and add below content
AZURE_OPENAI_EMBEDDING_API_KEY=sk-
AZURE_OPENAI_EMBEDDING_ENDPOINT=https://aiportalapi.stu-platform.live/jpe
AZURE_OPENAI_EMBED_MODEL=text-embedding-3-small

AZURE_OPENAI_LLM_API_KEY=sk-
AZURE_OPENAI_LLM_ENDPOINT=https://aiportalapi.stu-platform.live/jpe
AZURE_OPENAI_LLM_MODEL=GPT-4o-mini

TAVILY_API_KEY=tvly-dev-

Don't forget to use your real keys

2. install required packages
    pip install -r requirements.txt

3. run scripts
    python main.py

