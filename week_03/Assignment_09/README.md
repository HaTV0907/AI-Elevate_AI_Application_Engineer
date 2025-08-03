# 💻 Laptop Consultant Chatbot

This project builds a simple AI chatbot using **Azure OpenAI** and **ChromaDB** that helps users find laptops suited to their needs. It uses embedding models for context retrieval and LLMs for generating responses.

## 🧠 Features

- Conversational chatbot that asks about user laptop needs.
- Embedding-based retrieval for similarity search using ChromaDB.
- Response generation via Azure OpenAI Chat Completion API.
- Operates entirely within a Jupyter notebook or Python script.

## 🚀 How to Run

1. Clone this repo or copy the scripts into a folder.
2. Install dependencies:

pip install -r requirements.txt

3. create .env file with below content

AZURE_OPENAI_EMBEDDING_ENDPOINT=https://aiportalapi.stu-platform.live/jpe
AZURE_OPENAI_EMBED_MODEL=text-embedding-3-small
AZURE_OPENAI_EMBEDDING_API_KEY=sk-

AZURE_OPENAI_LLM_ENDPOINT=https://aiportalapi.stu-platform.live/jpe
AZURE_OPENAI_LLM_MODEL=GPT-4o-mini
AZURE_OPENAI_LLM_API_KEY=sk-

don't forget to add your real key

4. run python script:

```bash
python chatbot.py