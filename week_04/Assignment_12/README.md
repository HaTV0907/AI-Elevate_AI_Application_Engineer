# ☁️ Satellite Image Cloud Detection via Azure OpenAI

## 🔍 Objective
Build a lightweight CLI tool that uses Azure OpenAI's GPT-4o to classify satellite images as **"Cloudy"** or **"Clear"** — no ML model training required.

## 🚀 Features
- Accepts satellite images via URL
- Uses GPT-4o for multimodal inference
- Returns label + confidence score
- Prints logs for 3 mock images

## 🧠 Concepts Covered
- Azure OpenAI API usage
- Multimodal image classification
- LangChain structured output
- Base64 image encoding

## 🛠️ Setup Instructions
1. Clone the repo
2. Create .env file
    ensure your .env file contain below information with your real key
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT=https://aiportalapi.stu-platform.live/jpe
    AZURE_OPENAI_DEPLOYMENT_NAME=GPT-4o-mini
    AZURE_OPENAI_API_KEY=sk-
    AZURE_OPENAI_API_VERSION=2024-07-01-preview
3. Install dependencies:
   pip install -r requirements.txt
4. Run script
   python cloud-detector.py

LangChain Docs Referenc
-https://python.langchain.com/docs/concepts/structured_outputs/
-https://python.langchain.com/docs/concepts/multimodality/

Learnings- GPT-4o can perform basic image classification tasks without custom models
- Base64 encoding is essential for passing images to LLMs
- LangChain simplifies structured output parsing
- Azure OpenAI enables scalable, low-code inference pipelines
