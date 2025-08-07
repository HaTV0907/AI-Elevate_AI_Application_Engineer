# 🌤️ Weather & Search AI Agent

This project builds a Langchain-based AI assistant that answers real-time weather and web search queries using OpenWeather and Tavily APIs.

## 🚀 Features

- Weather queries via OpenWeather API
- Web search via Tavily Search API
- Langchain agent with tool routing
- Simulated conversational loop in Jupyter Notebook

## 🧰 Requirements

- Python 3.8+
- Jupyter Notebook

## 🚀 How to Run
### 1. Install requirements
    pip install -r requirements.txt
### 2, Create .env file
    ensure your .env file contain below information with your real key
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT=https://aiportalapi.stu-platform.live/jpe
    AZURE_OPENAI_DEPLOYMENT_NAME=GPT-4o-mini
    AZURE_OPENAI_API_KEY=sk-
    AZURE_OPENAI_API_VERSION=2024-07-01-preview
    # OpenWeather
    OPENWEATHERMAP_API_KEY=

    # Tavily
    TAVILY_API_KEY=tvly-dev-
### 3. Run scripts
    python agent.py