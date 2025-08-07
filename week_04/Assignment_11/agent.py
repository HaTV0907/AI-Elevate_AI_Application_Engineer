import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Load environment variables
load_dotenv()

# Weather tool
weather = OpenWeatherMapAPIWrapper()

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    print(f"Calling get_weather tool: Getting weather for {city}")
    return weather.run(city)

# Tavily search tool
tavily_search_tool = TavilySearch(
    max_results=1,
    topic="general",
)

# Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# Langchain agent
tools = [get_weather, tavily_search_tool]
agent = create_react_agent(model=llm, tools=tools)

# Simulated conversation loop
print("Welcome to the AI assistant. Type 'exit' to stop.")
messages = []

mock_questions = [
    "What's the weather in Hanoi?",
    "Tell me about the latest news in AI.",
    "Who won the last World Cup?",
    "exit",
]

for user_input in mock_questions:
    print("User:", user_input)
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    messages.append({"role": "user", "content": user_input})
    try:
        response = agent.invoke({"messages": messages})
        messages.append({"role": "assistant", "content": response["messages"][-1].content})
        print("AI:", response["messages"][-1].content)
    except Exception as e:
        print("Error during agent invocation:", e)
