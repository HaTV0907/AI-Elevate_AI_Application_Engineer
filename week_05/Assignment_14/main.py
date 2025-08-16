import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()

# --- Mock Medical Advice Documents ---
mock_chunks = [
    Document(page_content="Patients with a sore throat should drink warm fluids and avoid cold beverages."),
    Document(page_content="Mild fevers under 38.5°C can often be managed with rest and hydration."),
    Document(page_content="If a patient reports dizziness, advise checking their blood pressure and hydration level."),
    Document(page_content="Persistent coughs lasting more than 2 weeks should be evaluated for infections or allergies."),
    Document(page_content="Patients experiencing fatigue should consider iron deficiency or poor sleep as potential causes."),
]

# --- Embedding Setup ---
embedding_model = AzureOpenAIEmbeddings(
    model=os.getenv("AZURE_OPENAI_EMBED_MODEL"),
    api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    api_version="2024-02-15-preview",
)

db = FAISS.from_documents(mock_chunks, embedding_model)
retriever = db.as_retriever()

# --- Tool 1: Internal Advice Retrieval ---
@tool
def retrieve_advice(user_input: str) -> str:
    """Searches internal documents for relevant patient advice."""
    docs = retriever.invoke(user_input)
    return "\n".join(doc.page_content for doc in docs)

# --- Tool 2: Tavily Web Search ---
tavily_tool = TavilySearch(api_key=os.getenv("TAVILY_API_KEY"))

# --- LLM Setup ---
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_LLM_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_LLM_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_LLM_MODEL"),
    api_version="2024-02-15-preview",
)

llm_with_tools = llm.bind_tools([retrieve_advice, tavily_tool])

# --- Model Node ---
def call_model(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# --- Conditional Routing ---
def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# --- Build Graph ---
tool_node = ToolNode([retrieve_advice, tavily_tool])
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("call_model", call_model)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "call_model")
graph_builder.add_conditional_edges("call_model", should_continue, ["tools", END])
graph_builder.add_edge("tools", "call_model")
graph = graph_builder.compile()

# --- Dummy Inputs ---
if __name__ == "__main__":
    dummy_inputs = [
        "I feel tired and have a sore throat.",
        "I've had a cough for three weeks.",
        "I'm dizzy and feel weak.",
    ]

    for input_text in dummy_inputs:
        result = graph.invoke({
            "messages": [
                SystemMessage(content="You are a helpful medical assistant. Use tools if needed."),
                HumanMessage(content=input_text),
            ]
        })
        print(f"Input: {input_text}")
        print("Response:", result["messages"][-1].content)
        print("-" * 50)
