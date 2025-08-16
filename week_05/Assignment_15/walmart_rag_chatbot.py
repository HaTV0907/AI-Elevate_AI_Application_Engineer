import os
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph

# Load environment variables
load_dotenv()

# Walmart policy documents
raw_docs = [
    "Walmart customers may return electronics within 30 days with a receipt and original packaging.",
    "Grocery items at Walmart can be returned within 90 days with proof of purchase, except perishable products.",
    "Walmart offers a 1-year warranty on most electronics and appliances. See product details for exceptions.",
    "Walmart Plus members get free shipping with no minimum order amount.",
    "Prescription medications purchased at Walmart are not eligible for return or exchange.",
    "Open-box items are eligible for return at Walmart within the standard return period, but must include all original accessories.",
    "If a Walmart customer does not have a receipt, most returns are eligible for store credit with valid photo identification.",
    "Walmart allows price matching for identical items found on Walmart.com and local competitor ads.",
    "Walmart Vision Center purchases may be returned or exchanged within 60 days with a receipt.",
    "Returns on cell phones at Walmart require the device to be unlocked and all personal data erased.",
    "Walmart gift cards cannot be redeemed for cash except where required by law.",
    "Seasonal merchandise at Walmart (e.g., holiday decorations) may have modified return windows, see in-store signage.",
    "Bicycles purchased at Walmart can be returned within 90 days if not used outdoors and with all accessories present.",
    "For online Walmart orders, customers can return items in store or by mail using the prepaid label.",
    "Walmart reserves the right to deny returns suspected of fraud or abuse.",
]
docs = [Document(page_content=doc) for doc in raw_docs]

# Typed state
class RAGState(TypedDict):
    question: str
    context: Optional[str]
    answer: Optional[str]

# Embedding and vector store
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
    model=os.getenv("AZURE_OPENAI_EMBED_MODEL"),
    api_version="2024-07-01-preview",
)
vectorstore = FAISS.from_documents(
    docs,
    embeddings,
    docstore=InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)}),
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Azure OpenAI chat model
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_LLM_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_LLM_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_LLM_MODEL"),
    api_version="2024-07-01-preview",
    temperature=0,
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful Walmart support assistant. Use the provided information to answer product and policy questions. Always cite the retrieved info in your answer.",
    ),
    ("human", "{context}\n\nUser question: {question}"),
])

# Define LangGraph nodes
def retrieve_node(state: RAGState) -> RAGState:
    docs = retriever.get_relevant_documents(state["question"])
    context = "\n".join([doc.page_content for doc in docs])
    return {**state, "context": context}

def generate_node(state: RAGState) -> RAGState:
    formatted_prompt = prompt.format(context=state["context"], question=state["question"])
    answer = llm.invoke(formatted_prompt)
    return {**state, "answer": answer.content}

# Build LangGraph
builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)
builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "generate")
builder.set_finish_point("generate")
rag_graph = builder.compile()

# Demo questions
demo_questions = [
    "Can I return a Walmart bicycle if I've ridden it?",
    "What is Walmart's return policy for cell phones?",
    "Can I return groceries at Walmart?",
    "Do Walmart Plus members get free shipping?",
    "Can I return electronics without a receipt?",
]

if __name__ == "__main__":
    for q in demo_questions:
        result = rag_graph.invoke({"question": q})
        print(f"\n🧾 Question: {q}")
        print("📚 Retrieved Context:\n", result["context"])
        print("💬 Generated Answer:\n", result["answer"])