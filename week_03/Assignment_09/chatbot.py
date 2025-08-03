# chatbot.py

import os
import chromadb
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()


AZURE_OPENAI_EMBEDDING_API_KEY = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
AZURE_OPENAI_EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBED_MODEL")

AZURE_OPENAI_LLM_API_KEY = os.getenv("AZURE_OPENAI_LLM_API_KEY")
AZURE_OPENAI_LLM_ENDPOINT = os.getenv("AZURE_OPENAI_LLM_ENDPOINT")
AZURE_OPENAI_LLM_MODEL = os.getenv("AZURE_OPENAI_LLM_MODEL")

# ---- Clients ----
embedding_client = AzureOpenAI(
    api_key=AZURE_OPENAI_EMBEDDING_API_KEY,
    azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
    api_version="2024-07-01-preview"
)
llm_client = AzureOpenAI(
    api_key=AZURE_OPENAI_LLM_API_KEY,
    azure_endpoint=AZURE_OPENAI_LLM_ENDPOINT,
    api_version="2024-07-01-preview"
)

# ---- Embedding Generator ----
def get_embedding(text):
    response = embedding_client.embeddings.create(
        input=text,
        model=AZURE_OPENAI_EMBED_MODEL
    )
    return response.data[0].embedding

# ---- LLM Response ----
def ask_llm(context, user_input):
    system_prompt = (
        "You are a helpful assistant specializing in laptop recommendations. "
        "Use the provided context to recommend the best laptop(s) for the user's needs."
    )

    user_prompt = (
        f"User requirements: {user_input}\n\n"
        f"Context (top relevant laptops):\n{context}\n\n"
        "Based on the above, which laptop(s) would you recommend and why?"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = llm_client.chat.completions.create(
        model=AZURE_OPENAI_LLM_MODEL,
        messages=messages
    )

    return response.choices[0].message.content

# ---- ChromaDB Setup ----
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="laptops")

laptops = [
    {
        "id": "1",
        "name": "Gaming Beast Pro",
        "description": "A high-end gaming laptop with RTX 4080, 32GB RAM, and 1TB SSD. Perfect for hardcore gaming.",
        "tags": "gaming, high-performance, windows"
    },
    {
        "id": "2",
        "name": "Business Ultrabook X1",
        "description": "A lightweight business laptop with Intel i7, 16GB RAM, and long battery life. Great for productivity.",
        "tags": "business, ultrabook, lightweight"
    },
    {
        "id": "3",
        "name": "Student Basic",
        "description": "Affordable laptop with 8GB RAM, 256GB SSD, and a reliable battery. Ideal for students and general use.",
        "tags": "student, budget, general"
    },
]

# ---- Add Data to ChromaDB ----
for laptop in laptops:
    embedding = get_embedding(laptop["description"])
    collection.add(
        embeddings=[embedding],
        documents=[laptop["description"]],
        ids=[laptop["id"]],
        metadatas=[{
            "name": laptop["name"],
            "tags": laptop["tags"]
        }],
    )

# ---- Build Context ----
def build_context(results, n_context=3):
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    context_str = ""
    for doc, meta in zip(docs, metas):
        context_str += (
            f"Name: {meta['name']}\n"
            f"Description: {doc}\n"
            f"Tags: {meta['tags']}\n\n"
        )
    return context_str.strip()

# ---- Simulated Chat ----
user_queries = [
    "I want a lightweight laptop with long battery life for business trips.",
    "I need a laptop for gaming with the best graphics card available.",
    "Looking for a budget laptop suitable for student tasks and general browsing."
]

for user_input in user_queries:
    print("="*60)
    print(f"User input: {user_input}")
    
    query_embedding = get_embedding(user_input)
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    
    context = build_context(results)
    llm_output = ask_llm(context, user_input)
    
    print("\n💻 Recommended Laptop(s):\n")
    print(llm_output)
    print("="*60 + "\n")