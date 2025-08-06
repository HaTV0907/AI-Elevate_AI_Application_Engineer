import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load API keys from .env file
load_dotenv()

# Set your OpenAI API key
client = AzureOpenAI(
    api_version="2024-07-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create the index if it doesn't exist
index_name = "product-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",        # Using AWS as the cloud provider
            region="us-east-1"  # Region compatible with Free Tier
        )
    )

# Connect to index
index = pc.Index(index_name)

# Sample product descriptions
products = [
    {"id": "1", "description": "Lightweight cotton t-shirt perfect for summer"},
    {"id": "2", "description": "Stylish linen shirt for hot weather"},
    {"id": "3", "description": "Breathable summer dress for sunny days"},
    {"id": "4", "description": "Wool sweater suitable for winter"},
    {"id": "5", "description": "Insulated parka for cold weather"}
]

# Helper function to generate embedding
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Upsert vectors into Pinecone
vectors = [
    (product["id"], get_embedding(product["description"]), {"description": product["description"]})
    for product in products
]
index.upsert(vectors)

# Define search query
query = "clothing item for summer"
query_embedding = get_embedding(query)

# Search the index
results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

# Display results
print(f"\nTop 3 similar products for the query: '{query}':\n")
for match in results["matches"]:
    print(f"- {match['metadata']['description']}")
