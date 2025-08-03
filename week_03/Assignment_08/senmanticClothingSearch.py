import os
from openai import AzureOpenAI
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
load_dotenv()
# Step 1: Setup AzureOpenAI client
# Ensure you’ve set the following environment variables
# AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_DEPLOYMENT_NAME

client = AzureOpenAI(
    api_version="2024-07-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# Step 2: Sample expanded product data
products = [
    {
        "title": "Classic Blue Jeans",
        "short_description": "Comfortable blue denim jeans with a relaxed fit.",
        "price": 49.99,
        "category": "Jeans"
    },
    {
        "title": "Red Hoodie",
        "short_description": "Cozy red hoodie made from organic cotton.",
        "price": 39.99,
        "category": "Hoodies"
    },
    {
        "title": "Black Leather Jacket",
        "short_description": "Stylish black leather jacket with a slim fit design.",
        "price": 120.00,
        "category": "Jackets"
    },
    {
        "title": "Gray Sweatpants",
        "short_description": "Soft gray sweatpants, perfect for lounging or working out.",
        "price": 29.99,
        "category": "Bottoms"
    },
    {
        "title": "White Cotton T-Shirt",
        "short_description": "Simple white cotton t-shirt with a classic crew neck.",
        "price": 19.99,
        "category": "T-Shirts"
    },
    {
        "title": "Blue Windbreaker",
        "short_description": "Lightweight blue windbreaker jacket for outdoor activities.",
        "price": 59.99,
        "category": "Outerwear"
    },
    {
        "title": "Beige Cardigan Sweater",
        "short_description": "Warm beige cardigan sweater with front buttons.",
        "price": 42.50,
        "category": "Sweaters"
    },
    {
        "title": "Green Workout Tank",
        "short_description": "Breathable green tank top for gym and fitness.",
        "price": 24.99,
        "category": "Activewear"
    },
]

# Step 3: Function to generate embedding from description
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Step 4: Preprocess and embed all products
for product in products:
    full_text = f"{product['title']} {product['short_description']}".lower()
    product["embedding"] = get_embedding(full_text)

# Step 5: Define user query (simulate input)
query = "warm cotton sweatshirt for winter"

# Step 6: Generate embedding for query
query_embedding = get_embedding(query.lower())

# Step 7: Compute cosine similarity between query and products
def similarity_score(vec1, vec2):
    return 1 - cosine(vec1, vec2)

scores = []
for product in products:
    score = similarity_score(query_embedding, product["embedding"])
    scores.append((score, product))

# Step 8: Sort by similarity score (descending)
scores.sort(key=lambda x: x[0], reverse=True)

# Step 9: Display top N matches
top_n = 3
print(f"Top {top_n} matching products for query: '{query}'\n")
for score, product in scores[:top_n]:
    print(f"🔹 Title: {product['title']}")
    print(f"📝 Description: {product['short_description']}")
    print(f"💰 Price: ${product['price']:.2f}")
    print(f"🏷️ Category: {product['category']}")
    print(f"📊 Similarity Score: {score:.4f}\n")