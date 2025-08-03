# Clothing Product Semantic Search Engine

## 🧠 Objective
Build a semantic search engine using Azure OpenAI's `text-embedding-3-small` model to recommend similar clothing products based on product descriptions.

## 🚀 Features
- Embeds product descriptions into vector space
- Accepts a search query and finds semantically closest products
- Uses cosine similarity for ranking relevance

## 🧰 Tools & Libraries
- Azure OpenAI (via `openai` Python package)
- `scipy` for cosine similarity
- `os` for managing environment variables

## 🔍 How It Works
1. Create sample clothing product data
2. Generate text embeddings for product descriptions
3. Accept a search query (in text format)
4. Generate embedding for the query
5. Compute cosine similarity between the query and each product
6. Return top matches

## 📐 Cosine Similarity
Cosine similarity measures the angle between two vectors. A smaller angle means more similarity. This helps identify semantically similar products even if keywords differ.

## ⚠️ Limitations
- Currently uses a small hardcoded dataset
- Embedding generation is synchronous (not batched)
- No caching of embeddings

## 💡 Future Improvements
- Batch embeddings for efficiency
- Integrate with a real product database
- Support filtering by category, price, etc.

## 📝 How to Run
Set your Azure OpenAI credentials as environment variables:
```bash
export AZURE_OPENAI_ENDPOINT="https://aiportalapi.stu-platform.live/jpe"
export AZURE_OPENAI_API_KEY=""
export AZURE_DEPLOYMENT_NAME="text-embedding-3-small"