
---

## 🧠 `chatbot/rag_pipeline.py`

```python
import json
import os
# Pinecone SDK for initialization
import pinecone
# Langchain wrapper for vector store
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

def build_vector_store():
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV")
    )

    index_name = os.getenv("PINECONE_INDEX")
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=1536)

    index = pinecone.Index(index_name)

    loader = JSONLoader(file_path="data/mock_health_data.json", jq_schema=".[]")
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(docs, embeddings, index_name=index_name)

if __name__ == "__main__":
    build_vector_store()
