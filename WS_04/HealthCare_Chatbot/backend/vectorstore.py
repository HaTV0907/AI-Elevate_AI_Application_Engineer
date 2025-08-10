from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from backend.pinecone_init import init_pinecone
import os

def get_vectorstore():
    init_pinecone()

    embeddings = OpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_API_EMBEDDING_BASE"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_EMBEDDING_KEY"),
        deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_EMBEDDING_VERSION")
    )

    index_name = os.getenv("PINECONE_INDEX")
    return Pinecone.from_existing_index(index_name, embeddings)
