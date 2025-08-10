import openai
from pinecone import Pinecone, ServerlessSpec

def init_pinecone(api_key, env, index_name):
    pc = Pinecone(api_key=api_key)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=env)
        )

    return pc.Index(index_name)

def embed_text(text, endpoint, api_key, model):
    client = openai.OpenAI(
        base_url=endpoint,
        api_key=api_key
    )

    response = client.embeddings.create(
        model=model,
        input=[text]
    )

    return response.data[0].embedding

def store_in_pinecone(index, file_id, embedding, metadata):
    index.upsert([
        {
            "id": file_id,
            "values": embedding,
            "metadata": metadata
        }
    ])
