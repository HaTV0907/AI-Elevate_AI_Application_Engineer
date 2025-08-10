from backend.vectorstore import get_vectorstore

def get_retriever():
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever()
