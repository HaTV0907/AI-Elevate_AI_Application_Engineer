import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
import pinecone
from function_definitions import get_medication_info
 
load_dotenv()

st.title("🏥 Healthcare RAG Chatbot")

query = st.text_input("Ask a health-related question:")

if query:
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV")
    )
    embeddings = OpenAIEmbeddings()
    index_name = os.getenv("PINECONE_INDEX")

    # This uses the existing Pinecone index
    vectorstore = Pinecone.from_existing_index(index_name, embeddings)

    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    llm = ChatOpenAI(temperature=0)
    chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

    response = chain.run(query)
    st.write("💬 Chatbot:", response)

    if "Paracetamol" in query:
        st.write("📦 Function Call:", get_medication_info("Paracetamol"))
