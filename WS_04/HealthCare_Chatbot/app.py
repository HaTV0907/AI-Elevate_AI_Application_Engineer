import streamlit as st
from backend.chatbot import get_chatbot

st.set_page_config(page_title="HealthCare Chatbot", layout="centered")
st.title("🩺 HealthCare Chatbot")

if "chatbot" not in st.session_state:
    st.session_state.chatbot = get_chatbot()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask me anything about your health:")

if user_input:
    response = st.session_state.chatbot.run(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

for speaker, text in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {text}")
