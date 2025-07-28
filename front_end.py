import streamlit as st
from streamlit.runtime.scriptrunner import RerunException
from rag import answer_query

st.title("OVPRI Chat")

# initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# prompt for user input
user_input = st.chat_input("Ask me questions about HRPP!")

# answer query using rag
if user_input:
    st.session_state.chat_history.append(("user", user_input))

    # display chat history with user's message
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    # query the llm with the chat history + current question
    with st.spinner("Answering your question..."):
        chat_history_only = [msg for role, msg in st.session_state.chat_history if role == "user"]
        response = answer_query(user_input, history=chat_history_only)
        
    st.session_state.chat_history.append(("bot", response))

    st.rerun()

else:
    # display user & bot messages
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)