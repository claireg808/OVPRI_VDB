## Launch front end of chatbot using Streamlit

import json
import streamlit as st
from rag import answer_query

st.title('OVPRI AI Chat')

# initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# initialize log flag
if 'log_saved' not in st.session_state:
    st.session_state.log_saved = False

# prompt for user input
user_input = st.chat_input('Ask me about HRPP')

# answer query using rag
if user_input:
    st.session_state.chat_history.append(('user', user_input))

    # display chat history with user's message
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    # query the llm with the chat history + current question
    with st.spinner('Answering your question...'):
        user_chats = [msg for role, msg in st.session_state.chat_history if role == "user"]
        response, log = answer_query(user_input, history=user_chats)
        st.session_state.chat_history.append(('bot', response))

        if not st.session_state.log_saved:
            # create log of interaction
            with open('logs/rag_logs_B.jsonl', 'a', encoding='utf-8') as f:
                json.dump(log, f)
                f.write('\n')
            st.session_state.log_saved = True

    st.rerun()

else:
    # reset log flag for next interaction
    st.session_state.log_saved = False

    # display user & bot messages
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)