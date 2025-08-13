## Launch front end of chatbot using Streamlit

import json
import streamlit as st
from rag import answer_query
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import TransformersNlpEngine
from presidio_anonymizer import AnonymizerEngine

st.title('AI IRB Assistant')

# initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# initialize log flag
if 'log_saved' not in st.session_state:
    st.session_state.log_saved = False

# initialize analyzer and anonymizer
model_config = [{'lang_code': 'en', 'model_name': {
    'spacy': 'en_core_web_sm',
    'transformers': 'dslim/bert-base-NER'
    }
}]
nlp_engine = TransformersNlpEngine(models=model_config)
analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
anonymizer = AnonymizerEngine()

# prompt for user input
user_input = st.chat_input(placeholder='Ask me questions or upload documents for review', accept_file='multiple')

# anonymize input text
if user_input and user_input['text']:
    query = user_input['text']
    analyzer_results = analyzer.analyze(text=query, language='en')
    anonymized_query = anonymizer.anonymize(text=query, analyzer_results=analyzer_results).text

try:
    # answer document upload queries
    if user_input and user_input['files']:
        if query:
            st.session_state.chat_history.append(('user', anonymized_query))
        st.session_state.chat_history.append(('bot', 'I apologize, document upload is currently under development'))

        st.rerun()

    # answer text query using rag
    elif user_input and user_input['text']:
        st.session_state.chat_history.append(('user', anonymized_query))

        # display chat history with user's message
        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(message)

        # query the llm with the chat history + current question
        with st.spinner('Answering your question...'):
            user_chats = [msg for role, msg in st.session_state.chat_history if role == 'user']
            response, log = answer_query(anonymized_query, history=user_chats[:-1])
            st.session_state.chat_history.append(('bot', response))

            if not st.session_state.log_saved:
                # create log of interaction
                with open('logs/rag_logs.jsonl', 'a', encoding='utf-8') as f:
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

except Exception as e:
    st.markdown('Apologies, we are having technical difficulties with your request. Please try again or come back another time.')

    # log error
    with open('logs/errors.jsonl', 'a', encoding='utf-8') as f:
        json.dump({'error': str(e)}, f)
        f.write('\n')
