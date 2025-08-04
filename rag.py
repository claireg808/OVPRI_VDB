## Perform RAG based on local Chroma DB

import os
import json
import re
from dotenv import load_dotenv
from transformers import pipeline
from googletrans import Translator
from langdetect import detect
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI


load_dotenv()


# initialize embedding model & llm
embedding_model_name = os.environ['EMBEDDING_MODEL']
embedding_model = HuggingFaceEmbeddings(
                    model_name=embedding_model_name,
                    encode_kwargs={'normalize_embeddings': True}
                )

llm = ChatOpenAI(
    base_url=os.environ['BASE_URL'],
    api_key='dummy-key',
    model=os.environ['MODEL']
)


# access stored vector database
persist_dir = './chroma_hrpp'
vectorstore = Chroma(
    collection_name='hrpp_docs',
    persist_directory=persist_dir,
    embedding_function=embedding_model
)


# initialize retriever to get top 5 results
retriever = vectorstore.as_retriever(
                search_type='similarity',
                search_kwargs={'k': 10}
            )

# import english & spanish prompt versions
with open('en_prompt_template.txt', 'r', encoding='utf-8') as f:
    en_template = f.read()
with open('es_prompt_template.txt', 'r', encoding='utf-8') as f:
    es_template = f.read()


# concatenate retrieved documents
def combine_docs(docs):
    combined_texts = []
    for d in docs:
        name = d.metadata.get('document_name', '')
        content = d.page_content
        combined_texts.append(f'Document name: {name}\nContent: {content}')
    return '\n\n'.join(combined_texts)


# answer given query
def answer_query(query: str, history: list[str]) -> str:
    # determine query language: spanish or english
    lang = detect(query)
    if lang == 'es':
        # retrieve Spanish prompt
        prompt = PromptTemplate.from_template(es_template)

        # retrieve relevant context & translate to Spanish
        docs = retriever.invoke(query)
        english_combined_docs = combine_docs(docs)
        translator = Translator()
        combined_docs = translator.translate(english_combined_docs, src='en', dest='es')

        # utilize chat history
        history_text = '\n'.join([f'User: {q}' for q in history]) if history else ''

    else:
        # retieve English prompt
        prompt = PromptTemplate.from_template(en_template)

        # retrieve relevant context
        docs = retriever.invoke(query)
        combined_docs = combine_docs(docs)

        # utilize chat history
        history_text = '\n'.join([f'User: {q}' for q in history]) if history else ''

    # append retrieved context to query
    input_data = {
        'history': history_text,
        'documents': combined_docs,
        'question': query
    }

    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke(input_data)

    log_entry = {
        'user_query': query,
        'response': response,
        'retrieved_docs': [
            {
                'metadata': doc.metadata,
                'text': doc.page_content
            }
            for doc in docs
        ]
    }

    # return rag response
    return response, log_entry
