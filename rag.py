## Perform RAG based on local Chroma DB

import os
import json
import re
from dotenv import load_dotenv
from transformers import pipeline
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
                search_kwargs={'k': 5}
            )


# prompt template
prompt = PromptTemplate.from_template(
    'Use the following context to answer the question. Cite the source document name.\n'
    'Chat History:\n{history}\n\n'
    'Context:\n{documents}\n\n'
    'Question: {question}\n\n'
    'Answer:'
)

rag_chain = prompt | llm | StrOutputParser()


# concatenate retrieved documents
def combine_docs(docs):
    combined_texts = []
    for d in docs:
        name = d.metadata.get('document_name', '')
        content = d.page_content
        combined_texts.append(f'Document name: {name}\nContent: {content}')
    return '\n\n'.join(combined_texts)


# answer given query
logs = []
def answer_query(query: str, history: list[str]) -> str:
    # retrieve relevant context
    docs = retriever.invoke(query)
    combined_docs = combine_docs(docs)

    # utilize chat history
    history_text = "\n".join([f'User: {q}' for q in history]) if history else ''

    # append retrieved context to query
    input_data = {
        'history': history_text,
        'documents': combined_docs,
        'question': query
    }

    # return rag response
    return rag_chain.invoke(input_data)
