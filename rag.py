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
persist_dir = "./chroma_hrpp"
vectorstore = Chroma(
    collection_name="hrpp_docs",
    persist_directory=persist_dir,
    embedding_function=embedding_model
)


# prompt template
prompt = PromptTemplate.from_template(
    'Use the following context to answer the question. Cite the source document name.\n'
    'Context:\n{documents}\n\n'
    'Question: {question}\n\n'
    'Answer:'
)



# list of queries
queries = ['What do I need to know about conducting human research?',
           'What should an IRB member with a conflict of interest do?']


# concatenate retrieved documents
def combine_docs(docs):
    combined_texts = []
    for d in docs:
        name = d.metadata.get('document_name', '')
        content = d.page_content
        combined_texts.append(f'Document name: {name}\nContent: {content}')
    return "\n\n".join(combined_texts)


if __name__ == '__main__':
    # get top 5 results
    retriever = vectorstore.as_retriever(
                    search_type='similarity',
                    search_kwargs={"k": 5}
                )

    # answer user queries
    logs = []
    for query in queries:
        user_query = query
        results = retriever.invoke(user_query)
        combined_docs = combine_docs(results)

        # append retrieved context to query
        input_data = {
            "documents": combined_docs,
            "question": user_query
        }

        # query the llm
        rag_chain = prompt | llm | StrOutputParser()
        response = rag_chain.invoke(input_data)

        print(f'{user_query}\n{response}\n\n')

        # build a log entry
        log_entry = {
            "user_query": user_query,
            "response": response,
            "retrieved_docs": [
                {
                    "metadata": doc.metadata,
                    "text": doc.page_content
                }
                for doc in results
            ]
        }

        logs.append(log_entry)

    # save rag logs
    with open("rag_logs.json", "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
