## Perform RAG based on local Chroma DB

import os
import json
import re
from dotenv import load_dotenv
from transformers import pipeline
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain


load_dotenv()


# initialize embedding model & llm
embedding_model = HuggingFaceEmbeddings(model_name=os.environ['EMBEDDING_MODEL'])

pipe = pipeline('text-generation', 
                model=os.environ['MODEL'], 
                tokenizer=os.environ['MODEL'],
                return_full_text=False
        )

llm = HuggingFacePipeline(pipeline=pipe)


# access stored vector database
persist_dir = "./chroma_hrpp"
vectorstore = Chroma(
    collection_name="hrpp_docs",
    persist_directory=persist_dir,
    embedding_function=embedding_model
)


# prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the following context to answer the question."),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])


# list of queries
queries = ['What should I know about conducting human research?',
           'Who can serve as LAR?']


# concatenate retrieved documents
def combine_docs(docs):
    combined_texts = []
    for d in docs:
        name = d.metadata.get('document_name', '')
        content = d.page_content
        combined_texts.append(f'Document name: {name}\n{content}')
    return "\n\n".join(combined_texts)


if __name__ == '__main__':
    # get top 5 results
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # answer user queries
    logs = []
    for query in queries:
        user_query = query
        results = retriever.invoke(user_query)
        print('results:')
        print(results)
        print('\n\ncombined results:')
        context = combine_docs(results)
        print(context)

        # append retrieved context to query
        input_data = {
            "context": context,
            "question": user_query
        }

        # query the llm
        output_parser = StrOutputParser()
        rag_chain = prompt | llm
        raw_response = rag_chain.invoke(input_data)
        print("Raw LLM output:", raw_response)
        response = output_parser.parse(raw_response)
        print("Parsed response:", response)

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
