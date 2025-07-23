## Perform RAG based on local Chroma DB

import os
import json
import re
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


# initialize embedding model & llm
embedding_model = HuggingFaceEmbeddings(model_name=os.environ['EMBEDDING_MODEL'])

llm = ChatOpenAI(
    base_url=os.environ['BASE_URL'],
    api_key='dummy-key',
    model=os.environ['MODEL'],
    max_tokens=10000,
    timeout=float(os.environ['TIMEOUT'])
)


# access stored vector database
persist_dir = "./chroma_hrpp"
vectorstore = Chroma(
    collection_name="hrpp_docs",
    persist_directory=persist_dir,
    embedding_function=embedding_model
)


# prompt template
prompt = ChatPromptTemplate.from_template("""You are a helpful assistant.
Use the following context to answer the question.

Context:
{context}

Question:
{question}
""")


# list of queries
queries = ['What should I know about conducting human research?',
           'Who can serve as LAR?']


# concatenate retrieved documents
def combine_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


if __name__ == '__main__':
    # get top 5 results
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # answer user queries
    logs = []
    for query in queries:
        user_query = query
        results = retriever.invoke(user_query)
        context = combine_docs(results)

        # append retrieved context to query
        input_data = {
            "context": context,
            "question": user_query
        }

        # query the llm
        rag_chain = prompt | llm
        response = rag_chain.invoke(input_data).content

        # extract the response
        match = re.search(r".*</think>\s*(.*)", response, re.DOTALL)
        if match:
            response = match.group(1)

        print(response)

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
