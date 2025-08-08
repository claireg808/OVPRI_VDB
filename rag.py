## Perform RAG based on local Chroma DB

import os
import json
import re
from dotenv import load_dotenv
from googletrans import Translator
from langdetect import detect
from typing_extensions import List, TypedDict
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()


# initialize embedding model & llm
embedding_model_name = os.environ['EMBEDDING_MODEL']
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

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

# initialize memory for chat history
memory = MemorySaver()

# import prompt template
with open('prompt_template.txt', 'r', encoding='utf-8') as f:
    prompt_template = f.read()


# concatenate retrieved documents
def combine_docs(docs):
    combined_texts = []
    for d in docs:
        name = d.metadata.get('document_name', '')
        content = d.page_content
        combined_texts.append(f'Source: {name}\nContent: {content}')
    return '\n\n'.join(combined_texts)


# use retriever to find relevant documents
@tool(response_format='content_and_artifact')
def retrieve(query: str):
    docs = retriever.invoke(query)
    combined_docs = combine_docs(docs)
    return combined_docs, docs

# generate Message that can include a tool-call
def query_or_respond(state: MessagesState):
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state['messages'])
    return {'messages': [response]}

tools = ToolNode([retrieve])

# generate response
def generate(state: MessagesState):
    recent_tool_messages = []
    for message in reversed(state['messages']):
        if message.type == 'tool':
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]
    docs_content = "\n\n".join(doc.content for doc in tool_messages)

    prompt_template = PromptTemplate.from_template(prompt_template)
    prompt_template['documents'] = docs_content

    conversation_messages = [
        message
        for message in state['messages']
        if message.type in ('human', 'system')
        or (message.type == 'ai' and not message.tool_calls)
    ]

    prompt = [SystemMessage(prompt_template)] + conversation_messages

    rag_chain = prompt | llm | StrOutputParser()

    # if the query is in another language, prompt in that language
    lang = detect(state['messages'][-1])
    if lang != 'en':
        translator = Translator()
        prompt = translator.translate(prompt, src='en', dest=lang)

    response = rag_chain.invoke(prompt)
    return {'messages': response}


# compile application flow into a graph object & retrieve response
def response(question):
    graph_builder = StateGraph(state_schema=MessagesState)

    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point('query_or_respond')
    graph_builder.add_conditional_edges(
        'query_or_respond',
        tools_condition,
        {END: END, 'tools': 'tools'},
    )
    graph_builder.add_edge('tools', 'generate')
    graph_builder.add_edge('generate', END)

    graph = graph_builder.compile(checkpointer=memory)

    return graph.invoke({'question': question})
