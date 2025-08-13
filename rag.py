## Perform RAG based on local Chroma DB

import os
import textwrap
from langdetect import detect_langs
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
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
    # retrieve relevant context
    docs = retriever.invoke(query)
    combined_docs = combine_docs(docs)

    # utilize chat history
    history_text = '\n'.join([f'User: {q}' for q in history]) if history else ''

    # generate prompt
    with open('prompt_template.txt', 'r', encoding='utf-8') as f:
        prompt_template_txt = f.read()

    full_prompt = prompt_template_txt \
                .replace('{documents}', combined_docs) \
                .replace('{history}', history_text)

    # translate prompt if query language is not English
    detection = detect_langs(query)[0]
    lang, confidence = detection.lang, detection.prob
    if lang != 'en' and confidence>0.90:
        prompt_sections = textwrap.wrap(full_prompt, 5000)
        result = ''
        # retrieve relevant context & translate to query language
        for text in prompt_sections:
            result += GoogleTranslator(source='en', target=lang).translate(text)
        full_prompt = result

    # prompt LLM
    response = llm.invoke(full_prompt).content + '\n\nDisclaimer: This response is for educational purposes only. For official determinations, please consult the IRB through VIRBS.'

    log_entry = {
        'prompt': full_prompt,
        'language': lang,
        'user_query': query,
        'response': response,
        'retrieved_docs': [
            {
                'metadata': doc.metadata,
                'text': doc.page_content
            }
            for doc in docs
        ],
        'chat_history': history
    }

    # return rag response
    return response, log_entry