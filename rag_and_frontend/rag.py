## Perform RAG based on local Chroma DB

import os
import textwrap
from langdetect import detect_langs
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder


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
vectorstore = Chroma(
    collection_name='hrpp_docs',
    persist_directory='data/chroma_db',
    embedding_function=embedding_model
)


# initialize retriever to get top 10 results
retriever = vectorstore.as_retriever(
                search_type='similarity',
                search_kwargs={
                    'k': 20,
                    'include': ['documents', 'metadatas', 'distances'] 
                }
            )


# re-rank retrieved documents based on the query
def re_rank(query, docs):
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')

    # sort documents by cross-encoder score
    scored_documents = []
    for doc in docs:
        text = doc.page_content
        score = cross_encoder.predict((query, text))
        scored_documents.append((doc, score))
        scored_documents.sort(key=lambda x: x[1], reverse=True)
    
    # return just the sorted list of documents
    final_list = [d[0] for d in scored_documents]
    return final_list[:11]


# concatenate retrieved documents
def combine_docs(docs):
    combined_texts = []
    for d in docs:
        name = d.metadata.get('document_name', 'unknown')
        date = d.metadata.get('effective_date', 'unknown')
        content = d.page_content
        combined_texts.append(f'Document Name: {name}\nEffective Date: {date}\nContent: {content}')
    return '\n\n'.join(combined_texts)


# answer given query
def answer_query(query: str, history: list[str]) -> str:    
    # retrieve relevant context
    docs = retriever.invoke(query.lower())
    filtered_docs = re_rank(query.lower(), docs)
    combined_docs = combine_docs(filtered_docs)

    # utilize chat history
    history_text = '\n'.join([f'User: {q}' for q in history]) if history else ''

    # generate prompt
    with open('rag_and_frontend/prompt_template.txt', 'r', encoding='utf-8') as f:
        prompt_template_txt = f.read()

    full_prompt = prompt_template_txt \
                .replace('{history}', history_text) \
                .replace('{documents}', combined_docs) \
                .replace('{query}', query)

    # translate prompt if query language is not English
    detection = detect_langs(query)[0]
    lang, confidence = detection.lang, detection.prob
    if lang != 'en' and confidence>0.90 and len(query)>10:
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
        'retrieved_docs': combined_docs,
        'chat_history': history
    }

    # return rag response
    return response, 'log_entry'