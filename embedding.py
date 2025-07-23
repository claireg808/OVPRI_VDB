## Chunk documents into ~250 tokens per chunk
## Create embedding representation
## Attach metadata for each chunk
    # Document name
    # Document date
    # Chunk ID
    # Text
    # Embedding

import os
import re
import nltk
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from nltk.tokenize import sent_tokenize, word_tokenize


load_dotenv()
nltk.download('punkt')
nltk.download('punkt_tab')


# chunk text into 250 token size
def chunk_text(text, chunk_size=250, overlap=30):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        words = word_tokenize(sentence)
        if current_length + len(words) > chunk_size:
            chunks.append(" ".join(current_chunk))
            # Start new chunk with overlap
            overlap_words = current_chunk[-overlap:] if overlap > 0 else []
            current_chunk = overlap_words + words
            current_length = len(current_chunk)
        else:
            current_chunk.extend(words)
            current_length += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # remove possible empty chunks
    return [c for c in chunks if c.strip()]


# combine chunk data
chunked_records = []
def assemble_chunks(chunks, embeddings, doc_name, doc_date):
    for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        chunked_records.append({
            "text": chunk,
            "embedding": vector, 
            "metadata": {
                "chunk_number": idx,
                "document_name": doc_name,
                "document_date": doc_date
            }
        })


# convert assembled chunks to langchain Documents
def records_to_documents(records):
    docs = []
    for r in records:
        docs.append(
            Document(
                page_content=r["text"],
                metadata=r["metadata"]
            )
        )
    return docs


if __name__ == '__main__':
    folder = 'HRPP_normalized'
    text_files = [f for f in os.listdir(folder) if f.endswith('.txt')]

    embedding_model_name = os.environ['EMBEDDING_MODEL']
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    chunked_records = []

    for file in text_files:
        doc_name = os.path.splitext(os.path.basename(file))[0]
        print(f'Embedding {doc_name}')

        input_path = os.path.join(folder, file)
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
            doc_date_match = re.search(r"\| (\d{1,2}\/\d{1,2}\/\d{4})", text)
            doc_date = doc_date_match.group(1) if doc_date_match else ''

        chunks = chunk_text(text)
        embeddings = embedding_model.embed_documents(chunks)

        # add metadata
        assemble_chunks(chunks, embeddings, doc_name, doc_date)

    # convert records to LangChain Documents
    docs = records_to_documents(chunked_records)

    # create or update Chroma DB
    persist_dir = "./chroma_hrpp"
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        collection_name="hrpp_docs",
        persist_directory=persist_dir,
    )
    vectorstore.persist()
    print("Chroma DB built and persisted.")

    print(chunked_records[:2])
