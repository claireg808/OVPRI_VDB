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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


load_dotenv()


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

    # initialize embedding model
    embedding_model_name = os.environ['EMBEDDING_MODEL']
    embedding_model = HuggingFaceEmbeddings(
                        model_name=embedding_model_name,
                        encode_kwargs={"normalize_embeddings": True}
                    )

    # initialize tokenizer
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=0,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

    for file in text_files:
        doc_name = os.path.splitext(os.path.basename(file))[0]
        print(f'Embedding {doc_name}')

        input_path = os.path.join(folder, file)
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
            doc_date_match = re.search(r"\| (\d{1,2}\/\d{1,2}\/\d{4})", text)
            doc_date = doc_date_match.group(1) if doc_date_match else ''

            chunks = text_splitter.split_text(text)
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
