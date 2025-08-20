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
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from chromadb import PersistentClient


load_dotenv()


# combine chunk data
def assemble_chunks(chunks, embeddings, doc_name, doc_date):
    chunked_records = []
    for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        chunked_records.append({
            'text': chunk,
            'embedding': vector, 
            'metadata': {
                'chunk_number': idx,
                'document_name': doc_name,
                'effective_date': doc_date
            }
        })

    return chunked_records


# convert assembled chunks to langchain Documents
def records_to_documents(chunk_lists):
    docs = []
    for chunk_list in chunk_lists:
        for record in chunk_list:
            docs.append(
                Document(
                    page_content=record['text'],
                    metadata=record['metadata']
                )
            )
    return docs

# extract the date of document revision
def extract_revision_date(doc_name: str, file_text: str) -> str:
    # list of documents to skip
    skip = ['HRP General Documents', 'HRP Templates']

    # check the start for the revision date
    date_at_start_match = re.search(r"^[a-z]*-[0-9]*[a-z]? \|\s*(\d{1,2}\/\d{1,2}\/\d{4})", file_text)
    if date_at_start_match:
        date = date_at_start_match.group(1)
        formatted_date = datetime.strptime(date, '%m/%d/%Y').strftime('%m/%d/%Y')
        return formatted_date
    
    # list of possible identifications for the revision date
    list_of_date_formats = [
        "revised:?\s*(\d{1,2}\/\d{1,2}\/\d{4})",
        "revision\s*(?:date)?:?\s*(\d{1,2}\/\d{1,2}\/\d{4})"
    ]

    # find all revision dates in the document
    doc_date_matches = []
    for rgx in list_of_date_formats:
        doc_date_matches.extend(re.findall(rgx, file_text))

    # if no revision date is found
    if not doc_date_matches or doc_name in skip:
        print(f'[INFO] No revision date found: {doc_name}')
        return ''

    # convert to datetime objects and sort
    doc_date_series = pd.Series(doc_date_matches)
    sorted_dates = doc_date_series.apply(pd.to_datetime, format='%m/%d/%Y').sort_values()
    # choose latest date
    final_date = sorted_dates.iloc[-1].strftime('%m/%d/%Y')

    return final_date

# delete the Chroma collection if it already exists
def delete_collection(collection_name, path):
    try:
        chroma_client = PersistentClient(path=path)
        chroma_client.delete_collection(collection_name)
        print(f'Collection {collection_name} deleted successfully.')
    except Exception as e:
        print(f'Unable to delete collection: {e}')


if __name__ == '__main__':
    folder = 'data/HRPP_normalized'
    text_files = [f for f in os.listdir(folder) if f.endswith('.txt')]

    # initialize embedding model: 1024d
    embed_model_name = os.environ['EMBEDDING_MODEL']
    embedding_model = HuggingFaceEmbeddings(model_name=embed_model_name)

    # initialize tokenizer
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators = ['. ', ' '],
        keep_separator = False
    )

    complete_chunks = []
    for file in text_files:
        doc_name = os.path.splitext(os.path.basename(file))[0]

        print(f'[INFO] Embedding {doc_name}')

        input_path = os.path.join(folder, file)
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
            doc_date = extract_revision_date(doc_name, text)
            chunks = text_splitter.split_text(text)

        embeddings = embedding_model.embed_documents(chunks)

        # add metadata
        doc_complete_chunks = assemble_chunks(chunks, embeddings, doc_name, doc_date)
        complete_chunks.append(doc_complete_chunks)

    # convert records to LangChain Documents
    docs = records_to_documents(complete_chunks)

    # create or update Chroma DB
    collection_name = 'hrpp_docs'
    directory = './data/chroma_db'
    delete_collection(collection_name, directory)
    vectorstore = Chroma.from_documents(
        documents=docs,
        collection_name=collection_name,
        embedding=embedding_model,
        persist_directory=directory
    )

    print('[INFO] Chroma DB built and persisted.')
    for doc in docs:
        if doc.metadata.get('document_name') == 'HRP-302-WORKSHEET-ApprovalIntervals':
            print(doc)