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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


load_dotenv()


# combine chunk data
chunked_records = []
def assemble_chunks(chunks, embeddings, doc_name, doc_date):
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


# convert assembled chunks to langchain Documents
def records_to_documents(records):
    docs = []
    for r in records:
        docs.append(
            Document(
                page_content=r['text'],
                metadata=r['metadata']
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


if __name__ == '__main__':
    folder = 'data/HRPP_normalized'
    text_files = [f for f in os.listdir(folder) if f.endswith('.txt')]

    # initialize embedding model: 1024d
    embed_model_name = os.environ['EMBEDDING_MODEL']
    embedding_model = HuggingFaceEmbeddings(model_name=embed_model_name)

    # initialize tokenizer
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=0,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

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
        assemble_chunks(chunks, embeddings, doc_name, doc_date)

    # convert records to LangChain Documents
    docs = records_to_documents(chunked_records)

    # create or update Chroma DB
    vectorstore = Chroma.from_documents(
        documents=docs,
        collection_name='hrpp_docs',
        embedding=embedding_model,
        persist_directory='./data/chroma_hrpp'
    )

    print('[INFO] Chroma DB built and persisted.')
    print(len(chunked_records[0]['embedding']))
