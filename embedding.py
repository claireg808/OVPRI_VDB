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
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

chunked_records = []


# chunk text into 250 token size
def chunk_text(text, chunk_size=250, overlap=30):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def assemble_chunks(chunks, embeddings):
    for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        chunked_records.append({
            "chunk_id": idx,
            "text": chunk,
            "embedding": vector.tolist(),  # Convert numpy array to list (JSON-serializable)
            "metadata": {
                "document_name": document_name,
                "document_date": document_date
            }
        })



if __name__ == '__main__':
    # retrieve normalized text files
    folder = 'HRPP_normalized'
    text_files = [f for f in os.listdir(folder) if f.endswith('.txt')]

    # set up embedding model
    embedding_model_name = os.environ['EMBEDDING_MODEL']
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # create & embed chunks of each document
    for file in text_files:
        # extract the name
        doc_name = os.path.splitext(os.path.basename(file))[0]
        print(f'Embedding {doc_name}')

        with open(file, 'r', encoding='utf-8') as f:
            # extract the date
            doc_date_match = re.search(r"\| (\d{1,2}\/\d{1,2}\/\d{4})", f.read())
            doc_date = [doc_date_match.group(1) if doc_date_match else '']

            chunks = chunk_text(f, chunk_size=250, overlap=30)
        embeddings = embedding_model.encode(chunks, convert_to_numpy=True)


