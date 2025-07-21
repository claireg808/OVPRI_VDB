import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

qdrant_client = QdrantClient(
    url=os.environ['QDRANT_URL'], 
    api_key=os.environ['QDRANT_API'],
)

print(qdrant_client.get_collections())