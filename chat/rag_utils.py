import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

load_dotenv()
openai_api_key=os.getenv("OPENAI_API_KEY")

# CREATE A CHROMA CLIENT
chroma_client = chromadb.PersistentClient(path="./chroma_db")

embedding_function=OpenAIEmbeddingFunction(api_key=openai_api_key, model_name="text-embedding-3-small")

collection = chroma_client.get_or_create_collection(
    name="indian_laws",
    embedding_function= embedding_function
)

def populate_vector_db(docs,metadatas=None,ids=None):
    """
    Adds documents to the vector DB.
    - docs: list of text chunks
    - metadatas: list of dicts (one per chunk)
    - ids: list of unique IDs
    """
    collection.add(
        documents=docs,
        metadatas=metadatas,
        ids=ids
    )

