import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    CSVLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

VECTOR_STORE_PATH = "vector_store"
DATA_PATH = "data/rag_sources"

def create_vector_db():
    """
    This script is responsible for creating the MAIN, PUBLIC knowledge base
    for the chatbot. It reads all the documents from the 'data/rag_sources'
    folder and builds the primary vector store that all users can access.

    User-specific vector stores are created separately by the backend.py script.
    """
    
    # This logic now checks if a vector store exists and skips creation if it does.
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"Vector store already exists at '{VECTOR_STORE_PATH}'. Skipping creation.")
        print("If you want to rebuild the main knowledge base, please delete the 'vector_store' folder manually and run this script again.")
        return

    if not os.path.exists(DATA_PATH):
        print(f"Data directory '{DATA_PATH}' not found. Please create it and add your documents.")
        return

    loaders = [
        DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True),
        # Added encoding to the TextLoader for robustness
        DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader, show_progress=True, loader_kwargs={'encoding': 'utf8'}),
        DirectoryLoader(DATA_PATH, glob="**/*.csv", loader_cls=CSVLoader, show_progress=True, loader_kwargs={'encoding': 'utf8'}),
    ]
    
    loaded_documents = []
    for loader in loaders:
        try:
            loaded_documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading files: {e}")
            continue

    if not loaded_documents:
        print(f"No documents found in '{DATA_PATH}'.")
        return

    print(f"Loaded {len(loaded_documents)} document(s) from '{DATA_PATH}'.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(loaded_documents)
    print(f"Split into {len(texts)} chunks.")

    print("Initializing Google Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    print(f"Creating new vector store with {len(texts)} chunks. This may take a few minutes...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=VECTOR_STORE_PATH)
    print(f"Vector store created successfully and saved at: {VECTOR_STORE_PATH}")

if __name__ == "__main__":
    create_vector_db()
