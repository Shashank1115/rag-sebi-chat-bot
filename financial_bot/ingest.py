import os
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
# --- THIS IS THE CHANGE ---
# We now point specifically to the subfolder for RAG documents
DATA_PATH = "data/rag_sources"

def create_vector_db():
    if not os.path.exists(DATA_PATH):
        print(f"Data directory '{DATA_PATH}' not found. Please create it and add your documents.")
        return

    loaders = [
        DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True),
        DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader, show_progress=True),
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

    print(f"Creating vector store...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=VECTOR_STORE_PATH)
    print(f"Vector store created and saved at: {VECTOR_STORE_PATH}")

if __name__ == "__main__":
    create_vector_db()
