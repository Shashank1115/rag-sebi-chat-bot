# ingest.py
"""
Ingest scripts for financial_bot RAG sources.

Behavior:
 - By default uses local sentence-transformers ("all-MiniLM-L6-v2") for embeddings (384-dim).
 - If environment var USE_GOOGLE_EMBEDDINGS=1, it will attempt to use the Google embeddings provider
   (but beware quota / billing limits).
 - Replaces the vector store at VECTOR_STORE_PATH, backing up the old store first.
"""

import os
import sys
import shutil
import uuid
import logging
from pathlib import Path
from tqdm import tqdm

# langchain / loaders
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# embedding providers
USE_GOOGLE = os.getenv("USE_GOOGLE_EMBEDDINGS", "0") == "1"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GROQ_API_KEY")  # your setup may vary

# local embedder
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# fallback local sentence-transformers (if you prefer to call model directly)
try:
    from sentence_transformers import SentenceTransformer
    S2_AVAILABLE = True
except Exception:
    S2_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ingest")

PROJECT_ROOT = Path(__file__).parent.resolve()
RAG_SOURCES = PROJECT_ROOT / "data" / "rag_sources"
VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store"

def load_documents_from_rag_sources():
    """
    Load PDFs and text files from RAG_SOURCES. Returns list of Document-like objects with .page_content and .metadata.
    """
    docs = []
    if not RAG_SOURCES.exists():
        logger.error("RAG_SOURCES directory not found: %s", RAG_SOURCES)
        return docs

    # Use DirectoryLoader for PDFs, and simple read for text/csv
    # First, PDF loader:
    for root, _, files in os.walk(RAG_SOURCES):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            fp = Path(root) / fname
            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(str(fp))
                    d = loader.load()
                    docs.extend(d)
                elif ext in {".txt", ".md", ".csv"}:
                    text = fp.read_text(encoding="utf8", errors="ignore")
                    # create a simple document-like object
                    docs.append(type("D", (), {"page_content": text, "metadata": {"source": str(fp)}})())
                else:
                    logger.debug("Skipping unsupported file: %s", fp)
            except Exception as e:
                logger.warning("Failed to load %s: %s", fp, e)
    return docs

def choose_embedding_provider():
    """
    Return a LangChain-compatible embedding object and a descriptive name string.
    """
    if USE_GOOGLE:
        # Attempt to use Google embeddings via langchain_google_genai if available
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            if not GOOGLE_API_KEY:
                logger.warning("USE_GOOGLE set but GOOGLE_API_KEY not found. Falling back to HF.")
            else:
                emb = GoogleGenerativeAIEmbeddings(api_key=GOOGLE_API_KEY)
                logger.info("Using Google embeddings (be careful with quotas).")
                return emb, "google"
        except Exception as e:
            logger.warning("Google embeddings import failed: %s. Falling back to HF/local.", e)

    # Prefer HuggingFaceEmbeddings (uses sentence-transformers under the hood)
    if HF_AVAILABLE:
        model_name = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        emb = HuggingFaceEmbeddings(model_name=model_name)
        logger.info("Using HuggingFaceEmbeddings model: %s", model_name)
        return emb, "hf"

    # Last resort: direct sentence-transformers encoding (we'll wrap it later)
    if S2_AVAILABLE:
        model_name = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
        logger.info("HuggingFaceEmbeddings unavailable, using SentenceTransformer: %s", model_name)
        s2 = SentenceTransformer(model_name)
        # define a tiny wrapper object with embed_documents method expected by langchain_chroma.from_documents
        class SimpleSTWrapper:
            def embed_documents(self, texts):
                return s2.encode(texts, show_progress_bar=False).tolist()
            def embed_query(self, text):
                return s2.encode([text])[0].tolist()
        return SimpleSTWrapper(), "s2"

    raise RuntimeError("No embedding provider available. Install langchain_community or sentence-transformers.")

def backup_old_store():
    if VECTOR_STORE_PATH.exists():
        backup = VECTOR_STORE_PATH.with_name(VECTOR_STORE_PATH.name + ".backup." + uuid.uuid4().hex[:8])
        logger.info("Backing up existing vector store to %s", backup)
        shutil.move(str(VECTOR_STORE_PATH), str(backup))

def create_vector_store(docs, embeddings_obj):
    # split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    logger.info("Splitting documents into chunks...")
    texts = splitter.split_documents(docs)
    logger.info("Split into %d chunks.", len(texts))

    if len(texts) == 0:
        raise RuntimeError("No chunks to index. Aborting.")

    # create vector store
    logger.info("Creating Chroma vector store at %s ...", VECTOR_STORE_PATH)
    Chroma.from_documents(texts, embeddings_obj, persist_directory=str(VECTOR_STORE_PATH))
    logger.info("Vector store created successfully.")

def main():
    logger.info("Starting ingest.py")
    docs = load_documents_from_rag_sources()
    logger.info("Loaded %d document(s) from '%s'.", len(docs), RAG_SOURCES)

    if not docs:
        logger.error("No documents found to ingest. Exiting.")
        sys.exit(1)

    emb_obj, provider = choose_embedding_provider()
    logger.info("Embedding provider chosen: %s", provider)

    # backup and rebuild
    logger.info("Backing up old vector store (if any) and creating a fresh one.")
    try:
        backup_old_store()
        create_vector_store(docs, emb_obj)
        logger.info("Ingest completed.")
    except Exception as e:
        logger.exception("Ingest failed: %s", e)
        # if it's a Google quota error, give clear instruction
        msg = str(e).lower()
        if "quota" in msg or "429" in msg:
            logger.error("Detected quota/429 error during embedding. Either enable billing, increase quota, or use local embeddings by unsetting USE_GOOGLE_EMBEDDINGS.")
        sys.exit(2)

if __name__ == "__main__":
    main()
