"""
ingest_faiss.py

Replaces previous ingest.py with a FAISS-backed pipeline that:
 - Loads PDF/TXT/CSV from data/rag_sources
 - Splits into chunks using RecursiveCharacterTextSplitter
 - Builds embeddings using (preferred) HuggingFaceEmbeddings, or falls back to SentenceTransformer
 - Builds a FAISS index (exact IndexFlatIP for cosine similarity when vectors are normalized)
 - Persists the FAISS index and a separate metadata JSON file
 - Backs up the old vector_store atomically before swapping

Notes:
 - Install: pip install faiss-cpu sentence-transformers langchain langchain_community
 - If you have GPU and want faiss-gpu, install that instead.
 - LangChain APIs and import paths can vary between versions; this script attempts multiple import paths.

Exit codes:
 - 0 success
 - 1 no documents
 - 2 ingest / indexing error
"""

import os
import sys
import shutil
import uuid
import time
import json
import logging
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ingest_faiss")

PROJECT_ROOT = Path(__file__).parent.resolve()
RAG_SOURCES = PROJECT_ROOT / "data" / "rag_sources"
VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store"
BACKUP_SUFFIX = ".backup."

# --- Try to import LangChain + wrappers (best-effort) ---
try:
    # Try canonical langchain imports first
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    try:
        # newer langchain versions
        from langchain.vectorstores import FAISS as LC_FAISS
        from langchain.embeddings import HuggingFaceEmbeddings
        LC_FAISS_AVAILABLE = True
        HF_AVAILABLE = True
    except Exception:
        # community variant
        from langchain_community.vectorstores import FAISS as LC_FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        LC_FAISS_AVAILABLE = True
        HF_AVAILABLE = True
except Exception:
    # langchain not available (we'll fallback to direct sentence-transformers + faiss)
    RecursiveCharacterTextSplitter = None
    LC_FAISS_AVAILABLE = False
    HF_AVAILABLE = False

# Try sentence-transformers and native FAISS
try:
    from sentence_transformers import SentenceTransformer
    S2_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    S2_AVAILABLE = False

try:
    import faiss
    FAISS_NATIVE = True
except Exception:
    faiss = None
    FAISS_NATIVE = False

# --- Document loading ---
# We prefer langchain loaders if available (PyPDFLoader), else simple read
try:
    from langchain_community.document_loaders import PyPDFLoader
    LC_PDF_LOADER = True
except Exception:
    PyPDFLoader = None
    LC_PDF_LOADER = False


def load_documents_from_rag_sources():
    """Load PDFs and text files from RAG_SOURCES. Returns list of objects with .page_content and .metadata."""
    docs = []
    if not RAG_SOURCES.exists():
        logger.error("RAG_SOURCES directory not found: %s", RAG_SOURCES)
        return docs

    for root, _, files in os.walk(RAG_SOURCES):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            fp = Path(root) / fname
            try:
                if ext == ".pdf" and LC_PDF_LOADER:
                    loader = PyPDFLoader(str(fp))
                    d = loader.load()
                    docs.extend(d)
                elif ext in {".txt", ".md", ".csv"}:
                    text = fp.read_text(encoding="utf8", errors="ignore")
                    docs.append(type("D", (), {"page_content": text, "metadata": {"source": str(fp)}})())
                else:
                    logger.debug("Skipping unsupported file: %s", fp)
            except Exception as e:
                logger.warning("Failed to load %s: %s", fp, e)
    logger.info("Loaded %d document(s) from '%s'", len(docs), RAG_SOURCES)
    return docs


def choose_embedding_provider():
    """Return (emb_obj, provider_name, embed_fn)

    - emb_obj: object usable by LangChain FAISS.from_documents if available, or None
    - provider_name: string
    - embed_fn(texts) -> numpy.ndarray: a function that returns embeddings for a list of texts
    """
    # Prefer HuggingFaceEmbeddings (LangChain)
    if HF_AVAILABLE:
        try:
            emb = HuggingFaceEmbeddings(model_name=os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
            logger.info("Using HuggingFaceEmbeddings via LangChain")

            def fn(texts: List[str]):
                # LangChain wrapper may accept list and return list of vectors
                return emb.embed_documents(texts)

            return emb, "huggingface_langchain", fn
        except Exception as e:
            logger.warning("HuggingFaceEmbeddings via LangChain failed: %s", e)

    # Fallback: direct sentence-transformers
    if S2_AVAILABLE:
        model_name = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
        logger.info("Using sentence-transformers model: %s", model_name)
        s2 = SentenceTransformer(model_name)

        def fn(texts: List[str]):
            return s2.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        return None, "sentence_transformers", fn

    raise RuntimeError("No embedding provider available. Install langchain_community or sentence-transformers.")


def backup_old_store():
    if VECTOR_STORE_PATH.exists():
        backup = VECTOR_STORE_PATH.with_name(VECTOR_STORE_PATH.name + BACKUP_SUFFIX + uuid.uuid4().hex[:8])
        logger.info("Backing up existing vector store to %s", backup)
        shutil.move(str(VECTOR_STORE_PATH), str(backup))
        return backup
    return None


def write_metas(out_dir: Path, metas: List[dict]):
    with open(out_dir / "metas.json", "w", encoding="utf8") as fh:
        json.dump(metas, fh, ensure_ascii=False, indent=2)


def build_faiss_native(texts, metas, embed_fn, out_dir: Path):
    """Build FAISS index using native faiss + numpy embeddings. Writes faiss.index and metas.json to out_dir."""
    import numpy as np

    vecs = embed_fn(texts)
    # ensure numpy array
    vecs = np.asarray(vecs, dtype=np.float32)
    # Normalize vectors for cosine similarity when using inner product
    faiss.normalize_L2(vecs)
    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)  # exact inner-product search
    index.add(vecs)

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    write_metas(out_dir, metas)
    logger.info("Native FAISS index written to %s", out_dir)


def build_faiss_via_langchain(docs, emb_obj, out_dir: Path):
    """Use LangChain FAISS.from_documents (if available) and persist locally."""
    # Note: LangChain FAISS wrapper API differs across versions. We'll try common variants.
    try:
        lc_store = LC_FAISS.from_documents(docs, embedding=emb_obj)
        # try save_local if available
        try:
            lc_store.save_local(str(out_dir))
            logger.info("LangChain FAISS saved_local to %s", out_dir)
        except Exception:
            # try serialize -> bytes
            try:
                serialized = lc_store.serialize()
                with open(out_dir / "faiss_index.bin", "wb") as fh:
                    fh.write(serialized)
                logger.info("LangChain FAISS serialized to %s/faiss_index.bin", out_dir)
            except Exception:
                logger.warning("LangChain FAISS could not be saved using available methods")
        return
    except Exception as e:
        logger.exception("LangChain FAISS.from_documents failed: %s", e)
        raise


def split_documents_to_chunks(documents):
    # Use RecursiveCharacterTextSplitter if available; otherwise naive per-doc split
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=int(os.getenv('CHUNK_SIZE', '1000')),
                                                  chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '150')))
        chunks = splitter.split_documents(documents)
        logger.info("Split into %d chunks using RecursiveCharacterTextSplitter", len(chunks))
        return chunks
    else:
        # naive: each document becomes one chunk
        chunks = []
        for d in documents:
            text = getattr(d, 'page_content', '') or ''
            if text.strip():
                chunks.append(type('D', (), {'page_content': text, 'metadata': getattr(d, 'metadata', {})})())
        logger.info("Recursive splitter not available. Falling back to %d naive chunks.", len(chunks))
        return chunks


def create_vector_store(docs):
    # Split
    chunks = split_documents_to_chunks(docs)
    if not chunks:
        raise RuntimeError("No chunks to index.")

    # Prepare texts + metas lists
    texts = [getattr(c, 'page_content', '') for c in chunks]
    metas = [getattr(c, 'metadata', {}) or {} for c in chunks]

    emb_obj, provider, embed_fn = choose_embedding_provider()

    tmp_dir = VECTOR_STORE_PATH.with_name(VECTOR_STORE_PATH.name + ".tmp." + uuid.uuid4().hex[:8])
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    try:
        if LC_FAISS_AVAILABLE and emb_obj is not None:
            # Build via LangChain FAISS wrapper (preferred if available)
            logger.info("Building FAISS via LangChain wrapper (provider=%s)", provider)
            # Convert our chunks to LangChain Document objects if they aren't
            # Many LangChain wrappers accept simple dicts or Document objects; we will try to pass `chunks` directly.
            create_docs = chunks
            tmp_dir.mkdir(parents=True, exist_ok=True)
            build_faiss_via_langchain(create_docs, emb_obj, tmp_dir)
        elif FAISS_NATIVE and S2_AVAILABLE:
            logger.info("Building FAISS using native faiss + sentence-transformers (provider=%s)", provider)
            build_faiss_native(texts, metas, embed_fn, tmp_dir)
        else:
            raise RuntimeError("No FAISS path available: check langchain/faiss/sentence-transformers installation")

        # Atomic swap: backup old store, move tmp -> VECTOR_STORE_PATH
        backup = backup_old_store()
        if VECTOR_STORE_PATH.exists():
            logger.info("Removing existing path (shouldn't exist after backup): %s", VECTOR_STORE_PATH)
            shutil.rmtree(VECTOR_STORE_PATH)
        shutil.move(str(tmp_dir), str(VECTOR_STORE_PATH))
        logger.info("Vector store is now at %s", VECTOR_STORE_PATH)
        return True
    except Exception:
        logger.exception("Failed to build vector store; cleaning up tmp dir")
        try:
            if tmp_dir.exists(): shutil.rmtree(tmp_dir)
        except Exception:
            pass
        raise


def main():
    logger.info("Starting ingest_faiss.py")
    docs = load_documents_from_rag_sources()
    if not docs:
        logger.error("No documents found to ingest. Exiting.")
        sys.exit(1)
    try:
        create_vector_store(docs)
        logger.info("Ingest completed successfully.")
        sys.exit(0)
    except Exception as e:
        logger.exception("Ingest failed: %s", e)
        # If quota / rate limit type errors were involved, provide helpful guidance
        msg = str(e).lower()
        if "quota" in msg or "429" in msg:
            logger.error("Detected quota/429 error during embedding. Either enable billing, increase quota, or use local embeddings by unsetting USE_GOOGLE_EMBEDDINGS.")
        sys.exit(2)


if __name__ == "__main__":
    main()
