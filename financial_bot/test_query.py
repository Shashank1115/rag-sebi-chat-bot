# test_query.py
import sys
from pathlib import Path
import json

VSTORE = Path(__file__).parent.resolve() / "vector_store"
QUERY = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "what is neft"
TOP_K = 6

print("Vector store path:", VSTORE)
print("Query:", QUERY)
print("Attempting to load store...\n")

# Helper: embed with sentence-transformers (CPU)
def embed_with_st(model_name="all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(model_name)
        vec = st.encode([QUERY], convert_to_numpy=True)
        return vec.astype("float32")
    except Exception as e:
        print("SentenceTransformer embedding failed:", e)
        return None

# Try 1: LangChain FAISS wrapper
def try_langchain_faiss(vpath):
    try:
        try:
            from langchain.vectorstores import FAISS as LC_FAISS
        except Exception:
            from langchain_community.vectorstores import FAISS as LC_FAISS
        # try to instantiate same embedder as ingest (langchain huggingface wrapper)
        from langchain_community.embeddings import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        store = LC_FAISS.load_local(str(vpath), embeddings=emb)
        print("Loaded LangChain FAISS via LC_FAISS.load_local")
        return ("langchain", store, emb)
    except Exception as e:
        print("LangChain FAISS load_local failed:", e)
        return None

# Try 2: native faiss index + index.pkl (pickle metas)
def try_native_faiss(vpath):
    try:
        import faiss, numpy as np, pickle
        idx_file = vpath / "index.faiss"
        pkl_file = vpath / "index.pkl"
        if not idx_file.exists():
            print("Native faiss index.faiss not found at", idx_file)
            return None
        idx = faiss.read_index(str(idx_file))
        metas = []
        if pkl_file.exists():
            with open(pkl_file, "rb") as fh:
                metas = pickle.load(fh)
        else:
            # try metas.json
            mjson = vpath / "metas.json"
            if mjson.exists():
                with open(mjson, "r", encoding="utf8") as fh:
                    metas = json.load(fh)
        print("Loaded native FAISS index (index.faiss). Metas count:", len(metas))
        # Adapter with similarity_search_by_vector
        class Adapter:
            def __init__(self, index, metas):
                self.index = index
                self.metas = metas
            def similarity_search_by_vector(self, qvec, k=TOP_K):
                q = qvec.astype('float32')
                if q.ndim == 1:
                    q = q.reshape(1, -1)
                faiss.normalize_L2(q)
                D, I = self.index.search(q, k)
                out = []
                for dist, idx in zip(D[0], I[0]):
                    if idx < 0:
                        continue
                    meta = self.metas[idx] if idx < len(self.metas) else {}
                    # meta may be dict or string
                    page = meta.get('page_content') if isinstance(meta, dict) else ""
                    out.append((float(dist), idx, page, meta))
                return out
        return ("native_faiss", Adapter(idx, metas), None)
    except Exception as e:
        print("Native FAISS load failed:", e)
        return None

# Try 3: Chroma (sqlite) load
def try_chroma(vpath):
    try:
        # Many Chroma versions use chromadb or langchain-chroma wrapper.
        # We'll attempt langchain_community or langchain Chroma load_local
        try:
            from langchain_community.vectorstores import Chroma as LC_Chroma
            emb = None
            # try huggingface embedder
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            except Exception:
                emb = None
            store = LC_Chroma(persist_directory=str(vpath), embedding_function=emb)
            print("Loaded Chroma via langchain_community.Chroma")
            return ("chroma", store, emb)
        except Exception as e:
            print("Chroma load attempt failed:", e)
            return None
    except Exception as e:
        print("Chroma loader error:", e)
        return None

# Run tries
loaders = [try_langchain_faiss, try_native_faiss, try_chroma]
loaded = None
for fn in loaders:
    try:
        res = fn(VSTORE)
        if res:
            loaded = res
            break
    except Exception as e:
        print("Loader raised:", e)

if not loaded:
    print("\nNo vectorstore loader succeeded. You can still search raw source files in data/rag_sources.")
    sys.exit(1)

kind, store_obj, emb_obj = loaded
print("\nUsing loader:", kind)

# Create query vector
qvec = None
if emb_obj is not None:
    # Try to use embedding object's method if available
    try:
        if hasattr(emb_obj, "embed_query"):
            qv = emb_obj.embed_query(QUERY)
            import numpy as np
            qvec = np.asarray(qv, dtype="float32")
        elif hasattr(emb_obj, "embed_documents"):
            qv = emb_obj.embed_documents([QUERY])
            import numpy as np
            qvec = np.asarray(qv, dtype="float32")
    except Exception as e:
        print("Embedding with emb_obj failed:", e)
        qvec = None

if qvec is None:
    qvec = embed_with_st()
    if qvec is None:
        print("Cannot create query embedding — install sentence-transformers or confirm embedding object.")
        sys.exit(1)

# Query according to loader type
print("\nSearching for top", TOP_K, "results...\n")
try:
    if kind == "langchain":
        # LangChain store has similarity_search
        docs = store_obj.similarity_search(QUERY, k=TOP_K)
        for i, d in enumerate(docs, 1):
            pc = getattr(d, "page_content", "")[:400].replace("\n", " ")
            meta = getattr(d, "metadata", None)
            print(f"{i}. preview: {pc}\n   meta: {meta}\n")
    elif kind == "native_faiss":
        results = store_obj.similarity_search_by_vector(qvec, k=TOP_K)
        for i, (score, idx, page, meta) in enumerate(results, 1):
            preview = (page or "")[:400].replace("\n", " ")
            print(f"{i}. score={score:.4f} idx={idx} preview={preview}\n   meta={meta}\n")
    elif kind == "chroma":
        # Chroma store typical API: similarity_search or similar
        try:
            docs = store_obj.similarity_search(QUERY, k=TOP_K)
            for i, d in enumerate(docs, 1):
                pc = getattr(d, "page_content", "")[:400].replace("\n", " ")
                meta = getattr(d, "metadata", None)
                print(f"{i}. preview: {pc}\n   meta: {meta}\n")
        except Exception as e:
            # try query by vector
            try:
                # Chroma wrapper may expose similarity_search_by_vector
                docs = store_obj.similarity_search_by_vector(qvec[0], k=TOP_K)
                for i, d in enumerate(docs, 1):
                    pc = getattr(d, "page_content", "")[:400].replace("\n", " ")
                    meta = getattr(d, "metadata", None)
                    print(f"{i}. preview: {pc}\n   meta: {meta}\n")
            except Exception as e2:
                print("Chroma query failed:", e2)
except Exception as e:
    print("Search/query failed:", e)
