# load_with_langchain.py
from pathlib import Path
VSTORE = Path("vector_store")
try:
    try:
        from langchain.vectorstores import FAISS as LC_FAISS
    except Exception:
        from langchain_community.vectorstores import FAISS as LC_FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # BE CAREFUL: it's OK here because files are local to you
    store = LC_FAISS.load_local(str(VSTORE), embeddings=emb, allow_dangerous_deserialization=True)
    docs = store.similarity_search("what is neft", k=5)
    for i,d in enumerate(docs,1):
        print(i, getattr(d,'page_content','')[:400].replace("\n"," "))
except Exception as e:
    print("LangChain load_local failed:", e)
