# try_chroma_load.py
from pathlib import Path
VSTORE = Path("vector_store")
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = Chroma(persist_directory=str(VSTORE), embedding_function=emb)
    docs = store.similarity_search("what is neft", k=5)
    print("Chroma returned", len(docs), "docs")
    for i,d in enumerate(docs,1):
        print(i, getattr(d,'page_content','')[:300].replace("\n"," "))
except Exception as e:
    print("Chroma load failed:", e)
