# inspect_metas.py
import pickle, json, sys
from pathlib import Path

p = Path("vector_store")
pkl = p / "index.pkl"
mjson = p / "metas.json"

if pkl.exists():
    with open(pkl, "rb") as fh:
        metas = pickle.load(fh)
    print("Loaded index.pkl (type, len):", type(metas), len(metas) if hasattr(metas, "__len__") else "unknown")
    # print first 5 metas
    for i, m in enumerate(metas[:5]):
        print(i, repr(m)[:300])
elif mjson.exists():
    with open(mjson, "r", encoding="utf8") as fh:
        metas = json.load(fh)
    print("Loaded metas.json len:", len(metas))
    for i, m in enumerate(metas[:5]):
        print(i, m)
else:
    print("No index.pkl or metas.json found in vector_store")
