# final backend.py
"""
Full backend for SEBI Saathi (combined features + static SPA serving).
- Serves frontend build from frontend/dist (if exists)
- Contains DB models, auth, IPO analyzer, portfolio endpoints, news fetch, embeddings fallback, etc.
"""

import os
import shutil
import math
import tempfile
import traceback
import json
import random
import requests
from datetime import datetime
import urllib.parse
import time
import logging
import re
import pathlib
import numpy as np
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from urllib.parse import unquote, quote
from models import db, User, Portfolio, Holding, IpoReport


from flask import (
    Flask, request, jsonify, render_template, send_from_directory, session, Response, url_for, send_file, redirect
)
from flask_cors import CORS
from flask_login import current_user

# Auth / DB
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user, UserMixin
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
try:
    import pdfplumber
except Exception:
    pdfplumber = None

# LangChain / providers (placeholders; if not installed, initialization may fail but routes stay)
try:
    from langchain_chroma import Chroma
    from langchain.prompts import PromptTemplate
    from langchain_groq import ChatGroq
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    # Provide fallback names so file imports don't crash at import time; initialization will handle missing pieces.
    Chroma = None
    PromptTemplate = None
    ChatGroq = None
    HuggingFaceEmbeddings = None
    PyPDFLoader = None
    DirectoryLoader = None
    RecursiveCharacterTextSplitter = None

from functools import lru_cache
# markdown converter (server-side use)
from markdown import markdown
# from werkzeug.utils import secure_filename

# Local fallback embedder
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# --- FAISS / LangChain-FAISS detection (non-intrusive additions) ---
# Try LangChain FAISS wrapper at common import paths
try:
    try:
        from langchain.vectorstores import FAISS as LC_FAISS
        LC_FAISS_AVAILABLE = True
    except Exception:
        from langchain_community.vectorstores import FAISS as LC_FAISS
        LC_FAISS_AVAILABLE = True
except Exception:
    LC_FAISS = None
    LC_FAISS_AVAILABLE = False

# Try native faiss if installed
try:
    import faiss
    FAISS_NATIVE = True
except Exception:
    faiss = None
    FAISS_NATIVE = False
# --------------------------------------------------------------------

# --- Simple in-memory news cache ---
NEWS_CACHE = {}
NEWS_CACHE_TTL = 30 * 60  # 30 minutes

TITLE_RE = re.compile(r"^\s*(draft\s+red\s+herring\s+prospectus|red\s+herring\s+prospectus|prospectus|ipo)\b.*", re.I)

def derive_title_from_text(text: str) -> str:
    lines = [l.strip() for l in text.splitlines()]
    for l in lines[:200]:
        if not l or l.startswith(("http", "www", "scan", "qr", "page")): continue
        if len(l) > 8 and (l.isupper() or TITLE_RE.search(l)): return l[:180]
    for l in lines:
        if l and not l.lower().startswith(("please scan", "qr code", "table of contents")):
            return l[:180]
    return "IPO Report"

def summarize_with_llm(text: str, facts: dict, checks: dict):
    """
    Use LLM to summarize DRHP into a short report.
    """
    if not text.strip():
        return "No text to summarize."

    if not llm:
        return "LLM not available (using fallback)."

    prompt = f"""
    You are an analyst. Summarize the following IPO draft prospectus into:
    1. Company Overview
    2. Key Financial Highlights
    3. SEBI Compliance / Governance Issues
    4. Risks for Investors

    Facts extracted:
    - Revenue: {facts.get('revenue')}
    - Profit: {facts.get('profit')}
    - Debt: {facts.get('debt')}
    - Promoter mentioned: {facts.get('promoter_mentioned')}

    Heuristic checks:
    {json.dumps(checks, indent=2)}

    Document excerpt (truncated):
    {text[:8000]}  # limit to ~8k chars so prompt isn’t huge
    """

    try:
        res = llm.invoke(prompt)
        return res.content if hasattr(res, "content") else str(res)
    except Exception as e:
        app_logger().exception("LLM summarization failed")
        return f"LLM summarization failed: {e}"


def compute_ipo_score(text: str) -> int:
    t = text.lower()
    score = 50
    if "profit" in t or "pat" in t: score += 10
    if "revenue" in t or "total income" in t: score += 5
    if "dividend" in t: score += 3
    if "promoter" in t: score += 2
    for key, p in [("loss",10),("negative cash flow",10),("high debt",10),
                   ("pledge",8),("qualified opinion",12),("material uncertainty",10),
                   ("related party",4),("litigation",6)]:
        if key in t: score -= p
    return max(0, min(100, score))

FIN_NUM = re.compile(r"(?P<label>revenue|total\s+income|profit|pat|loss|debt|borrowings)[:\s₹$]*([\d,]+(\.\d+)?)\s*(cr|crore|bn|billion|mn|million|lakh|k)?", re.I)

def extract_quick_facts(text: str):
    facts = {"revenue": None, "profit": None, "debt": None, "promoter_mentioned": ("promoter" in text.lower())}
    for m in FIN_NUM.finditer(text[:120_000]):
        label = m.group("label").lower()
        val = m.group(0)
        if "revenue" in label or "total income" in label:
            facts["revenue"] = facts["revenue"] or val
        elif "profit" in label or "pat" in label or "loss" in label:
            facts["profit"]  = facts["profit"]  or val
        elif "debt" in label or "borrowings" in label:
            facts["debt"]    = facts["debt"]    or val
    return facts

# --- Helpers for news ---
def _is_probable_ticker(text: str) -> bool:
    t = (text or "").strip()
    if not t: return False
    if " " in t: return False
    return len(t) <= 20 and any(c.isalpha() for c in t)

def _safe_combine_title_desc(title: str, desc: str) -> str:
    title = (title or "").strip()
    desc = (desc or "").strip()
    if not title and not desc: return ""
    if not desc or desc.lower() in title.lower(): return title
    return f"{title} — {desc}"

def _recent_date_range(days: int = 14):
    try:
        to_dt = datetime.utcnow().date()
        from_dt = to_dt.fromordinal(to_dt.toordinal() - max(1, days))
        return from_dt.isoformat(), to_dt.isoformat()
    except Exception:
        return "2024-01-01", datetime.utcnow().date().isoformat()

def _fetch_news_finnhub(symbol: str, max_headlines: int):
    token = (os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_TOKEN") or "").strip()
    if not token: return []
    frm, to = _recent_date_range(14)
    try:
        url = "https://finnhub.io/api/v1/company-news"
        su = (symbol or "").upper()
        base = su.split('.')[0] if '.' in su else su
        if su.endswith(('.NS', '.NSE')):
            norm = f"{base}.NS"
        elif su.endswith(('.BO', '.BSE')):
            norm = f"{base}.BO"
        else:
            norm = su
        params = {"symbol": norm, "from": frm, "to": to, "token": token}
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json() or []
        out = []
        for item in data[:max_headlines]:
            title = item.get("headline") or item.get("title")
            summary = item.get("summary") or ""
            combined = _safe_combine_title_desc(title, summary)
            if combined: out.append(combined)
        return out
    except Exception as e:
        app_logger().warning(f"Finnhub news error for {symbol}: {e}")
        return []

def _fetch_news_marketaux(symbol: str, max_headlines: int):
    token = (os.getenv("MARKETAUX_API_KEY") or os.getenv("MARKETAUX_TOKEN") or "").strip()
    if not token: return []
    try:
        url = "https://api.marketaux.com/v1/news/all"
        su = (symbol or "").upper()
        norm = su.split('.')[0] if '.' in su else symbol
        params = {
            "symbols": norm,
            "filter_entities": "true",
            "language": "en",
            "countries": os.getenv("NEWS_COUNTRIES", "in"),
            "limit": max_headlines,
            "sort": "published_at:desc",
            "api_token": token,
        }
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json() or {}
        articles = data.get("data") or []
        if not articles:
            search_params = {
                "search": norm,
                "filter_entities": "true",
                "language": "en",
                "countries": os.getenv("NEWS_COUNTRIES", "in"),
                "limit": max_headlines,
                "sort": "published_at:desc",
                "api_token": token,
            }
            r2 = requests.get(url, params=search_params, timeout=8)
            r2.raise_for_status()
            data2 = r2.json() or {}
            articles = data2.get("data") or []
        out = []
        for a in articles[:max_headlines]:
            title = a.get("title")
            desc = a.get("description") or ""
            combined = _safe_combine_title_desc(title, desc)
            if combined: out.append(combined)
        return out
    except Exception as e:
        app_logger().warning(f"Marketaux news error for {symbol}: {e}")
        return []

def _fetch_news_alphavantage(symbol: str, max_headlines: int):
    key = (os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHAVANTAGE_TOKEN") or "").strip()
    if not key: return []
    try:
        url = "https://www.alphavantage.co/query"
        su = (symbol or "").upper()
        norm = su.split('.')[0] if '.' in su else symbol
        params = {"function":"NEWS_SENTIMENT","tickers":norm,"sort":"LATEST","apikey":key}
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json() or {}
        feed = data.get("feed") or []
        out = []
        for item in feed[:max_headlines]:
            title = item.get("title")
            summary = item.get("summary") or ""
            combined = _safe_combine_title_desc(title, summary)
            if combined: out.append(combined)
        return out
    except Exception as e:
        app_logger().warning(f"AlphaVantage news error for {symbol}: {e}")
        return []

def _fetch_news_newsapi(query: str, max_headlines: int, api_key: str):
    try:
        q = f'"{query}" (stock OR shares OR company) (NSE OR BSE OR India)'
        params = {"q": q, "language": "en", "sortBy":"publishedAt", "pageSize": max_headlines, "apiKey": api_key}
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=8)
        r.raise_for_status()
        data = r.json() or {}
        if data.get("status") == "ok":
            out = []
            for a in (data.get("articles") or [])[:max_headlines]:
                title = (a.get("title") or "").strip()
                desc  = (a.get("description") or "").strip()
                combined = _safe_combine_title_desc(title, desc)
                if combined: out.append(combined)
            return out
    except Exception as e:
        app_logger().warning(f"NewsAPI error for {query}: {e}")
    return []

def _fetch_news_google(query: str, max_headlines: int):
    try:
        q = urllib.parse.quote_plus(f"{query} stock")
        search_url = f"https://www.google.com/search?q={q}&tbm=nws"
        headers = {"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari:537.36"}
        r = requests.get(search_url, headers=headers, timeout=8)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        items = []
        for g in soup.select("div.dbsr"):
            title_el = g.select_one("div.JheGif, div.JheGif.nDgy9d, div[role='heading']")
            title = title_el.get_text(strip=True) if title_el else ""
            if title: items.append(title)
            if len(items) >= max_headlines: break
        if len(items) < max_headlines:
            for a in soup.select("a.DY5T1d"):
                title = a.get_text(strip=True)
                if title: items.append(title)
                if len(items) >= max_headlines: break
        return items
    except Exception as e:
        app_logger().warning(f"Google news scrape failed for {query}: {e}")
        return []

def fetch_stock_headlines(symbol: str, max_headlines: int = 5) -> list:
    s = (symbol or "").strip()
    if not s: return []
    if _is_probable_ticker(s):
        for fetcher in (_fetch_news_finnhub, _fetch_news_marketaux, _fetch_news_alphavantage):
            try:
                items = fetcher(s, max_headlines)
            except Exception:
                items = []
            if items: return items
    items = _fetch_news_google(s, max_headlines)
    if items: return items
    newsapi_key = (os.getenv("NEWSAPI_KEY") or "").strip()
    if newsapi_key:
        items = _fetch_news_newsapi(s, max_headlines, newsapi_key)
        if items: return items
    return []

def cached_headlines(key: str, fetch_fn, ttl: int = NEWS_CACHE_TTL):
    now = time.time()
    cached = NEWS_CACHE.get(key)
    if cached and (now - cached['time']) < ttl:
        return cached['payload']
    try:
        items = fetch_fn() or []
    except Exception:
        items = []
    NEWS_CACHE[key] = {'time': now, 'payload': items}
    return items

# --- Load env ---
load_dotenv()

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("backend")
def app_logger():
    return logger

# --- Paths & Flask app init ---
basedir = os.path.abspath(os.path.dirname(__file__))
# FRONTEND_DIST is default frontend/dist relative to this backend file
FRONTEND_DIST = os.path.join(basedir, "frontend", "dist")
# If not present, check alternative places (e.g., backend/static/dist)
if not os.path.isdir(FRONTEND_DIST):
    alt = os.path.join(basedir, "static", "dist")
    if os.path.isdir(alt):
        FRONTEND_DIST = alt

app = Flask(__name__, static_folder=None, template_folder=os.path.join(basedir, "templates"))
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///financial_bot.db")
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)
with app.app_context():
    db.create_all()
# We set static_folder=None because we serve the SPA build via custom routes
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

app_logger().info("FRONTEND_DIST resolved to: %s", FRONTEND_DIST)

# Cookie/session settings helpful for cross-port dev (adjust for production!)
app.config['SESSION_COOKIE_NAME'] = os.getenv('SESSION_COOKIE_NAME', 'sebisession')
app.config['SESSION_COOKIE_HTTPONLY'] = True
# For local dev where frontend is on different port, allow cross-site cookies.
# In production you may want 'Lax' or 'Strict' and set SESSION_COOKIE_SECURE = True
same_site = os.getenv('SESSION_COOKIE_SAMESITE', 'Lax')  # changed default to 'Lax' for local dev
app.config['SESSION_COOKIE_SAMESITE'] = same_site
app.config['SESSION_COOKIE_SECURE'] = os.getenv('SESSION_COOKIE_SECURE', 'False').lower() in ('true', '1', 'yes')

# Enable credentialed CORS so browser sends cookies from your frontend origins
CORS(app,
     supports_credentials=True,
     resources={r"/*": {"origins": [
         "http://localhost:3000",
         "http://127.0.0.1:3000",
         "http://localhost:5173",
         "http://127.0.0.1:5173",
         "http://localhost:5174",
         "http://127.0.0.1:5174"
     ]}})

# --- Database config ---
# DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///financial_bot.db")
# app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# db = SQLAlchemy(app)

# --- Flask-Login ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login_page"

# IMPORTANT: For API/XHR requests return JSON 401 (instead of redirecting to login HTML).
@login_manager.unauthorized_handler
def unauthorized_callback():
    wants_json = (
        request.accept_mimetypes.accept_json
        or request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or request.is_json
        or request.headers.get("Accept") == "application/json"
    )
    if wants_json:
        return jsonify({"error": "authentication_required"}), 401
    return redirect(url_for("login_page"))

# --- Global variables & directories ---
VECTOR_STORE_PATH_MAIN = os.path.join(basedir, "vector_store")
USER_DATA_PATH = os.path.join(basedir, "user_data")
DATA_PATH = os.path.join(basedir, "data")
RAG_SOURCES_PATH = os.path.join(DATA_PATH, "rag_sources")

# LLM / embeddings / vector DB placeholders
llm = None
embeddings = None
db_main = None
qa_prompt = None
analysis_prompt = None
scam_data = []
myth_data = []

# Fallback embedding config
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_CACHE_SIZE = int(os.getenv("EMBED_CACHE_SIZE", "1024"))
EMBED_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "4"))
EMBED_BACKOFF_BASE = float(os.getenv("EMBED_BACKOFF_BASE", "1.0"))

# --- Flask-Login user loader ---
@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except Exception:
        return None

# --- Helpers ---
def safe_load_json(path):
    if not os.path.exists(path):
        logger.warning(f"Missing JSON file: {path}")
        return []
    try:
        with open(path, 'r', encoding='utf8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return []

# -----------------------
# Embedding fallback helpers
# -----------------------
_local_embedder = None
def get_local_embedder():
    global _local_embedder
    if _local_embedder is None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed. Install with: pip install sentence-transformers")
        logger.info("Loading SentenceTransformers model: %s", LOCAL_EMBED_MODEL)
        _local_embedder = SentenceTransformer(LOCAL_EMBED_MODEL)
    return _local_embedder

@lru_cache(maxsize=EMBED_CACHE_SIZE)
def _cached_local_embed(text: str):
    model = get_local_embedder()
    vec = model.encode(text)
    return tuple(float(x) for x in vec.tolist())

def local_embed_list(text: str): return list(_cached_local_embed(text))

def with_retries(fn, max_attempts=EMBED_MAX_RETRIES):
    attempt = 1
    while True:
        try:
            return fn()
        except Exception as e:
            if attempt >= max_attempts:
                logger.error("with_retries: max attempts reached (%d). Raising.", max_attempts)
                raise
            sleep_for = EMBED_BACKOFF_BASE * (2 ** (attempt - 1)) + random.random()
            logger.warning("with_retries: attempt %d/%d failed: %s. sleeping %.2fs before retry",
                           attempt, max_attempts, repr(e), sleep_for)
            time.sleep(sleep_for); attempt += 1

def similarity_search_with_fallback(query: str, k: int = 4, user_vector_store_path: str = None):
    global db_main, embeddings
    target_db = None
    try:
        if user_vector_store_path:
            if os.path.exists(user_vector_store_path) and Chroma is not None:
                target_db = Chroma(persist_directory=user_vector_store_path, embedding_function=embeddings)
                logger.debug("Using user vectorstore at %s", user_vector_store_path)
            else:
                logger.debug("User vectorstore path not found or Chroma unavailable: %s", user_vector_store_path)
                target_db = None
        if target_db is None:
            if db_main is None:
                raise RuntimeError("Main vectorstore not initialized.")
            target_db = db_main
        try:
            logger.debug("Primary similarity_search via vectorstore (query='%s' k=%d)", query[:120], k)
            return with_retries(lambda: target_db.similarity_search(query, k=k))
        except Exception as primary_exc:
            logger.warning("Primary similarity_search failed: %s. Falling back to local embedding.", repr(primary_exc))
            qvec = local_embed_list(query)
            if hasattr(target_db, "similarity_search_by_vector"):
                return target_db.similarity_search_by_vector(qvec, k=k)
            elif hasattr(target_db, "similarity_search_with_score_by_vector"):
                return target_db.similarity_search_with_score_by_vector(qvec, k=k)
            else:
                return target_db.similarity_search(qvec, k=k)
    except Exception:
        logger.exception("Both primary and fallback similarity_search attempts failed.")
        raise

# -----------------------
# FAISS fallback loader (non-intrusive)
# -----------------------
def try_load_faiss_store(path: str, embeddings_obj=None):
    """
    Try to load a FAISS-backed store from `path`.
    Order:
      1) Prefer LangChain wrapper load_local(..., allow_dangerous_deserialization=True)
      2) Fallback: native faiss.index + metas.json / index.pkl adapter
    Returns an object that exposes similarity_search(query, k) or similarity_search_by_vector(vec, k)
    """
    if not os.path.exists(path):
        app_logger().info("try_load_faiss_store: path does not exist: %s", path)
        return None

    # 1) Try LangChain wrapper load_local with allow_dangerous_deserialization=True
    if LC_FAISS_AVAILABLE and LC_FAISS is not None:
        try:
            # Some LangChain versions accept 'embeddings=' parameter name
            try:
                store = LC_FAISS.load_local(path, embeddings=embeddings_obj, allow_dangerous_deserialization=True)
            except TypeError:
                # older/newer signatures might differ; try positional
                store = LC_FAISS.load_local(path, allow_dangerous_deserialization=True)
            app_logger().info("Loaded FAISS store via LangChain wrapper from %s (deserialized)", path)
            return store
        except Exception as e:
            app_logger().warning("LC_FAISS.load_local (deserialized) failed for %s: %s", path, e)

    # 2) Try native FAISS + metas.json / index.pkl adapter
    try:
        faiss_index_path = os.path.join(path, 'index.faiss')
        metas_path_json = os.path.join(path, 'metas.json')
        metas_path_pkl = os.path.join(path, 'index.pkl')
        if FAISS_NATIVE and os.path.exists(faiss_index_path):
            idx = faiss.read_index(faiss_index_path)
            # Load metas: prefer metas.json, then index.pkl
            metas = []
            if os.path.exists(metas_path_json):
                try:
                    with open(metas_path_json, 'r', encoding='utf8') as fh:
                        metas = json.load(fh)
                except Exception:
                    app_logger().warning("Failed to load metas.json; will try index.pkl")
            if not metas and os.path.exists(metas_path_pkl):
                try:
                    import pickle
                    with open(metas_path_pkl, 'rb') as fh:
                        metas = pickle.load(fh)
                    # If pickle is a tuple like (InMemoryDocstore, mapping), try to extract mapping values
                    if isinstance(metas, tuple) and len(metas) >= 2:
                        # some langchain versions persist (docstore, id_to_uuid_mapping)
                        docstore, idmap = metas[0], metas[1]
                        # if docstore has get method, reconstruct metas by idmap order
                        try:
                            reconstructed = []
                            if isinstance(idmap, dict):
                                for i in range(len(idmap)):
                                    uuid = idmap.get(i) or idmap.get(str(i))
                                    if uuid and hasattr(docstore, "search") is False:
                                        # try to fetch doc by key if available (docstore may expose dict-like)
                                        try:
                                            doc = docstore._dict.get(uuid) if hasattr(docstore, "_dict") else None
                                            if doc:
                                                reconstructed.append({"page_content": getattr(doc, "page_content", ""), **getattr(doc, "metadata", {})})
                                                continue
                                        except Exception:
                                            pass
                            if reconstructed:
                                metas = reconstructed
                        except Exception:
                            pass
                except Exception:
                    app_logger().warning("Failed to unpickle index.pkl; metas may be incomplete.")
            # If still no metas, keep metas as empty list
            if not metas:
                app_logger().warning("Loaded faiss index but no metas found (path=%s). Search may return empty metadata.", path)

            class FaissAdapter:
                def __init__(self, index, metas, embed_fn):
                    self.index = index
                    self.metas = metas or []
                    self.embed_fn = embed_fn

                def similarity_search(self, query, k=4):
                    try:
                        qv = self.embed_fn([query])
                        qv = np.asarray(qv, dtype=np.float32)
                        faiss.normalize_L2(qv)
                        D, I = self.index.search(qv, k)
                        out = []
                        for idx_i, dist in zip(I[0], D[0]):
                            if idx_i < 0: continue
                            meta = self.metas[idx_i] if idx_i < len(self.metas) else {}
                            class Doc:
                                def __init__(self, page_content, metadata):
                                    self.page_content = page_content
                                    self.metadata = metadata
                            page = meta.get('page_content') or meta.get('text') or meta.get('content') or ''
                            out.append(Doc(page, meta))
                        return out
                    except Exception:
                        app_logger().exception('FaissAdapter search failed')
                        return []

                def similarity_search_by_vector(self, vec, k=4):
                    try:
                        v = np.asarray(vec, dtype=np.float32)
                        faiss.normalize_L2(v)
                        D, I = self.index.search(v, k)
                        out = []
                        for idx_i in I[0]:
                            if idx_i < 0: continue
                            meta = self.metas[idx_i] if idx_i < len(self.metas) else {}
                            class Doc:
                                def __init__(self, page_content, metadata):
                                    self.page_content = page_content
                                    self.metadata = metadata
                            page = meta.get('page_content') or meta.get('text') or meta.get('content') or ''
                            out.append(Doc(page, meta))
                        return out
                    except Exception:
                        app_logger().exception('FaissAdapter vector search failed')
                        return []

            # Determine embedding function: prefer embeddings_obj.embed_query/embed_documents if present; else SentenceTransformer
            embed_fn = None
            if embeddings_obj is not None:
                try:
                    if hasattr(embeddings_obj, 'embed_query') and hasattr(embeddings_obj, 'embed_documents'):
                        embed_fn = lambda texts: embeddings_obj.embed_documents(texts)
                    elif hasattr(embeddings_obj, 'embed_documents'):
                        embed_fn = lambda texts: embeddings_obj.embed_documents(texts)
                except Exception:
                    embed_fn = None
            if embed_fn is None:
                if SentenceTransformer is not None:
                    st = SentenceTransformer(os.getenv('LOCAL_EMBED_MODEL','all-MiniLM-L6-v2'))
                    embed_fn = lambda texts: st.encode(texts, convert_to_numpy=True)
                else:
                    app_logger().warning("No embedding function available to use with native FAISS adapter.")
                    return None
            return FaissAdapter(idx, metas, embed_fn)
    except Exception:
        app_logger().exception('Failed to load native FAISS store')
        return None

    return None


# -----------------------
# Initialize models, LLM, embeddings, prompts, vector DB
# -----------------------
def initialize_app():
    global llm, embeddings, db_main, qa_prompt, analysis_prompt, scam_data, myth_data
    logger.info("--- STARTING INITIALIZATION ---")
    try:
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not set. LLM may fail.")
        # instantiate LLM only if ChatGroq is available
        if ChatGroq is not None:
            llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
        else:
            llm = None
        if HuggingFaceEmbeddings is not None:
            try:
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            except Exception:
                embeddings = None
        else:
            embeddings = None

        os.makedirs(VECTOR_STORE_PATH_MAIN, exist_ok=True)

        # Prefer Chroma if available and it appears to contain data; otherwise fall back to FAISS loader.
        db_main = None
        if Chroma is not None and embeddings is not None:
            try:
                app_logger().info("Attempting to open Chroma vectorstore at: %s", VECTOR_STORE_PATH_MAIN)
                chroma_store = None
                try:
                    chroma_store = Chroma(persist_directory=VECTOR_STORE_PATH_MAIN, embedding_function=embeddings)
                except Exception as e_ch:
                    app_logger().warning("Failed to instantiate Chroma store: %s", e_ch)
                    chroma_store = None

                # If we created a chroma_store, do a small probe to ensure it actually has documents.
                got_any = False
                if chroma_store is not None:
                    try:
                        # Best-effort probe: try similarity_search("test") if available.
                        if hasattr(chroma_store, "similarity_search"):
                            probe = None
                            try:
                                probe = chroma_store.similarity_search("test", k=1)
                                got_any = bool(probe)
                            except Exception:
                                # Some Chroma wrappers raise on unknown queries; mark as unknown and continue
                                got_any = False
                        else:
                            # If store has a method to list collections/ids, try that (best-effort)
                            if hasattr(chroma_store, "get"):
                                got_any = True
                            else:
                                # Unknown store shape — assume it may have data to avoid false negative
                                got_any = True
                    except Exception as probe_exc:
                        app_logger().warning("Chroma probe failed: %s", probe_exc)
                        got_any = False

                if got_any:
                    db_main = chroma_store
                    app_logger().info("Chroma vectorstore opened and appears to contain data — using Chroma.")
                else:
                    # Chroma opened but looked empty (or probe failed). Close/ignore and try FAISS fallback.
                    try:
                        # attempt to cleanup chroma store if it has a close method
                        if chroma_store is not None and hasattr(chroma_store, "persist"):
                            try:
                                chroma_store.persist()
                            except Exception:
                                pass
                    except Exception:
                        pass
                    app_logger().warning("Chroma store appears empty or probe failed — falling back to FAISS loader.")
                    faiss_store = try_load_faiss_store(VECTOR_STORE_PATH_MAIN, embeddings_obj=embeddings)
                    if faiss_store is not None:
                        db_main = faiss_store
                        app_logger().info("FAISS store loaded into db_main (fallback).")
                    else:
                        db_main = None
                        app_logger().warning("FAISS fallback did not find a usable store.")
            except Exception:
                app_logger().exception("Unexpected error initializing Chroma/FAISS fallback.")
                db_main = None
        else:
            # Chroma or embeddings not available — try FAISS load directly
            app_logger().info("Chroma not available or embeddings missing; attempting FAISS load as fallback")
            faiss_store = try_load_faiss_store(VECTOR_STORE_PATH_MAIN, embeddings_obj=embeddings)
            if faiss_store is not None:
                db_main = faiss_store
                app_logger().info("FAISS store loaded into db_main (direct).")
            else:
                db_main = None
                app_logger().warning("No vectorstore found (Chroma unavailable and FAISS fallback failed).")

        scam_data = safe_load_json(os.path.join(DATA_PATH, 'scam_examples.json'))
        myth_data = safe_load_json(os.path.join(DATA_PATH, 'myths.json'))
        logger.info("Engagement data loaded (scam examples: %d, myths: %d).", len(scam_data), len(myth_data))

        qa_prompt_template = (
            "CONTEXT: {context}\n"
            "QUESTION: {question}\n"
            "INSTRUCTIONS: Based ONLY on the context, answer the user's question. If the answer is not in the context, say so."
        )
        qa_prompt = PromptTemplate(template=qa_prompt_template, input_variables=["context", "question"]) if PromptTemplate is not None else None

        analysis_prompt_template = (
            "You are 'SEBI Saathi', an expert portfolio analyst. Analyze the user's portfolio based on the data provided. "
            "Provide a 'Portfolio Health Check' as Markdown. Do NOT give investment advice.\n"
            "USER'S PORTFOLIO DATA:\n{portfolio_data}\nANALYSIS:"
        )
        analysis_prompt = PromptTemplate(template=analysis_prompt_template, input_variables=["portfolio_data"]) if PromptTemplate is not None else None

        os.makedirs(USER_DATA_PATH, exist_ok=True)
        return True
    except Exception:
        logger.exception("FATAL ERROR DURING INITIALIZATION")
        return False

# -----------------------
# --- Static SPA serving helpers ---
# -----------------------
def dist_file_path(*parts):
    return os.path.join(FRONTEND_DIST, *parts)

# Serve asset directories if frontend build exists
if os.path.isdir(FRONTEND_DIST):
    app_logger().info("Serving SPA from: %s", FRONTEND_DIST)

    @app.route('/assets/<path:filename>')
    def _serve_assets(filename):
        assets_dir = os.path.join(FRONTEND_DIST, "assets")
        candidate = os.path.join(assets_dir, filename)
        if os.path.isfile(candidate):
            return send_from_directory(assets_dir, filename)
        return jsonify({'error': 'Not found'}), 404

    @app.route('/dist/<path:filename>')
    def _serve_dist(filename):
        candidate = dist_file_path(filename)
        if os.path.isfile(candidate):
            return send_from_directory(FRONTEND_DIST, filename)
        return jsonify({'error': 'Not found'}), 404

    # Serve root-level static files referenced by index.html (like favicon)
    @app.route('/static_dist/<path:filename>')
    def _serve_static_dist(filename):
        candidate = dist_file_path(filename)
        if os.path.isfile(candidate):
            return send_from_directory(FRONTEND_DIST, filename)
        return jsonify({'error': 'Not found'}), 404

# If dist not present, logging will show fallback to templates
else:
    app_logger().warning("Frontend dist not found at %s — SPA won't be served from dist. Falling back to templates if available.", FRONTEND_DIST)

# Serve SPA index for root and for client-side routes. If dist available, use it.
# Serve the SPA index for client-side routes (do NOT require login here)
@app.route('/dashboard')
def dashboard_page():
    idx = dist_file_path('index.html')
    if os.path.exists(idx):
        # Serve the SPA entry point. React Router will decide what to render on the client.
        return send_file(idx)
    # Fallback dev-mode template (only used if you kept a server-side template)
    try:
        return render_template('dashboard.html')
    except Exception:
        return "<h3>Dashboard template not found and frontend not built.</h3>", 500

# Also ensure /, /login, /register behave the same (serve index.html)
@app.route('/')
def index():
    idx = dist_file_path('index.html')
    if os.path.exists(idx):
        return send_file(idx)
    if 'anon_id' not in session:
        session['anon_id'] = os.urandom(8).hex()
    try:
        return render_template('index.html')
    except Exception:
        return "<h3>Frontend not built. Run `npm run build` in frontend/ and restart backend.</h3>", 500

@app.route('/login')
def login_page():
    idx = dist_file_path('index.html')
    if os.path.exists(idx):
        return send_file(idx)
    try:
        return render_template('login.html')
    except Exception:
        return "<h3>Login page not found and frontend not built.</h3>", 500

@app.route('/register')
def register_page():
    idx = dist_file_path('index.html')
    if os.path.exists(idx):
        return send_file(idx)
    try:
        return render_template('register.html')
    except Exception:
        return "<h3>Register page not found and frontend not built.</h3>", 500

# -----------------------
# --- API Routes (restored from your original code)
# -----------------------

@app.route('/sebi/circulars')
def sebi_circulars():
    url = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=7&smid=0"
    try:
        res = requests.get(url, timeout=10); res.raise_for_status()
    except Exception as e:
        return jsonify({'circulars': [], 'error': f'Failed to fetch SEBI site: {e}'}), 502
    soup = BeautifulSoup(res.content, 'html.parser')
    table = soup.find('table')
    items = []
    if table:
        rows = table.find_all('tr')[1:6]
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 2:
                date = cols[0].get_text(strip=True)
                title_tag = cols[1].find('a')
                if not title_tag:
                    title = cols[1].get_text(strip=True); full_link = url
                else:
                    title = title_tag.get_text(strip=True); link = title_tag.get('href', '')
                    full_link = requests.compat.urljoin(url, link)
                items.append({'date': date, 'title': title, 'url': full_link})
    return jsonify({'circulars': items})

@app.route('/market/live')
def market_live():
    try:
        tickers = {"NIFTY 50":"^NSEI","SENSEX":"^BSESN","CDSL":"CDSL.NS","KFINTECH":"KFINTECH.NS"}
        data = {}
        for name, symbol in tickers.items():
            try:
                ticker = yf.Ticker(symbol); hist = ticker.history(period="2d")
                if len(hist) >= 2:
                    prev_close = hist['Close'].iloc[-2]; last_close = hist['Close'].iloc[-1]
                    change = last_close - prev_close
                    pct_change = (change / prev_close) * 100 if prev_close else 0
                    data[name] = {"price": round(float(last_close),2), "change": round(float(change),2), "pct_change": round(float(pct_change),2)}
                else:
                    data[name] = {"price": None, "change": None, "pct_change": None}
            except Exception as inner_e:
                data[name] = {"error": f"failed to fetch {symbol}: {inner_e}"}
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json(force=True, silent=True) or {}
    question = data.get('question', '')
    search_scope = data.get('scope', 'all')
    user_id = session.get('user_id')
    if not question: return jsonify({'error': 'No question provided.'}), 400
    try:
        docs = []
        user_vector_store_path = os.path.join(USER_DATA_PATH, str(user_id or ''), "vector_store")
        if search_scope == 'user_only' and os.path.exists(user_vector_store_path):
            try:
                docs = similarity_search_with_fallback(question, k=4, user_vector_store_path=user_vector_store_path)
            except Exception as e:
                logger.warning("User-only search failed, falling back to main store: %s", repr(e))
                docs = similarity_search_with_fallback(question, k=4, user_vector_store_path=None)
        else:
            docs = similarity_search_with_fallback(question, k=4, user_vector_store_path=None)
        context = "\n\n".join([getattr(doc, 'page_content', str(doc)) for doc in (docs or [])])
        if not context.strip(): context = "No relevant information was found in the selected knowledge base."
        formatted_prompt = qa_prompt.format(context=context, question=question) if qa_prompt is not None else f"CONTEXT:{context}\nQUESTION:{question}"
        if llm is None:
            # fallback simple response
            return jsonify({'answer': "LLM not configured on server. Please check GROQ_API_KEY."})
        result = llm.invoke(formatted_prompt)
        answer = getattr(result, 'content', result) if result is not None else "No response from LLM."
        return jsonify({'answer': answer})
    except Exception:
        logger.exception("--- AN ERROR OCCURRED IN /ask ---")
        return jsonify({'error': 'A server error occurred.'}), 500
    
# Add this near your other /ask endpoints in backend.py

# backend.py (add near your other /ask routes)

# Add near your other /ask endpoints

# backend: add or replace this route in your Flask app

# backend_endpoints.py

import os, time, traceback
from flask import request, jsonify, session
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

# langchain loaders
try:
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    PyPDFLoader = DirectoryLoader = RecursiveCharacterTextSplitter = None

# global paths
USER_DATA_PATH = os.getenv("USER_DATA_PATH", "./user_data")
ALLOWED_USER_DOC_EXTS = {".pdf", ".txt"}

# embeddings wrapper (replace with your embedder)
try:
    from sentence_transformers import SentenceTransformer
    _embed_model = SentenceTransformer(os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2"))
    class _EmbWrap:
        def embed_documents(self, texts):
            return _embed_model.encode(texts, convert_to_numpy=True).tolist()
    emb = _EmbWrap()
except Exception:
    emb = None


# ------------------------------------------------------------------
# /ask_user
# ------------------------------------------------------------------
# Add required imports near top of your file if not already present:
# import os, time, shutil, traceback
# from flask import request, jsonify, session
# from werkzeug.utils import secure_filename
# Ensure you already have: PyPDFLoader, DirectoryLoader, RecursiveCharacterTextSplitter, embeddings (or local_embed_list)
# and app_logger(), USER_DATA_PATH, ALLOWED_USER_DOC_EXTS defined in your module.

# ---------------------------
# /ask_user
# ---------------------------
@app.route('/ask_user', methods=['POST'])
@login_required
def ask_user():
    payload = request.get_json() or {}
    question = (payload.get('question') or "").strip()
    if not question:
        return jsonify({'error': 'Missing question'}), 400

    try:
        user_id = current_user.id
    except Exception:
        user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'not authenticated'}), 401

    user_vector_dir = os.path.join(USER_DATA_PATH, str(user_id), "user_vector_store")
    user_collection_name = "user_docs"

    # quick: no vector store -> return friendly message
    if not os.path.exists(user_vector_dir):
        ctx_preview = "No relevant information was found in your library."
        return jsonify({
            'answer': "No relevant information was found in your library.",
            'context_preview': ctx_preview,
            'debug': {'reason': 'no_store'}
        })

    # Query vector store via chromadb (new client)
    top_k = int(payload.get('top_k', 3))
    snippets = []
    try:
        import chromadb
        from chromadb.config import Settings

        # Create client pointing to user's vector dir
        client = chromadb.Client(Settings(persist_directory=user_vector_dir))

        # Get collection; if missing, return no-store message
        try:
            collection = client.get_collection(user_collection_name)
        except Exception:
            collection = None

        if not collection:
            ctx_preview = "No relevant information was found in your library."
            return jsonify({'answer': ctx_preview, 'context_preview': ctx_preview, 'debug': {'reason': 'collection-not-found'}})

        # Create query embedding using your embedding wrapper object `emb`
        # emb must have embed_documents(list[str]) -> list[vectors]
        q_embs = None
        try:
            q_embs = emb.embed_documents([question])
            if not (isinstance(q_embs, list) and len(q_embs) == 1):
                raise RuntimeError("embed_documents did not return a single vector")
            q_emb = q_embs[0]
        except Exception as ex:
            app_logger().exception("Embedding for query failed")
            ctx_preview = "No relevant information was found in your library."
            return jsonify({'answer': ctx_preview, 'context_preview': ctx_preview, 'debug': {'embed_error': str(ex)}})

        # Query the collection using query_embeddings (new API)
        try:
            query_res = collection.query(
                query_embeddings=[q_emb],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
        except Exception as ex:
            app_logger().exception("Collection query failed")
            ctx_preview = "No relevant information was found in your library."
            return jsonify({'answer': ctx_preview, 'context_preview': ctx_preview, 'debug': {'query_error': str(ex)}})

        # Parse results - chromadb returns lists-of-lists
        docs_lists = query_res.get('documents', [[]])[0] if isinstance(query_res.get('documents'), list) else []
        dists = query_res.get('distances', [[]])[0] if isinstance(query_res.get('distances'), list) else []
        metas_lists = query_res.get('metadatas', [[]])[0] if isinstance(query_res.get('metadatas'), list) else []

        for i, doc_txt in enumerate(docs_lists):
            dist = dists[i] if i < len(dists) else None
            meta = metas_lists[i] if i < len(metas_lists) else {}
            preview = (doc_txt or "")[:1200]  # limit length
            if dist is None:
                snippets.append(f"{preview}")
            else:
                snippets.append(f"({dist:.4f}) {preview} — source: {meta.get('source') or meta.get('file') or 'unknown'}")

    except Exception as ex:
        app_logger().exception("ask_user vector query failed")
        ctx_preview = "No relevant information was found in your library."
        return jsonify({'answer': ctx_preview, 'context_preview': ctx_preview, 'debug': {'exception': str(ex), 'trace': traceback.format_exc()}})

    # Prepare context preview to show to user
    if not snippets:
        ctx_preview = "No relevant information was found in your library."
    else:
        ctx_preview = "\n\n---\n\n".join(snippets)

    # DEBUG: Always show snippets
    app_logger().info("Retrieved snippets: %s", ctx_preview)


    # Minimal behaviour: return number of snippets and preview.
    # You can replace the following with an LLM call that synthesizes an answer using `snippets`.
    answer = f"I found {len(snippets)} snippet(s) in your library. Ask again to synthesize an answer using retrieved context."

    return jsonify({'answer': answer, 'context_preview': ctx_preview, 'debug': {'snippets_returned': len(snippets)}})


# ---------------------------
# /upload_and_ingest
# ---------------------------
@app.route('/upload_and_ingest', methods=['POST'])
@login_required
def upload_and_ingest():
    """
    Save uploaded PDF, split into chunks, embed and add to a chromadb collection.
    Uses the new chromadb.Client API (no client.persist()).
    Falls back to a fresh `_v2` directory if the existing store triggers legacy errors.
    """
    file = request.files.get('userFile')
    try:
        user_id = current_user.id
    except Exception:
        user_id = session.get('user_id')
    if not file or not user_id:
        return jsonify({'error': 'Missing file or user session'}), 400

    user_id_str = str(user_id)
    user_docs_path = os.path.join(USER_DATA_PATH, user_id_str, "docs")
    user_vector_dir = os.path.join(USER_DATA_PATH, user_id_str, "user_vector_store")
    os.makedirs(user_docs_path, exist_ok=True)

    # 1) save file
    filename = secure_filename(file.filename) or f"uploaded_{int(time.time())}.pdf"
    file_path = os.path.join(user_docs_path, filename)
    try:
        file.save(file_path)
    except Exception as ex:
        app_logger().exception("Failed to save uploaded file")
        return jsonify({'error': f'Could not save file: {ex}'}), 500
    app_logger().info("Saved uploaded file to %s", file_path)

    # 2) load PDFs -> langchain Documents
    try:
        if PyPDFLoader is None or DirectoryLoader is None:
            raise RuntimeError("PDF loader is not available in environment")
        loader = DirectoryLoader(user_docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
    except Exception as ex:
        app_logger().exception("PDF loader failed")
        return jsonify({'error': f'Failed to read PDF(s): {ex}'}), 500

    if not documents:
        return jsonify({'error': 'No text extracted from the uploaded PDF(s).'}), 400

    # 3) split into chunks
    try:
        if RecursiveCharacterTextSplitter:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_documents(documents)
        else:
            chunks = [d for d in documents if getattr(d, "page_content", "").strip()]
    except Exception as ex:
        app_logger().exception("Text splitting failed")
        return jsonify({'error': f'Text splitting failed: {ex}'}), 500

    chunks = [c for c in chunks if getattr(c, "page_content", "").strip()]
    app_logger().info("Chunks to embed: %s", len(chunks))
    if not chunks:
        return jsonify({'error': 'No chunks to embed.'}), 400

    # 4) prepare embedding wrapper `emb`
    if embeddings is not None:
        emb_local = embeddings
        if callable(emb_local) and not hasattr(emb_local, "embed_documents"):
            class _Wrap:
                def embed_documents(self, texts): return [emb_local(t) for t in texts]
            emb_local = _Wrap()
    else:
        class _LocalEmb:
            def embed_documents(self, texts): return [local_embed_list(t) for t in texts]
        emb_local = _LocalEmb()

    # 4a) embed sanity check
    try:
        test_vecs = emb_local.embed_documents(["hello world"])
        if not isinstance(test_vecs, list) or not isinstance(test_vecs[0], (list, tuple)):
            raise RuntimeError("embed_documents must return list-of-vectors")
        app_logger().info("Embed test OK – dim=%s", len(test_vecs[0]))
    except Exception as ex:
        app_logger().exception("Embed function broken")
        return jsonify({'error': f'Embed test failed: {ex}'}), 500

    # 5) attempt to create / open chromadb client and add documents
    collection_name = "user_docs"
    added_total = 0
    used_vector_dir = user_vector_dir
    try:
        import chromadb
        from chromadb.config import Settings

        def _create_client(persist_dir):
            return chromadb.Client(Settings(persist_directory=persist_dir))

        # first try original path, but if ValueError (legacy config) occurs, fallback to _v2
        try:
            client = _create_client(used_vector_dir)
        except ValueError as ve:
            app_logger().warning("Chroma client init failed (legacy store?) - falling back to new directory: %s", str(ve))
            used_vector_dir = user_vector_dir + "_v2"
            os.makedirs(used_vector_dir, exist_ok=True)
            client = _create_client(used_vector_dir)

        # get or create collection
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            collection = client.create_collection(name=collection_name)

        # add chunks in batches
        batch = 50
        ts_base = int(time.time() * 1000)
        for i in range(0, len(chunks), batch):
            batch_chunks = chunks[i:i+batch]
            texts = [getattr(c, "page_content", "") for c in batch_chunks]
            embs = emb_local.embed_documents(texts)
            if not (isinstance(embs, list) and len(embs) == len(texts)):
                app_logger().error("Embeddings length mismatch: texts=%s emb=%s", len(texts), len(embs) if isinstance(embs, list) else type(embs))
                return jsonify({'error': 'Embedding API returned wrong shape'}), 500

            ids = [f"{user_id_str}-{ts_base + i}-{j}" for j in range(len(texts))]
            metadatas = []
            for c in batch_chunks:
                md = getattr(c, "metadata", {}) or {}
                # prefer an explicit source path in metadata if available
                src = md.get("source") or md.get("file_path") or filename
                md.update({"source": src})
                metadatas.append(md)

            # new Chroma client: collection.add(...) persists automatically to persist_directory
            collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=texts,
                embeddings=embs
            )
            added_total += len(texts)

        # no client.persist() in new API

    except Exception as ex:
        app_logger().exception("Chroma/chromadb add_documents failed")
        return jsonify({'error': f'Embedding/store failed: {ex}', 'trace': traceback.format_exc()}), 500

    # 6) verify count (best-effort)
    cnt = None
    try:
        client2 = chromadb.Client(Settings(persist_directory=used_vector_dir))
        collection2 = client2.get_collection(collection_name)
        try:
            cnt = collection2.count()
        except Exception:
            # fallback: try query length
            try:
                res = collection2.query(n_results=1, query_texts=["dummy"], include=["ids"])
                cnt = len(res.get("ids", [[]])[0]) if isinstance(res, dict) else None
            except Exception:
                cnt = None
    except Exception:
        app_logger().exception("Could not probe vector count")

    app_logger().info("Vector-store contains %s embeddings (added_total=%s) at %s", cnt, added_total, used_vector_dir)
    if cnt == 0 or cnt is None:
        app_logger().warning("Zero vectors stored – check chunks/embedding and collection config")

    # 7) return updated library listing
    try:
        user_lib = sorted(
            f for f in os.listdir(user_docs_path)
            if os.path.isfile(os.path.join(user_docs_path, f)) and
            os.path.splitext(f)[1].lower() in ALLOWED_USER_DOC_EXTS
        )
    except Exception:
        user_lib = []

    response = {'success': True, 'user_library': user_lib, 'vectors_added': added_total, 'vector_count': cnt}
    if used_vector_dir != user_vector_dir:
        response['note'] = 'Used a fresh vector directory to avoid legacy Chroma migration.'
        response['vector_dir'] = used_vector_dir

    return jsonify(response)



@app.route('/delete_user_file', methods=['POST'])
@login_required
def delete_user_file():
    """
    Delete a file from the current user's library (docs folder).
    Expects JSON body: { "filename": "something.pdf" }
    Returns updated user_library list.
    """
    try:
        data = request.get_json() or {}
        filename = data.get("filename")
        if not filename:
            return jsonify({'error': 'Missing filename'}), 400

        # determine user_id
        try:
            user_id = current_user.id
        except Exception:
            user_id = session.get('user_id')

        if not user_id:
            return jsonify({'error': 'Missing user session'}), 401

        user_docs_path = os.path.join(USER_DATA_PATH, str(user_id), "docs")
        file_path = os.path.join(user_docs_path, filename)

        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404

        os.remove(file_path)
        app_logger().info("Deleted file %s for user %s", filename, user_id)

        # update user library
        try:
            files = sorted(os.listdir(user_docs_path))
        except Exception:
            files = []

        return jsonify({'success': True, 'user_library': files})

    except Exception as e:
        logger.exception("Delete user file failed")
        return jsonify({'error': f'Delete failed: {e}'}), 500


ALLOWED_USER_DOC_EXTS = {'.pdf', '.txt', '.csv'}

@app.route('/get_user_library', methods=['GET'])
def get_user_library():
    """
    Return a JSON list of files in the current user's user_data/<id>/docs directory.
    Works with Flask-Login current_user or session['user_id'] fallback.
    """
    try:
        # Prefer flask-login current_user if authenticated
        uid = None
        try:
            if current_user and getattr(current_user, "is_authenticated", False):
                uid = current_user.id
        except Exception:
            uid = None

        if uid is None:
            uid = session.get('user_id')

        if not uid:
            # Not authenticated — return empty library (frontend expects this)
            app_logger().debug("get_user_library: no user id in session")
            return jsonify({'user_library': []})

        user_docs_path = os.path.join(USER_DATA_PATH, str(uid), "docs")
        if not os.path.isdir(user_docs_path):
            return jsonify({'user_library': []})

        try:
            files = sorted([
                f for f in os.listdir(user_docs_path)
                if os.path.isfile(os.path.join(user_docs_path, f)) and os.path.splitext(f)[1].lower() in ALLOWED_USER_DOC_EXTS
            ])
        except Exception as e:
            app_logger().exception("Failed to list user library files")
            files = []

        return jsonify({'user_library': files})
    except Exception:
        app_logger().exception("Unhandled error in /get_user_library")
        return jsonify({'user_library': []}), 500
# Serve a user's raw uploaded file (PDF/TXT/CSV) from user_data/<user_id>/docs
@app.route('/user_source_files/<path:filename>', methods=['GET'])
@login_required
def serve_user_source_file(filename):
    """
    Serve files uploaded by the current user (safe path resolution).
    URL: /user_source_files/<filename>
    """
    try:
        # decode/normalize
        fname = unquote(filename or "").strip()
        if not fname:
            return jsonify({'error': 'File not found'}), 404

        user_docs_path = os.path.join(USER_DATA_PATH, str(current_user.id), "docs")
        # safety: resolve absolute and ensure it is inside user's docs
        candidate = os.path.abspath(os.path.join(user_docs_path, fname))
        base_abs = os.path.abspath(user_docs_path)
        if not (candidate == base_abs or candidate.startswith(base_abs + os.sep)):
            return jsonify({'error': 'File not found'}), 404
        if not os.path.exists(candidate) or not os.path.isfile(candidate):
            return jsonify({'error': 'File not found'}), 404

        ext = os.path.splitext(candidate)[1].lower()
        basename = os.path.basename(candidate)
        if ext == '.pdf':
            # serve inline PDF
            resp = send_from_directory(user_docs_path, basename, as_attachment=False)
            resp.headers['Content-Disposition'] = f'inline; filename="{basename}"'
            return resp
        if ext in ('.txt', '.csv'):
            with open(candidate, 'r', encoding='utf-8', errors='replace') as fh:
                text = fh.read()
            html = f"""
            <!doctype html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
            <title>{basename}</title>
            <style>body{{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial;padding:1rem}} pre{{white-space:pre-wrap;word-wrap:break-word}}</style>
            </head><body>
            <a href="/get_user_library">← Back to library</a>
            <h3>{basename}</h3>
            <pre>{text}</pre>
            </body></html>"""
            return Response(html, mimetype='text/html')
        return jsonify({'error': 'Unsupported file type'}), 400
    except Exception:
        app_logger().exception("Error serving user source file")
        return jsonify({'error': 'Internal server error'}), 500


# Viewer HTML wrapper (iframe) for user docs
@app.route('/user_source_view/<path:filename>', methods=['GET'])
@login_required
def user_source_viewer(filename):
    try:
        fname = unquote(filename or '').strip()
        if not fname: return jsonify({'error': 'File not found'}), 404
        user_docs_path = os.path.join(USER_DATA_PATH, str(current_user.id), "docs")
        candidate = os.path.abspath(os.path.join(user_docs_path, fname))
        base_abs = os.path.abspath(user_docs_path)
        if not (candidate == base_abs or candidate.startswith(base_abs + os.sep)):
            return jsonify({'error': 'File not found'}), 404
        if not os.path.exists(candidate) or not os.path.isfile(candidate): return jsonify({'error': 'File not found'}), 404

        basename = os.path.basename(candidate)
        file_url = url_for('serve_user_source_file', filename=quote(basename, safe=''), _external=False)
        viewer_html = f"""
        <!doctype html>
        <html>
          <head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
            <title>Viewing: {basename}</title>
            <style>body,html{{height:100%;margin:0}} .topbar{{padding:8px;background:#f3f4f6;border-bottom:1px solid #e5e7eb}} .iframe-wrap{{height:calc(100vh - 52px)}}</style>
          </head>
          <body>
            <div class="topbar">
              <a href="/get_user_library" style="margin-right:12px">← Back to my library</a>
              <strong>{basename}</strong>
              <a style="float:right" href="{file_url}" target="_blank" rel="noopener">Open raw</a>
            </div>
            <div class="iframe-wrap"><iframe src="{file_url}" style="width:100%;height:100%;border:0;"></iframe></div>
          </body>
        </html>
        """
        return Response(viewer_html, mimetype='text/html')
    except Exception:
        app_logger().exception("Error building user source viewer")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'portfolioFile' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['portfolioFile']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            file.save(tmp.name); tmp_path = tmp.name
        if tmp_path.endswith('.csv'): df = pd.read_csv(tmp_path)
        elif tmp_path.endswith(('.xlsx', '.xls')): df = pd.read_excel(tmp_path)
        else:
            os.remove(tmp_path); return jsonify({'error': 'Unsupported file format'}), 400
        if not {'Quantity', 'Current Price', 'Sector'}.issubset(df.columns):
            possible_qty = next((c for c in df.columns if c.lower() in ('quantity','qty','shares')), None)
            possible_price = next((c for c in df.columns if c.lower() in ('current price','price','last price','ltp')), None)
            possible_sector = next((c for c in df.columns if c.lower() in ('sector','industry')), None)
            if possible_qty and possible_price:
                df = df.rename(columns={possible_qty: 'Quantity', possible_price: 'Current Price'})
                if possible_sector: df = df.rename(columns={possible_sector: 'Sector'})
            else:
                return jsonify({'error': 'CSV/XLSX must include Quantity and Current Price columns (or similar headers)'}), 400
        df['Investment Value'] = df['Quantity'] * df['Current Price']
        sector_allocation = df.groupby('Sector')['Investment Value'].sum().round(2).to_dict()
        portfolio_string = df.to_string()
        formatted_prompt = analysis_prompt.format(portfolio_data=portfolio_string) if analysis_prompt is not None else portfolio_string
        if llm is None:
            return jsonify({'analysis_markdown': "LLM not configured", 'chart_data': sector_allocation})
        result = llm.invoke(formatted_prompt)
        analysis_markdown = getattr(result, 'content', result)
        return jsonify({'analysis_markdown': analysis_markdown, 'chart_data': sector_allocation})
    except Exception:
        logger.exception("--- AN ERROR OCCURRED DURING ANALYSIS ---")
        return jsonify({'error': 'An error occurred during analysis.'}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)

# ---------- Portfolio helpers & endpoints ----------
POSITIVE_WORDS = {"beats","upgrade","raised","approved","good","strong","positive","gain","growth","outperform","record","profit","surge"}
NEGATIVE_WORDS = {"downgrade","cut","loss","falls","fall","decline","weak","negative","recall","scandal","fraud","drops","hit","miss"}

_price_cache = {}
CACHE_TTL = 300  # 5 minutes

def _normalize_ticker_for_yf(sym: str) -> str:
    s = (sym or "").strip()
    if not s: return s
    # If user already provided an exchange suffix keep it
    if '.' in s:
        return s
    # Heuristic: treat plain tickers as NSE tickers (common in your CSVs)
    return f"{s}.NS"

def fetch_latest_prices(symbols):
    """
    Return dict {symbol: price_or_None}. Accepts list of symbols as stored in DB (like 'RELIANCE' or 'RELIANCE.NS').
    Uses batch yf.download when possible, tries per-symbol fallback when batch fails.
    """
    out = {}
    now = time.time()
    need_fetch = []

    # TTL cache check
    for sym in symbols:
        if sym in _price_cache and (now - _price_cache[sym]["time"]) < CACHE_TTL:
            out[sym] = _price_cache[sym]["price"]
        else:
            need_fetch.append(sym)

    if not need_fetch:
        return out

    # Normalize tickers for yfinance
    ticker_map = {sym: _normalize_ticker_for_yf(sym) for sym in need_fetch}
    yf_tickers = list(set(ticker_map.values()))

    # Try batch download first
    try:
        df = yf.download(" ".join(yf_tickers), period="1d", group_by="ticker", progress=False, threads=True, auto_adjust=False)
        # If batch returned a multiindex, iterate
        if isinstance(df.columns, pd.MultiIndex):
            for orig_sym, yf_sym in ticker_map.items():
                price = None
                try:
                    if yf_sym in df:
                        sub = df[yf_sym]
                        if 'Close' in sub.columns and not sub.empty:
                            v = sub['Close'].iloc[-1]
                            price = float(v) if not (pd.isna(v)) else None
                except Exception:
                    price = None
                _price_cache[orig_sym] = {"price": price, "time": now}
                out[orig_sym] = price
        else:
            # single-dataframe returned (when only one ticker)
            for orig_sym, yf_sym in ticker_map.items():
                price = None
                try:
                    if 'Close' in df.columns and not df.empty:
                        v = df['Close'].iloc[-1]
                        price = float(v) if not (pd.isna(v)) else None
                except Exception:
                    price = None
                _price_cache[orig_sym] = {"price": price, "time": now}
                out[orig_sym] = price
    except Exception as e:
        app_logger().debug("fetch_latest_prices: batch download failed: %s", e)
        # fallback: per-symbol
        for orig_sym, yf_sym in ticker_map.items():
            price = None
            try:
                t = yf.Ticker(yf_sym)
                hist = t.history(period="1d", interval="1d")
                if hist is not None and not hist.empty and 'Close' in hist.columns:
                    v = hist['Close'].iloc[-1]
                    price = float(v) if not (pd.isna(v)) else None
            except Exception as e2:
                app_logger().debug("fetch_latest_prices: per-symbol fetch failed for %s (%s): %s", orig_sym, yf_sym, e2)
                price = None
            _price_cache[orig_sym] = {"price": price, "time": now}
            out[orig_sym] = price

    return out

def fetch_headlines_for_symbol(symbol, api_key=None, max_headlines=5):
    symbol = (symbol or "").strip()
    if not symbol: return []
    if api_key:
        try:
            q = f'"{symbol}" OR {symbol} stock OR {symbol} share OR {symbol} company'
            params = {"q": q, "language":"en","sortBy":"publishedAt","pageSize":max_headlines,"apiKey": api_key}
            r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=8)
            r.raise_for_status()
            data = r.json()
            if data.get("status") == "ok":
                headlines = []
                for a in data.get("articles", [])[:max_headlines]:
                    title = (a.get("title") or "").strip()
                    desc  = (a.get("description") or "").strip()
                    combined = (title + " — " + desc).strip(" — ")
                    if combined: headlines.append(combined)
                return headlines
        except Exception as e:
            app_logger().warning(f"NewsAPI error for {symbol}: {e}")
    try:
        query = urllib.parse.quote_plus(f"{symbol} stock")
        search_url = f"https://www.google.com/search?q={query}&tbm=nws"
        headers = {"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari:537.36"}
        r = requests.get(search_url, headers=headers, timeout=8)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        items = []
        for g in soup.select("div.dbsr")[:max_headlines]:
            title_el = g.select_one("div.JheGif, div.JheGif.nDgy9d")
            title = title_el.get_text(strip=True) if title_el else ""
            if title: items.append(title)
        return items
    except Exception as e:
        app_logger().warning(f"Google news scrape failed for {symbol}: {e}")
        return []

def news_for_symbol_cached(symbol, max_headlines=5):
    cache_key = f"news:{symbol}:{max_headlines}"
    def _fetch(): return fetch_stock_headlines(symbol, max_headlines=max_headlines)
    return cached_headlines(cache_key, _fetch, ttl=NEWS_CACHE_TTL)

def simple_sentiment_estimate(headlines):
    score = 0
    for h in headlines:
        txt = h.lower()
        for w in POSITIVE_WORDS:
            if w in txt: score += 1
        for w in NEGATIVE_WORDS:
            if w in txt: score -= 1
    if not headlines: return 0.0
    return max(-1.0, min(1.0, score / (len(headlines) * 3.0)))

def discover_company_name(sym):
    try:
        t = yf.Ticker(sym); info = {}
        try:
            info = t.get_info() if hasattr(t, "get_info") else getattr(t, "info", {}) or {}
        except Exception:
            info = getattr(t, "info", {}) or {}
        return info.get("longName") or info.get("shortName")
    except Exception:
        return None

def build_dashboard_payload(user_id):
    """
    Build dashboard payload with numeric fields cleaned (NaN -> None).
    """
    user = db.session.get(User, int(user_id))
    if not user: return {'error': 'user not found'}
    portfolio = Portfolio.query.filter_by(user_id=user.id).first()
    if not portfolio: return {'holdings': [], 'allocations': {}, 'alerts': []}
    holdings = portfolio.holdings or []
    symbols = [h.symbol for h in holdings]
    prices = fetch_latest_prices(symbols)
    holdings_payload = []; total_value = 0.0
    for h in holdings:
        last_price = prices.get(h.symbol)
        current_value = None
        try:
            if last_price is None or (isinstance(last_price, float) and (np.isnan(last_price) or np.isinf(last_price))):
                current_value = None
            else:
                current_value = round(float(last_price) * float(h.quantity), 2)
                total_value += current_value
        except Exception:
            current_value = None

        holdings_payload.append({
            'symbol': h.symbol,
            'quantity': float(h.quantity),
            'avg_price': (float(h.avg_price) if h.avg_price is not None else None),
            'last_price': (float(last_price) if last_price is not None else None),
            'current_value': (current_value if current_value is not None else None)
        })

    allocations = {}
    if total_value and total_value > 0:
        for item in holdings_payload:
            cv = item.get('current_value') or 0.0
            try:
                allocations[item['symbol']] = round((cv / total_value) * 100.0, 2)
            except Exception:
                allocations[item['symbol']] = 0.0
    else:
        for item in holdings_payload:
            allocations[item['symbol']] = 0.0

    alerts = []
    sorted_holdings = sorted(holdings_payload, key=lambda x: x.get('current_value') or 0.0, reverse=True)[:5]
    def _get_headlines_cached(query, max_headlines=5, ttl=NEWS_CACHE_TTL):
        key = f"dash_news:{query}:{max_headlines}"
        return cached_headlines(key, lambda: fetch_stock_headlines(query, max_headlines=max_headlines), ttl=ttl)
    for h in sorted_holdings:
        company_name = discover_company_name(h['symbol'])
        headlines = _get_headlines_cached(h['symbol'], 5)
        if not headlines and company_name:
            headlines = _get_headlines_cached(company_name, 5)
        sent = simple_sentiment_estimate(headlines)
        qty_factor = math.log1p(h['quantity'])
        est_pct = round(sent * 3.0 * qty_factor, 2)
        sentiment_label = "Neutral"
        if sent > 0.15: sentiment_label = "Positive"
        elif sent < -0.15: sentiment_label = "Negative"
        alerts.append({
            'symbol': h['symbol'],
            'quantity': h['quantity'],
            'sentiment': sentiment_label,
            'estimated_pct_move': est_pct,
            'headlines': headlines[:5] if headlines else []
        })
    return {
        'user_id': user.id,
        'username': user.username,
        'total_value': (round(total_value, 2) if total_value and not math.isnan(total_value) else None),
        'holdings': holdings_payload,
        'allocations': allocations,
        'alerts': alerts
    }

@app.route('/portfolio/upload', methods=['POST'])
@login_required
def upload_portfolio():
    file = request.files.get('file')
    if not file: return jsonify({'error': 'No file provided'}), 400
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            file.save(tmp.name); tmp_path = tmp.name
        if tmp_path.endswith('.csv'): df = pd.read_csv(tmp_path)
        elif tmp_path.endswith(('.xls','.xlsx')): df = pd.read_excel(tmp_path)
        else: return jsonify({'error': 'Unsupported file type'}), 400
        upsert_portfolio_from_df(current_user.id, df, portfolio_name="Default Portfolio")
        payload = build_dashboard_payload(current_user.id)
        return jsonify({'success': True, 'dashboard': payload})
    except Exception as e:
        logger.exception("Error uploading portfolio: %s", e)
        return jsonify({'error': str(e)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.route('/portfolio', methods=['GET'])
@login_required
def get_portfolio():
    payload = build_dashboard_payload(current_user.id)
    return jsonify(payload)

@app.route('/portfolio/history', methods=['GET'])
@login_required
def portfolio_history():
    try:
        months = int(request.args.get('months', 6))

        portfolio = Portfolio.query.filter_by(user_id=current_user.id).first()
        if not portfolio or not getattr(portfolio, "holdings", None):
            return jsonify({'history': []})

        symbols = [h.symbol for h in portfolio.holdings]
        if not symbols:
            return jsonify({'history': []})

        end_date = datetime.today()
        start_date = end_date - pd.DateOffset(months=months)

        # store per-symbol close series
        hist_data = {}

        for sym in symbols:
            try:
                yf_sym = _normalize_ticker_for_yf(sym)
                df = yf.download(yf_sym,
                                 start=start_date,
                                 end=end_date,
                                 progress=False,
                                 interval="1wk",
                                 auto_adjust=True,
                                 threads=False)
                if df is None or df.empty:
                    logger.debug("No price data for %s (yf symbol=%s)", sym, yf_sym)
                    continue

                close = None
                if 'Close' in df.columns:
                    close = df['Close']
                elif 'Adj Close' in df.columns:
                    close = df['Adj Close']
                else:
                    if isinstance(df.columns, pd.MultiIndex):
                        try:
                            close = df.xs('Close', axis=1, level=1, drop_level=False)
                            if isinstance(close, pd.DataFrame):
                                close = close.iloc[:, 0]
                        except Exception:
                            close = None

                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]

                if close is None or close.empty:
                    logger.debug("No close series for %s", sym)
                    continue

                close.index = pd.to_datetime(close.index, errors='coerce')
                close = close.dropna()
                if close.empty:
                    logger.debug("Close series empty after dropna for %s", sym)
                    continue

                # coerce to numeric and scalarize arrays if needed
                if close.dtype == object:
                    def scalarize(x):
                        try:
                            if isinstance(x, (list, tuple, np.ndarray)):
                                x0 = np.asarray(x).ravel()
                                return float(x0[0]) if x0.size > 0 else np.nan
                            return float(x)
                        except Exception:
                            return np.nan
                    close = close.apply(scalarize)
                else:
                    close = pd.to_numeric(close, errors='coerce')

                close = close.dropna()
                if close.empty:
                    logger.debug("After coercion no numeric close values for %s", sym)
                    continue

                hist_data[sym] = close

            except Exception as e:
                logger.warning("Failed to fetch history for %s: %s", sym, getattr(e, 'message', str(e)))
                continue

        # Build portfolio time series by weighting each symbol series by its holding quantity
        portfolio_series = None
        for h in portfolio.holdings:
            sym = h.symbol
            qty = getattr(h, 'quantity', 0) or 0
            if sym not in hist_data:
                continue
            try:
                series_val = hist_data[sym] * float(qty)
            except Exception:
                # if hist_data[sym] contains non-scalar weird types, attempt to coerce
                try:
                    series_val = hist_data[sym].apply(lambda v: float(np.asarray(v).ravel()[0]) if hasattr(v, '__iter__') else float(v)) * float(qty)
                except Exception:
                    continue

            if portfolio_series is None:
                portfolio_series = series_val
            else:
                portfolio_series = portfolio_series.add(series_val, fill_value=0)

        if portfolio_series is None or portfolio_series.empty:
            return jsonify({'history': []})

        ps = portfolio_series.dropna().sort_index()
        ps.index = pd.to_datetime(ps.index, errors='coerce')
        ps = ps[~ps.index.isna()]
        if ps.empty:
            return jsonify({'history': []})

        history = []
        for ts, val in ps.items():
            try:
                if isinstance(val, (list, tuple, np.ndarray)):
                    arr = np.asarray(val).ravel()
                    if arr.size > 0:
                        fval = float(arr[0])
                    else:
                        continue
                else:
                    fval = float(val)
                history.append({'date': pd.to_datetime(ts).strftime('%Y-%m-%d'), 'value': round(fval, 2)})
            except Exception:
                continue

        return jsonify({'history': history})

    except Exception as e:
        logger.exception("Failed to compute portfolio history")
        return jsonify({'error': str(e)}), 500

# ---------- IPO analyzer helpers & routes ----------
IPO_CHECKLIST = [
    ("financials", "Financial disclosures (revenue, profits, margins, debt levels)"),
    ("governance", "Corporate governance & promoter/shareholding structure"),
    ("use_of_proceeds", "Use of IPO proceeds (expansion, debt repayment, acquisitions)"),
    ("business_risks", "Business risks, litigation, regulatory exposures"),
    ("market_position", "Market position, competitive moat and growth drivers"),
    ("compliance", "Regulatory / SEBI disclosures (DRHP completeness, approvals)"),
]

def extract_text_from_pdf(path, max_chars=100000):
    try:
        if PyPDFLoader is None:
            raise RuntimeError("PyPDFLoader not available")
        loader = PyPDFLoader(path); docs = loader.load()
        text = "\n\n".join([d.page_content for d in docs])
        return text[:max_chars]
    except Exception:
        try:
            with open(path, 'rb') as f:
                return f.read().decode('utf-8', errors='ignore')[:max_chars]
        except Exception:
            return ""

def simple_rule_extract(text):
    res = {}; lower = text.lower()
    m = re.search(r'(revenue|turnover)[^\d{0,6}]*([\d,\.]+\s*(?:crore|cr|₹|rs|rs\.|m|bn|b|million|billion)?)', text, re.I)
    if m: res['revenue'] = m.group(2).strip()
    m2 = re.search(r'(profit|net profit|net income)[^\d{0,6}]*([\d,\,\.\s]+(?:crore|cr|₹|rs|m|bn|b|million|billion)?)', text, re.I)
    if m2: res['profit'] = m2.group(2).strip()
    m3 = re.search(r'(debt|total debt|borrowings)[^\d{0,6}]*([\d,\,\.\s]+(?:crore|cr|₹|rs|m|bn|b|million|billion)?)', text, re.I)
    if m3: res['debt'] = m3.group(2).strip()
    if 'promoter' in lower: res['promoter_mentioned'] = True
    return res
# -----------------------
# SEBI-checks analyzer & scoring
# -----------------------

# helper to parse numeric amounts (re-uses FIN_NUM regex)
def _parse_fin_number(text):
    if not text: return None
    try:
        m = FIN_NUM.search(text)
        if not m:
            s = re.sub(r'[^\d\.]', '', text)
            return float(s) if s else None
        num_part = m.group(2) or m.group(0)
        s = re.sub(r'[^\d\.]', '', num_part)
        if not s: return None
        val = float(s)
        unit = (m.group(3) or "").lower() if m.lastindex and m.lastindex >= 3 else ""
        if unit:
            if "lakh" in unit: val *= 1e5
            if "crore" in unit or unit == "cr": val *= 1e7
            if "mn" in unit or "million" in unit: val *= 1e6
            if "bn" in unit or "billion" in unit: val *= 1e9
        return val
    except Exception:
        return None

def analyze_sebi_checks(text: str, facts: dict = None):
    """
    Heuristic SEBI checks. Returns a dict of checks/flags and some values.
    """
    t = (text or "").lower()
    facts = facts or {}
    checks = {}

    # Financials
    rev = facts.get("revenue") if isinstance(facts, dict) else None
    prof = facts.get("profit") if isinstance(facts, dict) else None
    rev_val = _parse_fin_number(rev) if rev else None
    prof_val = _parse_fin_number(prof) if prof else None
    checks["financials_present"] = bool(rev or prof)
    checks["revenue_value"] = rev or None
    checks["profit_value"] = prof or None
    checks["profitability"] = bool(prof_val and prof_val > 0)

    # Governance / promoter checks
    checks["promoter_mentioned"] = bool(re.search(r'\bpromoter(s)?\b', t))
    checks["promoter_pledge"] = bool(re.search(r'\bpledg(ed|e)?\b', t))
    checks["related_party"] = bool(re.search(r'related[\s-]*party', t))

    # Auditor opinions
    checks["qualified_opinion"] = bool(re.search(r'qualified opinion|qualified audit opinion', t))
    checks["material_uncertainty"] = bool(re.search(r'material uncertainty|going concern', t))

    # Debt & leverage
    checks["debt_mentioned"] = bool(re.search(r'\b(debt|borrowings|loans|interest bearing)\b', t))
    debt_hint = facts.get("debt")
    if not debt_hint:
        m = re.search(r'(debt|borrowings)[^\d]{0,8}([\d,\.]+\s*(?:crore|cr|₹|rs|m|bn|b|million|billion)?)', text, re.I)
        if m:
            debt_hint = m.group(2)
    checks["debt_value"] = debt_hint or None
    debt_val = _parse_fin_number(str(debt_hint)) if debt_hint else None
    checks["debt_high"] = bool(debt_val and debt_val > 5e8)  # heuristic >50 crore

    # Use-of-proceeds clarity
    checks["use_of_proceeds_clear"] = bool(re.search(r'use of proceeds|purpose of the issue|utiliz(e|ation) of proceeds', t))
    checks["use_of_proceeds_vague"] = bool(re.search(r'general corporate purposes|general corporate use', t))

    # Litigation / regulatory exposure
    checks["litigation_mentioned"] = bool(re.search(r'\b(litigat|court|lawsuit|legal\s+proceeding|penalty|regulator|show cause)\b', t))

    # Revenue trend signals
    checks["revenue_growth_words"] = bool(re.search(r'\b(grow|increase|expan|year on year|yoy|y/y|rise|surge)\b', t))
    checks["revenue_decline_words"] = bool(re.search(r'\b(declin|fall|drop|contract)\b', t))

    # Fraud / material weakness words
    checks["fraud_terms"] = bool(re.search(r'\b(fraud|scam|insolv|bankrupt|material weakness)\b', t))

    # Lock-in mentions
    checks["lockin_mentioned"] = bool(re.search(r'lock-in|lockin', t))

    return checks

def compute_ipo_score_from_checks(checks: dict):
    """
    Convert checks dict into an integer 0..100 score (heuristic).
    """
    score = 50.0
    if checks.get("financials_present"): score += 10
    if checks.get("profitability"): score += 8
    if checks.get("promoter_mentioned"): score += 3
    if checks.get("promoter_pledge"): score -= 12
    if checks.get("related_party"): score -= 6
    if checks.get("qualified_opinion"): score -= 15
    if checks.get("material_uncertainty"): score -= 10
    if checks.get("debt_mentioned"): score -= 2
    if checks.get("debt_high"): score -= 10
    if checks.get("use_of_proceeds_clear"): score += 6
    if checks.get("use_of_proceeds_vague"): score -= 6
    if checks.get("litigation_mentioned"): score -= 8
    if checks.get("revenue_growth_words"): score += 4
    if checks.get("revenue_decline_words"): score -= 4
    if checks.get("fraud_terms"): score -= 20
    if checks.get("lockin_mentioned"): score += 2

    score = max(0, min(100, score))
    return int(round(score))


def save_ipo_report_to_db(user_id, title, filename, content_md, raw_text,
                          sebi_score=None, sebi_checks=None, llm_score=None,
                          heuristic_meta=None, checklist=None, overall_score=None):
    try:
        rpt = IpoReport(
            user_id=user_id,
            title=(title or (filename or "IPO Report"))[:512],
            filename=(filename or None),
            content_md=(content_md or ""),
            raw_text=(raw_text or "")[:250000],
            excerpt=(raw_text or "")[:120],
            heuristic_meta=json.dumps(heuristic_meta or {}),
            checklist=json.dumps(checklist or {}),
            overall_score=(float(overall_score) if overall_score is not None else None),
            sebi_score=(float(sebi_score) if sebi_score is not None else None),
            sebi_checks=(sebi_checks if sebi_checks is not None else None),
            llm_score=(int(llm_score) if llm_score is not None else None)
        )
        db.session.add(rpt)
        db.session.commit()
        return rpt.id
    except Exception:
        db.session.rollback()
        logger.exception("DB commit failed while saving IPOReport")
        raise

def list_user_ipo_reports(user_id):
    rows = IpoReport.query.filter_by(user_id=user_id).order_by(IpoReport.created_at.desc()).all()
    out = []
    for r in rows:
        out.append({
            "id": r.id, "title": r.title, "filename": r.filename,
            "created_at": r.created_at.isoformat(), "overall_score": r.overall_score,
            "sebi_score": r.sebi_score, "llm_score": r.llm_score
        })
    return out

@app.post("/ipo/analyze")
@login_required
def ipo_analyze():
    """
    Full IPO analyze endpoint:
    - Accepts uploaded PDF (form field 'ipoFile') or JSON body { "content": "..." }
    - Extracts text, runs heuristics, builds SEBI checklist, computes score
    - Calls LLM for a short human-readable summary (if configured)
    - Saves report to DB and returns summary + checks
    """
    try:
        text = None
        filename = None
        tmp_path = None

        # 1) Get text either from uploaded file or JSON body
        if "ipoFile" in request.files and request.files["ipoFile"]:
            f = request.files["ipoFile"]
            filename = secure_filename(f.filename or "ipo.pdf")
            # Save to a temp file so we can pass to pdfplumber or loaders
            with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(filename).suffix) as tmp:
                f.save(tmp.name)
                tmp_path = tmp.name

            if filename.lower().endswith(".pdf"):
                if pdfplumber is None:
                    # Try simple binary read fallback
                    try:
                        with open(tmp_path, "rb") as fh:
                            text = fh.read().decode("utf-8", errors="ignore")
                    except Exception:
                        # cleanup
                        try: os.remove(tmp_path)
                        except Exception: pass
                        return jsonify({"error": "PDF support not installed (pdfplumber). Install pdfplumber or upload plain text."}), 400
                else:
                    try:
                        pages = []
                        with pdfplumber.open(tmp_path) as pdf:
                            for p in pdf.pages[:400]:  # limit pages to avoid huge extraction
                                pages.append(p.extract_text() or "")
                        text = "\n".join(pages)
                    except Exception:
                        # fallback to raw read
                        try:
                            with open(tmp_path, "rb") as fh:
                                text = fh.read().decode("utf-8", errors="ignore")
                        except Exception:
                            text = ""
                    finally:
                        try: os.remove(tmp_path)
                        except Exception: pass
            else:
                # non-pdf, read as text
                try:
                    with open(tmp_path, "r", encoding="utf-8", errors="ignore") as fh:
                        text = fh.read()
                except Exception:
                    with open(tmp_path, "rb") as fh:
                        text = fh.read().decode("utf-8", errors="ignore")
                finally:
                    try: os.remove(tmp_path)
                    except Exception: pass

        # If JSON body with content (paste)
        elif request.is_json:
            text = (request.json or {}).get("content", "")

        if not text or not text.strip():
            return jsonify({"error": "No IPO content found."}), 400

        # Truncate for internal processing but keep raw_text for DB small enough
        raw_text = text[:200_000]

        # 2) Title, quick facts, heuristics, score
        title = derive_title_from_text(text) or "IPO Report"
        quick_facts = extract_quick_facts(text)
        simple_meta = simple_rule_extract(text)
        sebi_score = compute_ipo_score(text)

        # 3) Build more detailed SEBI checks (heuristic rules)
        lower = text.lower()
        def contains_any(words):
            return any(w in lower for w in words)

        checks = {
            "financials_present": bool(quick_facts.get("revenue") or quick_facts.get("profit") or simple_meta.get("revenue")),
            "revenue_value": quick_facts.get("revenue") or simple_meta.get("revenue"),
            "profit_value": quick_facts.get("profit") or simple_meta.get("profit"),
            "profitability": bool((quick_facts.get("profit") or simple_meta.get("profit")) and not contains_any(["loss", "losses", "negative"])),
            "promoter_mentioned": bool("promoter" in lower or "promoters" in lower),
            "promoter_pledge": bool("pledge" in lower or "pledged" in lower),
            "related_party": bool("related party" in lower or "related-party" in lower),
            "qualified_opinion": bool("qualified opinion" in lower or "qualified report" in lower),
            "material_uncertainty": bool("material uncertainty" in lower),
            "debt_mentioned": bool("debt" in lower or "borrowings" in lower or "total borrowings" in lower),
            "debt_value": quick_facts.get("debt") or simple_meta.get("debt"),
            "debt_high": False,  # heuristic: could be set based on parsed numeric debt vs revenue (advanced)
            "useofproceeds_clear": not bool(contains_any(["general corporate purposes", "general corporate", "working capital"])) and "use of proceeds" in lower,
            "useofproceeds_vague": bool("general corporate purposes" in lower or "general corporate" in lower or ("use of proceeds" in lower and contains_any(["general","unspecified"]))),
            "litigation_mentioned": bool("litigation" in lower or "suit" in lower or "case" in lower or "claim" in lower),
            "revenuegrowthwords": contains_any(["growth", "growing", "increase in revenue", "y-o-y", "year on year"]),
            "revenuedeclinewords": contains_any(["decline", "decrease", "drop in revenue", "de-growth", "revenue declined"]),
            "fraud_terms": contains_any(["fraud", "forgery", "embezzlement", "scam"]),
            "lockin_mentioned": bool("lock-in" in lower or "lock in" in lower or "lockin" in lower),
        }

        # Simple derived flag for debt_high (very naive): if debt mentions and no revenue found -> mark high
        try:
            if checks["debt_mentioned"] and not checks["financials_present"]:
                checks["debt_high"] = True
            else:
                checks["debt_high"] = False
        except Exception:
            checks["debt_high"] = False

        # 4) Build markdown report skeleton (we will include LLM summary if available)
        md_header = f"# {title}\n\n**SEBI Heuristic Score:** {sebi_score}/100\n\n"
        md_quick = (
            f"**Revenue:** {quick_facts.get('revenue') or simple_meta.get('revenue') or '—'}  \n"
            f"**Profit:** {quick_facts.get('profit') or simple_meta.get('profit') or '—'}  \n"
            f"**Debt:** {quick_facts.get('debt') or simple_meta.get('debt') or '—'}  \n"
            f"**Promoter Mentioned:** {'Yes' if checks.get('promoter_mentioned') else 'No'}\n\n"
            f"---\n\n"
        )

        # 5) LLM summary (short). Use limited excerpt to keep prompt size manageable
        llm_summary = None
        try:
            if llm is not None:
                # Prepare compact prompt
                prompt_parts = [
                    "You are SEBI Saathi, an analyst. Produce a concise IPO summary (about 6-10 bullet points) and a one-sentence recommendation-style summary (NOT investment advice). Base statements only on the text given.",
                    "",
                    "EXTRACTED FACTS:",
                    f"- Revenue: {quick_facts.get('revenue') or simple_meta.get('revenue') or 'Unknown'}",
                    f"- Profit: {quick_facts.get('profit') or simple_meta.get('profit') or 'Unknown'}",
                    f"- Debt: {quick_facts.get('debt') or simple_meta.get('debt') or 'Unknown'}",
                    f"- Promoter mentioned: {checks.get('promoter_mentioned')}",
                    "",
                    "KEY HEURISTICS (SEBI CHECKS):",
                    json.dumps(checks, indent=2),
                    "",
                    "DOCUMENT EXCERPT (truncated):",
                    text[:16000],  # keep prompt bounded
                    "",
                    "OUTPUT FORMAT:\n1) A short title line\n2) 6-10 bullet points summarizing the most important items for an investor to scan (financials, governance, material risks, use of proceeds, promoter issues)\n3) One-sentence summary closing line.\n\nBe factual and concise."
                ]
                prompt = "\n".join(prompt_parts)
                res = llm.invoke(prompt)
                llm_summary = getattr(res, "content", str(res))
            else:
                llm_summary = "LLM not configured on server. Enable your LLM (GROQ_API_KEY / ChatGroq) to get a generated summary."
        except Exception:
            app_logger().exception("LLM summarization failed")
            llm_summary = "LLM summarization failed or raised an error."

        # 6) Compose final markdown for storing/display
        md_body = md_header + md_quick + "## LLM Summary\n\n" + (llm_summary or "") + "\n\n---\n\n## SEBI Checks\n\n"
        for k, v in checks.items():
            md_body += f"- **{k}**: {v}\n"

        # Optionally include a truncated excerpt for reference
        md_body += "\n\n---\n\n## Extracted Text (truncated)\n\n" + text[:20_000]

        # 7) Save to DB (use helper; it will filter unknown kwargs if you used defensive version)
        try:
            report_id = save_ipo_report_to_db(
                user_id=current_user.id,
                title=title,
                filename=filename,
                content_md=md_body,
                raw_text=raw_text,
                sebi_score=sebi_score,
                sebi_checks=checks,
                llm_score=None,
                heuristic_meta=simple_meta,
                checklist=checks,
                overall_score=sebi_score
            )
        except Exception:
            app_logger().exception("Failed to save IPO report")
            return jsonify({"error": "failed to save report"}), 500

        # 8) Return response
        return jsonify({
            "report_id": report_id,
            "title": title,
            "sebi_score": sebi_score,
            "sebi_checks": checks,
            "quick_facts": quick_facts,
            "llm_summary": llm_summary,
            "ipo_report_md": md_body[:10000]  # return only first 10k chars of markdown to keep response reasonable
        })
    except Exception:
        app_logger().exception("Unexpected error in /ipo/analyze")
        return jsonify({"error": "analysis failed"}), 500



# -----------------------
# IPO list / fetch endpoints (add near your IPO analyzer routes)
# -----------------------
from flask import abort

@app.route("/ipo/list", methods=["GET"])
@login_required
def ipo_list():
    """
    Return a list of IPO reports for the current user.
    Frontend expects an array of { id, title, filename, created_at, overall_score, sebi_score, llm_score }.
    """
    try:
        user_id = current_user.id
        reports = list_user_ipo_reports(user_id)
        return jsonify({"reports": reports})
    except Exception:
        logger.exception("Failed to list IPO reports")
        return jsonify({"error": "failed to list reports"}), 500

@app.route("/ipo/get/<int:report_id>", methods=["GET"])
@login_required
def ipo_get_report(report_id):
    """
    Return full report data for a single IPO report (markdown + metadata).
    """
    try:
        rpt = IpoReport.query.filter_by(id=report_id, user_id=current_user.id).first()
        if not rpt:
            return jsonify({"error": "report not found"}), 404
        try:
            sebi_checks = rpt.sebi_checks if isinstance(rpt.sebi_checks, dict) else (json.loads(rpt.sebi_checks) if rpt.sebi_checks else {})
        except Exception:
            sebi_checks = rpt.sebi_checks
        payload = {
            "id": rpt.id,
            "title": rpt.title,
            "filename": rpt.filename,
            "created_at": rpt.created_at.isoformat() if rpt.created_at else None,
            "content_md": rpt.content_md,
            "raw_text": rpt.raw_text,
            "excerpt": getattr(rpt, "excerpt", None),
            "overall_score": getattr(rpt, "overall_score", None),
            "sebi_score": getattr(rpt, "sebi_score", None),
            "llm_score": getattr(rpt, "llm_score", None),
            "sebi_checks": sebi_checks,
        }
        return jsonify({"report": payload})
    except Exception:
        logger.exception("Failed to fetch IPO report %s", report_id)
        return jsonify({"error": "failed to fetch report"}), 500


@app.get("/ipo/compare")
@login_required
def ipo_compare():
    ids = request.args.get("ids", "")
    id_list = [int(i) for i in ids.split(",") if i.isdigit()]
    rows = IpoReport.query.filter(
        IpoReport.user_id == current_user.id,
        IpoReport.id.in_(id_list)
    ).all()

    out = []
    for r in rows:
        out.append({
            "title": r.title,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "revenue": r.revenue,
            "profit": r.profit,
            "debt": r.debt,
            "promoter_mentioned": bool(r.promoter_mentioned),
            "overall_score": r.sebi_score if r.sebi_score is not None else r.llm_score,
        })
    return jsonify({"comparison": out})


# ---------- Quiz / myth / SIP ----------
@app.route('/quiz/next_question', methods=['GET'])
def next_quiz_question():
    if scam_data: return jsonify(random.choice(scam_data))
    return jsonify({'error': 'No quiz data available'}), 404

@app.route('/get_myth', methods=['GET'])
def get_myth():
    if myth_data: return jsonify(random.choice(myth_data))
    return jsonify({'error': 'No myth data available'}), 404

@app.route('/calculate_sip', methods=['POST'])
def calculate_sip():
    data = request.get_json(force=True, silent=True) or {}
    try:
        future_value = float(data['amount']); years = float(data['years'])
        rate = float(data.get('rate', 12)) / 100
        i = rate / 12; n = years * 12
        if i == 0:
            sip = future_value / n if n > 0 else 0
        else:
            sip = future_value * (i / ((1 + i)**n - 1))
        growth_data = []
        for year in range(1, int(years) + 1):
            invested = sip * 12 * year
            value = sip * (((1 + i)**(year*12) - 1) / i) * (1 + i) if i > 0 else invested
            growth_data.append({'year': year, 'invested': round(invested), 'value': round(value)})
        return jsonify({'monthly_sip': round(sip, 2), 'growth_data': growth_data})
    except Exception as e:
        return jsonify({'error': f'Invalid input. {e}'}), 400

# ---------- RAG sources ----------
ALLOWED_SOURCE_EXTS = {'.pdf', '.txt', '.csv'}

@app.route('/sources', methods=['GET'])
def list_sources():
    try:
        if not os.path.isdir(RAG_SOURCES_PATH):
            return jsonify({'sources': [], 'error': 'rag_sources directory not found'}), 200
        files = []
        for entry in sorted(os.listdir(RAG_SOURCES_PATH), key=lambda s: s.lower()):
            ext = os.path.splitext(entry)[1].lower()
            if ext in ALLOWED_SOURCE_EXTS:
                enc = quote(entry, safe='')
                viewer_url = url_for('source_viewer', filename=enc, _external=False)
                raw_url = url_for('serve_source_file', filename=enc, _external=False)
                files.append({'name': entry, 'viewer_url': viewer_url, 'url': raw_url})
        return jsonify({'sources': files})
    except Exception:
        return jsonify({'sources': [], 'error': traceback.format_exc()}), 500

def _resolve_source_path(encoded_filename: str):
    try:
        fname = unquote(encoded_filename or '').strip()
        if not fname: return None
        candidate = os.path.abspath(os.path.join(RAG_SOURCES_PATH, fname))
        base_abs = os.path.abspath(RAG_SOURCES_PATH)
        if not (candidate == base_abs or candidate.startswith(base_abs + os.sep)):
            return None
        if not os.path.exists(candidate) or not os.path.isfile(candidate): return None
        return candidate
    except Exception:
        return None

@app.route('/source_files/<path:filename>', methods=['GET'])
def serve_source_file(filename):
    try:
        full_path = _resolve_source_path(filename)
        if not full_path: return jsonify({'error': 'File not found'}), 404
        basename = os.path.basename(full_path)
        ext = os.path.splitext(basename)[1].lower()
        if ext not in ALLOWED_SOURCE_EXTS:
            return jsonify({'error': 'Only PDF, TXT, or CSV files are allowed.'}), 400
        if ext == '.pdf':
            directory = os.path.dirname(full_path)
            resp = send_from_directory(directory, basename, as_attachment=False)
            resp.headers['Content-Disposition'] = f'inline; filename="{basename}"'
            return resp
        if ext in ('.txt', '.csv'):
            with open(full_path, 'r', encoding='utf-8', errors='replace') as fh:
                text = fh.read()
            html = f"""
            <!doctype html>
            <html>
              <head>
                <meta charset="utf-8"/>
                <meta name="viewport" content="width=device-width,initial-scale=1"/>
                <title>{basename}</title>
                <style>body{{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial; padding:1rem}} pre{{white-space:pre-wrap;word-wrap:break-word}}</style>
              </head>
              <body>
                <a href="/sources">← Back to sources</a>
                <h3>{basename}</h3>
                <pre>{text}</pre>
              </body>
            </html>
            """
            return Response(html, mimetype='text/html')
        return jsonify({'error': 'Unsupported file type'}), 400
    except Exception:
        app.logger.exception("Error serving source file")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/source_view/<path:filename>', methods=['GET'])
def source_viewer(filename):
    try:
        full_path = _resolve_source_path(filename)
        if not full_path: return jsonify({'error': 'File not found'}), 404
        basename = os.path.basename(full_path)
        file_url = url_for('serve_source_file', filename=quote(basename, safe=''), _external=False)
        viewer_html = f"""
        <!doctype html>
        <html>
          <head>
            <meta charset="utf-8"/>
            <meta name="viewport" content="width=device-width,initial-scale=1"/>
            <title>Viewing: {basename}</title>
            <style>body,html{{height:100%;margin:0}} .topbar{{padding:8px;background:#f3f4f6;border-bottom:1px solid #e5e7eb}} .iframe-wrap{{height:calc(100vh - 52px)}}</style>
          </head>
          <body>
            <div class="topbar">
              <a href="/sources" style="margin-right:12px">← Back to sources</a>
              <strong>{basename}</strong>
              <a style="float:right" href="{file_url}" target="_blank" rel="noopener">Open raw</a>
            </div>
            <div class="iframe-wrap">
              <iframe src="{file_url}" style="width:100%;height:100%;border:0;"></iframe>
            </div>
          </body>
        </html>
        """
        return Response(viewer_html, mimetype='text/html')
    except Exception:
        app.logger.exception("Error building source viewer")
        return jsonify({'error': 'Internal server error'}), 500

# ---------- Auth pages ----------
# Note: /login and /register now serve SPA index.html when frontend/dist exists (enables client-side routing)

# ---------- Auth APIs ----------
@app.route("/auth/register", methods=["POST"])
def auth_register():
    try:
        data = request.get_json(force=True, silent=True) or {}
        username = data.get("username", "").strip()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")

        if not username or not email or not password:
            return jsonify({"error": "All fields are required"}), 400

        # check if already exists
        if User.query.filter_by(email=email).first():
            return jsonify({"error": "Email already registered"}), 400
        if User.query.filter_by(username=username).first():
            return jsonify({"error": "Username already taken"}), 400

        user = User(username=username, email=email); user.set_password(password)
        db.session.add(user); db.session.commit()
        login_user(user)

        return jsonify({"success": True, "username": user.username, "user_id": user.id})
    except Exception as e:
        logger.exception("Registration failed")
        return jsonify({"error": "Registration failed", "details": str(e)}), 500


@app.route('/auth/login', methods=['POST'])
def auth_login():
    data = request.get_json(force=True, silent=True) or {}
    username = (data.get('username') or "").strip()
    password = data.get('password') or ""
    if not username or not password: return jsonify({'error': 'username and password required'}), 400
    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        return jsonify({'error': 'invalid credentials'}), 401
    try:
        login_user(user); session['user_id'] = user.id
        payload = build_dashboard_payload(user.id)
        return jsonify({'success': True, 'dashboard': payload, 'user_id': user.id, 'username': user.username})
    except Exception:
        logger.exception("Login failed"); return jsonify({'error': 'Login failed'}), 500

@app.route('/auth/logout', methods=['POST'])
@login_required
def auth_logout():
    try:
        logout_user(); session.pop('user_id', None)
    except Exception:
        pass
    return jsonify({'success': True})

# NEW: endpoint your frontend may call for session/user info
@app.route('/auth/whoami', methods=['GET'])
def auth_whoami():
    if current_user and getattr(current_user, "is_authenticated", False):
        return jsonify({'user_id': current_user.id, 'username': current_user.username})
    uid = session.get('user_id')
    if uid:
        u = db.session.get(User, int(uid))
        if u:
            return jsonify({'user_id': u.id, 'username': u.username})
    return jsonify({'error': 'not_authenticated'}), 401

# ---------- Portfolio helpers ----------
def upsert_portfolio_from_df(user_id, df, portfolio_name="Default Portfolio"):
    df = df.rename(columns={c: c.strip() for c in df.columns})
    symbol_col = next((c for c in df.columns if c.lower() in ('symbol','ticker','scrip','stock')), None)
    qty_col = next((c for c in df.columns if c.lower() in ('quantity','qty','shares')), None)
    price_col = next((c for c in df.columns if c.lower() in ('avg_price','avg price','price','cost','current price','ltp')), None)
    if not symbol_col or not qty_col:
        raise ValueError("Uploaded file must include symbol and quantity columns (headers like 'symbol','ticker' and 'quantity').")
    portfolio = Portfolio.query.filter_by(user_id=user_id, name=portfolio_name).first()
    if not portfolio:
        portfolio = Portfolio(name=portfolio_name, user_id=user_id)
        db.session.add(portfolio); db.session.commit()
    Holding.query.filter_by(portfolio_id=portfolio.id).delete()
    for _, row in df.iterrows():
        symbol = str(row[symbol_col]).strip().upper()
        try: quantity = float(row[qty_col])
        except Exception: quantity = 0.0
        try: avg_price = float(row[price_col]) if price_col and not pd.isna(row[price_col]) else None
        except Exception: avg_price = None
        if symbol:
            h = Holding(symbol=symbol, quantity=quantity, avg_price=avg_price, portfolio_id=portfolio.id)
            db.session.add(h)
    db.session.commit(); return portfolio

# ---------- Startup ----------
def create_db_and_seed(app):
    with app.app_context():
        db.create_all()
        logger.info("Database created/checked.")
        if not User.query.filter_by(username="test").first():
            u = User(username="test", email="test@example.com"); u.set_password("test123")
            db.session.add(u); db.session.commit()
            p = Portfolio(name="Default Portfolio", user_id=u.id)
            db.session.add(p); db.session.commit()
            logger.info("Seeded test user (username=test / password=test123)")

if __name__ == '__main__':
    create_db_and_seed(app)
    ok = initialize_app()
    if not ok:
        logger.warning("Initialization had errors. LLM/embeddings may not work until configured correctly.")
    else:
        logger.info("Initialization complete.")
    port = int(os.getenv("PORT", 5001))
    logger.info("Starting Flask server on http://0.0.0.0:%s ...", port)
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
