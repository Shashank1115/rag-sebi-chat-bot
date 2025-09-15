# backend.py
"""
Robust backend for financial_bot.

Features:
 - Exponential backoff + retries wrapper for transient embedding/search failures
 - LRU cache for local embeddings
 - Local SentenceTransformers fallback for embeddings when remote provider fails
 - similarity_search wrapper with fallback to local embeddings to avoid 500s
 - IPO analyzer endpoint (file or pasted content)
 - Portfolio upload / dashboard endpoints, SEBI circulars scraping, live market data
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
import pathlib
import re
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from urllib.parse import unquote, quote
from sqlalchemy import func
from slugify import slugify  

from flask import (
    Flask, request, jsonify, render_template, send_from_directory, session ,Response,url_for,render_template_string
)
from flask_cors import CORS

# Auth / DB
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user, UserMixin
)
from werkzeug.security import generate_password_hash, check_password_hash

# LangChain / providers
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings   # local HF embeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from functools import lru_cache
# markdown converter (server-side use)
from markdown import markdown
from werkzeug.utils import secure_filename

# Local fallback embedder
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# --- Simple in-memory news cache ---
NEWS_CACHE = {}
NEWS_CACHE_TTL = 10 * 60  # 10 minutes

# --- Load env ---
load_dotenv()

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("backend")

# --- Flask app init ---
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))
CORS(app)

# --- Database config ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///financial_bot.db")
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# --- Flask-Login ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login_page"

# --- Global variables & directories ---
basedir = os.path.abspath(os.path.dirname(__file__))
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

# --- Models (embedded here for a single-file drop-in) ---
class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    portfolios = db.relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Portfolio(db.Model):
    __tablename__ = "portfolios"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), default="My Portfolio")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    user = db.relationship("User", back_populates="portfolios")
    holdings = db.relationship("Holding", back_populates="portfolio", cascade="all, delete-orphan")

class Holding(db.Model):
    __tablename__ = "holdings"
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(32), nullable=False, index=True)
    quantity = db.Column(db.Float, nullable=False, default=0.0)
    avg_price = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    portfolio_id = db.Column(db.Integer, db.ForeignKey("portfolios.id"), nullable=False)

    portfolio = db.relationship("Portfolio", back_populates="holdings")


try:
    from slugify import slugify
except Exception:
    def slugify(s):
        # Minimal fallback - replace non-alphanumerics with hyphens
        if not s:
            return ""
        return "".join(c if c.isalnum() else "-" for c in str(s)).strip("-").lower()

class IPOReport(db.Model):
    __tablename__ = "ipo_reports"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    title = db.Column(db.String(512))
    filename = db.Column(db.String(512), nullable=True)          # original uploaded filename (if any)
    content_md = db.Column(db.Text)                              # markdown report returned by LLM / fallback
    raw_text = db.Column(db.Text)                                # extracted raw text from PDF
    excerpt = db.Column(db.String(512), nullable=True)           # short excerpt / title guess
    heuristic_meta = db.Column(db.Text, nullable=True)           # JSON-dumped heuristic facts
    checklist = db.Column(db.Text, nullable=True)                # JSON-dumped per-check results
    overall_score = db.Column(db.Float, nullable=True)           # normalized score (e.g. 0-100)
    sebi_score = db.Column(db.Float, nullable=True)              # (optional) earlier schema
    sebi_checks = db.Column(db.JSON, nullable=True)              # legacy per-check results (keeps compatibility)
    llm_score = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User", backref="ipo_reports")

def _ensure_user_ipo_dir(user_id):
    base = os.path.join(USER_DATA_PATH, str(user_id), "ipo_reports")
    os.makedirs(base, exist_ok=True)
    return base

def save_ipo_report_to_db(user_id, title, filename, content_md, raw_text,
                          sebi_score=None, sebi_checks=None, llm_score=None,
                          heuristic_meta=None, checklist=None, overall_score=None):
    """
    Persist IPOReport; returns created IPOReport.id
    """
    try:
        rpt = IPOReport(
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
    except Exception:
        db.session.rollback()
        logger.exception("DB commit failed while saving IPOReport")
        raise

    # also save a JSON copy to disk for quick inspection (non-fatal)
    try:
        base = _ensure_user_ipo_dir(user_id)
        fname = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{slugify(rpt.title)[:120] or 'report'}.json"
        path = os.path.join(base, fname)
        with open(path, 'w', encoding='utf8') as fh:
            json.dump({
                "id": rpt.id,
                "user_id": rpt.user_id,
                "title": rpt.title,
                "filename": rpt.filename,
                "created_at": rpt.created_at.isoformat(),
                "sebi_score": rpt.sebi_score,
                "sebi_checks": rpt.sebi_checks,
                "heuristic_meta": json.loads(rpt.heuristic_meta or "{}"),
                "checklist": json.loads(rpt.checklist or "{}"),
                "overall_score": rpt.overall_score,
                "llm_score": rpt.llm_score,
                "content_md": rpt.content_md
            }, fh, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Failed to write IPO report JSON file to disk (non-fatal).")

    return rpt.id

def list_user_ipo_reports(user_id):
    rows = IPOReport.query.filter_by(user_id=user_id).order_by(IPOReport.created_at.desc()).all()
    out = []
    for r in rows:
        out.append({
            "id": r.id,
            "title": r.title,
            "filename": r.filename,
            "created_at": r.created_at.isoformat(),
            "overall_score": r.overall_score,
            "sebi_score": r.sebi_score,
            "llm_score": r.llm_score
        })
    return out

def get_user_ipo_report(user_id, report_id):
    r = IPOReport.query.filter_by(user_id=user_id, id=report_id).first()
    if not r:
        return None
    try:
        return {
            "id": r.id,
            "title": r.title,
            "filename": r.filename,
            "created_at": r.created_at.isoformat(),
            "content_md": r.content_md,
            "raw_text": r.raw_text,
            "excerpt": r.excerpt,
            "heuristic_meta": json.loads(r.heuristic_meta or "{}"),
            "checklist": json.loads(r.checklist or "{}"),
            "overall_score": r.overall_score,
            "sebi_score": r.sebi_score,
            "sebi_checks": r.sebi_checks,
            "llm_score": r.llm_score
        }
    except Exception:
        logger.exception("Failed to deserialize IPO report JSON fields")
        # Fallback: return raw strings if JSON load fails
        return {
            "id": r.id,
            "title": r.title,
            "filename": r.filename,
            "created_at": r.created_at.isoformat(),
            "content_md": r.content_md,
            "raw_text": r.raw_text,
            "excerpt": r.excerpt,
            "heuristic_meta": r.heuristic_meta,
            "checklist": r.checklist,
            "overall_score": r.overall_score,
            "sebi_score": r.sebi_score,
            "sebi_checks": r.sebi_checks,
            "llm_score": r.llm_score
        }

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
# Embedding fallback helpers (LRU cache + local model)
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

def local_embed_list(text: str):
    return list(_cached_local_embed(text))

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
            time.sleep(sleep_for)
            attempt += 1

def similarity_search_with_fallback(query: str, k: int = 4, user_vector_store_path: str = None):
    global db_main, embeddings

    target_db = None
    try:
        if user_vector_store_path:
            if os.path.exists(user_vector_store_path):
                target_db = Chroma(persist_directory=user_vector_store_path, embedding_function=embeddings)
                logger.debug("Using user vectorstore at %s", user_vector_store_path)
            else:
                logger.debug("User vectorstore path not found: %s", user_vector_store_path)
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
            logger.debug("Local embedding computed (len=%d). Calling similarity_search_by_vector.", len(qvec))
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
# Initialize models, LLM, embeddings, prompts, vector DB
# -----------------------
def initialize_app():
    global llm, embeddings, db_main, qa_prompt, analysis_prompt, scam_data, myth_data
    logger.info("--- STARTING INITIALIZATION ---")
    try:
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")

        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not set. LLM may fail.")

        llm = ChatGroq(
            temperature=0,
            model_name="llama-3.1-8b-instant",
            api_key=GROQ_API_KEY
        )

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        os.makedirs(VECTOR_STORE_PATH_MAIN, exist_ok=True)
        db_main = Chroma(persist_directory=VECTOR_STORE_PATH_MAIN, embedding_function=embeddings)
        logger.info("Models and MAIN vector store loaded successfully (or opened).")

        scam_data = safe_load_json(os.path.join(DATA_PATH, 'scam_examples.json'))
        myth_data = safe_load_json(os.path.join(DATA_PATH, 'myths.json'))
        logger.info("Engagement data loaded (scam examples: %d, myths: %d).", len(scam_data), len(myth_data))

        qa_prompt_template = (
            "CONTEXT: {context}\n"
            "QUESTION: {question}\n"
            "INSTRUCTIONS: Based ONLY on the context, answer the user's question. If the answer is not in the context, say so."
        )
        qa_prompt = PromptTemplate(template=qa_prompt_template, input_variables=["context", "question"])

        analysis_prompt_template = (
            "You are 'SEBI Saathi', an expert portfolio analyst. Analyze the user's portfolio based on the data provided. "
            "Provide a 'Portfolio Health Check' as Markdown. Do NOT give investment advice.\n"
            "USER'S PORTFOLIO DATA:\n{portfolio_data}\nANALYSIS:"
        )
        analysis_prompt = PromptTemplate(template=analysis_prompt_template, input_variables=["portfolio_data"])

        os.makedirs(USER_DATA_PATH, exist_ok=True)
        return True
    except Exception:
        logger.exception("FATAL ERROR DURING INITIALIZATION")
        return False

# -----------------------
# --- ROUTES (API) -----
# -----------------------

@app.route('/')
def index():
    if 'anon_id' not in session:
        session['anon_id'] = os.urandom(8).hex()
    return render_template('index.html')

@app.route('/sebi/circulars')
def sebi_circulars():
    url = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=7&smid=0"
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
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
                    title = cols[1].get_text(strip=True)
                    full_link = url
                else:
                    title = title_tag.get_text(strip=True)
                    link = title_tag.get('href', '')
                    full_link = requests.compat.urljoin(url, link)
                items.append({'date': date, 'title': title, 'url': full_link})
    return jsonify({'circulars': items})

@app.route('/market/live')
def market_live():
    try:
        tickers = {
            "NIFTY 50": "^NSEI",
            "SENSEX": "^BSESN",
            "CDSL": "CDSL.NS",
            "KFINTECH": "KFINTECH.NS",
        }
        data = {}
        for name, symbol in tickers.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                if len(hist) >= 2:
                    prev_close = hist['Close'].iloc[-2]
                    last_close = hist['Close'].iloc[-1]
                    change = last_close - prev_close
                    pct_change = (change / prev_close) * 100 if prev_close else 0
                    data[name] = {
                        "price": round(float(last_close), 2),
                        "change": round(float(change), 2),
                        "pct_change": round(float(pct_change), 2)
                    }
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

    if not question:
        return jsonify({'error': 'No question provided.'}), 400

    try:
        docs = []
        user_vector_store_path = os.path.join(USER_DATA_PATH, user_id or '', "vector_store")

        if search_scope == 'user_only' and os.path.exists(user_vector_store_path):
            try:
                docs = similarity_search_with_fallback(question, k=4, user_vector_store_path=user_vector_store_path)
            except Exception as e:
                logger.warning("User-only search failed, falling back to main store: %s", repr(e))
                docs = similarity_search_with_fallback(question, k=4, user_vector_store_path=None)
        else:
            docs = similarity_search_with_fallback(question, k=4, user_vector_store_path=None)

        context = "\n\n".join([getattr(doc, 'page_content', str(doc)) for doc in (docs or [])])
        if not context.strip():
            context = "No relevant information was found in the selected knowledge base."

        formatted_prompt = qa_prompt.format(context=context, question=question)
        result = llm.invoke(formatted_prompt)
        answer = getattr(result, 'content', result) if result is not None else "No response from LLM."
        return jsonify({'answer': answer})
    except Exception:
        logger.exception("--- AN ERROR OCCURRED IN /ask ---")
        return jsonify({'error': 'A server error occurred.'}), 500

@app.route('/upload_and_ingest', methods=['POST'])
@login_required
def upload_and_ingest():
    # require login using flask-login decorator; current_user is available
    file = request.files.get('userFile')
    # prefer current_user.id (flask-login)
    try:
        user_id = current_user.id
    except Exception:
        # fallback to session (older behavior)
        user_id = session.get('user_id')

    if not all([file, user_id]):
        return jsonify({'error': 'Missing file or user session'}), 400

    user_id_str = str(user_id)
    user_docs_path = os.path.join(USER_DATA_PATH, user_id_str, "docs")
    user_vector_store_path = os.path.join(USER_DATA_PATH, user_id_str, "vector_store")
    os.makedirs(user_docs_path, exist_ok=True)

    filename = secure_filename(file.filename) or "uploaded.pdf"
    file_path = os.path.join(user_docs_path, filename)
    file.save(file_path)

    try:
        # rebuild user vector store from PDFs
        if os.path.exists(user_vector_store_path):
            shutil.rmtree(user_vector_store_path)

        loader = DirectoryLoader(user_docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        if not documents:
            return jsonify({'error': 'No PDFs found in uploaded docs.'}), 400

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        texts = text_splitter.split_documents(documents)
        Chroma.from_documents(texts, embeddings, persist_directory=user_vector_store_path)
        return jsonify({'success': True, 'user_library': sorted(os.listdir(user_docs_path))})
    except Exception:
        logger.exception("--- AN ERROR OCCURRED DURING USER INGESTION ---")
        return jsonify({'error': 'Failed to process the document.'}), 500


@app.route('/delete_user_file', methods=['POST'])
def delete_user_file():
    data = request.get_json(force=True, silent=True) or {}
    filename = data.get('filename')
    user_id = data.get('user_id', session.get('user_id'))
    if not all([filename, user_id]):
        return jsonify({'error': 'Missing filename or user session'}), 400

    user_docs_path = os.path.join(USER_DATA_PATH, user_id, "docs")
    user_vector_store_path = os.path.join(USER_DATA_PATH, user_id, "vector_store")
    file_to_delete = os.path.join(user_docs_path, filename)

    try:
        if os.path.exists(file_to_delete):
            os.remove(file_to_delete)
            if os.path.exists(user_vector_store_path):
                shutil.rmtree(user_vector_store_path)

            remaining_files = os.listdir(user_docs_path)
            if remaining_files:
                loader = DirectoryLoader(user_docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                texts = text_splitter.split_documents(documents)
                Chroma.from_documents(texts, embeddings, persist_directory=user_vector_store_path)
            return jsonify({'success': True, 'user_library': remaining_files})
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception:
        logger.exception("--- ERROR deleting file ---")
        return jsonify({'error': 'Failed to delete file.'}), 500

@app.route('/get_user_library', methods=['GET'])
def get_user_library():
    # user_id may be int (from session or current_user). ensure it's a string for os.path.join
    user_id = session.get('user_id')
    # also allow logged-in user via flask-login
    try:
        if not user_id and current_user and getattr(current_user, 'is_authenticated', False):
            user_id = current_user.id
    except Exception:
        user_id = user_id

    if not user_id:
        return jsonify({'user_library': []})

    # ensure str for path building
    user_id_str = str(user_id)
    user_docs_path = os.path.join(USER_DATA_PATH, user_id_str, "docs")
    if os.path.exists(user_docs_path):
        try:
            files = sorted(os.listdir(user_docs_path))
        except Exception:
            files = []
        return jsonify({'user_library': files})
    return jsonify({'user_library': []})


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'portfolioFile' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['portfolioFile']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        if tmp_path.endswith('.csv'):
            df = pd.read_csv(tmp_path)
        elif tmp_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(tmp_path)
        else:
            os.remove(tmp_path)
            return jsonify({'error': 'Unsupported file format'}), 400

        if not {'Quantity', 'Current Price', 'Sector'}.issubset(df.columns):
            possible_qty = next((c for c in df.columns if c.lower() in ('quantity','qty','shares')), None)
            possible_price = next((c for c in df.columns if c.lower() in ('current price','price','last price','ltp')), None)
            possible_sector = next((c for c in df.columns if c.lower() in ('sector','industry')), None)
            if possible_qty and possible_price:
                df = df.rename(columns={possible_qty: 'Quantity', possible_price: 'Current Price'})
                if possible_sector:
                    df = df.rename(columns={possible_sector: 'Sector'})
            else:
                return jsonify({'error': 'CSV/XLSX must include Quantity and Current Price columns (or similar headers)'}), 400

        df['Investment Value'] = df['Quantity'] * df['Current Price']
        sector_allocation = df.groupby('Sector')['Investment Value'].sum().round(2).to_dict()
        portfolio_string = df.to_string()

        formatted_prompt = analysis_prompt.format(portfolio_data=portfolio_string)
        result = llm.invoke(formatted_prompt)
        analysis_markdown = getattr(result, 'content', result)
        return jsonify({'analysis_markdown': analysis_markdown, 'chart_data': sector_allocation})
    except Exception:
        logger.exception("--- AN ERROR OCCURRED DURING ANALYSIS ---")
        return jsonify({'error': 'An error occurred during analysis.'}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route('/portfolio/history')
@login_required
def portfolio_history():
    try:
        portfolio = Portfolio.query.filter_by(user_id=current_user.id).first()
        if not portfolio or not portfolio.holdings:
            return jsonify({'history': []})

        symbols = [h.symbol for h in portfolio.holdings]
        end_date = datetime.today()
        start_date = end_date - pd.DateOffset(months=6)

        hist_data = {}
        for sym in symbols:
            try:
                df = yf.download(sym, start=start_date, end=end_date, progress=False, interval="1wk")
                if not df.empty:
                    hist_data[sym] = df['Close']
            except Exception:
                continue

        # combine into portfolio total
        portfolio_series = None
        for h in portfolio.holdings:
            if h.symbol in hist_data:
                series_val = hist_data[h.symbol] * h.quantity
                portfolio_series = series_val if portfolio_series is None else portfolio_series.add(series_val, fill_value=0)

        if portfolio_series is None:
            return jsonify({'history': []})

        history = [{'date': d.strftime('%Y-%m-%d'), 'value': round(float(v), 2)}
                   for d, v in portfolio_series.dropna().items()]

        return jsonify({'history': history})
    except Exception as e:
        logger.exception("Failed to compute portfolio history")
        return jsonify({'error': str(e)}), 500


# IPO analyzer helpers and route
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
        loader = PyPDFLoader(path)
        docs = loader.load()
        text = "\n\n".join([d.page_content for d in docs])
        return text[:max_chars]
    except Exception:
        try:
            with open(path, 'rb') as f:
                return f.read().decode('utf-8', errors='ignore')[:max_chars]
        except Exception:
            return ""

def simple_rule_extract(text):
    res = {}
    lower = text.lower()
    import re
    m = re.search(r'(revenue|turnover)[^\d{0,6}]*([\d,\.]+\s*(?:crore|cr|₹|rs|rs\.|m|bn|b|million|billion)?)', text, re.I)
    if m:
        res['revenue'] = m.group(2).strip()
    m2 = re.search(r'(profit|net profit|net income)[^\d{0,6}]*([\d,\,\.\s]+(?:crore|cr|₹|rs|m|bn|b|million|billion)?)', text, re.I)
    if m2:
        res['profit'] = m2.group(2).strip()
    m3 = re.search(r'(debt|total debt|borrowings)[^\d{0,6}]*([\d,\,\.\s]+(?:crore|cr|₹|rs|m|bn|b|million|billion)?)', text, re.I)
    if m3:
        res['debt'] = m3.group(2).strip()
    if 'promoter' in lower:
        res['promoter_mentioned'] = True
    return res
def parse_overall_score_from_md(md_text: str):
    """
    Try to find an 'Overall' score like 'Overall: Score: 18/30' or 'Score: 18/30'.
    Returns float (e.g., 18.0) or None.
    """
    if not md_text:
        return None
    # pattern: capture something like 18/30
    m = re.search(r'([Oo]verall[^:\n]*[:\-\s]*)(?:Score[:\s]*)?([0-9]{1,3})(?:\s*/\s*([0-9]{1,3}))', md_text)
    if m:
        try:
            val = float(m.group(2))
            denom = float(m.group(3)) if m.group(3) else None
            # If denom present and is 30, return val (assume scale is 30). Else return raw val (caller may interpret).
            if denom and denom != 0:
                # normalize to out-of-30 if denom is known and not 30
                if denom != 30:
                    return round((val / denom) * 30.0, 2)
                return val
            return val
        except Exception:
            return None
    # fallback: find 'Score: X/Y'
    m2 = re.search(r'[Ss]core[:\s]*([0-9]{1,3})\s*/\s*([0-9]{1,3})', md_text)
    if m2:
        try:
            return float(m2.group(1))
        except:
            return None
    return None

def save_ipo_report(user_id, title, content_md, excerpt, heuristic_meta_dict=None, checklist_obj=None):
    """
    Persist IPOReport for user. heuristic_meta_dict and checklist_obj will be JSON-dumped.
    Returns the saved IPOReport object.
    """
    try:
        meta_json = json.dumps(heuristic_meta_dict or {})
        checklist_json = json.dumps(checklist_obj or [])
        overall = parse_overall_score_from_md(content_md)
        rpt = IPOReport(
            user_id=user_id,
            title=title or (excerpt[:120] if excerpt else "IPO report"),
            content_md=content_md or "",
            excerpt=(excerpt or "")[:120],
            heuristic_meta=meta_json,
            checklist=checklist_json,
            overall_score=overall
        )
        db.session.add(rpt)
        db.session.commit()
        return rpt
    except Exception:
        logger.exception("Failed to save IPOReport to DB")
        db.session.rollback()
        raise
@app.route('/ipo/analyze', methods=['POST'])
@login_required
def ipo_analyze():
    try:
        text = ""
        uploaded_filename = None
        if 'ipoFile' in request.files and request.files['ipoFile'].filename:
            f = request.files['ipoFile']
            uploaded_filename = secure_filename(f.filename)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_filename)[1])
            f.save(tmp.name)
            try:
                if uploaded_filename.lower().endswith('.pdf'):
                    text = extract_text_from_pdf(tmp.name)
                else:
                    with open(tmp.name, 'rb') as fh:
                        text = fh.read().decode('utf-8', errors='ignore')
            finally:
                try:
                    os.remove(tmp.name)
                except Exception:
                    pass
        elif request.form.get('content'):
            text = request.form.get('content')[:150000]
        elif request.json and request.json.get('content'):
            text = request.json.get('content')[:150000]

        if not text or not text.strip():
            return jsonify({'error': 'No IPO content provided.'}), 400

        # basic heuristic metadata
        meta = simple_rule_extract(text)
        checklist_out = []
        for key, label in IPO_CHECKLIST:
            checklist_out.append({'key': key, 'label': label, 'status': 'Unknown', 'notes': ''})

        # ---------------------------
        # SEBI-style rule-based checks
        # ---------------------------
        lower = text.lower()
        # mapping: checklist key -> list of indicative keywords/phrases
        SEBI_KEYWORDS = {
            "financials": ["revenue", "turnover", "net profit", "profit", "loss", "audited", "financial statements", "balance sheet", "profit and loss", "cash flow"],
            "governance": ["promoter", "promoters", "shareholding", "board", "independent director", "chairman", "audit committee", "related party"],
            "use_of_proceeds": ["use of proceeds", "objects of the issue", "utilisation", "utilization", "funds will be used", "proceeds of the issue"],
            "business_risks": ["risk factors", "risks", "contingent liabilities", "litigation", "regulatory", "competition", "uncertain"],
            "market_position": ["market share", "industry", "competition", "market position", "competitive", "growth drivers"],
            "compliance": ["sebi", "regulatory approval", "approval from", "filing with sebi", "compliance", "disclosure", "red herring prospectus", "drhp", "rhp"]
        }
        # weights for scoring (sum -> 100)
        WEIGHTS = {
            "financials": 25,
            "governance": 20,
            "use_of_proceeds": 20,
            "business_risks": 15,
            "market_position": 10,
            "compliance": 10
        }

        per_check = {}
        total_score = 0.0
        for key, label in IPO_CHECKLIST:
            kws = SEBI_KEYWORDS.get(key, [])
            found = False
            reason = ""
            for kw in kws:
                if kw in lower:
                    found = True
                    reason = f"Found keyword '{kw}'"
                    break
            status = "Verified" if found else "Missing"
            per_check[key] = {"label": label, "status": status, "notes": reason}
            if found:
                total_score += WEIGHTS.get(key, 0)

        # normalize to percentage 0-100 (since WEIGHTS already sum to 100 in our mapping)
        sebi_score = round(total_score, 2)

        # ---------------------------
        # Try LLM-based analysis (existing behavior)
        # ---------------------------
        report_md = ""
        try:
            if llm is None:
                raise Exception("LLM not initialized")

            excerpt = text[:8000]
            prompt = [
                "You are 'SEBI Saathi', an IPO analyst. Analyze the following IPO prospectus/excerpt and produce a structured report.",
                "Follow these steps exactly:",
                "1) For each checklist item, say: status = {Verified | Partially Verified | Missing}, short reason (1-2 lines).",
                "2) Extract key facts: revenue, profit, debt amounts (if present), promoter stake (if present), stated use of proceeds.",
                "3) Provide growth-positive factors (bullet list), risks (bullet list).",
                "4) Provide a 1-5 priority score per checklist item and an overall IPO score (out of 30).",
                "5) Output final result as MARKDOWN only. No apologies, no meta text.",
                "",
                "CHECKLIST:",
            ]
            for _, label in IPO_CHECKLIST:
                prompt.append(f"- {label}")
            prompt.append("\nPROSPECTUS EXCERPT:\n")
            prompt.append(excerpt)
            formatted_prompt = "\n".join(prompt)

            result = llm.invoke(formatted_prompt)
            report_md = getattr(result, 'content', str(result))
        except Exception as e:
            logger.exception("LLM-based IPO analysis failed; using heuristic fallback: %s", e)
            # fallback heuristic markdown
            parts = []
            parts.append("# IPO Quick Analysis (heuristic fallback)")
            parts.append("")
            parts.append("**Extracted facts (heuristic):**")
            if meta:
                for k, v in meta.items():
                    parts.append(f"- **{k.replace('_',' ').title()}:** {v}")
            else:
                parts.append("- No clear numeric facts found in the provided excerpt.")
            parts.append("")
            parts.append("**Checklist (heuristic / not exhaustive):**")
            for k, v in per_check.items():
                parts.append(f"- **{v['label']}**: **{v['status']}**. {v['notes']}")
            parts.append("")
            parts.append("**Growth positives (heuristic):**")
            parts.append("- Could not determine automatically — provide more text or use LLM.")
            parts.append("")
            parts.append("**Risks (heuristic):**")
            parts.append("- Not analyzed in-depth in fallback mode.")
            parts.append("")
            parts.append(f"**SEBI-style compliance score (heuristic):** {sebi_score}/100")
            report_md = "\n".join(parts)

        # Save the report for this user
        user_id = current_user.id if current_user else session.get('user_id')
        title_guess = None
        # try to extract title line from text
        first_line = (text.splitlines() or [''])[0]
        if first_line and len(first_line.strip()) > 5:
            title_guess = first_line.strip()[:200]
        title = title_guess or uploaded_filename or "IPO analysis"

        try:
            new_id = save_ipo_report_to_db(
                user_id=user_id,
                title=title,
                filename=uploaded_filename,
                content_md=report_md,
                raw_text=text,
                sebi_score=sebi_score,
                sebi_checks=per_check,
                llm_score=None
            )
        except Exception:
            logger.exception("Failed to save IPO report to DB")
            new_id = None

        return jsonify({'ipo_report_md': report_md, 'report_id': new_id, 'sebi_score': sebi_score})
    except Exception:
        app.logger.error("IPO analyze error:\n" + traceback.format_exc())
        return jsonify({'error': 'Internal server error during IPO analysis.'}), 500


@app.route('/ipo/list', methods=['GET'])
@login_required
def ipo_list():
    try:
        user_id = current_user.id
        reports = list_user_ipo_reports(user_id)
        return jsonify({'reports': reports})
    except Exception:
        logger.exception("Failed to list IPO reports")
        return jsonify({'error': 'Failed to retrieve reports'}), 500

@app.route('/ipo/get/<int:report_id>', methods=['GET'])
@login_required
def ipo_get(report_id):
    r = IPOReport.query.get(report_id)
    if not r or r.user_id != current_user.id:
        return jsonify({'error': 'report not found'}), 404
    try:
        return jsonify({
            'id': r.id,
            'title': r.title,
            'created_at': r.created_at.isoformat() if r.created_at else None,
            'content_md': r.content_md,
            'heuristic_meta': json.loads(r.heuristic_meta or "{}"),
            'checklist': json.loads(r.checklist or "[]"),
            'overall_score': r.overall_score
        })
    except Exception:
        logger.exception("Failed to load IPO report")
        return jsonify({'error': 'Failed to load report'}), 500

@app.route('/ipo/compare', methods=['GET'])
@login_required
def ipo_compare():
    ids_raw = request.args.get('ids', '')
    if not ids_raw:
        return jsonify({'error': 'ids query param required (comma-separated)'}), 400
    try:
        ids = [int(x) for x in ids_raw.split(',') if x.strip().isdigit()]
    except Exception:
        return jsonify({'error': 'Invalid ids param'}), 400

    reports = IPOReport.query.filter(IPOReport.id.in_(ids), IPOReport.user_id == current_user.id).all()
    if not reports:
        return jsonify({'error': 'No reports found for given ids'}), 404

    cmp_rows = []
    for r in reports:
        try:
            meta = json.loads(r.heuristic_meta or "{}")
        except:
            meta = {}
        cmp_rows.append({
            'id': r.id,
            'title': r.title,
            'created_at': r.created_at.isoformat() if r.created_at else None,
            'revenue': meta.get('revenue'),
            'profit': meta.get('profit'),
            'debt': meta.get('debt'),
            'promoter_mentioned': meta.get('promoter_mentioned', False),
            'overall_score': r.overall_score
        })
    return jsonify({'comparison': cmp_rows})


@app.route('/quiz/next_question', methods=['GET'])
def next_quiz_question():
    if scam_data:
        return jsonify(random.choice(scam_data))
    return jsonify({'error': 'No quiz data available'}), 404

@app.route('/get_myth', methods=['GET'])
def get_myth():
    if myth_data:
        return jsonify(random.choice(myth_data))
    return jsonify({'error': 'No myth data available'}), 404

@app.route('/calculate_sip', methods=['POST'])
def calculate_sip():
    data = request.get_json(force=True, silent=True) or {}
    try:
        future_value = float(data['amount'])
        years = float(data['years'])
        rate = float(data.get('rate', 12)) / 100
        i = rate / 12
        n = years * 12
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

ALLOWED_SOURCE_EXTS = {'.pdf', '.txt', '.csv'}

@app.route('/sources', methods=['GET'])
def list_sources():
    try:
        if not os.path.isdir(RAG_SOURCES_PATH):
            return jsonify({'sources': [], 'error': 'rag_sources directory not found'}), 200

        allowed_exts = ALLOWED_SOURCE_EXTS
        files = []
        for entry in sorted(os.listdir(RAG_SOURCES_PATH), key=lambda s: s.lower()):
            ext = os.path.splitext(entry)[1].lower()
            if ext in allowed_exts:
                # build viewer/raw URLs with percent-encoded filename
                enc = quote(entry, safe='')
                viewer_url = url_for('source_viewer', filename=enc, _external=False)
                raw_url = url_for('serve_source_file', filename=enc, _external=False)
                files.append({'name': entry, 'viewer_url': viewer_url, 'url': raw_url})
        return jsonify({'sources': files})
    except Exception:
        return jsonify({'sources': [], 'error': traceback.format_exc()}), 500


def _resolve_source_path(encoded_filename: str):
    """
    Decode percent-encoded filename, preserve filename characters,
    ensure the resolved path is inside RAG_SOURCES_PATH and exists.
    Returns the absolute file path if valid, otherwise None.
    """
    try:
        fname = unquote(encoded_filename or '')
        fname = fname.strip()
        if not fname:
            return None

        # resolve candidate path
        candidate = os.path.abspath(os.path.join(RAG_SOURCES_PATH, fname))

        # ensure candidate is inside RAG_SOURCES_PATH to prevent traversal
        base_abs = os.path.abspath(RAG_SOURCES_PATH)
        if not (candidate == base_abs or candidate.startswith(base_abs + os.sep)):
            return None

        if not os.path.exists(candidate) or not os.path.isfile(candidate):
            return None

        return candidate
    except Exception:
        return None


@app.route('/source_files/<path:filename>', methods=['GET'])
def serve_source_file(filename):
    """
    Serve files located in RAG_SOURCES_PATH. 'filename' is percent-encoded in the URL.
    Uses _resolve_source_path to find the real file on disk.
    """
    try:
        full_path = _resolve_source_path(filename)
        if not full_path:
            return jsonify({'error': 'File not found'}), 404

        basename = os.path.basename(full_path)
        ext = os.path.splitext(basename)[1].lower()
        if ext not in ALLOWED_SOURCE_EXTS:
            return jsonify({'error': 'Only PDF, TXT, or CSV files are allowed.'}), 400

        if ext == '.pdf':
            directory = os.path.dirname(full_path)
            resp = send_from_directory(directory, basename, as_attachment=False)
            resp.headers['Content-Disposition'] = f'inline; filename="{basename}"'
            return resp

        # txt / csv -> wrap in simple HTML so it displays nicely in browser
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
    """
    Viewer page that embeds the raw file URL (served by /source_files/<filename>) in an iframe.
    """
    try:
        full_path = _resolve_source_path(filename)
        if not full_path:
            return jsonify({'error': 'File not found'}), 404

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

@app.route('/register')
def register_page():
    # Renders the register form page
    return render_template('register.html')

@app.route('/login')
def login_page():
    # Renders the login form page
    return render_template('login.html')

@app.route('/auth/register', methods=['POST'])
def auth_register():
    """
    Expects JSON: { "username": "...", "password": "...", "email": "..." }
    Returns JSON: { success: True, user_id, username } or error
    """
    data = request.get_json(force=True, silent=True) or {}
    username = (data.get('username') or "").strip()
    password = data.get('password') or ""
    email = (data.get('email') or "").strip()

    if not username or not password:
        return jsonify({'error': 'username and password required'}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'username already exists'}), 400

    try:
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        # create default portfolio
        portfolio = Portfolio(name="Default Portfolio", user_id=user.id)
        db.session.add(portfolio)
        db.session.commit()

        login_user(user)                    # Flask-Login
        session['user_id'] = user.id        # keep session key consistent for your upload endpoints

        return jsonify({'success': True, 'user_id': user.id, 'username': user.username})
    except Exception as e:
        logger.exception("Registration failed")
        db.session.rollback()
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/auth/login', methods=['POST'])
def auth_login():
    """
    Expects JSON: { "username": "...", "password": "..." }
    Returns JSON: { success: True, dashboard: {...} } on success
    """
    data = request.get_json(force=True, silent=True) or {}
    username = (data.get('username') or "").strip()
    password = data.get('password') or ""

    if not username or not password:
        return jsonify({'error': 'username and password required'}), 400

    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        return jsonify({'error': 'invalid credentials'}), 401

    try:
        login_user(user)
        session['user_id'] = user.id   # IMPORTANT: set this for other endpoints using session
        payload = build_dashboard_payload(user.id)
        return jsonify({'success': True, 'dashboard': payload, 'user_id': user.id, 'username': user.username})
    except Exception:
        logger.exception("Login failed")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/auth/logout', methods=['POST'])
@login_required
def auth_logout():
    try:
        logout_user()
        session.pop('user_id', None)
    except Exception:
        pass
    return jsonify({'success': True})

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
        db.session.add(portfolio)
        db.session.commit()

    Holding.query.filter_by(portfolio_id=portfolio.id).delete()

    for _, row in df.iterrows():
        symbol = str(row[symbol_col]).strip().upper()
        try:
            quantity = float(row[qty_col])
        except Exception:
            quantity = 0.0
        try:
            avg_price = float(row[price_col]) if price_col and not pd.isna(row[price_col]) else None
        except Exception:
            avg_price = None
        if symbol:
            h = Holding(symbol=symbol, quantity=quantity, avg_price=avg_price, portfolio_id=portfolio.id)
            db.session.add(h)
    db.session.commit()
    return portfolio

@app.route('/portfolio/upload', methods=['POST'])
@login_required
def upload_portfolio():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        if tmp_path.endswith('.csv'):
            df = pd.read_csv(tmp_path)
        elif tmp_path.endswith(('.xls','.xlsx')):
            df = pd.read_excel(tmp_path)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400

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

POSITIVE_WORDS = {"beats","upgrade","raised","approved","good","strong","positive","gain","growth","outperform","record","profit","surge"}
NEGATIVE_WORDS = {"downgrade","cut","loss","falls","fall","decline","weak","negative","recall","scandal","fraud","drops","hit","miss"}


_price_cache = {}         # in-memory cache for stock prices
CACHE_TTL = 900  
HEADLINES_CACHE = {}
HEADLINES_TTL = 900  # 15 minutes
NEWSAPI_COOLDOWN = 900  # 15 minutes
_last_newsapi_fail = 0

def fetch_headlines_for_symbol(symbol, api_key=None, max_headlines=5):
    global _last_newsapi_fail
    symbol = (symbol or "").strip()
    if not symbol:
        return []

    cache_key = f"{symbol}:{max_headlines}"
    now = time.time()
    if cache_key in HEADLINES_CACHE and (now - HEADLINES_CACHE[cache_key]["time"]) < HEADLINES_TTL:
        return HEADLINES_CACHE[cache_key]["payload"]

    headlines = []

    # --- Try NewsAPI if not in cooldown ---
    if api_key and (now - _last_newsapi_fail) > NEWSAPI_COOLDOWN:
        try:
            params = {
                "q": f'"{symbol}" OR {symbol} stock OR {symbol} share OR {symbol} company',
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": max_headlines,
                "apiKey": api_key,
            }
            r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=8)
            if r.status_code == 429:
                app.logger.warning(f"NewsAPI rate-limited for {symbol}. Entering cooldown.")
                _last_newsapi_fail = now
            else:
                r.raise_for_status()
                data = r.json()
                if data.get("status") == "ok":
                    for a in data.get("articles", [])[:max_headlines]:
                        title = (a.get("title") or "").strip()
                        desc = (a.get("description") or "").strip()
                        combined = (title + " — " + desc).strip(" — ")
                        if combined:
                            headlines.append(combined)
        except Exception as e:
            app.logger.warning(f"NewsAPI error for {symbol}: {e}")
            _last_newsapi_fail = now

    # --- Fallback: Google News ---
    if not headlines:
        try:
            query = urllib.parse.quote_plus(f"{symbol} stock")
            url = f"https://www.google.com/search?q={query}&tbm=nws"
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=8)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for g in soup.select("div.dbsr")[:max_headlines]:
                title_el = g.select_one("div.JheGif, h3")
                if title_el:
                    headlines.append(title_el.get_text(strip=True))
        except Exception as e:
            app.logger.warning(f"Google scrape failed for {symbol}: {e}")

    HEADLINES_CACHE[cache_key] = {"time": now, "payload": headlines}
    return headlines


def fetch_latest_prices(symbols):
    out = {}
    now = time.time()
    need_fetch = []

    for sym in symbols:
        if sym in _price_cache and (now - _price_cache[sym]["time"]) < CACHE_TTL:
            out[sym] = _price_cache[sym]["price"]
        else:
            need_fetch.append(sym)

    if need_fetch:
        try:
            data = yf.download(need_fetch, period="1d", group_by="ticker", progress=False)
            for sym in need_fetch:
                price = None
                try:
                    if sym in data and not data[sym].empty:
                        price = float(data[sym]["Close"].iloc[-1])
                    elif "Close" in data and not data.empty:
                        price = float(data["Close"].iloc[-1])
                except Exception:
                    pass

                # --- fallback: try BSE if NSE fails ---
                if price is None and sym.endswith(".NS"):
                    alt = sym.replace(".NS", ".BO")
                    try:
                        alt_data = yf.download(alt, period="1d", progress=False)
                        if not alt_data.empty:
                            price = float(alt_data["Close"].iloc[-1])
                            app.logger.warning(f"Fallback used: {sym} → {alt}")
                    except Exception as e:
                        app.logger.warning(f"Fallback failed for {sym}: {e}")

                _price_cache[sym] = {"price": price, "time": now}
                out[sym] = price
        except Exception as e:
            app.logger.debug(f"fetch_latest_prices: batch failed: {e}")
            for sym in need_fetch:
                out[sym] = None
                _price_cache[sym] = {"price": None, "time": now}

    return out


@app.route('/news/<symbol>', methods=['GET'])
@login_required
def news_for_symbol(symbol):
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return jsonify({'error': 'symbol required'}), 400

    try:
        max_headlines = int(request.args.get('max', 5))
    except Exception:
        max_headlines = 5

    cache_key = f"news:{symbol}:{max_headlines}"
    now = time.time()
    cached = NEWS_CACHE.get(cache_key)
    if cached and (now - cached['time']) < NEWS_CACHE_TTL:
        return jsonify({'source': 'cache', 'symbol': symbol, 'headlines': cached['payload']})

    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "").strip() or None
    headlines = fetch_headlines_for_symbol(symbol, api_key=NEWSAPI_KEY, max_headlines=max_headlines)

    NEWS_CACHE[cache_key] = {'time': now, 'payload': headlines}

    source = 'newsapi' if NEWSAPI_KEY and headlines else 'google_scrape' if headlines else 'none'
    return jsonify({'source': source, 'symbol': symbol, 'headlines': headlines})

def simple_sentiment_estimate(headlines):
    score = 0
    for h in headlines:
        txt = h.lower()
        for w in POSITIVE_WORDS:
            if w in txt: score += 1
        for w in NEGATIVE_WORDS:
            if w in txt: score -= 1
    if not headlines:
        return 0.0
    return max(-1.0, min(1.0, score / (len(headlines) * 3.0)))

def discover_company_name(sym):
    try:
        t = yf.Ticker(sym)
        info = {}
        try:
            info = t.get_info() if hasattr(t, "get_info") else getattr(t, "info", {}) or {}
        except Exception:
            info = getattr(t, "info", {}) or {}
        return info.get("longName") or info.get("shortName")
    except Exception:
        return None

def build_dashboard_payload(user_id):
    user = db.session.get(User, int(user_id))
    if not user:
        return {'error': 'user not found'}
    portfolio = Portfolio.query.filter_by(user_id=user.id).first()
    if not portfolio:
        return {'holdings': [], 'allocations': {}, 'alerts': []}

    holdings = portfolio.holdings
    symbols = [h.symbol for h in holdings]
    prices = fetch_latest_prices(symbols)

    holdings_payload = []
    total_value = 0.0
    for h in holdings:
        last_price = prices.get(h.symbol)
        current_value = (last_price or 0.0) * h.quantity
        total_value += current_value
        holdings_payload.append({
            'symbol': h.symbol,
            'quantity': h.quantity,
            'avg_price': h.avg_price,
            'last_price': last_price,
            'current_value': round(current_value, 2)
        })

    allocations = {}
    if total_value > 0:
        for item in holdings_payload:
            allocations[item['symbol']] = round((item['current_value'] / total_value) * 100, 2)
    else:
        for item in holdings_payload:
            allocations[item['symbol']] = 0.0

    alerts = []
    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
    sorted_holdings = sorted(holdings_payload, key=lambda x: x['current_value'], reverse=True)[:5]
    for h in sorted_holdings:
        company_name = discover_company_name(h['symbol'])
        query_key = company_name if company_name else h['symbol']
        headlines = fetch_headlines_for_symbol(query_key, api_key=NEWSAPI_KEY, max_headlines=5)
        sent = simple_sentiment_estimate(headlines)
        qty_factor = math.log1p(h['quantity'])
        est_pct = round(sent * 3.0 * qty_factor, 2)
        sentiment_label = "Neutral"
        if sent > 0.15:
            sentiment_label = "Positive"
        elif sent < -0.15:
            sentiment_label = "Negative"
        alerts.append({
            'symbol': h['symbol'],
            'quantity': h['quantity'],
            'sentiment': sentiment_label,
            'estimated_pct_move': est_pct,
            'headlines': headlines[:5]
        })

    return {
        'user_id': user.id,
        'username': user.username,
        'total_value': round(total_value, 2),
        'holdings': holdings_payload,
        'allocations': allocations,
        'alerts': alerts
    }

# -----------------------
# Startup: create DB & initialize LLM
def create_db_and_seed(app):
    with app.app_context():
        db.create_all()
        logger.info("Database created/checked.")
        if not User.query.filter_by(username="test").first():
            u = User(username="test", email="test@example.com")
            u.set_password("test123")
            db.session.add(u)
            db.session.commit()
            p = Portfolio(name="Default Portfolio", user_id=u.id)
            db.session.add(p)
            db.session.commit()
            logger.info("Seeded test user (username=test / password=test123)")

if __name__ == '__main__':
    create_db_and_seed(app)
    ok = initialize_app()
    if not ok:
        logger.warning("Initialization had errors. LLM/embeddings may not work until configured correctly.")
    else:
        logger.info("Initialization complete.")
    logger.info("Starting Flask server on http://0.0.0.0:%s ...", os.getenv("PORT", "5001"))
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5001)), debug=True, use_reloader=False)
