import os
import pandas as pd
import traceback
import json
import random
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from flask import jsonify

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, send_from_directory, session

from flask_cors import CORS
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
from markdown import markdown

# --- Initializations ---
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# --- Global Variables & Paths ---
llm = None
embeddings = None
db_main = None
qa_prompt = None
analysis_prompt = None
scam_data = []
myth_data = []

basedir = os.path.abspath(os.path.dirname(__file__))
VECTOR_STORE_PATH_MAIN = os.path.join(basedir, "vector_store")
USER_DATA_PATH = os.path.join(basedir, "user_data")
DATA_PATH = os.path.join(basedir, "data")
RAG_SOURCES_PATH = os.path.join(DATA_PATH, "rag_sources")




def initialize_app():
    """Loads all models, data, and initializes prompts."""
    global llm, embeddings, db_main, qa_prompt, analysis_prompt, scam_data, myth_data
    print("--- STARTING INITIALIZATION ---")
    try:
        llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
        db_main = Chroma(persist_directory=VECTOR_STORE_PATH_MAIN, embedding_function=embeddings)
        print("Models and MAIN vector store loaded successfully.")
        
        with open(os.path.join(DATA_PATH, 'scam_examples.json'), 'r') as f: scam_data = json.load(f)
        with open(os.path.join(DATA_PATH, 'myths.json'), 'r') as f: myth_data = json.load(f)
        print("Engagement data loaded successfully.")

        qa_prompt_template = "CONTEXT: {context}\nQUESTION: {question}\nINSTRUCTIONS: Based ONLY on the context, answer the user's question. If the answer is not in the context, say so."
        qa_prompt = PromptTemplate(template=qa_prompt_template, input_variables=["context", "question"])
        
        analysis_prompt_template = "You are 'SEBI Saathi', an expert portfolio analyst. Analyze the user's portfolio based on the data provided. Provide a 'Portfolio Health Check' as Markdown. Do NOT give investment advice.\nUSER'S PORTFOLIO DATA:\n{portfolio_data}\nANALYSIS:"
        analysis_prompt = PromptTemplate(template=analysis_prompt_template, input_variables=["portfolio_data"])
        
        os.makedirs(USER_DATA_PATH, exist_ok=True)
        print("Prompts initialized successfully.")
        return True
    except Exception as e:
        print(f"--- FATAL ERROR DURING INITIALIZATION ---\n{traceback.format_exc()}")
        return False

# --- Flask Routes ---

@app.route('/sebi/circulars')
def sebi_circulars():
    url = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=7&smid=0"
    res = requests.get(url)
    soup = BeautifulSoup(res.content, 'html.parser')

    table = soup.find('table')
    items = []
    if table:
        rows = table.find_all('tr')[1:6]  # top 5 recent circulars
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 2:
                date = cols[0].get_text(strip=True)
                title_tag = cols[1].find('a')
                title = title_tag.get_text(strip=True)
                link = title_tag['href']
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
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")  # need 2 days to calculate change
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
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})



@app.route('/')
def index():
    if 'user_id' not in session:
        session['user_id'] = os.urandom(8).hex()
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')
    search_scope = data.get('scope', 'all')
    user_id = session.get('user_id')
    
    try:
        docs = []
        user_vector_store_path = os.path.join(USER_DATA_PATH, user_id, "vector_store")

        if search_scope == 'user_only' and os.path.exists(user_vector_store_path):
            db_user = Chroma(persist_directory=user_vector_store_path, embedding_function=embeddings)
            docs = db_user.similarity_search(question, k=4)
        else:
            docs = db_main.similarity_search(question, k=4)

        context = "\n\n".join([doc.page_content for doc in docs])
        if not context.strip():
            context = "No relevant information was found in the selected knowledge base."

        formatted_prompt = qa_prompt.format(context=context, question=question)
        result = llm.invoke(formatted_prompt)
        return jsonify({'answer': result.content})
    except Exception as e:
        print(f"--- AN ERROR OCCURRED IN /ask ---\n{traceback.format_exc()}")
        return jsonify({'error': 'A server error occurred.'}), 500

@app.route('/upload_and_ingest', methods=['POST'])
def upload_and_ingest():
    file = request.files.get('userFile')
    user_id = session.get('user_id')
    if not all([file, user_id]): return jsonify({'error': 'Missing file or user session'}), 400

    user_docs_path = os.path.join(USER_DATA_PATH, user_id, "docs")
    user_vector_store_path = os.path.join(USER_DATA_PATH, user_id, "vector_store")
    os.makedirs(user_docs_path, exist_ok=True)
    
    file_path = os.path.join(user_docs_path, file.filename)
    file.save(file_path)

    try:
        if os.path.exists(user_vector_store_path): shutil.rmtree(user_vector_store_path)
        loader = DirectoryLoader(user_docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        texts = text_splitter.split_documents(documents)
        Chroma.from_documents(texts, embeddings, persist_directory=user_vector_store_path)
        return jsonify({'success': True, 'user_library': os.listdir(user_docs_path)})
    except Exception as e:
        print(f"--- AN ERROR OCCURRED DURING USER INGESTION ---\n{traceback.format_exc()}")
        return jsonify({'error': 'Failed to process the document.'}), 500

@app.route('/delete_user_file', methods=['POST'])
def delete_user_file():
    data = request.get_json()
    filename = data.get('filename')
    user_id = session.get('user_id')
    if not all([filename, user_id]): return jsonify({'error': 'Missing filename or user session'}), 400

    user_docs_path = os.path.join(USER_DATA_PATH, user_id, "docs")
    user_vector_store_path = os.path.join(USER_DATA_PATH, user_id, "vector_store")
    file_to_delete = os.path.join(user_docs_path, filename)

    try:
        if os.path.exists(file_to_delete):
            os.remove(file_to_delete)
            if os.path.exists(user_vector_store_path): shutil.rmtree(user_vector_store_path)
            
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
    except Exception as e:
        return jsonify({'error': 'Failed to delete file.'}), 500

@app.route('/get_user_library', methods=['GET'])
def get_user_library():
    user_id = session.get('user_id')
    if not user_id: return jsonify({'user_library': []})
    user_docs_path = os.path.join(USER_DATA_PATH, user_id, "docs")
    if os.path.exists(user_docs_path):
        return jsonify({'user_library': os.listdir(user_docs_path)})
    return jsonify({'user_library': []})

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'portfolioFile' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['portfolioFile']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400

    if file:
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name

            if tmp_path.endswith('.csv'): df = pd.read_csv(tmp_path)
            elif tmp_path.endswith(('.xlsx', '.xls')): df = pd.read_excel(tmp_path)
            else:
                os.remove(tmp_path)
                return jsonify({'error': 'Unsupported file format'}), 400

            df['Investment Value'] = df['Quantity'] * df['Current Price']
            sector_allocation = df.groupby('Sector')['Investment Value'].sum().round(2).to_dict()
            portfolio_string = df.to_string()
            
            formatted_prompt = analysis_prompt.format(portfolio_data=portfolio_string)
            result = llm.invoke(formatted_prompt)
            analysis_markdown = result.content

            return jsonify({'analysis_markdown': analysis_markdown, 'chart_data': sector_allocation})
            
        except Exception as e:
            print(f"--- AN ERROR OCCURRED DURING ANALYSIS ---\n{traceback.format_exc()}")
            return jsonify({'error': 'An error occurred during analysis.'}), 500
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
    return jsonify({'error': 'File processing failed.'}), 500

# --- Engagement Routes ---
@app.route('/quiz/next_question', methods=['GET'])
def next_quiz_question():
    return jsonify(random.choice(scam_data))

@app.route('/get_myth', methods=['GET'])
def get_myth():
    return jsonify(random.choice(myth_data))

@app.route('/calculate_sip', methods=['POST'])
def calculate_sip():
    data = request.get_json()
    try:
        future_value = float(data['amount'])
        years = float(data['years'])
        rate = float(data.get('rate', 12)) / 100
        i = rate / 12
        n = years * 12
        if i == 0: sip = future_value / n if n > 0 else 0
        else: sip = future_value * (i / ((1 + i)**n - 1))
        
        growth_data = []
        for year in range(1, int(years) + 1):
            invested = sip * 12 * year
            value = sip * (((1 + i)**(year*12) - 1) / i) * (1 + i) if i > 0 else invested
            growth_data.append({'year': year, 'invested': round(invested), 'value': round(value)})
        return jsonify({'monthly_sip': round(sip, 2), 'growth_data': growth_data})
    except Exception as e:
        return jsonify({'error': f'Invalid input. {e}'}), 400

# --- Sources (RAG) Routes ---
@app.route('/sources', methods=['GET'])
def list_sources():
    """Return a list of source files (PDF/TXT/CSV) from data/rag_sources."""
    try:
        if not os.path.isdir(RAG_SOURCES_PATH):
            return jsonify({
                'sources': [],
                'error': 'rag_sources directory not found'
            }), 200

        allowed_exts = {'.pdf', '.txt', '.csv'}
        files = []
        for entry in os.listdir(RAG_SOURCES_PATH):
            ext = os.path.splitext(entry)[1].lower()
            if ext in allowed_exts:
                files.append({
                    'name': entry,
                    'url': f"/source_files/{entry}"
                })
        # Sort by name for stable order
        files.sort(key=lambda x: x['name'].lower())
        return jsonify({'sources': files})
    except Exception as e:
        return jsonify({'sources': [], 'error': str(e)}), 500


@app.route('/source_files/<path:filename>', methods=['GET'])
def serve_source_file(filename):
    """Serve a source file from data/rag_sources securely (PDF/TXT/CSV only)."""
    allowed_exts = {'.pdf', '.txt', '.csv'}
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_exts:
        return jsonify({'error': 'Only PDF, TXT, or CSV files are allowed.'}), 400
    try:
        return send_from_directory(RAG_SOURCES_PATH, filename, as_attachment=False)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404


if __name__ == '__main__':
    if initialize_app():
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
    else:
        print("Could not start Flask server due to initialization errors.")

