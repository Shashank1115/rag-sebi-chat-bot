# import os
# import traceback
# from dotenv import load_dotenv
# from langchain_chroma import Chroma
# from langchain.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, CSVLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Load environment variables from the .env file you uploaded
# load_dotenv()
# print("API Keys loaded.")

# # ==============================================================================
# # Step 3: Build the Vector Database (This runs only once)
# # ==============================================================================
# VECTOR_STORE_PATH = "vector_store"
# DATA_PATH = "data" # The folder you created and uploaded your documents to

# def create_vector_db():
#     print("\n--- Building Vector Database ---")
#     if os.path.exists(VECTOR_STORE_PATH):
#         print("Vector store already exists. Loading...")
#         return
        
#     loaders = [
#         DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True),
#         DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader, show_progress=True),
#         DirectoryLoader(DATA_PATH, glob="**/*.csv", loader_cls=CSVLoader, show_progress=True, loader_kwargs={'encoding': 'utf8'}),
#     ]
    
#     loaded_documents = []
#     for loader in loaders:
#         try:
#             loaded_documents.extend(loader.load())
#         except Exception as e:
#             print(f"Error loading files with {loader.__class__.__name__}: {e}")
#             continue

#     if not loaded_documents:
#         print("No documents found in the data directory.")
#         return

#     print(f"Loaded {len(loaded_documents)} document(s).")
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
#     texts = text_splitter.split_documents(loaded_documents)
#     print(f"Split into {len(texts)} chunks.")

#     print("Initializing Google Embeddings...")
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))

#     print(f"Creating vector store with {len(texts)} chunks. This may take a few minutes...")
#     db = Chroma.from_documents(texts, embeddings, persist_directory=VECTOR_STORE_PATH)
#     print(f"Vector store created and saved at: {VECTOR_STORE_PATH}")

# # Run the database creation function
# create_vector_db()

# # ==============================================================================
# # Step 4: Load the RAG components for the chatbot
# # ==============================================================================
# print("\n--- Initializing Chatbot ---")
# try:
#     llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
#     db = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)
    
#     qa_prompt_template = """
#     You are a helpful Financial Assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer from the context, just say that you don't know.
#     CONTEXT: {context}
#     QUESTION: {question}
#     ANSWER:
#     """
#     qa_prompt = PromptTemplate(template=qa_prompt_template, input_variables=["context", "question"])
    
#     print("Chatbot initialized successfully.")
# except Exception as e:
#     print("--- FATAL ERROR DURING INITIALIZATION ---")
#     print(traceback.format_exc())

# # ==============================================================================
# # Step 5: Define the chat function and run the chat loop
# # ==============================================================================
# def ask_question(question):
#     """The core RAG logic for answering a question."""
#     try:
#         retriever = db.as_retriever(search_kwargs={"k": 3})
#         docs = retriever.invoke(question)
#         context = "\n\n".join([doc.page_content for doc in docs])
#         formatted_prompt = qa_prompt.format(context=context, question=question)
#         result = llm.invoke(formatted_prompt)
#         return result.content
#     except Exception as e:
#         print(traceback.format_exc())
#         return "Sorry, an error occurred while processing your question."

# print("\n--- SEBI Saathi is Ready ---")
# print("Ask any question about Indian finance. Type 'exit' to quit.")

# while True:
#     user_question = input("\nYour question: ")
#     if user_question.lower() == 'exit':
#         print("Goodbye!")
#         break
#     if user_question:
#         answer = ask_question(user_question)
#         print("\nAnswer:")
#         print(answer)

import os
import pandas as pd
import traceback
import json
import random
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import tempfile
from markdown import markdown

# --- Initializations ---
load_dotenv()
app = Flask(__name__)
CORS(app)

# --- Global Variables ---
llm = None
embeddings = None
db = None
qa_prompt = None
analysis_prompt = None
scam_data = []
myth_data = []

basedir = os.path.abspath(os.path.dirname(__file__))
VECTOR_STORE_PATH = os.path.join(basedir, "vector_store")
# --- THIS PATH IS CORRECT ---
# It looks in the main data folder for the JSON files
DATA_PATH = os.path.join(basedir, "data")

def initialize_app():
    """Loads all models, data, and initializes prompts."""
    global llm, embeddings, db, qa_prompt, analysis_prompt, scam_data, myth_data
    print("--- STARTING INITIALIZATION ---")
    try:
        llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
        db = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)
        print("Models and vector store loaded successfully.")

        # Load engagement data from the main 'data' folder
        with open(os.path.join(DATA_PATH, 'scam_examples.json'), 'r') as f:
            scam_data = json.load(f)
        with open(os.path.join(DATA_PATH, 'myths.json'), 'r') as f:
            myth_data = json.load(f)
        print("Engagement data loaded successfully.")

        # Prompts
        qa_prompt_template = "CONTEXT: {context}\nQUESTION: {question}\nINSTRUCTIONS: Based only on the context, answer the user's question. If the answer is not in the context, say so."
        qa_prompt = PromptTemplate(template=qa_prompt_template, input_variables=["context", "question"])

        analysis_prompt_template = "You are 'SEBI Saathi', an expert portfolio analyst. Analyze the user's portfolio based on the data provided. Provide a 'Portfolio Health Check' as Markdown. Do NOT give investment advice.\nUSER'S PORTFOLIO DATA:\n{portfolio_data}\nANALYSIS:"
        analysis_prompt = PromptTemplate(template=analysis_prompt_template, input_variables=["portfolio_data"])
        
        print("Prompts initialized successfully.")
        return True
    except Exception as e:
        print(f"--- FATAL ERROR DURING INITIALIZATION ---\n{traceback.format_exc()}")
        return False

# --- Flask Routes ---
# (The rest of your backend.py file remains the same)
# ...

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    # ... (code remains the same)
    data = request.get_json()
    question = data.get('question')
    try:
        docs = db.similarity_search(question, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])
        formatted_prompt = qa_prompt.format(context=context, question=question)
        result = llm.invoke(formatted_prompt)
        return jsonify({'answer': result.content})
    except Exception as e:
        return jsonify({'error': 'A server error occurred.'}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    # ... (code remains the same)
    file = request.files.get('portfolioFile')
    if not file: return "No file part", 400
    # ... (rest of the function)
    df = pd.read_csv(file) # Simplified for brevity
    df['Investment Value'] = df['Quantity'] * df['Current Price']
    sector_allocation = df.groupby('Sector')['Investment Value'].sum().round(2).to_dict()
    portfolio_string = df.to_string()
    formatted_prompt = analysis_prompt.format(portfolio_data=portfolio_string)
    result = llm.invoke(formatted_prompt)
    analysis_html = markdown(result.content)
    return render_template('portfolio.html', analysis_html=analysis_html, chart_data=sector_allocation)

# --- NEW ENGAGEMENT ROUTES ---

@app.route('/quiz/next_question', methods=['GET'])
def next_quiz_question():
    if not scam_data:
        return jsonify({'error': 'Scam data not loaded'}), 500
    question = random.choice(scam_data)
    # Return the full object so the frontend can check the answer
    return jsonify(question)

@app.route('/get_myth', methods=['GET'])
def get_myth():
    if not myth_data:
        return jsonify({'error': 'Myth data not loaded'}), 500
    myth = random.choice(myth_data)
    return jsonify(myth)

@app.route('/calculate_sip', methods=['POST'])
def calculate_sip():
    data = request.get_json()
    try:
        future_value = float(data['amount'])
        years = float(data['years'])
        rate_of_return = float(data.get('rate', 12)) / 100 # Default 12% annual return
        
        # SIP formula
        i = rate_of_return / 12
        n = years * 12
        sip = future_value * (i / ((1 + i)**n - 1))
        
        # Calculate year-by-year growth
        growth_data = []
        invested_amount = 0
        for year in range(1, int(years) + 1):
            invested_amount = sip * 12 * year
            # Simplified future value calculation for chart
            current_value = sip * (((1 + i)**(year*12) - 1) / i) * (1 + i)
            growth_data.append({'year': year, 'invested': round(invested_amount), 'value': round(current_value)})

        return jsonify({
            'monthly_sip': round(sip, 2),
            'growth_data': growth_data
        })
    except (ValueError, KeyError, ZeroDivisionError) as e:
        return jsonify({'error': f'Invalid input for calculation. {e}'}), 400


if __name__ == '__main__':
    if initialize_app():
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
    else:
        print("Could not start Flask server due to initialization errors.")
