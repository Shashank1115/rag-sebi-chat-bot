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

basedir = os.path.abspath(os.path.dirname(__file__))
VECTOR_STORE_PATH = os.path.join(basedir, "vector_store")

def initialize_app():
    """Loads all models and initializes prompts."""
    global llm, embeddings, db, qa_prompt, analysis_prompt
    print("--- STARTING INITIALIZATION ---")
    try:
        llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
        db = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)
        print("Models and vector store loaded successfully.")

        qa_prompt_template = """
        You are an expert financial analyst. Use the following pieces of context to answer the question at the end. If you don't know the answer from the context, just say that you don't know.
        CONTEXT: {context}
        QUESTION: {question}
        ANSWER:
        """
        qa_prompt = PromptTemplate(template=qa_prompt_template, input_variables=["context", "question"])

        analysis_prompt_template = """
        You are "SEBI Saathi", an expert portfolio analyst. Analyze the user's portfolio based on the data provided. Provide a "Portfolio Health Check" as a Markdown formatted response covering Sector Concentration, Diversification, Strengths, Risks, and SEBI-Compliant Guidance. Do NOT give investment advice.
        USER'S PORTFOLIO DATA:
        {portfolio_data}
        ANALYSIS:
        """
        analysis_prompt = PromptTemplate(template=analysis_prompt_template, input_variables=["portfolio_data"])
        
        print("Prompts initialized successfully.")
        return True
    except Exception as e:
        print(f"--- FATAL ERROR DURING INITIALIZATION ---\n{traceback.format_exc()}")
        return False

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    if not all([llm, db, qa_prompt]):
        return jsonify({'error': 'Application not initialized correctly.'}), 500
    data = request.get_json()
    question = data.get('question')
    try:
        docs = db.similarity_search(question, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])
        formatted_prompt = qa_prompt.format(context=context, question=question)
        result = llm.invoke(formatted_prompt)
        return jsonify({'answer': result.content})
    except Exception as e:
        print(f"--- AN ERROR OCCURRED IN /ask ---\n{traceback.format_exc()}")
        return jsonify({'error': 'A server error occurred.'}), 500

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

            # Calculate investment value for the chart
            df['Investment Value'] = df['Quantity'] * df['Current Price']
            sector_allocation = df.groupby('Sector')['Investment Value'].sum().round(2).to_dict()
            
            # Prepare data for the LLM
            portfolio_string = df.to_string()
            
            formatted_prompt = analysis_prompt.format(portfolio_data=portfolio_string)
            result = llm.invoke(formatted_prompt)
            analysis_markdown = result.content

            # Return all necessary data as a single JSON object
            return jsonify({
                'analysis_markdown': analysis_markdown,
                'chart_data': sector_allocation
            })
            
        except Exception as e:
            print(f"--- AN ERROR OCCURRED DURING ANALYSIS ---\n{traceback.format_exc()}")
            return jsonify({'error': 'An error occurred during analysis.'}), 500
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
    return jsonify({'error': 'File processing failed.'}), 500

if __name__ == '__main__':
    if initialize_app():
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
    else:
        print("Could not start Flask server due to initialization errors.")
