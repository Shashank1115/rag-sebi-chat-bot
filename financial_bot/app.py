import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


# Load environment variables
load_dotenv()

# --- LLM and Embeddings Configuration ---
llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Vector Store Configuration ---
VECTOR_STORE_PATH = "vector_store"
db = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)

# --- General Q&A Chain ---
qa_prompt_template = """
You are a helpful Financial Assistant. Your goal is to provide accurate and clear answers
about Indian finance, banking rules, and the budget, based ONLY on the context provided.
CONTEXT: {context}
QUESTION: {question}
INSTRUCTIONS:
1. Answer the question based solely on the provided context.
2. If the answer is not found, say "I'm sorry, the answer is not available in my current knowledge base."
3. Be concise and do not give investment advice.
"""
QA_PROMPT = PromptTemplate(template=qa_prompt_template, input_variables=["context", "question"])
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT},
)

# --- Portfolio Analysis Chain ---
analysis_prompt_template = """
You are "SEBI Saathi", an expert portfolio analyst. Your role is to provide an objective, educational analysis of the user's stock portfolio based on the data provided and the financial context you have. You MUST NOT give any investment advice (e.g., "buy this", "sell that").

FINANCIAL CONTEXT FROM KNOWLEDGE BASE:
{context}

USER'S PORTFOLIO DATA:
{question}

INSTRUCTIONS:
Analyze the user's portfolio based on the data and the financial context. Provide a "Portfolio Health Check" covering the following points:
1.  **Overall Summary:** Briefly describe the portfolio (e.g., number of stocks, total investment).
2.  **Sector Concentration:** Identify the top 3 sectors by investment value. Explain the risks of high concentration in a single sector.
3.  **Diversification:** Comment on the level of diversification across different sectors.
4.  **Capitalization Mix:** (If possible to infer) Comment on the mix between large-cap, mid-cap, and small-cap stocks.
5.  **Educational Insight:** Provide one educational insight based on the portfolio. For example, if there is high concentration, explain the benefits of diversification using the provided context.
"""
ANALYSIS_PROMPT = PromptTemplate(template=analysis_prompt_template, input_variables=["context", "question"])
analysis_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 2}), # Retrieve 2 relevant educational chunks
    return_source_documents=True,
    chain_type_kwargs={"prompt": ANALYSIS_PROMPT},
)

def analyze_portfolio(file_path):
    """Loads a portfolio from a CSV/Excel file and provides an analysis."""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            print("Unsupported file format. Please use CSV or Excel.")
            return

        # Convert dataframe to a string format for the LLM
        portfolio_string = df.to_string()
        
        print("\nAnalyzing your portfolio...")
        result = analysis_chain.invoke(portfolio_string)
        
        print("\n--- Portfolio Health Check ---")
        print(result["result"])
        print("----------------------------")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

def main():
    """Main function to run the chatbot interface."""
    print("Welcome to SEBI Saathi! (Powered by Groq & Local Embeddings)")
    print("Modes:")
    print("1. Ask a question directly.")
    print("2. To analyze a portfolio, type: analyze <path_to_your_file.csv>")
    print("   (e.g., analyze C:\\Users\\shash\\Desktop\\my_portfolio.csv)")
    print("3. Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYour command: ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        if user_input.lower().startswith('analyze '):
            # Portfolio Analysis Mode
            parts = user_input.split(' ', 1)
            if len(parts) > 1:
                file_path = parts[1].strip()
                analyze_portfolio(file_path)
            else:
                print("Please provide a file path after 'analyze'.")
        
        elif user_input:
            # General Q&A Mode
            result = qa_chain.invoke(user_input)
            print("\nAnswer:")
            print(result["result"])

if __name__ == "__main__":
    main()
