
# agent.py

import os
import pickle
from langchain.schema import Document
import faiss
from dotenv import load_dotenv
import warnings

# Suppress specific deprecation warnings (optional and temporary)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables from .env file
load_dotenv()

# Updated Imports from langchain_community and langchain_huggingface
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Ensure correct import path

from langchain.chains import RetrievalQA

# Configuration
DOCUMENTS_FOLDER = 'documents'      # Folder to store your text documents
VECTOR_STORE_PATH = 'vector_store.index'
DOCUMENTS_PICKLE = 'documents.pkl'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Fetch from environment variable

# Validate API Key
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable in the .env file.")

def load_documents(folder_path):
    """
    Load documents from the specified folder and convert them to Document objects.
    """
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                document = Document(page_content=content, metadata={"source": filename})
                documents.append(document)
    return documents

def create_vector_store(docs, embeddings, index_path, pickle_path):
    """
    Create a FAISS vector store from the provided Document objects and embeddings.
    """
    # Create FAISS vector store from documents and embeddings
    vector_store = FAISS.from_documents(documents=docs, embedding=embeddings)
    # Save the FAISS index locally
    vector_store.save_local(index_path)
    # Save the documents for future reference
    with open(pickle_path, 'wb') as f:
        pickle.dump(docs, f)

def initialize_agent():
    """
    Initialize the AI agent by setting up the vector store and the retrieval QA chain.
    """
    # Check if vector store and documents pickle exist
    if not os.path.exists(VECTOR_STORE_PATH) or not os.path.exists(DOCUMENTS_PICKLE):
        print("Creating vector store...")
        docs = load_documents(DOCUMENTS_FOLDER)
        # Initialize Embeddings
        embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        create_vector_store(docs, embeddings, VECTOR_STORE_PATH, DOCUMENTS_PICKLE)
        print("Vector store created.")
    else:
        print("Vector store already exists.")
        # Load documents from pickle
        with open(DOCUMENTS_PICKLE, 'rb') as f:
            docs = pickle.load(f)
    
    # Initialize Embeddings (ensure consistency)
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    
    # Load FAISS vector store from the local index
    try:
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    except ValueError as ve:
        print(f"Error loading vector store: {ve}")
        print("If you trust the source of the vector store, ensure 'allow_dangerous_deserialization' is set to True.")
        raise ve
    
    # Initialize OpenAI model
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    
    # Create Retrieval QA Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    
    return qa

if __name__ == '__main__':
    qa = initialize_agent()
    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            print("Exiting AI Agent. Goodbye!")
            break
        response = qa(query)
        print("AI Agent:", response['result'])