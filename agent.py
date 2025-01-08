# agent.py

import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Updated Imports from langchain_community
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

from langchain.chains import RetrievalQA

# Configuration
DOCUMENTS_FOLDER = 'documents'  # Folder to store your text documents
VECTOR_STORE_PATH = 'vector_store.index'
DOCUMENTS_PICKLE = 'documents.pkl'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Fetch from environment variable

# Validate API Key
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable in the .env file.")

# Initialize Embedding Model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents

def create_vector_store(docs, embeddings, index_path, pickle_path):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    with open(pickle_path, 'wb') as f:
        pickle.dump(docs, f)

def initialize_agent():
    # Check if vector store exists
    if not os.path.exists(VECTOR_STORE_PATH) or not os.path.exists(DOCUMENTS_PICKLE):
        print("Creating vector store...")
        docs = load_documents(DOCUMENTS_FOLDER)
        embeddings = embedding_model.encode(docs, show_progress_bar=True)
        create_vector_store(docs, embeddings, VECTOR_STORE_PATH, DOCUMENTS_PICKLE)
        print("Vector store created.")
    else:
        print("Vector store already exists.")

    # Load documents and index
    with open(DOCUMENTS_PICKLE, 'rb') as f:
        docs = pickle.load(f)

    index = faiss.read_index(VECTOR_STORE_PATH)

    # Initialize FAISS with embeddings
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    
    # **Refactored FAISS Initialization: Removed 'documents' Argument**
    vector_store = FAISS(
        embedding_function=embeddings.embed_query,
        index=index,
        index_name=VECTOR_STORE_PATH
    )

    # Initialize OpenAI model
    llm = OpenAI(api_key=OPENAI_API_KEY)

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
            break
        response = qa(query)
        print("AI Agent:", response['result'])
