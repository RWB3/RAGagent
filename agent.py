 # agent.py

import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from langchain import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA

# Configuration
DOCUMENTS_FOLDER = 'documents'  # Folder to store your text documents
VECTOR_STORE_PATH = 'vector_store.index'
DOCUMENTS_PICKLE = 'documents.pkl'
OPENAI_API_KEY = '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'  # Replace with your OpenAI API key

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
         vector_store = FAISS(embeddings.embed_query, index, docs, VECTOR_STORE_PATH)

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
