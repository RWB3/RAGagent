# app.py
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import logging
from rag_agent import RAGAgent
from dotenv import load_dotenv


# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize the RAGAgent
agent = RAGAgent()

@app.route("/")
def index():
    global agent
    if agent is None:
        return "<h1>Error: RAGAgent failed to initialize. Check logs.</h1>"
    # Automatically load session
    agent.load_session()
    conversation_history = agent.conversation_history
    return render_template("index.html", conversation_history=conversation_history)

@app.route("/get_response", methods=["POST"])
def get_response():
    global agent
    # Ensure the knowledge base exists
    if not os.path.exists("knowledge_base"):
        logging.error("Knowledge base directory 'knowledge_base' not found.")
        return jsonify({"response": "Error: Knowledge base directory 'knowledge_base' not found."})
    # Updated: Check the collection via the ChromaDB client
    if agent.chromadb_client.collection is None:
        logging.error("Collection is not initialized.")
        return jsonify({"response": "Error: Collection is not initialized."})
    
    # Load new documents (duplicates are handled inside load_documents)
    agent.chromadb_client.load_documents("knowledge_base")
    
    user_message = request.form.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "Please provide a message.", "conversation_history": agent.conversation_history})
    
    logging.info(f"User Query: {user_message}")
    response_text = agent.generate_answer(user_message)
    agent.save_session()  # Save session after processing
    return jsonify({"response": response_text, "conversation_history": agent.conversation_history})

@app.route("/analyze_code", methods=["POST"])
def analyze_code():
    if agent is None:
        return jsonify({"analysis": "Error: RAGAgent failed to initialize."})
    analysis = agent.analyze_code("rag_agent.py")
    return jsonify({"analysis": analysis})

@app.route("/save_session", methods=["POST"])
def save_current_session():
    if agent is None:
        return jsonify({"status": "error", "message": "RAGAgent failed to initialize."})
    try:
        agent.save_session()
        return jsonify({"status": "success", "message": "Session saved successfully."})
    except Exception as e:
        logging.error(f"Error saving session: {e}")
        return jsonify({"status": "error", "message": f"Error saving session: {e}"})

@app.route("/load_session", methods=["POST"])
def load_saved_session():
    global agent
    # Delete the existing agent and reinitialize it
    if agent is not None:
        del agent
    agent = RAGAgent()
    if agent is None:
        return jsonify({"status": "error", "message": "Error loading session. RAGAgent failed to initialize."})
    return jsonify({"status": "success", "message": "Session loaded successfully.", "conversation_history": agent.conversation_history})

if __name__ == "__main__":
    app.run(debug=True)
