from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import logging
from rag_agent import RAGAgent  # Import your updated RAGAgent class
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the RAGAgent (move this outside the route)
agent = None  # Initialize agent to None

def initialize_agent():
    """Initializes the RAGAgent. Returns the agent or None if initialization fails."""
    global agent  # Use the global agent variable
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in the environment variables. Please set it.")

        # Specify the correct model name here
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o")  # Updated default to gpt-4o

        agent = RAGAgent(openai_api_key, model_name=model_name)
        logging.info("RAGAgent initialized successfully.")
        return agent
    except Exception as e:
        logging.error(f"Error initializing RAGAgent: {e}")
        agent = None
        return None

# Initialize the agent when the app starts
agent = initialize_agent()

@app.route("/")
def index():
    """Renders the main chat interface."""
    global agent
    if agent is None:
        return "<h1>Error: RAGAgent failed to initialize. Check the logs.</h1>"
    conversation_history = agent.conversation_history
    return render_template("index.html", conversation_history=conversation_history)


@app.route("/get_response", methods=["POST"])
def get_response():
    """Handles user input and returns the agent's response."""
    global agent

    # Check that the knowledge base directory exists.
    if not os.path.exists("knowledge_base"):
        logging.error("Knowledge base directory 'knowledge_base' not found.")
        return jsonify({"response": "Error: Knowledge base directory 'knowledge_base' not found."})

    # Ensure the collection is initialized
    if agent.collection is None:
        logging.error("Collection is not initialized.")
        return jsonify({"response": "Error: Collection is not initialized."})

    # Load documents; duplicates are handled within load_documents
    agent.load_documents("knowledge_base")

    user_message = request.form.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "Please provide a message.", "conversation_history": agent.conversation_history})

    logging.info(f"User Query: {user_message}")
    response = agent.run(user_message)
    agent.save_session()  # Save the session after each interaction
    return jsonify({"response": response, "conversation_history": agent.conversation_history})


@app.route("/analyze_code", methods=["POST"])
def analyze_code():
    """Analyzes the agent's code and returns the analysis."""
    if agent is None:
        return jsonify({"analysis": "Error: RAGAgent failed to initialize."})

    analysis = agent.analyze_code("rag_agent.py")  # Replace with the actual filename if different
    return jsonify({"analysis": analysis})


@app.route("/save_session", methods=["POST"])
def save_current_session():
    """Saves the current session."""
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
    """Loads a saved session."""
    global agent  # Use the global agent variable

    # Delete existing agent instance
    if agent is not None:
        del agent

    # Attempt to initialize agent and load session
    agent = initialize_agent()

    if agent is None:
        return jsonify({"status": "error", "message": "Error loading session. RAGAgent failed to initialize."})

    return jsonify({"status": "success", "message": "Session loaded successfully.", "conversation_history": agent.conversation_history})


if __name__ == "__main__":
    app.run(debug=True)
