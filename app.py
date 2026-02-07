# app.py

# To run the FastAPI server, execute the following command in the terminal:
#     uvicorn app:app --reload

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Optional
import os
import logging
import traceback
from rag_agent import RAGAgent
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

app = FastAPI()

# Check if the 'static' directory exists
STATIC_DIR = "static"
if not os.path.exists(STATIC_DIR):
    logging.warning(f"Static directory '{STATIC_DIR}' not found. Creating it...")
    os.makedirs(STATIC_DIR, exist_ok=True)  # Create the directory if it doesn't exist

# Mount the 'static' directory to serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Configure Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Initialize the RAGAgent
try:
    agent = RAGAgent()
except Exception as e:
    logging.error(f"Error initializing RAGAgent: {e}")
    traceback.print_exc()
    agent = None

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    global agent
    if agent is None:
        return HTMLResponse(content="<h1>Error: RAGAgent failed to initialize. Check logs.</h1>")
    # Automatically load session
    agent.load_session()
    conversation_history = agent.conversation_history
    return templates.TemplateResponse("index.html", {"request": request, "conversation_history": conversation_history})

@app.post("/get_response")
async def get_response(request: Request, message: str = Form(...)):
    global agent
    # Ensure the knowledge base exists
    if not os.path.exists("knowledge_base"):
        logging.error("Knowledge base directory 'knowledge_base' not found.")
        return JSONResponse(content={"response": "Error: Knowledge base directory 'knowledge_base' not found."})
    # Updated: Check the collection via the ChromaDB client
    if agent.chromadb_client.collection is None:
        logging.error("Collection is not initialized.")
        return JSONResponse(content={"response": "Error: Collection is not initialized."})
    
    # Load new documents (duplicates are handled inside load_documents)
    agent.chromadb_client.load_documents("knowledge_base")
    
    user_message = message.strip()
    if not user_message:
        return JSONResponse(content={"response": "Please provide a message.", "conversation_history": agent.conversation_history})
    
    logging.info(f"User Query: {user_message}")
    response_text = await agent.generate_answer_async(user_message)
    agent.conversation_history.append({"role": "user", "content": user_message})
    agent.conversation_history.append({"role": "agent", "content": response_text})
    agent.save_session()  # Save session after processing
    return JSONResponse(content={"response": response_text, "conversation_history": agent.conversation_history})

@app.post("/analyze_code")
async def analyze_code():
    global agent
    if agent is None:
        return JSONResponse(content={"analysis": "Error: RAGAgent failed to initialize."})
    analysis = agent.analyze_code("rag_agent.py")
    return JSONResponse(content={"analysis": analysis})

@app.post("/save_session")
async def save_current_session():
    global agent
    if agent is None:
        return JSONResponse(content={"status": "error", "message": "RAGAgent failed to initialize."})
    try:
        agent.save_session()
        return JSONResponse(content={"status": "success", "message": "Session saved successfully."})
    except Exception as e:
        logging.error(f"Error saving session: {e}")
        return JSONResponse(content={"status": "error", "message": f"Error saving session: {e}"})

@app.post("/load_session")
async def load_saved_session():
    global agent
    # Delete the existing agent and reinitialize it
    if agent is not None:
        del agent
    try:
        agent = RAGAgent()
    except Exception as e:
        logging.error(f"Error re-initializing agent: {e}")
        return JSONResponse(content={"status": "error", "message": "Error loading session. RAGAgent failed to initialize."})
    if agent is None:
        return JSONResponse(content={"status": "error", "message": "Error loading session. RAGAgent failed to initialize."})
    
    # Ensure conversation history is a list of dictionaries with 'role' and 'content'
    formatted_history = []
    for msg in agent.conversation_history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            formatted_history.append({"role": msg["role"], "content": msg["content"]})
        else:
            logging.warning(f"Skipping invalid message format: {msg}")
    
    return JSONResponse(content={"status": "success", "message": "Session loaded successfully.", "conversation_history": formatted_history})

@app.post("/run_tool")
async def run_tool_endpoint(tool_name: str = Form(...), tool_input: str = Form(...)):
    global agent
    if not tool_name:
        return JSONResponse(content={"error": "Tool name is required."}), 400
    result = agent.run_tool(tool_name, tool_input)
    return JSONResponse(content={"result": result})
