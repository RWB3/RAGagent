# RoBo-RAG-Agent

## Overview

RoBo-RAG-Agent is a web-based Retrieval-Augmented Generation (RAG) AI agent built using FastAPI. It integrates ChromaDB for document retrieval and supports asynchronous model interactions. The agent can chat, analyze code for improvements, manage conversation sessions, and run tools.

## Features

-   **Interactive Chat Interface**: Real-time conversations via a user-friendly web interface.
-   **Knowledge Retrieval**: Uses ChromaDB to fetch relevant documents from a `knowledge_base`.
-   **Code Analysis**: Analyzes Python code to suggest improvements in clarity, efficiency, and security.
-   **Session Management**: Automatically saves and loads conversation sessions in `agent_session.json`.
-   **Asynchronous Processing**: Supports asynchronous model completions using FastAPI.
-   **Tool Execution**: Dynamically imports and runs tools from the `tools` directory.
-   **Robust Logging**: Detailed logging for debugging and monitoring.

## Technologies Used

-   **Backend**: Python, FastAPI, uvicorn, dotenv, logging
-   **Document Retrieval**: ChromaDB
-   **Model Interaction**: Requests and HTTPX for API calls (Ollama backend)
-   **Frontend**: HTML, CSS, JavaScript, Jinja2

## Prerequisites

-   **Python**: 3.8 or higher
-   **ChromaDB**: Ensure it is set up and accessible.
-   **Knowledge Base**: Create a `knowledge_base` directory with `.txt` and `.pdf` documents.
-   **Ollama**: Ensure it is installed and running.
-   **Environment Variables**: Configure using a `.env` file as detailed below.

## Installation

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/rag-ai-agent.git
    cd rag-ai-agent
    ```

2.  **Create a Virtual Environment**

    ```bash
    python3 -m venv venv
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**

    Create a `.env` file in the project root with at least the following:

    ```env
    CHROMADB_PERSIST_DIRECTORY=chroma_db
    CHROMADB_COLLECTION=my_collection
    KNOWLEDGE_BASE_DIRECTORY=knowledge_base
    MODEL_BACKEND=ollama
    OLLAMA_HOST=http://localhost:11434
    OLLAMA_TIMEOUT=120
    USE_MODEL=llama3.2
    ```

## Usage

1.  **Start the FastAPI Application**

    ```bash
    uvicorn app:app --reload
    ```

    The app will run at [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

2.  **Interact with the AI Agent**

    -   Use the web interface to chat with the agent.
    -   Analyze code by sending a request to the `/analyze_code` endpoint.
    -   Manage sessions using `/save_session` and `/load_session` endpoints.

## API Endpoints

-   **`GET /`**: Renders the main chat interface.
-   **`POST /get_response`**: Receives user messages and returns AI responses.
-   **`POST /analyze_code`**: Analyzes the agent’s code for improvements.
-   **`POST /save_session`**: Saves the current conversation session.
-   **`POST /load_session`**: Loads a previously saved conversation session.
-   **`POST /run_tool`**: Runs a specified tool with the given input.

## File Structure

-   **`app.py`**: Main FastAPI application managing routes and integrating the RAGAgent.
-   **`rag_agent.py`**: Contains the RAGAgent class for AI interactions, tool execution, and session management.
-   **`model_client.py`**: Handles synchronous and asynchronous model API calls.
-   **`chromadb_client.py`**: Manages ChromaDB collection creation, document loading, and retrieval.
-   **`templates/`**: Contains HTML templates (e.g., `index.html`) for the frontend.
-   **`static/`**: Contains static files (e.g., CSS, JavaScript) for the frontend.
-   **`knowledge_base/`**: Directory housing text and PDF documents used for context.
-   **`chroma_db/`**: Directory where ChromaDB persists its data.
-   **`tools/`**: Directory containing tool modules that can be dynamically loaded and executed.
-   **`.env`**: Environment variable configuration file.
-   **`requirements.txt`**: Python dependency list.

## Logging & Error Handling

-   Logs are configured to capture key events, errors, and detailed messages.
-   The application handles errors related to missing API keys, document load failures, session management issues, and tool execution failures.

## Contributing

Contributions are welcome! Follow these steps:

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/YourFeature`).
3.  Commit your changes (`git commit -m "Add new feature"`).
4.  Push to the branch (`git push origin feature/YourFeature`).
5.  Open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).