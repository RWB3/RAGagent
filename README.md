# RoBo-RAG-Agent

## Overview

This **RAG Capable AI Agent** is a web-based application that leverages Retrieval-Augmented Generation (RAG) techniques to provide informed and contextually relevant responses to user queries. Powered by OpenAI's advanced language models and ChromaDB for document retrieval, this AI agent can interact with users, analyze code snippets, and manage conversation sessions seamlessly. (Okay, seamlessly is a little bit aspirational at this time!)

## Features

- **Interactive Chat Interface**: Engage in real-time conversations with the AI agent through a user-friendly web interface.
- **Knowledge Retrieval**: Retrieves relevant documents from a predefined knowledge base to enhance response accuracy.
- **Code Analysis**: Analyze Python code for improvements, potential bugs, efficiency enhancements, and adherence to best practices.
- **Session Management**: Save and load conversation sessions to maintain context across interactions.
- **Scalable Architecture**: Built with Flask, allowing easy scaling and integration with more complex applications.
- **Cross-Origin Resource Sharing (CORS)**: Enabled to facilitate interactions from different origins.

## Technologies Used

- **Backend**: Python, Flask, OpenAI API, ChromaDB (Ollama models comming soon!)
- **Frontend**: HTML, CSS, JavaScript
- **Database**: ChromaDB for document storage and retrieval
- **Utilities**: dotenv for environment variable management, logging for monitoring

## Prerequisites

- **Python**: Version 3.8 or higher
- **OpenAI API Key**: Sign up at [OpenAI](https://platform.openai.com/) to obtain an API key.
- **ChromaDB**: Ensure ChromaDB is set up and accessible.
- **Knowledge Base**: Prepare a `knowledge_base` directory with `.txt`, and (comming soon!`.pdf`) documents to serve as the knowledge repository.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/rag-ai-agent.git
   cd rag-ai-agent
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**

   Create a `.env` file in the project root with the following content:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_MODEL=o1-mini 
   ```
   gpt-4o and o1-mini have been tested so far. 

5. **Prepare the Knowledge Base**

   Ensure there is a `knowledge_base` directory in the project root containing (currently only .txt) documents that the AI agent can reference.

## Usage

1. **Start the Flask Application**

   ```bash
   python app.py
   ```

   The application will start on `http://127.0.0.1:5000/` by default.

2. **Interact with the AI Agent**

   - Open your web browser and navigate to `http://127.0.0.1:5000/`.
   - Use the chat interface to send messages to the AI agent.
   - Utilize action buttons to analyze code, save sessions, or load previous conversations.

## API Endpoints

- **`/`**: Renders the main chat interface.
- **`/get_response`**: Handles user messages and returns AI responses.
- **`/analyze_code`**: Analyzes the agent's codebase and provides feedback.
- **`/save_session`**: Saves the current conversation session.
- **`/load_session`**: Loads a previously saved conversation session.

## File Structure

- **`app.py`**: Main Flask application that manages routes and integrates the RAGAgent.
- **`rag_agent.py`**: Defines the `RAGAgent` class responsible for handling AI interactions and document retrieval.
- **`templates/index.html`**: Frontend template for the chat interface.
- **`knowledge_base/`**: Directory containing documents for the AI agent's knowledge base.
- **`chroma_db/`**: Directory where ChromaDB stores its data.
- **`.env`**: Environment variables configuration file.
- **`requirements.txt`**: Python dependencies.

## Logging

The application is configured to log important events and errors. Logs include timestamps, log levels, and messages, facilitating easier debugging and monitoring.

- **Log Level**: INFO by default. Can be adjusted in `app.py` and `rag_agent.py`.
- **Format**: `%(asctime)s - %(levelname)s - %(message)s`

## Error Handling

The application includes comprehensive error handling to manage issues such as:

- Missing or invalid API keys.
- Failures in initializing the RAGAgent.
- Issues with document loading and retrieval.
- Problems during code analysis.
- Session management errors.

Errors are logged with detailed messages to assist in troubleshooting.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add awesome feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE).

---

*Happy Chatting with RoBoRAG Agent!*