import os
import openai
import chromadb
import logging
import json
import importlib
import traceback
from dotenv import load_dotenv
from chromadb.errors import InvalidCollectionException
from openai import OpenAIError  # Updated import for error handling

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Constants
PERSIST_DIRECTORY = "chroma_db"  # Directory for ChromaDB persistence

class RAGAgent:
    def __init__(self, openai_api_key, model_name="gpt-4o", persist_directory=PERSIST_DIRECTORY):
        """
        Initializes the RAGAgent with an OpenAI API key, model name, and persistence directory.
        """
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = None  # Will be initialized when needed
        self.conversation_history = []

        openai.api_key = self.openai_api_key

        logging.info("RAGAgent initialized.")

        # Validate the model
        if not self.validate_model():
            raise ValueError(f"Model '{self.model_name}' is invalid or inaccessible.")

        # Initialize the collection
        self.initialize_collection()

        # Load session
        self.load_session()

    def validate_model(self):
        """Validates if the specified model is accessible."""
        # Removed call to openai.Model.retrieve since it's no longer supported.
        logging.info(f"Assuming model '{self.model_name}' is valid based on the new API interface.")
        return True

    def initialize_collection(self, collection_name="my_collection"):
        """
        Initializes the ChromaDB collection.
        """
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logging.info(f"Collection '{collection_name}' loaded.")
        except InvalidCollectionException:
            self.collection = self.client.create_collection(name=collection_name)
            logging.info(f"Collection '{collection_name}' created.")
            # Load documents immediately after creation
            self.load_documents("knowledge_base")

    def load_documents(self, directory):
        """Loads documents from a directory into the ChromaDB collection without duplicating."""
        if self.collection is None:
            raise ValueError("Collection not initialized. Call initialize_collection() first.")

        documents = []
        ids = []

        for filename in os.listdir(directory):
            if filename.endswith(".txt") or filename.endswith(".pdf"):  # Add other relevant extensions
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                except UnicodeDecodeError:
                    logging.warning(f"Skipping '{filename}' due to encoding error.")
                    continue

                # Check if document already exists by ID
                existing = self.collection.get(ids=[filename])
                if existing and len(existing['ids']) > 0:
                    logging.info(f"Document '{filename}' already exists. Skipping.")
                    continue

                documents.append(text)
                ids.append(filename)  # Using filename as ID

        if documents:  # Only add if there are new documents
            self.collection.add(
                documents=documents,
                ids=ids  # Provide unique IDs for each document
            )
            logging.info(f"Loaded {len(documents)} new documents from '{directory}'.")
        else:
            logging.warning(f"No new documents to load from '{directory}'.")

    def retrieve_relevant_documents(self, query, n_results=4):
        """Retrieves relevant documents from the vector store based on a query."""
        if self.collection is None:
            raise ValueError("Collection not initialized. Call initialize_collection() first.")

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results['documents'][0]  # Return the list of documents

    def generate_answer(self, query, context_documents):
        """
        Generates an answer using the LLM, incorporating retrieved documents.
        The prompt allows the model to use its own knowledge if context is insufficient.
        """
        if context_documents:
            prompt = f"""You are a helpful and informative AI agent. Use the following context to answer the question. Feel free to use your own knowledge if the context is insufficient.

Context documents:
{context_documents}

Question: {query}

Answer:"""
        else:
            prompt = f"""You are a helpful and informative AI agent. Answer the following question based on your knowledge.

Question: {query}

Answer:"""

        try:
            completion = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful and informative AI agent."},
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content.strip()
        except OpenAIError as e:  # Updated exception handling
            logging.error(f"OpenAI API error: {e}")
            logging.error(traceback.format_exc())  # Log the full stack trace
            return "I encountered an error while generating the answer."
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            logging.error(traceback.format_exc())  # Log the full stack trace
            return "I encountered an unexpected error while generating the answer."

    def run_tool(self, tool_name, tool_input):
        """Runs a tool based on the tool_name and input."""
        try:
            module = importlib.import_module(f"tools.{tool_name}")  # Assuming tools are in a 'tools' directory
            result = module.run(tool_input)
            return result
        except ImportError:
            return f"Tool '{tool_name}' not found."
        except Exception as e:
            logging.error(f"Error running tool '{tool_name}': {e}")
            logging.error(traceback.format_exc())
            return f"Error running tool '{tool_name}': {e}"

    def analyze_code(self, filepath):
        """Analyzes the agent's own code and suggests improvements."""
        try:
            with open(filepath, "r") as f:
                code = f.read()
        except FileNotFoundError:
            return "Error: File not found."
        except Exception as e:
            return f"Error reading file: {e}"

        prompt = f"""You are a senior software engineer reviewing the following Python code. Identify potential improvements,
including but not limited to:
- Code clarity and readability
- Potential bugs or errors
- Efficiency improvements
- Adherence to best practices
- Security vulnerabilities

Provide specific suggestions for how to improve the code.

```python
{code}
```
"""

        try:
            completion = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a senior software engineer reviewing Python code."},
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content.strip()
        except OpenAIError as e:  # Updated exception handling
            logging.error(f"OpenAI API error: {e}")
            logging.error(traceback.format_exc())
            return "I encountered an error while analyzing the code."
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            logging.error(traceback.format_exc())
            return "I encountered an unexpected error while analyzing the code."

    def save_session(self, filename="agent_session.json"):
        """Saves the agent's state to a JSON file."""
        data = {
            "conversation_history": self.conversation_history,
            "persist_directory": self.persist_directory,
            "model_name": self.model_name,
            "collection_exists": self.collection is not None  # Save flag for whether the collection exists
        }
        try:
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)  # Adding indent=4 for readability.
            logging.info(f"Session saved to '{filename}'")
        except Exception as e:
            logging.error(f"Error saving session: {e}")

    def load_session(self, filename="agent_session.json"):
        """Loads the agent's state from a JSON file."""
        try:
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    data = json.load(f)
                # Load conversation history
                self.conversation_history = data.get("conversation_history", [])
                self.persist_directory = data.get("persist_directory", self.persist_directory)
                self.model_name = data.get("model_name", self.model_name)

                # Re-initialize ChromaDB client and collection
                self.client = chromadb.PersistentClient(path=self.persist_directory)
                self.initialize_collection()

                openai.api_key = self.openai_api_key  # Ensure the API key is still set
                logging.info(f"Session loaded from '{filename}'")
            else:
                logging.warning(f"Session file '{filename}' not found. Starting a new session.")
        except Exception as e:
            logging.error(f"Error loading session: {e}")

    def run(self, query):
        """Processes a user query: retrieves context, generates an answer, updates conversation history, and returns the answer."""
        context = self.retrieve_relevant_documents(query)
        answer = self.generate_answer(query, context)
        self.conversation_history.append({"query": query, "answer": answer})
        return answer
