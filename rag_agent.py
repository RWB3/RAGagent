import os
import openai
import chromadb
import logging
import json
import importlib
import traceback
from dotenv import load_dotenv
from chromadb.errors import InvalidCollectionException
from openai import OpenAIError  # Import error handling from OpenAI

# Configure logging with a readable format.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load variables from .env file (like API keys and model names)
load_dotenv()

# Constant that defines where ChromaDB will persist data.
PERSIST_DIRECTORY = "chroma_db"

class RAGAgent:
    def __init__(self, openai_api_key, model_name="gpt-4o", persist_directory=PERSIST_DIRECTORY):
        """
        Initialize the RAGAgent with the necessary keys, model name, and persistence directory.
        """
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.persist_directory = persist_directory
        # Create a persistent client for ChromaDB.
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = None  # Collection will be set later when needed.
        self.conversation_history = []  # To store query/response history.

        # Set the API key for OpenAI.
        openai.api_key = self.openai_api_key

        logging.info("RAGAgent initialized.")

        # Check if the model is valid before proceeding.
        if not self.validate_model():
            raise ValueError(f"Model '{self.model_name}' is invalid or inaccessible.")

        # Initialize the document collection.
        self.initialize_collection()
        # Try to load a previous session.
        self.load_session()

    def validate_model(self):
        """
        Since the model validation API call is no longer supported,
        we assume the provided model is valid.
        """
        logging.info(f"Assuming model '{self.model_name}' is valid based on the new API interface.")
        return True

    def initialize_collection(self, collection_name="my_collection"):
        """
        Try to load an existing ChromaDB collection.
        If not found, create a new one and load documents immediately.
        """
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logging.info(f"Collection '{collection_name}' loaded.")
        except InvalidCollectionException:
            self.collection = self.client.create_collection(name=collection_name)
            logging.info(f"Collection '{collection_name}' created.")
            # Load documents from the knowledge base folder after collection creation.
            self.load_documents("knowledge_base")

    def load_documents(self, directory):
        """
        Loads documents from the given directory into the collection.
        It prevents duplicate loading by checking for an existing document.
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call initialize_collection() first.")

        documents = []
        ids = []

        for filename in os.listdir(directory):
            # Only process text and PDF files (additional formats can be added as needed).
            if filename.endswith(".txt") or filename.endswith(".pdf"):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                except UnicodeDecodeError:
                    logging.warning(f"Skipping '{filename}' due to encoding error.")
                    continue

                # Check if a document with the same filename already exists.
                existing = self.collection.get(ids=[filename])
                if existing and len(existing['ids']) > 0:
                    logging.info(f"Document '{filename}' already exists. Skipping.")
                    continue

                documents.append(text)
                ids.append(filename)  # Use the filename as the unique document ID.

        if documents:  # Only add new documents.
            self.collection.add(
                documents=documents,
                ids=ids
            )
            logging.info(f"Loaded {len(documents)} new documents from '{directory}'.")
        else:
            logging.warning(f"No new documents to load from '{directory}'.")

    def retrieve_relevant_documents(self, query, n_results=4):
        """
        Queries the ChromaDB collection to find the most relevant documents.
        Returns a list of document texts.
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call initialize_collection() first.")

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results['documents'][0]  # Return the first set of matched documents.

    def supports_system_role(self):
        """
        Returns True if the current model supports the 'system' role in messages.
        You can update this list as more models are added.
        """
        supported_models = {"gpt-4o", "gpt-4o-realtime-preview"}
        return self.model_name in supported_models

    def generate_answer(self, query, context_documents):
        """
        Generates an answer by building a prompt from context documents and the query,
        then sending it to the appropriate OpenAI model.
        """
        if context_documents:
            prompt = (
                "You are a helpful and informative AI agent. Use the following context to answer the question. "
                "Feel free to use your own knowledge if the context is insufficient.\n\n"
                f"Context documents:\n{context_documents}\n\n"
                f"Question: {query}\n\nAnswer:"
            )
        else:
            prompt = (
                "You are a helpful and informative AI agent. Answer the following question based on your knowledge.\n\n"
                f"Question: {query}\n\nAnswer:"
            )

        # Build the messages list according to whether the model supports system messages.
        if self.supports_system_role():
            messages = [
                {"role": "system", "content": "You are a helpful and informative AI agent."},
                {"role": "user", "content": prompt}
            ]
        else:
            # For models that do not support a separate system role, prepend instructions into the user's message.
            messages = [
                {"role": "user", "content": "You are a helpful and informative AI agent. " + prompt}
            ]

        try:
            completion = openai.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            # Return the cleaned-up answer text.
            return completion.choices[0].message.content.strip()
        except OpenAIError as e:
            logging.error(f"OpenAI API error: {e}")
            logging.error(traceback.format_exc())
            return "I encountered an error while generating the answer."
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            logging.error(traceback.format_exc())
            return "I encountered an unexpected error while generating the answer."

    def run_tool(self, tool_name, tool_input):
        """
        From the tools directory, imports and runs a tool based on tool_name.
        Designed to extend functionality easily.
        """
        try:
            module = importlib.import_module(f"tools.{tool_name}")
            result = module.run(tool_input)
            return result
        except ImportError:
            return f"Tool '{tool_name}' not found."
        except Exception as e:
            logging.error(f"Error running tool '{tool_name}': {e}")
            logging.error(traceback.format_exc())
            return f"Error running tool '{tool_name}': {e}"

    def analyze_code(self, filepath):
        """
        Reads the content of the file at filepath and sends it to the OpenAI model
        for code review suggestions. Returns the analysis as text.
        """
        try:
            with open(filepath, "r") as f:
                code = f.read()
        except FileNotFoundError:
            return "Error: File not found."
        except Exception as e:
            return f"Error reading file: {e}"

        prompt = (
            "You are a senior software engineer reviewing the following Python code. "
            "Identify potential improvements, including but not limited to:\n"
            "- Code clarity and readability\n"
            "- Potential bugs or errors\n"
            "- Efficiency improvements\n"
            "- Adherence to best practices\n"
            "- Security vulnerabilities\n\n"
            "Provide specific suggestions for how to improve the code.\n\n"
            "```python\n" + code + "\n```"
        )

        try:
            completion = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content.strip()
        except OpenAIError as e:
            logging.error(f"OpenAI API error: {e}")
            logging.error(traceback.format_exc())
            return "I encountered an error while analyzing the code."
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            logging.error(traceback.format_exc())
            return "I encountered an unexpected error while analyzing the code."

    def save_session(self, filename="agent_session.json"):
        """
        Saves the agent's current conversation history and configuration to a JSON file.
        This allows resuming a session later.
        """
        data = {
            "conversation_history": self.conversation_history,
            "persist_directory": self.persist_directory,
            "model_name": self.model_name,
            "collection_exists": self.collection is not None  # Indicates if the collection was initialized.
        }
        try:
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)
            logging.info(f"Session saved to '{filename}'")
        except Exception as e:
            logging.error(f"Error saving session: {e}")

    def load_session(self, filename="agent_session.json"):
        """
        Loads the agent's state from a JSON file.
        If the file is not present, a new session is started.
        """
        try:
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    data = json.load(f)
                # Restore conversation history and configurations.
                self.conversation_history = data.get("conversation_history", [])
                self.persist_directory = data.get("persist_directory", self.persist_directory)
                self.model_name = data.get("model_name", self.model_name)

                # Reinitialize the ChromaDB client and its collection.
                self.client = chromadb.PersistentClient(path=self.persist_directory)
                self.initialize_collection()

                # Ensure the OpenAI API key is still set.
                openai.api_key = self.openai_api_key
                logging.info(f"Session loaded from '{filename}'")
            else:
                logging.warning(f"Session file '{filename}' not found. Starting a new session.")
        except Exception as e:
            logging.error(f"Error loading session: {e}")

    def run(self, query):
        """
        Main method to process a user's query:
         - Retrieves relevant documents.
         - Generates an answer using the model.
         - Updates and returns the conversation history.
        """
        context = self.retrieve_relevant_documents(query)
        answer = self.generate_answer(query, context)
        # Save the query and the model's answer.
        self.conversation_history.append({"query": query, "answer": answer})
        return answer

# End of RAGAgent class.
