import os
import openai
import chromadb
import logging
import json
import importlib
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Constants
PERSIST_DIRECTORY = "chroma_db"  # Directory for ChromaDB persistence

class RAGAgent:
    def __init__(self, openai_api_key, model_name="gpt-4-1106-preview", persist_directory=PERSIST_DIRECTORY):
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

    def initialize_collection(self, collection_name="my_collection"):
        """
        Initializes the ChromaDB collection.
        """
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logging.info(f"Collection '{collection_name}' loaded.")
        except ValueError:
            self.collection = self.client.create_collection(name=collection_name)
            logging.info(f"Collection '{collection_name}' created.")

    def load_documents(self, directory):
        """Loads documents from a directory into the ChromaDB collection."""
        if self.collection is None:
            raise ValueError("Collection not initialized. Call initialize_collection() first.")

        import os
        from tqdm import tqdm  # Import tqdm here

        documents = []
        ids = []

        for filename in tqdm(os.listdir(directory), desc="Loading Documents"):
            if filename.endswith(".txt") or filename.endswith(".pdf"):  # Add other relevant extensions
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                except UnicodeDecodeError:
                    logging.warning(f"Skipping {filename} due to encoding error.")
                    continue

                documents.append(text)
                ids.append(filename)  # Using filename as ID

        if documents:  # Only upsert if there are documents
            self.collection.add(
                documents=documents,
                ids=ids  # Provide unique IDs for each document
            )
            logging.info(f"Loaded {len(documents)} documents from {directory}.")
        else:
            logging.warning(f"No documents found in {directory}.")


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
        """Generates an answer using the LLM, incorporating retrieved documents."""
        prompt = f"""You are a helpful and informative AI agent. Use the following context to answer the question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context documents:
        {context_documents}

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
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            return "I encountered an error while generating the answer."

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
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error analyzing code: {e}")
            return "I encountered an error while analyzing the code."

    def save_session(self, filename="agent_session.json"):
        """Saves the agent's state to a JSON file."""
        data = {
            "conversation_history": self.conversation_history,
            "persist_directory": self.persist_directory,
            "model_name": self.model_name,
            "collection_exists": self.collection is not None #Save flag for whether the collection exists
        }
        try:
            with open(filename, "w") as f:
                json.dump(data, f)
            logging.info(f"Session saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving session: {e}")

    def load_session(self, filename="agent_session.json"):
        """Loads the agent's state from a JSON file."""
        try:
            if os.path.exists(filename):  # Check if the session file exists
                with open(filename, "r") as f:
                    data = json.load(f)
                self.conversation_history = data["conversation_history"]
                self.persist_directory = data["persist_directory"]
                self.model_name = data["model_name"]
                #collection_exists = data.get("collection_exists", False)

                #Re-initialize chromadb, openai, and model in case they are different from initial values
                self.client = chromadb.PersistentClient(path=self.persist_directory)

                openai.api_key = self.openai_api_key  # Make sure the API key is still set
                logging.info(f"Session loaded from {filename}")

            else:
                logging.warning("Session file not found. Starting a new session.")


        except FileNotFoundError:
            logging.warning("Session file not found. Starting a new session.")
        except Exception as e:
            logging.error(f"Error loading session: {e}")

    def run(self, query):
        """Main function to run the agent."""
        logging.info(f"User Query: {query}")

        # 1. Determine if a tool is needed
        tool_name = None  # Placeholder, replace with actual logic to determine tool usage
        tool_input = None  # Placeholder, replace with actual logic to extract tool input

        if "calculate" in query.lower():  # Example: Simple tool trigger based on keyword
            tool_name = "calculator"  # Assuming you have a 'calculator' tool
            tool_input = query.lower().replace("calculate", "").strip()  # Extract input

        if tool_name:
            logging.info(f"Using tool: {tool_name} with input: {tool_input}")
            tool_result = self.run_tool(tool_name, tool_input)
            response = f"Tool '{tool_name}' result: {tool_result}"
        else:
            # 2. Retrieve relevant documents
            context_documents = self.retrieve_relevant_documents(query)

            # 3. Generate an answer
            response = self.generate_answer(query, context_documents)

        # 4. Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

# Example Usage (for testing in the same script - remove for separate execution)
if __name__ == "__main__":
    # Get the OpenAI API key from the environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Ensure the API key is set
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in the environment variables. Please set it.")

    # Initialize the agent
    agent = RAGAgent(openai_api_key)

    # Load a previous session or create a new one
    agent.load_session()  # Loads from "agent_session.json" by default

    # Initialize the ChromaDB collection
    agent.initialize_collection()

    # Load documents from a directory
    agent.load_documents("knowledge_base")  # Replace "knowledge_base" with your directory name

    # Example query
    query = "What are the benefits of using Chromadb?"
    response = agent.run(query)
    print(f"Agent's Response: {response}")

    # Example of analyzing the agent's code
    analysis = agent.analyze_code("rag_agent.py")  # Replace with the actual filename
    print(f"Code Analysis:\n{analysis}")

    # Save the session
    agent.save_session()