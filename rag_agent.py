# rag_agent.py
import os
import json
import logging
import traceback
from dotenv import load_dotenv
from model_client import ModelClient
from chromadb_client import ChromaDBClient
import importlib

load_dotenv()
logger = logging.getLogger(__name__)

SESSION_FILENAME = "agent_session.json"
DEFAULT_PERSIST_DIRECTORY = os.getenv("CHROMADB_PERSIST_DIRECTORY", "chroma_db")
DEFAULT_COLLECTION_NAME = os.getenv("CHROMADB_COLLECTION", "my_collection")
DEFAULT_KB_DIRECTORY = os.getenv("KNOWLEDGE_BASE_DIRECTORY", "knowledge_base")

class RAGAgent:
    def __init__(self, model_name=None, backend=None, persist_directory=DEFAULT_PERSIST_DIRECTORY):
        self.model_name = model_name or os.getenv("USE_MODEL", "llama3.2")
        self.backend = backend or os.getenv("MODEL_BACKEND", "ollama")
        self.model_client = ModelClient(backend=self.backend, model_name=self.model_name)
        self.chromadb_client = ChromaDBClient(
            persist_directory=persist_directory,
            collection_name=DEFAULT_COLLECTION_NAME,
            knowledge_base_directory=DEFAULT_KB_DIRECTORY
        )
        self.conversation_history = []
        self.load_session()

    @property
    def collection(self):
        return self.chromadb_client.collection

    def retrieve_relevant_documents(self, query, n_results=4):
        docs = self.chromadb_client.retrieve_relevant_documents(query, n_results)
        logger.info(f"Retrieved documents for query '{query}': {docs}")
        return docs

    def generate_answer(self, query):
        context = self.retrieve_relevant_documents(query)
        raw_output = self.model_client.generate_completion_sync(query, context)
        logging.info(f"Raw LLM output: {raw_output}")

        # Look for the TOOL_CALL marker anywhere in the output.
        # If found, extract the tool command and run the tool.
        # This is required because thinking/resoning LLMs will output their
        # thought process before the final answer or tool command.
        marker = "TOOL_CALL:"
        marker_index = raw_output.find(marker)
        if marker_index != -1:
            # Extract everything after the marker.
            command_str = raw_output[marker_index + len(marker):].strip()
            try:
                command = json.loads(command_str)
                tool_name = command.get("tool")
                tool_input = command.get("input", "")
                final_answer = command.get("final_answer", "")
                tool_result = self.run_tool(tool_name, tool_input)
                answer = f"{final_answer}\nTool result: {tool_result}"
                logging.info(f"Tool command processed: {command}")
            except Exception as e:
                logging.error(f"Error processing tool command: {e}")
                answer = raw_output  # fallback to raw output if parsing fails
        else:
            answer = raw_output

        self.conversation_history.append({"query": query, "answer": answer})
        return answer


    async def generate_answer_async(self, query):
        context = self.retrieve_relevant_documents(query)
        answer = await self.model_client.generate_completion_async(query, context)
        self.conversation_history.append({"query": query, "answer": answer})
        return answer

    def analyze_code(self, filepath):
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
        return self.model_client.generate_custom_prompt_sync(prompt)

    def run_tool(self, tool_name, tool_input):
        """
        Dynamically imports and runs a tool from the 'tools' directory.
        The tool module should have a 'run' function that accepts tool_input.
        """
        try:
            # Ensure the tools folder is in the Python path if not already
            import sys
            tools_path = os.path.join(os.path.dirname(__file__), "tools")
            if tools_path not in sys.path:
                sys.path.append(tools_path)
            module = importlib.import_module(f"tools.{tool_name}")
            result = module.run(tool_input)
            logger.info(f"Tool '{tool_name}' executed with input '{tool_input}'. Result: {result}")
            return result
        except ImportError:
            error_msg = f"Tool '{tool_name}' not found."
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error running tool '{tool_name}': {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return error_msg

    def save_session(self, filename=SESSION_FILENAME):
        data = {"conversation_history": self.conversation_history}
        try:
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)
            logger.info(f"Session saved to '{filename}'")
        except Exception as e:
            logger.error(f"Error saving session: {e}")

    def load_session(self, filename=SESSION_FILENAME):
        try:
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    data = json.load(f)
                self.conversation_history = data.get("conversation_history", [])
                logger.info(f"Session loaded from '{filename}'")
            else:
                logger.warning(f"Session file '{filename}' not found. Starting a new session.")
        except Exception as e:
            logger.error(f"Error loading session: {e}")
