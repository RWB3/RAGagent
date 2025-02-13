# model_client.py
import os
from dotenv import load_dotenv
import requests
import httpx
import logging

# Load environment variables from .env file
load_dotenv()
logger = logging.getLogger(__name__)

class ModelClient:
    def __init__(self, backend=None, ollama_host=None, model_name=None):
        self.backend = backend or os.getenv("MODEL_BACKEND", "ollama")
        self.ollama_host = (ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model_name = model_name or os.getenv("USE_MODEL", "llama3.2")
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT", 120))

    def _build_prompt(self, query, context_documents=None):
        if context_documents:
            context_str = "\n".join(context_documents)
            prompt = (
                "You are a helpful and informative AI agent.\n\n Process the following based primarily on your own knowledge: {query}\n\n"
                f"Here is some additional information. Use it only if relevant. "
                f"If it is not relevant, ignore it:\n{context_str}\n\nAnswer:"
            )
        else:
            prompt = (
                "You are a helpful and informative AI agent. \n\nProcess the following based on your knowledge.\n\n"
                f"Question: {query}\n\nAnswer:"
            )
        # Log the complete prompt sent to the model
        logger.info(f"Generated prompt: {prompt}")
        return prompt

    def generate_completion_sync(self, query, context_documents=None):
        if self.backend != "ollama":
            raise NotImplementedError("Only the Ollama backend is implemented.")
        prompt = self._build_prompt(query, context_documents)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(f"{self.ollama_host}/api/generate", json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Model response data: {data}")
            return data.get("response", "").strip()
        except Exception as e:
            error_msg = f"Error generating completion: {e}"
            logger.error(error_msg)
            return error_msg

    async def generate_completion_async(self, query, context_documents=None):
        if self.backend != "ollama":
            raise NotImplementedError("Only the Ollama backend is implemented.")
        prompt = self._build_prompt(query, context_documents)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(f"{self.ollama_host}/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
                logger.info(f"Async model response data: {data}")
                return data.get("response", "").strip()
            except Exception as e:
                error_msg = f"Error generating async completion: {e}"
                logger.error(error_msg)
                return error_msg

    def generate_custom_prompt_sync(self, custom_prompt):
        """
        Generates a completion using a custom prompt (bypassing the default prompt builder).
        Useful for tasks like code analysis.
        """
        logger.info(f"Custom prompt sent to model: {custom_prompt}")
        payload = {
            "model": self.model_name,
            "prompt": custom_prompt,
            "stream": False
        }
        try:
            response = requests.post(f"{self.ollama_host}/api/generate", json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Custom prompt response data: {data}")
            return data.get("response", "").strip()
        except Exception as e:
            error_msg = f"Error generating custom prompt: {e}"
            logger.error(error_msg)
            return error_msg
