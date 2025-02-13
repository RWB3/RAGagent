# chromadb_client.py
import os
import logging
import chromadb
from chromadb.errors import InvalidCollectionException

logger = logging.getLogger(__name__)

class ChromaDBClient:
    def __init__(self, persist_directory="chroma_db", collection_name="my_collection", knowledge_base_directory="knowledge_base"):
        """
        Initializes the ChromaDB persistent client, loads (or creates) the collection,
        and loads documents from the given knowledge base directory.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.knowledge_base_directory = knowledge_base_directory
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = None
        self.initialize_collection()

    def initialize_collection(self):
        """Attempts to load an existing collection; if not found, creates a new one and loads documents."""
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' loaded.")
        except InvalidCollectionException:
            self.collection = self.client.create_collection(name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' created.")
            self.load_documents(self.knowledge_base_directory)

    def load_documents(self, directory):
        """
        Loads text or PDF documents from the specified directory into the collection,
        skipping documents that are already loaded.
        """
        documents = []
        ids = []
        if not os.path.exists(directory):
            logger.warning(f"Knowledge base directory '{directory}' not found.")
            return
        for filename in os.listdir(directory):
            if filename.endswith(".txt") or filename.endswith(".pdf"):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                except UnicodeDecodeError:
                    logger.warning(f"Skipping '{filename}' due to encoding error.")
                    continue

                # Check if a document with this filename already exists.
                try:
                    existing = self.collection.get(ids=[filename])
                    if existing and len(existing.get('ids', [])) > 0:
                        logger.info(f"Document '{filename}' already exists. Skipping.")
                        continue
                except Exception as e:
                    logger.warning(f"Error checking document '{filename}': {e}")

                documents.append(text)
                ids.append(filename)
        if documents:
            self.collection.add(documents=documents, ids=ids)
            logger.info(f"Loaded {len(documents)} new documents from '{directory}'.")
        else:
            logger.info(f"No new documents to load from '{directory}'.")

    def retrieve_relevant_documents(self, query, n_results=4):
        """
        Retrieves a list of relevant document texts for the given query.
        Logs the raw results and returns the first set of documents.
        """
        if self.collection is None:
            logger.error("Collection not initialized.")
            return []
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)
            logger.info(f"ChromaDB query for '{query}' returned: {results}")
            return results.get('documents', [[]])[0]
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
