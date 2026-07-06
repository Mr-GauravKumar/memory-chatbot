"""
core/vector_store.py
ChromaDB wrapper. Manages 3 independent persistent collections:
semantic_memory, factual_memory, episodic_memory.
"""

import chromadb
from typing import List, Dict, Any

from config import (
    CHROMA_DIR,
    COLLECTION_SEMANTIC,
    COLLECTION_FACTUAL,
    COLLECTION_EPISODIC,
)
from utils.logger import get_logger

logger = get_logger(__name__)

TYPE_TO_COLLECTION = {
    "Semantic": COLLECTION_SEMANTIC,
    "Factual": COLLECTION_FACTUAL,
    "Episodic": COLLECTION_EPISODIC,
}


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collections = {
            "Semantic": self.client.get_or_create_collection(COLLECTION_SEMANTIC),
            "Factual": self.client.get_or_create_collection(COLLECTION_FACTUAL),
            "Episodic": self.client.get_or_create_collection(COLLECTION_EPISODIC),
        }

    def add_memory(self, memory_type: str, memory_id: str, text: str, embedding: List[float]):
        """Add a single memory embedding to the correct collection."""
        if memory_type not in self.collections:
            raise ValueError(f"Invalid memory type: {memory_type}")

        self.collections[memory_type].add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{"type": memory_type}],
        )
        logger.info(f"Added {memory_type} memory to vector store: {memory_id}")

    def query(self, memory_type: str, query_embedding: List[float], top_k: int) -> Dict[str, Any]:
        """Query a single collection for top_k nearest memories."""
        collection = self.collections[memory_type]
        count = collection.count()
        if count == 0:
            return {"ids": [[]], "documents": [[]], "distances": [[]]}

        effective_k = min(top_k, count)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_k,
        )
        return results

    def delete_memory(self, memory_type: str, memory_id: str):
        """Delete a single memory by id from its collection."""
        self.collections[memory_type].delete(ids=[memory_id])

    def clear_all(self):
        """Wipe all three collections completely (used by 'Clear LTM' button)."""
        for mem_type, collection_name in TYPE_TO_COLLECTION.items():
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass  # collection might not exist yet
        # Recreate empty collections
        self.collections = {
            "Semantic": self.client.get_or_create_collection(COLLECTION_SEMANTIC),
            "Factual": self.client.get_or_create_collection(COLLECTION_FACTUAL),
            "Episodic": self.client.get_or_create_collection(COLLECTION_EPISODIC),
        }
        logger.info("Cleared all vector store collections.")

    def count(self, memory_type: str) -> int:
        return self.collections[memory_type].count()