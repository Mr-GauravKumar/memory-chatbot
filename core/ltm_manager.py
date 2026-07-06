"""
core/ltm_manager.py
High-level orchestrator for Long-Term Memory.
Ties together: classifier -> embedding -> vector store -> metadata store.
Also handles retrieval: embed query -> search all 3 collections -> merge.
"""

from typing import List, Optional

from core.gemini_client import GeminiClient
from core.classifier import MemoryClassifier
from core.vector_store import VectorStore
from core.metadata_store import MetadataStore
from core.memory_models import RetrievedMemory
from config import LTM_TOP_K, VALID_MEMORY_TYPES, IMPORTANCE_DEFAULT
from utils.helpers import generate_id, compute_importance_score
from utils.logger import get_logger

logger = get_logger(__name__)


class LTMManager:
    def __init__(self):
        self.gemini = GeminiClient()
        self.classifier = MemoryClassifier(self.gemini)
        self.vector_store = VectorStore()
        self.metadata_store = MetadataStore()

    # ------------------------------------------------------------------
    # STORAGE PATH
    # ------------------------------------------------------------------
    def process_and_store(self, user_message: str) -> Optional[str]:
        """
        Classify a user message and store it if relevant.
        Returns the memory type stored, or None if nothing was stored.
        """
        result = self.classifier.classify(user_message)

        if not result.store:
            return None

        try:
            embedding = self.gemini.embed_text(result.memory)
        except Exception as e:
            logger.error(f"Skipping storage due to embedding failure: {e}")
            return None

        memory_id = generate_id()
        importance = compute_importance_score(result.type, result.memory)

        self.vector_store.add_memory(result.type, memory_id, result.memory, embedding)
        self.metadata_store.insert_memory(memory_id, result.type, result.memory, importance)

        logger.info(f"Stored new {result.type} memory: {result.memory}")
        return result.type

    # ------------------------------------------------------------------
    # RETRIEVAL PATH
    # ------------------------------------------------------------------
    def retrieve_relevant_memories(self, query_text: str, top_k: int = LTM_TOP_K) -> List[RetrievedMemory]:
        """
        Embed the query, search all three collections, merge and return
        the most relevant memories across all categories.
        Never fabricates — only returns what's actually found in the DB.
        """
        try:
            query_embedding = self.gemini.embed_text(query_text)
        except Exception as e:
            logger.error(f"Query embedding failed, skipping retrieval: {e}")
            return []

        all_results: List[RetrievedMemory] = []

        for mem_type in VALID_MEMORY_TYPES:
            raw = self.vector_store.query(mem_type, query_embedding, top_k)

            ids = raw.get("ids", [[]])[0]
            documents = raw.get("documents", [[]])[0]
            distances = raw.get("distances", [[]])[0]

            for i in range(len(ids)):
                all_results.append(
                    RetrievedMemory(
                        id=ids[i],
                        type=mem_type,
                        content=documents[i],
                        distance=distances[i],
                        importance=IMPORTANCE_DEFAULT,
                    )
                )
                # Mark as accessed (bumps usage_count, updates last_accessed)
                self.metadata_store.update_access(ids[i])

        # Sort by distance ascending (most similar first)
        all_results.sort(key=lambda m: m.distance)
        return all_results

    # ------------------------------------------------------------------
    # STATS + CLEAR
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        return self.metadata_store.get_stats()

    def clear_all_memory(self):
        self.vector_store.clear_all()
        self.metadata_store.clear_all()
        logger.info("All long-term memory cleared.")