"""
core/memory_models.py
Data models used across the memory system.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClassificationResult:
    """Output of the classifier for a single user message."""
    store: bool
    type: Optional[str] = None      # "Semantic" | "Factual" | "Episodic"
    memory: Optional[str] = None    # Cleaned memory text to store


@dataclass
class MemoryRecord:
    """A single long-term memory record (metadata + content)."""
    id: str
    type: str                # Semantic | Factual | Episodic
    content: str
    created_at: str
    last_accessed: str
    importance: float
    usage_count: int = 0


@dataclass
class RetrievedMemory:
    """A memory returned from a retrieval query, with relevance info."""
    id: str
    type: str
    content: str
    distance: float          # similarity distance from Chroma (lower = closer)
    importance: float