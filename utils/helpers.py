"""
utils/helpers.py
Small reusable utility functions used across the project.
"""

import uuid
from datetime import datetime


def generate_id() -> str:
    """Generate a unique ID for a memory record."""
    return str(uuid.uuid4())


def current_timestamp() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.utcnow().isoformat()


def compute_importance_score(memory_type: str, text: str) -> float:
    """
    Simple heuristic importance scorer (0.0 - 1.0).
    Factual memories (identity facts) are weighted higher than episodic
    (transient events), since they're more likely to be useful long-term.
    This is intentionally simple and deterministic — no LLM call needed.
    """
    base_scores = {
        "Factual": 0.8,
        "Semantic": 0.6,
        "Episodic": 0.4,
    }
    score = base_scores.get(memory_type, 0.5)

    # Slight boost for longer, more detailed statements
    if len(text.split()) > 12:
        score = min(1.0, score + 0.1)

    return round(score, 2)