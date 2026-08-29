"""
config.py
Central configuration for the Memory-Aware Chatbot.
Handles safe API key loading (Streamlit Secrets first, .env fallback).
"""

import os
from dotenv import load_dotenv

# Load .env for local development (no-op if file doesn't exist)
load_dotenv()

def get_api_key() -> str:
    """
    Safely retrieve the Gemini API key.
    Priority:
      1. Streamlit Secrets (used on Streamlit Community Cloud)
      2. Environment variable / .env file (used for local dev)
    Raises a clear error if not found anywhere.
    """
    api_key = None

    # Try Streamlit secrets first (only works inside a Streamlit runtime)
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        # st.secrets not available (e.g. running outside streamlit, or no secrets.toml)
        pass

    # Fallback to environment variable / .env
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. "
            "Set it in a local .env file (for dev) or in Streamlit "
            "Cloud -> App Settings -> Secrets (for deployment)."
        )

    return api_key


# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------
GEMINI_CHAT_MODEL = "gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"

# ---------------------------------------------------------------------
# Memory configuration
# ---------------------------------------------------------------------
STM_MAX_MESSAGES = 12          # Max messages kept in short-term memory (rolling window)
LTM_TOP_K = 3                  # Top-k results retrieved PER category (semantic/factual/episodic)
IMPORTANCE_DEFAULT = 0.5        # Default importance score for new memories

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "db")
CHROMA_DIR = os.path.join(DB_DIR, "chroma_db")
SQLITE_PATH = os.path.join(DB_DIR, "metadata.sqlite3")

os.makedirs(CHROMA_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Chroma collection names
# ---------------------------------------------------------------------
COLLECTION_SEMANTIC = "semantic_memory"
COLLECTION_FACTUAL = "factual_memory"
COLLECTION_EPISODIC = "episodic_memory"

VALID_MEMORY_TYPES = ["Semantic", "Factual", "Episodic"]
