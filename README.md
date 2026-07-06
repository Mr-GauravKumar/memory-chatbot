# Memory-Aware AI Chatbot

A production-quality chatbot built with Google Gemini, implementing both
Short-Term Memory (STM) and Long-Term Memory (LTM: Semantic, Factual, Episodic).

## Stack
- Streamlit (UI)
- Google Gemini API (chat + embeddings)
- ChromaDB (vector storage)
- SQLite (metadata)

## Setup (Local)

1. Clone the repo
2. `pip install -r requirements.txt`
3. Copy `.env.example` → `.env` and add your real `GEMINI_API_KEY`
4. `streamlit run app.py`

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub (`.env` is gitignored — it will NOT be uploaded)
2. Deploy on share.streamlit.io, pointing to `app.py`
3. Go to **App Settings → Secrets** and add:
```toml
   GEMINI_API_KEY = "your_real_key_here"
```
4. Save — the app will pick up the key automatically via `st.secrets`.

## Memory Architecture
- **STM**: last N messages, session-only, never persisted
- **LTM**: classified into Semantic / Factual / Episodic, embedded via Gemini,
  stored in ChromaDB (vectors) + SQLite (metadata: importance, usage count, timestamps)
- Retrieval merges STM + top-k LTM matches per category before calling Gemini