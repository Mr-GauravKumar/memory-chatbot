# 🧠 Memory-Aware AI Chatbot
A production-ready AI chatbot that combines **Short-Term Memory (STM)** and **Long-Term Memory (LTM)** to deliver context-aware, personalized conversations. The chatbot automatically classifies memories into **Semantic**, **Factual**, and **Episodic** categories, enabling intelligent retrieval and long-term context retention using Google Gemini embeddings.

--- 


## 🚀 Live Demo


**Streamlit App:**  


---

https://memory-chatbot-u9nal3ycbosdiqbucjcgaz.streamlit.app/

---

## ✨ Features

- 💬 Context-aware conversations using Google Gemini
- 🧠 Short-Term Memory (session-based context)
- 📚 Long-Term Memory with automatic classification:
  - Semantic Memory
  - Factual Memory
  - Episodic Memory
- 🔍 Semantic search using Gemini Embeddings
- 📦 ChromaDB vector database for efficient memory retrieval
- 🗄️ SQLite for metadata management
- ⚡ Fast Streamlit web interface
- ☁️ One-click deployment on Streamlit Community Cloud

---


## 🏗️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| LLM | Google Gemini |
| Embeddings | Gemini Embeddings |
| Vector Database | ChromaDB |
| Metadata Database | SQLite |
| Language | Python |

---

## 📂 Memory Architecture

### Short-Term Memory (STM)
- Maintains the most recent conversation history.
- Session-based only.
- Cleared automatically when the session ends.
- Provides immediate conversational context.

### Long-Term Memory (LTM)
Stores important information permanently by classifying memories into:


- **Semantic Memory** – General knowledge and learned concepts.
- **Factual Memory** – User-specific facts and preferences.
- **Episodic Memory** – Past interactions and experiences.

Each memory is:
- Embedded using Gemini Embeddings
- Stored as vectors in ChromaDB
- Indexed with metadata in SQLite
- Ranked using similarity search before retrieval

Before every response, the chatbot combines:
- Recent STM conversation
- Top relevant memories from LTM

This creates a richer and more personalized context for Gemini.

---

## 📁 Project Structure

```
Memory-Chatbot/
│
├── app.py
├── requirements.txt
├── .env.example
├── chroma_db/
├── database/
├── memory/
├── utils/
└── README.md
```

---

## ⚙️ Local Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Memory-Chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file from `.env.example` and add your Gemini API key.

```env
GEMINI_API_KEY=your_api_key
```

### 4. Run the Application

```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push the repository to GitHub.
2. Open **Streamlit Community Cloud**.
3. Deploy the repository and select `app.py` as the entry point.
4. Navigate to **App Settings → Secrets**.
5. Add your Gemini API key:

```toml
GEMINI_API_KEY = "your_real_api_key"
```

6. Save the secrets and redeploy if necessary.

---

## 🔮 Future Improvements

- Procedural Memory support
- Memory editing and deletion
- Importance scoring using LLM
- Hybrid BM25 + Vector Retrieval
- Multi-user authentication
- Conversation summarization
- Memory visualization dashboard

---

## 👨‍💻 Author

**Gaurav Kumar**

If you found this project useful, consider giving it a ⭐ on GitHub.
