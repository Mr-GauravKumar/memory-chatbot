"""
app.py
Streamlit entrypoint: Chat UI + Sidebar (memory stats, clear buttons, debug mode).
"""

import streamlit as st

st.set_page_config(page_title="Memory-Aware AI Chatbot", page_icon="🧠", layout="centered")

from core.stm import ShortTermMemory
from core.ltm_manager import LTMManager
from core.gemini_client import GeminiClient

CHAT_SYSTEM_PROMPT = """You are a helpful, friendly AI assistant with long-term memory about the user.
You will be given relevant memories retrieved from a database (Semantic, Factual, Episodic).
Rules:
- Only use memories that are explicitly provided to you below. Never invent or assume facts about the user that aren't given.
- If no relevant memory is provided for something, just say you don't know it yet.
- Be natural and conversational — don't say "according to my memory" every time, just use the info naturally.
- Keep responses concise and helpful.
"""

if "stm" not in st.session_state:
    st.session_state.stm = ShortTermMemory()

if "ltm_manager" not in st.session_state:
    st.session_state.ltm_manager = LTMManager()

if "gemini_client" not in st.session_state:
    st.session_state.gemini_client = GeminiClient()

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = []

stm = st.session_state.stm
ltm = st.session_state.ltm_manager
gemini = st.session_state.gemini_client

with st.sidebar:
    st.title("Memory Dashboard")

    stats = ltm.get_stats()
    st.subheader("Long-Term Memory Stats")
    col1, col2 = st.columns(2)
    col1.metric("Semantic", stats.get("Semantic", 0))
    col2.metric("Factual", stats.get("Factual", 0))
    col3, col4 = st.columns(2)
    col3.metric("Episodic", stats.get("Episodic", 0))
    col4.metric("Total", stats.get("total", 0))

    st.subheader("Short-Term Memory")
    st.write(f"Messages in current session: **{len(stm.get_history())}**")

    st.divider()

    st.session_state.debug_mode = st.toggle("Debug mode (show retrieved memories)", value=st.session_state.debug_mode)

    st.divider()

    if st.button("Clear Chat (STM)", use_container_width=True):
        stm.clear()
        st.session_state.last_retrieved = []
        st.rerun()

    if st.button("Clear Long-Term Memory", use_container_width=True, type="secondary"):
        ltm.clear_all_memory()
        st.rerun()

    st.divider()
    st.caption("Built with Streamlit + Gemini + ChromaDB + SQLite")

st.title("Memory-Aware AI Chatbot")
st.caption("Demonstrates Short-Term + Long-Term (Semantic / Factual / Episodic) memory")

for msg in stm.get_history():
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])

if st.session_state.debug_mode and st.session_state.last_retrieved:
    with st.expander("Retrieved Memories (last query)", expanded=True):
        for mem in st.session_state.last_retrieved:
            st.markdown(
                f"**[{mem.type}]** {mem.content}  \n"
                f"`distance: {mem.distance:.4f}`"
            )

user_input = st.chat_input("Type your message...")

if user_input:
    stm.add_message("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        stored_type = ltm.process_and_store(user_input)
    except Exception as e:
        stored_type = None
        if st.session_state.debug_mode:
            st.warning(f"Memory storage skipped due to error: {e}")

    try:
        retrieved = ltm.retrieve_relevant_memories(user_input)
    except Exception:
        retrieved = []

    st.session_state.last_retrieved = retrieved

    if retrieved:
        memory_context = "\n".join(
            f"- [{m.type}] {m.content}" for m in retrieved
        )
        memory_block = f"\n\nRelevant memories about the user:\n{memory_context}"
    else:
        memory_block = "\n\nNo relevant long-term memories found for this query."

    full_system_prompt = CHAT_SYSTEM_PROMPT + memory_block

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            contents = stm.to_gemini_format()
            response_text = gemini.generate_response(full_system_prompt, contents)
            st.markdown(response_text)

    stm.add_message("model", response_text)