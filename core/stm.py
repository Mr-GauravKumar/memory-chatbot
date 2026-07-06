"""
core/stm.py
Short-Term Memory: in-session conversation history only.
Never persisted to disk. Resets completely on 'Clear Chat' or new session.
"""

from typing import List, Dict
from config import STM_MAX_MESSAGES


class ShortTermMemory:
    def __init__(self):
        self.messages: List[Dict[str, str]] = []  # [{"role": "user"/"model", "content": "..."}]

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._trim()

    def _trim(self):
        """Keep only the most recent STM_MAX_MESSAGES messages."""
        if len(self.messages) > STM_MAX_MESSAGES:
            self.messages = self.messages[-STM_MAX_MESSAGES:]

    def get_history(self) -> List[Dict[str, str]]:
        return self.messages

    def to_gemini_format(self) -> List[Dict]:
        """Convert STM into Gemini's expected 'contents' format."""
        gemini_role_map = {"user": "user", "model": "model"}
        return [
            {"role": gemini_role_map.get(m["role"], "user"), "parts": [m["content"]]}
            for m in self.messages
        ]

    def clear(self):
        self.messages = []