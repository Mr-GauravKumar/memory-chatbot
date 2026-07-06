"""
core/gemini_client.py
Wraps all calls to the Google Gemini API: chat completion + embeddings.
Uses the current `google-genai` SDK (the old `google-generativeai`
package is deprecated and no longer receives updates).
"""

from google import genai
from google.genai import types
from typing import List, Dict

from config import get_api_key, GEMINI_CHAT_MODEL, GEMINI_EMBEDDING_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)

# Single shared client, configured once with the API key
_client = genai.Client(api_key=get_api_key())


class GeminiClient:
    def __init__(self):
        self.client = _client

    def embed_text(self, text: str) -> List[float]:
        """Generate an embedding vector for a given text using Gemini Embedding API."""
        try:
            result = self.client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=text,
            )
            return result.embeddings[0].values
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    def generate_response(self, system_prompt: str, contents: List[Dict]) -> str:
        """
        Generate a chat response.
        `contents` is a list of {"role": "user"/"model", "parts": [text]} dicts (STM history).
        """
        try:
            gemini_contents = [
                types.Content(
                    role=c["role"],
                    parts=[types.Part(text=p) for p in c["parts"]],
                )
                for c in contents
            ]
            response = self.client.models.generate_content(
                model=GEMINI_CHAT_MODEL,
                contents=gemini_contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                ),
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Chat generation failed: {e}")
            return "Sorry, I ran into an error generating a response. Please try again."

    def classify_raw(self, system_prompt: str, user_message: str) -> str:
        """
        Calls Gemini with the classifier system prompt + user message.
        Returns raw text (expected to be JSON) — parsing happens in classifier.py.
        """
        try:
            response = self.client.models.generate_content(
                model=GEMINI_CHAT_MODEL,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                ),
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Classification call failed: {e}")
            return '{"store": false}'