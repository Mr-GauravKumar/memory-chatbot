"""
core/classifier.py
Classifies each user message: should it be stored as long-term memory,
and if so, under which category (Semantic / Factual / Episodic)?
Uses the exact classification prompt specified for this project.
"""

import json
from typing import Optional

from core.gemini_client import GeminiClient
from core.memory_models import ClassificationResult
from config import VALID_MEMORY_TYPES
from utils.logger import get_logger

logger = get_logger(__name__)

CLASSIFIER_SYSTEM_PROMPT = """You are a memory classification engine.
Your only job is to decide whether the user message should be stored as long-term memory.
There are only three memory types.

Semantic
Stores stable preferences, personality, skills, interests, likes/dislikes.
Examples:
I love football.
I enjoy programming.
I prefer Python.
----------------------------
Factual
Stores objective facts about the user.
Examples:
My name is Gaurav.
I am studying at IIT Madras.
I work at Infosys.
----------------------------
Episodic
Stores experiences, events and actions that happened at a particular time.
Examples:
Yesterday I attended an interview.
Last week I visited Goa.
Today I completed my assignment.

If nothing important exists
Return
{
"store":false
}

Otherwise return
{
"store":true,
"type":"Semantic",
"memory":"I enjoy football."
}

Return JSON only. No markdown, no explanation, no code fences. Only raw JSON."""


class MemoryClassifier:
    def __init__(self, gemini_client: GeminiClient):
        self.client = gemini_client

    def classify(self, user_message: str) -> ClassificationResult:
        raw_output = self.client.classify_raw(CLASSIFIER_SYSTEM_PROMPT, user_message)
        return self._parse(raw_output)

    def _parse(self, raw_output: str) -> ClassificationResult:
        """Safely parse the model's JSON output, defaulting to store=False on any error."""
        cleaned = raw_output.strip()

        # Strip accidental markdown fences if the model adds them despite instructions
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json", "", 1).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"Classifier returned non-JSON output: {raw_output}")
            return ClassificationResult(store=False)

        store = data.get("store", False)
        if not store:
            return ClassificationResult(store=False)

        mem_type = data.get("type")
        memory_text = data.get("memory")

        # Validate type and content strictly — never trust blindly
        if mem_type not in VALID_MEMORY_TYPES or not memory_text:
            logger.warning(f"Classifier returned invalid structure: {data}")
            return ClassificationResult(store=False)

        return ClassificationResult(store=True, type=mem_type, memory=memory_text.strip())