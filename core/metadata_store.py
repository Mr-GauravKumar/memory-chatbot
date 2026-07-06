"""
core/metadata_store.py
SQLite wrapper for memory metadata (id, type, timestamps, importance, usage_count).
This mirrors what's stored in Chroma, but lets us do fast stats/queries
without touching the vector DB.
"""

import sqlite3
from typing import List, Optional
from contextlib import contextmanager

from config import SQLITE_PATH
from utils.logger import get_logger
from utils.helpers import current_timestamp

logger = get_logger(__name__)


class MetadataStore:
    def __init__(self):
        self.db_path = SQLITE_PATH
        self._init_db()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    importance REAL NOT NULL,
                    usage_count INTEGER NOT NULL DEFAULT 0
                )
            """)

    def insert_memory(self, memory_id: str, memory_type: str, content: str, importance: float):
        now = current_timestamp()
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO memories (id, type, content, created_at, last_accessed, importance, usage_count)
                   VALUES (?, ?, ?, ?, ?, ?, 0)""",
                (memory_id, memory_type, content, now, now, importance),
            )
        logger.info(f"Inserted metadata for memory {memory_id} ({memory_type})")

    def update_access(self, memory_id: str):
        """Bump usage_count and last_accessed when a memory is retrieved and used."""
        now = current_timestamp()
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE memories
                   SET usage_count = usage_count + 1, last_accessed = ?
                   WHERE id = ?""",
                (now, memory_id),
            )

    def get_stats(self) -> dict:
        """Return counts per memory type + total, used for sidebar stats."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT type, COUNT(*) as cnt FROM memories GROUP BY type"
            ).fetchall()

        stats = {"Semantic": 0, "Factual": 0, "Episodic": 0}
        for row in rows:
            stats[row["type"]] = row["cnt"]
        stats["total"] = sum(stats.values())
        return stats

    def get_all(self, memory_type: Optional[str] = None) -> List[sqlite3.Row]:
        with self._get_conn() as conn:
            if memory_type:
                return conn.execute(
                    "SELECT * FROM memories WHERE type = ? ORDER BY created_at DESC", (memory_type,)
                ).fetchall()
            return conn.execute("SELECT * FROM memories ORDER BY created_at DESC").fetchall()

    def delete_memory(self, memory_id: str):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

    def clear_all(self):
        """Wipe all metadata (used by 'Clear LTM' button)."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM memories")
        logger.info("Cleared all metadata records.")