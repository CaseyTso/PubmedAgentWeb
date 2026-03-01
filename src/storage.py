"""
Deduplication Storage

SQLite-based storage to track which PMIDs have already been processed,
preventing duplicate notifications across runs.
"""

import logging
import os
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)


class SeenStore:
    """SQLite store for tracking processed article PMIDs."""

    def __init__(self, db_path: str = "data/seen_articles.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create the table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS seen_articles (
                    pmid TEXT PRIMARY KEY,
                    title TEXT,
                    score INTEGER,
                    first_seen TEXT
                )
            """)
            conn.commit()

    def filter_unseen(self, pmids: list[str]) -> set[str]:
        """
        Given a list of PMIDs, return the subset that has NOT been seen before.
        """
        if not pmids:
            return set()
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ",".join("?" for _ in pmids)
            cursor = conn.execute(
                f"SELECT pmid FROM seen_articles WHERE pmid IN ({placeholders})",
                pmids,
            )
            seen = {row[0] for row in cursor.fetchall()}
        return set(pmids) - seen

    def mark_seen(self, pmid: str, title: str = "", score: int = 0):
        """Mark a single PMID as seen."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO seen_articles (pmid, title, score, first_seen) VALUES (?, ?, ?, ?)",
                (pmid, title, score, datetime.now().isoformat()),
            )
            conn.commit()

    def mark_batch_seen(self, articles: list[tuple[str, str, int]]):
        """
        Mark multiple PMIDs as seen.

        Args:
            articles: List of (pmid, title, score) tuples.
        """
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO seen_articles (pmid, title, score, first_seen) VALUES (?, ?, ?, ?)",
                [(pmid, title, score, now) for pmid, title, score in articles],
            )
            conn.commit()
        logger.debug(f"Marked {len(articles)} articles as seen.")

    def count(self) -> int:
        """Return total number of tracked PMIDs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM seen_articles")
            return cursor.fetchone()[0]
