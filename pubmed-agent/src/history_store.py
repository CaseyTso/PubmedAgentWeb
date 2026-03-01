"""
Extended History Storage for Web UI

Stores full article data + LLM scores in a separate SQLite table
for browsing history, searching, and exporting reports.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)


class HistoryStore:
    """Extended storage for the web UI — stores full article data + scores."""

    def __init__(self, db_path: str = "data/seen_articles.db"):
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scored_history (
                    pmid TEXT PRIMARY KEY,
                    title TEXT,
                    abstract TEXT,
                    authors TEXT,
                    journal TEXT,
                    pub_date TEXT,
                    doi TEXT,
                    keywords TEXT,
                    score INTEGER,
                    reason TEXT,
                    search_query TEXT,
                    scored_at TEXT
                )
            """)
            conn.commit()

    def save_scored_article(self, scored_article, query: str = ""):
        """Store a scored article. Updates score/reason if PMID already exists."""
        a = scored_article.article
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO scored_history
                (pmid, title, abstract, authors, journal, pub_date, doi, keywords, score, reason, search_query, scored_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                a.pmid,
                a.title,
                a.abstract,
                json.dumps(a.authors, ensure_ascii=False),
                a.journal,
                a.pub_date,
                a.doi,
                json.dumps(a.keywords, ensure_ascii=False),
                scored_article.score,
                scored_article.reason,
                query,
                now,
            ))
            conn.commit()

    def save_batch(self, scored_articles, query: str = ""):
        """Store multiple scored articles."""
        now = datetime.now().isoformat()
        rows = []
        for sa in scored_articles:
            a = sa.article
            rows.append((
                a.pmid, a.title, a.abstract,
                json.dumps(a.authors, ensure_ascii=False),
                a.journal, a.pub_date, a.doi,
                json.dumps(a.keywords, ensure_ascii=False),
                sa.score, sa.reason, query, now,
            ))
        with self._conn() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO scored_history
                (pmid, title, abstract, authors, journal, pub_date, doi, keywords, score, reason, search_query, scored_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()

    def get_history(
        self,
        page: int = 1,
        per_page: int = 50,
        min_score: int = 0,
        search: str = "",
    ) -> dict:
        """
        Paginated history query.

        Returns:
            {"items": [...], "total": int, "page": int, "per_page": int}
        """
        conditions = []
        params = []

        if min_score > 0:
            conditions.append("score >= ?")
            params.append(min_score)
        if search:
            conditions.append("(title LIKE ? OR journal LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%"])

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            # Count
            count_row = conn.execute(
                f"SELECT COUNT(*) as cnt FROM scored_history {where}", params
            ).fetchone()
            total = count_row["cnt"]

            # Fetch page
            offset = (page - 1) * per_page
            rows = conn.execute(
                f"SELECT * FROM scored_history {where} ORDER BY scored_at DESC, score DESC LIMIT ? OFFSET ?",
                params + [per_page, offset],
            ).fetchall()

        items = [self._row_to_dict(r) for r in rows]
        return {"items": items, "total": total, "page": page, "per_page": per_page}

    def get_articles_by_pmids(self, pmids: list[str]) -> list[dict]:
        """Fetch specific articles for export."""
        if not pmids:
            return []
        placeholders = ",".join("?" for _ in pmids)
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM scored_history WHERE pmid IN ({placeholders}) ORDER BY score DESC",
                pmids,
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_stats(self) -> dict:
        """Dashboard stats."""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM scored_history").fetchone()[0]
            avg_score = conn.execute("SELECT AVG(score) FROM scored_history WHERE score > 0").fetchone()[0]
            high_relevance = conn.execute("SELECT COUNT(*) FROM scored_history WHERE score >= 6").fetchone()[0]
        return {
            "total_articles": total,
            "avg_score": round(avg_score, 1) if avg_score else 0,
            "high_relevance": high_relevance,
        }

    @staticmethod
    def _row_to_dict(row) -> dict:
        d = dict(row)
        # Parse JSON arrays
        for field in ("authors", "keywords"):
            if d.get(field):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    d[field] = []
            else:
                d[field] = []
        d["url"] = f"https://pubmed.ncbi.nlm.nih.gov/{d['pmid']}/"
        return d
