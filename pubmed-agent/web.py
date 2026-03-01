#!/usr/bin/env python3
"""
PubMed Literature Monitor — Web Application

FastAPI backend that wraps the existing PubMed client, LLM scorer,
and storage modules into an interactive web API with SSE streaming.

Usage:
    python web.py                  # Start on port 8080
    python web.py --port 9000      # Custom port
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.pubmed_client import PubMedClient
from src.scorer import RelevanceScorer, ScoredArticle
from src.storage import SeenStore
from src.history_store import HistoryStore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("web")

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="PubMed Monitor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.environ.get("DB_PATH", "data/seen_articles.db")


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="PubMed search query")
    research_description: str = Field(..., min_length=1)
    days: int = Field(default=2, ge=1, le=30)
    max_results: int = Field(default=100, ge=1, le=1000)
    min_score: int = Field(default=6, ge=0, le=10)
    llm_api_key: str = Field(..., min_length=1)
    llm_base_url: str = ""
    llm_model: str = "deepseek-chat"
    llm_temperature: float = Field(default=0.1, ge=0, le=2)
    skip_seen: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def serialize_article(sa: ScoredArticle) -> dict:
    """Convert a ScoredArticle to a JSON-serializable dict."""
    a = sa.article
    return {
        "pmid": a.pmid,
        "title": a.title,
        "abstract": a.abstract,
        "authors": a.authors,
        "journal": a.journal,
        "pub_date": a.pub_date,
        "doi": a.doi,
        "keywords": a.keywords,
        "url": a.url,
        "score": sa.score,
        "reason": sa.reason,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=FileResponse)
@app.head("/")
async def index():
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "static", "index.html"),
        media_type="text/html",
    )


@app.post("/api/search")
async def search(req: SearchRequest):
    """
    Run the full search + score pipeline with SSE streaming.
    Streams progress events, article results, and a final summary.
    """

    async def event_generator():
        store = SeenStore(db_path=DB_PATH)
        history = HistoryStore(db_path=DB_PATH)

        # --- Stage 1: Search PubMed ---
        yield sse_event("progress", {
            "stage": "search", "message": "正在搜索 PubMed...", "percent": 5,
        })

        try:
            client = PubMedClient(email="pubmed_monitor@example.com")
            articles = await asyncio.to_thread(
                client.search_and_fetch, req.query, req.days, 200, req.max_results,
            )
        except Exception as e:
            yield sse_event("error", {"message": f"PubMed 搜索失败: {e}"})
            return

        yield sse_event("progress", {
            "stage": "search",
            "message": f"找到 {len(articles)} 篇文献",
            "percent": 15,
            "total_found": len(articles),
        })

        if not articles:
            yield sse_event("complete", {
                "total_found": 0, "new_articles": 0,
                "total_scored": 0, "passed": 0,
            })
            return

        # --- Stage 2: Dedup ---
        if req.skip_seen:
            all_pmids = [a.pmid for a in articles]
            unseen = store.filter_unseen(all_pmids)
            articles = [a for a in articles if a.pmid in unseen]

        yield sse_event("progress", {
            "stage": "dedup",
            "message": f"{len(articles)} 篇为新文献（去重后）",
            "percent": 20,
            "new_articles": len(articles),
        })

        if not articles:
            yield sse_event("complete", {
                "total_found": len(all_pmids) if req.skip_seen else 0,
                "new_articles": 0, "total_scored": 0, "passed": 0,
            })
            return

        # Filter articles with abstracts
        articles_with_abs = [a for a in articles if a.has_abstract]
        articles_no_abs = [a for a in articles if not a.has_abstract]

        yield sse_event("progress", {
            "stage": "scoring",
            "message": f"{len(articles_with_abs)} 篇有摘要，准备 LLM 评分...",
            "percent": 22,
        })

        # Mark no-abstract articles as seen
        if articles_no_abs:
            store.mark_batch_seen([(a.pmid, a.title, 0) for a in articles_no_abs])

        # --- Stage 3: LLM Scoring ---
        try:
            scorer = RelevanceScorer(
                api_key=req.llm_api_key,
                model=req.llm_model,
                base_url=req.llm_base_url or None,
                temperature=req.llm_temperature,
            )
        except Exception as e:
            yield sse_event("error", {"message": f"LLM 初始化失败: {e}"})
            return

        results = []
        total = len(articles_with_abs)

        for i, article in enumerate(articles_with_abs):
            pct = 22 + int((i / max(total, 1)) * 70)
            yield sse_event("progress", {
                "stage": "scoring",
                "message": f"正在评分 {i + 1}/{total}...",
                "current": i + 1,
                "total": total,
                "percent": pct,
            })

            try:
                scored = await asyncio.to_thread(
                    scorer.score_article, article, req.research_description,
                )
            except Exception as e:
                logger.error(f"Scoring error for PMID {article.pmid}: {e}")
                scored = ScoredArticle(article=article, score=0, reason=f"评分出错: {e}")

            # Save to history and mark seen
            history.save_scored_article(scored, query=req.query)
            store.mark_seen(article.pmid, article.title, scored.score)

            if scored.score >= req.min_score:
                results.append(scored)
                yield sse_event("article", serialize_article(scored))

        # --- Stage 4: Complete ---
        yield sse_event("complete", {
            "total_found": len(all_pmids) if req.skip_seen else total,
            "new_articles": len(articles),
            "total_scored": total,
            "passed": len(results),
            "percent": 100,
        })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/history")
async def history(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=50, ge=1, le=200),
    min_score: int = Query(default=0, ge=0, le=10),
    search: str = Query(default=""),
):
    """Browse previously scored articles."""
    hs = HistoryStore(db_path=DB_PATH)
    return hs.get_history(page=page, per_page=per_page, min_score=min_score, search=search)


@app.get("/api/stats")
async def stats():
    """Dashboard statistics."""
    hs = HistoryStore(db_path=DB_PATH)
    return hs.get_stats()


@app.get("/api/export")
async def export(
    pmids: str = Query(default="", description="Comma-separated PMIDs"),
):
    """Generate a downloadable HTML report for given PMIDs."""
    hs = HistoryStore(db_path=DB_PATH)

    if pmids:
        pmid_list = [p.strip() for p in pmids.split(",") if p.strip()]
        articles = hs.get_articles_by_pmids(pmid_list)
    else:
        result = hs.get_history(page=1, per_page=100, min_score=1)
        articles = result["items"]

    html = _build_export_html(articles)
    date_str = datetime.now().strftime("%Y-%m-%d")
    return HTMLResponse(
        content=html,
        headers={
            "Content-Disposition": f'attachment; filename="pubmed-report-{date_str}.html"',
        },
    )


def _build_export_html(articles: list[dict]) -> str:
    """Build a self-contained HTML report."""
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = ""
    for i, a in enumerate(articles, 1):
        authors = ", ".join(a.get("authors", [])[:4])
        if len(a.get("authors", [])) > 4:
            authors += " et al."
        score = a.get("score", 0)
        color = "#059669" if score >= 8 else "#d97706" if score >= 5 else "#dc2626"
        abstract = a.get("abstract", "").replace("\n", "<br>")
        rows += f"""
        <div style="border:1px solid #e5e7eb;border-radius:8px;padding:20px;margin-bottom:16px;page-break-inside:avoid">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
            <span style="background:{color};color:#fff;border-radius:50%;width:36px;height:36px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px;flex-shrink:0">{score}</span>
            <h3 style="margin:0;font-size:15px;line-height:1.4"><a href="{a.get('url','')}" target="_blank" style="color:#1d4ed8;text-decoration:none">{a.get('title','')}</a></h3>
          </div>
          <p style="margin:4px 0;font-size:13px;color:#6b7280">{a.get('journal','')} | {a.get('pub_date','')} | PMID: {a.get('pmid','')}</p>
          <p style="margin:4px 0;font-size:13px;color:#6b7280">{authors}</p>
          <p style="margin:8px 0 4px;font-size:13px;color:#4338ca;font-style:italic">{a.get('reason','')}</p>
          <details style="margin-top:8px"><summary style="cursor:pointer;font-size:12px;color:#9ca3af">Abstract</summary>
            <p style="margin:8px 0;font-size:12px;color:#374151;line-height:1.6">{abstract}</p>
          </details>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>PubMed 文献报告 — {date_str}</title>
<style>
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:800px;margin:0 auto;padding:24px;color:#1f2937;background:#fff}}
  h1{{font-size:22px;border-bottom:2px solid #3b82f6;padding-bottom:8px}}
  .meta{{color:#6b7280;font-size:13px;margin-bottom:24px}}
  @media print{{body{{padding:0}} details{{display:block}} summary{{display:none}} details>p{{display:block}}}}
</style></head><body>
<h1>PubMed 文献筛选报告</h1>
<p class="meta">生成时间: {date_str} | 共 {len(articles)} 篇文献</p>
{rows}
<p style="text-align:center;color:#9ca3af;font-size:11px;margin-top:32px">Generated by PubMed Monitor Agent</p>
</body></html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PubMed Monitor Web")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    logger.info(f"Starting PubMed Monitor Web at http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
