#!/usr/bin/env python3
"""
PubMed Literature Monitor Agent — Main Entry Point

Orchestrates the full pipeline:
  1. Search PubMed for recently indexed articles
  2. Deduplicate against previously seen articles
  3. Score relevance using an LLM
  4. Send notifications via Feishu / WeChat Work webhook

Usage:
  # Run once immediately:
  python main.py --config config.yaml

  # Run once (dry-run, no notifications):
  python main.py --config config.yaml --dry-run

  # Run as a scheduled daemon:
  python main.py --config config.yaml --daemon
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import yaml

from src.pubmed_client import PubMedClient
from src.scorer import RelevanceScorer
from src.notifier import create_notifier
from src.storage import SeenStore


def load_config(path: str) -> dict:
    """Load and validate YAML configuration."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validate required fields
    if not config.get("research", {}).get("query"):
        raise ValueError("Config error: research.query is required.")
    if not config.get("llm", {}).get("api_key"):
        raise ValueError("Config error: llm.api_key is required.")
    if not config.get("research", {}).get("description"):
        raise ValueError("Config error: research.description is required.")

    return config


def setup_logging(config: dict):
    """Configure logging from config."""
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("file", "")

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def run_pipeline(config: dict, dry_run: bool = False):
    """Execute the full monitoring pipeline once."""
    logger = logging.getLogger("pipeline")
    date_str = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"=== PubMed Monitor Pipeline Start ({date_str}) ===")

    # --- Config shortcuts ---
    pm_cfg = config.get("pubmed", {})
    research_cfg = config["research"]
    llm_cfg = config["llm"]
    notify_cfg = config.get("notify", {})
    storage_cfg = config.get("storage", {})

    # --- 1. Search PubMed ---
    client = PubMedClient(
        api_key=pm_cfg.get("api_key", ""),
        email=pm_cfg.get("email", ""),
        tool_name=pm_cfg.get("tool_name", "pubmed_monitor"),
    )

    articles = client.search_and_fetch(
        query=research_cfg["query"],
        days=pm_cfg.get("search_days", 2),
        batch_size=pm_cfg.get("batch_size", 200),
    )
    logger.info(f"Fetched {len(articles)} articles from PubMed.")

    if not articles:
        logger.info("No new articles found. Pipeline complete.")
        return

    # --- 2. Deduplicate ---
    store = SeenStore(db_path=storage_cfg.get("db_path", "data/seen_articles.db"))
    all_pmids = [a.pmid for a in articles]
    unseen_pmids = store.filter_unseen(all_pmids)
    articles = [a for a in articles if a.pmid in unseen_pmids]
    logger.info(f"{len(articles)} articles are new (not previously seen).")

    if not articles:
        logger.info("All articles already seen. Pipeline complete.")
        return

    # Filter out articles without abstracts before LLM scoring (save cost)
    articles_with_abstract = [a for a in articles if a.has_abstract]
    articles_no_abstract = [a for a in articles if not a.has_abstract]
    logger.info(
        f"{len(articles_with_abstract)} articles have abstracts, "
        f"{len(articles_no_abstract)} do not (will be skipped for LLM scoring)."
    )

    # --- 3. LLM Relevance Scoring ---
    scorer = RelevanceScorer(
        api_key=llm_cfg["api_key"],
        model=llm_cfg.get("model", "gpt-4o-mini"),
        base_url=llm_cfg.get("base_url") or None,
        temperature=llm_cfg.get("temperature", 0.1),
        max_tokens=llm_cfg.get("max_tokens", 512),
    )

    min_score = research_cfg.get("min_score", 6)
    scored = scorer.score_batch(
        articles=articles_with_abstract,
        research_description=research_cfg["description"],
        min_score=min_score,
    )
    logger.info(f"{len(scored)} articles scored >= {min_score} (will be notified).")

    # Limit notification count
    max_articles = notify_cfg.get("max_articles", 20)
    scored = scored[:max_articles]

    # --- 4. Mark all fetched articles as seen (including low-score ones) ---
    seen_entries = [(a.pmid, a.title, 0) for a in articles_no_abstract]
    seen_entries += [(sa.article.pmid, sa.article.title, sa.score) for sa in scored]
    # Also mark articles that were scored but below threshold
    all_scored_pmids = {sa.article.pmid for sa in scored}
    for a in articles_with_abstract:
        if a.pmid not in all_scored_pmids:
            seen_entries.append((a.pmid, a.title, 0))
    store.mark_batch_seen(seen_entries)
    logger.info(f"Total tracked PMIDs in database: {store.count()}")

    # --- 5. Send Notifications ---
    if dry_run:
        logger.info("=== DRY RUN — Skipping notifications ===")
        for sa in scored:
            logger.info(
                f"  [{sa.score}/10] {sa.article.title}\n"
                f"           {sa.article.url}\n"
                f"           {sa.reason}"
            )
    else:
        notifiers = create_notifier(notify_cfg)
        for notifier in notifiers:
            notifier.send(scored, date_str)

    logger.info(f"=== Pipeline Complete ({date_str}) ===")


def run_daemon(config: dict):
    """Run as a scheduled daemon using APScheduler."""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    logger = logging.getLogger("daemon")
    schedule_cfg = config.get("schedule", {})
    cron_expr = schedule_cfg.get("cron", "0 8 * * *")
    timezone = schedule_cfg.get("timezone", "Asia/Shanghai")

    scheduler = BlockingScheduler()

    # Parse cron expression: "minute hour day month day_of_week"
    parts = cron_expr.split()
    trigger = CronTrigger(
        minute=parts[0] if len(parts) > 0 else "0",
        hour=parts[1] if len(parts) > 1 else "8",
        day=parts[2] if len(parts) > 2 else "*",
        month=parts[3] if len(parts) > 3 else "*",
        day_of_week=parts[4] if len(parts) > 4 else "*",
        timezone=timezone,
    )

    scheduler.add_job(run_pipeline, trigger, args=[config], id="pubmed_monitor")
    logger.info(f"Scheduler started. Cron: {cron_expr}, Timezone: {timezone}")
    logger.info("Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="PubMed Literature Monitor Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main.py --config config.yaml              # Run once
  python main.py --config config.yaml --dry-run     # Run once, no notifications
  python main.py --config config.yaml --daemon      # Run on schedule
""",
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline but skip sending notifications (print results instead).",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as a scheduled daemon (uses cron from config).",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    if args.daemon:
        run_daemon(config)
    else:
        run_pipeline(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
