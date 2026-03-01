"""
Notification Module

Sends filtered article summaries via Feishu or WeChat Work webhooks.
"""

import hashlib
import hmac
import base64
import logging
import time as time_module
from datetime import datetime

import requests

from .scorer import ScoredArticle

logger = logging.getLogger(__name__)


class FeishuNotifier:
    """Send notifications to a Feishu (飞书) group chat via custom bot webhook."""

    def __init__(self, webhook_url: str, secret: str = ""):
        self.webhook_url = webhook_url
        self.secret = secret

    def _sign(self) -> tuple[str, str]:
        """Generate timestamp and HMAC-SHA256 signature for Feishu webhook."""
        timestamp = str(int(time_module.time()))
        string_to_sign = f"{timestamp}\n{self.secret}"
        hmac_code = hmac.new(
            string_to_sign.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        sign = base64.b64encode(hmac_code).decode("utf-8")
        return timestamp, sign

    def send(self, scored_articles: list[ScoredArticle], date_str: str = ""):
        """
        Send a formatted message card with article summaries.

        Args:
            scored_articles: Articles to include (already filtered and sorted).
            date_str: Date string for the report title.
        """
        if not scored_articles:
            logger.info("No articles to notify. Skipping Feishu notification.")
            return

        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        # Build markdown content for the card
        md_lines = [
            f"共筛选出 **{len(scored_articles)}** 篇相关文献\n",
            "---\n",
        ]

        for i, sa in enumerate(scored_articles, 1):
            a = sa.article
            authors_str = ", ".join(a.authors[:3])
            if len(a.authors) > 3:
                authors_str += " et al."

            md_lines.append(f"**{i}. [{a.title}]({a.url})**")
            md_lines.append(f"*{a.journal}* | {a.pub_date} | PMID: {a.pmid}")
            if authors_str:
                md_lines.append(f"作者: {authors_str}")
            md_lines.append(f"相关度: **{sa.score}/10** — {sa.reason}")
            md_lines.append("")  # blank line separator

        content = "\n".join(md_lines)

        # Feishu interactive card message
        payload = {
            "msg_type": "interactive",
            "card": {
                "schema": "2.0",
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"📚 PubMed 文献日报 | {date_str}",
                    },
                    "template": "blue",
                },
                "body": {
                    "direction": "vertical",
                    "padding": "12px 12px 12px 12px",
                    "elements": [
                        {
                            "tag": "markdown",
                            "content": content[:18000],  # Stay under 20KB limit
                            "text_align": "left",
                            "text_size": "normal_v2",
                        }
                    ],
                },
            },
        }

        # Add signature if secret is configured
        if self.secret:
            timestamp, sign = self._sign()
            payload["timestamp"] = timestamp
            payload["sign"] = sign

        try:
            resp = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=15,
            )
            resp.raise_for_status()
            result = resp.json()
            if result.get("code", 0) != 0:
                logger.error(f"Feishu API error: {result}")
            else:
                logger.info(f"Feishu notification sent: {len(scored_articles)} articles.")
        except Exception as e:
            logger.error(f"Failed to send Feishu notification: {e}", exc_info=True)


class WeComNotifier:
    """Send notifications to a WeChat Work (企业微信) group chat via webhook."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, scored_articles: list[ScoredArticle], date_str: str = ""):
        """
        Send a markdown message with article summaries.

        Args:
            scored_articles: Articles to include (already filtered and sorted).
            date_str: Date string for the report title.
        """
        if not scored_articles:
            logger.info("No articles to notify. Skipping WeCom notification.")
            return

        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        # WeCom markdown has a 4096 byte limit.
        # We build the message and truncate if needed.
        lines = [
            f"# PubMed 文献日报 | {date_str}",
            f"> 共筛选出 <font color=\"info\">{len(scored_articles)}</font> 篇相关文献\n",
        ]

        for i, sa in enumerate(scored_articles, 1):
            a = sa.article
            authors_str = ", ".join(a.authors[:2])
            if len(a.authors) > 2:
                authors_str += " et al."

            lines.append(f"**{i}. [{a.title}]({a.url})**")
            lines.append(f"> {a.journal} | 相关度: **{sa.score}/10**")
            lines.append(f"> {sa.reason}")
            lines.append("")

        content = "\n".join(lines)

        # WeCom markdown limit is 4096 bytes
        content_bytes = content.encode("utf-8")
        if len(content_bytes) > 4000:
            # Truncate and add overflow note
            content = content_bytes[:3900].decode("utf-8", errors="ignore")
            content += f"\n\n> ... 更多文献请查看完整报告"

        payload = {
            "msgtype": "markdown",
            "markdown": {"content": content},
        }

        try:
            resp = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=15,
            )
            resp.raise_for_status()
            result = resp.json()
            if result.get("errcode", 0) != 0:
                logger.error(f"WeCom API error: {result}")
            else:
                logger.info(f"WeCom notification sent: {len(scored_articles)} articles.")
        except Exception as e:
            logger.error(f"Failed to send WeCom notification: {e}", exc_info=True)


def create_notifier(config: dict):
    """
    Factory function to create the appropriate notifier(s) from config.

    Returns:
        A list of notifier instances.
    """
    platform = config.get("platform", "feishu")
    notifiers = []

    if platform in ("feishu", "both"):
        feishu_cfg = config.get("feishu", {})
        url = feishu_cfg.get("webhook_url", "")
        if url:
            notifiers.append(FeishuNotifier(
                webhook_url=url,
                secret=feishu_cfg.get("secret", ""),
            ))
        else:
            logger.warning("Feishu webhook_url not configured.")

    if platform in ("wecom", "both"):
        wecom_cfg = config.get("wecom", {})
        url = wecom_cfg.get("webhook_url", "")
        if url:
            notifiers.append(WeComNotifier(webhook_url=url))
        else:
            logger.warning("WeCom webhook_url not configured.")

    if not notifiers:
        logger.error("No notification channel configured!")

    return notifiers
