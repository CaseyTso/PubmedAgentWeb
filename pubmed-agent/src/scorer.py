"""
LLM-based Relevance Scorer

Uses an LLM (OpenAI-compatible API) to read article abstracts and score
their relevance to the user's research profile.
"""

import json
import logging
from dataclasses import dataclass

from openai import OpenAI

from .pubmed_client import Article

logger = logging.getLogger(__name__)

SCORING_SYSTEM_PROMPT = """\
你是一位生物医学研究助手。你的任务是评估一篇科学论文与研究者特定兴趣的相关程度。

你将收到：
1. 研究者的研究方向描述
2. 一篇论文的标题、摘要和关键词

请按 0-10 分评估论文的相关度：
- 0: 完全无关
- 1-3: 边缘相关（提到了部分关键词，但研究方向不同）
- 4-5: 中等相关（主题有重叠，但不直接有用）
- 6-7: 高度相关（直接涉及研究方向）
- 8-9: 非常相关（核心主题，可能包含有价值的发现）
- 10: 极度相关（直接回答关键研究问题）

请务必用中文回复。仅返回以下格式的 JSON，不要输出任何其他内容：
{
  "score": <0-10的整数>,
  "reason": "<1-2句中文解释>"
}
"""

SCORING_USER_PROMPT = """\
## Researcher's Profile
{research_description}

## Article to Evaluate
**Title:** {title}
**Journal:** {journal}
**Keywords:** {keywords}

**Abstract:**
{abstract}
"""


@dataclass
class ScoredArticle:
    """An article with its LLM-assigned relevance score."""
    article: Article
    score: int
    reason: str


class RelevanceScorer:
    """Scores articles for relevance using an LLM."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ):
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def score_article(
        self,
        article: Article,
        research_description: str,
    ) -> ScoredArticle:
        """
        Score a single article's relevance to the research profile.

        Args:
            article: The article to score.
            research_description: Natural language description of research interests.

        Returns:
            ScoredArticle with score and reasoning.
        """
        if not article.has_abstract:
            logger.debug(f"PMID {article.pmid}: No abstract, skipping LLM scoring.")
            return ScoredArticle(article=article, score=0, reason="No abstract available.")

        user_msg = SCORING_USER_PROMPT.format(
            research_description=research_description,
            title=article.title,
            journal=article.journal,
            keywords=", ".join(article.keywords) if article.keywords else "N/A",
            abstract=article.abstract[:3000],  # Truncate very long abstracts
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content.strip()
            # Parse JSON response
            # Handle potential markdown code fences
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            result = json.loads(content)
            score = int(result.get("score", 0))
            score = max(0, min(10, score))  # Clamp to 0-10
            reason = result.get("reason", "")

            logger.debug(f"PMID {article.pmid}: score={score}, reason={reason}")
            return ScoredArticle(article=article, score=score, reason=reason)

        except json.JSONDecodeError as e:
            logger.warning(f"PMID {article.pmid}: Failed to parse LLM response: {e}")
            return ScoredArticle(article=article, score=0, reason="LLM response parsing error.")
        except Exception as e:
            logger.error(f"PMID {article.pmid}: LLM scoring error: {e}", exc_info=True)
            return ScoredArticle(article=article, score=0, reason=f"Scoring error: {e}")

    def score_batch(
        self,
        articles: list[Article],
        research_description: str,
        min_score: int = 0,
    ) -> list[ScoredArticle]:
        """
        Score a batch of articles and optionally filter by minimum score.

        Args:
            articles: List of articles to score.
            research_description: Research interest description.
            min_score: Minimum score to include in results (0 = include all).

        Returns:
            List of ScoredArticle, sorted by score (descending).
        """
        logger.info(f"Scoring {len(articles)} articles with LLM ({self.model})...")

        scored = []
        for i, article in enumerate(articles):
            result = self.score_article(article, research_description)
            scored.append(result)
            if (i + 1) % 10 == 0:
                logger.info(f"Scored {i + 1}/{len(articles)} articles.")

        # Filter and sort
        if min_score > 0:
            scored = [s for s in scored if s.score >= min_score]
            logger.info(
                f"{len(scored)} articles passed the minimum score threshold ({min_score})."
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored
