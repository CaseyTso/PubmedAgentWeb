"""
PubMed E-utilities API Client

Searches PubMed for recent articles and fetches their metadata/abstracts.
Uses NCBI E-utilities: ESearch + EFetch pipeline with History server.
"""

import time
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


@dataclass
class Article:
    """Represents a PubMed article with key metadata."""
    pmid: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    pub_date: str = ""
    doi: str = ""
    keywords: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    @property
    def has_abstract(self) -> bool:
        return bool(self.abstract and self.abstract.strip())


class PubMedClient:
    """Client for NCBI PubMed E-utilities API."""

    def __init__(
        self,
        api_key: str = "",
        email: str = "",
        tool_name: str = "pubmed_monitor",
    ):
        self.api_key = api_key
        self.email = email
        self.tool_name = tool_name
        self.session = requests.Session()
        # Rate limit: 10 req/s with key, 3 req/s without
        self._delay = 0.11 if api_key else 0.35

    def _base_params(self) -> dict:
        """Common parameters for all NCBI requests."""
        params = {"email": self.email, "tool": self.tool_name}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _throttle(self):
        """Respect NCBI rate limits."""
        time.sleep(self._delay)

    def search_recent(
        self,
        query: str,
        days: int = 2,
        max_results: int = 500,
    ) -> tuple[int, str, str]:
        """
        Search PubMed for articles indexed within the last `days` days.

        Returns:
            (count, webenv, query_key) for use with fetch_articles().
        """
        params = {
            **self._base_params(),
            "db": "pubmed",
            "term": query,
            "reldate": days,
            "datetype": "edat",  # Entrez date = when indexed in PubMed
            "retmax": 0,         # Only need count + history
            "usehistory": "y",
            "retmode": "json",
            "sort": "pub_date",
        }

        logger.info(f"Searching PubMed: reldate={days}, query={query[:80]}...")
        resp = self.session.get(f"{BASE_URL}esearch.fcgi", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        result = data.get("esearchresult", {})
        count = int(result.get("count", 0))
        webenv = result.get("webenv", "")
        query_key = result.get("querykey", "")

        logger.info(f"PubMed search returned {count} results.")

        if count > max_results:
            logger.warning(
                f"Result count ({count}) exceeds max_results ({max_results}). "
                f"Only the first {max_results} will be fetched."
            )
            count = max_results

        return count, webenv, query_key

    def fetch_articles(
        self,
        count: int,
        webenv: str,
        query_key: str,
        batch_size: int = 200,
    ) -> list[Article]:
        """
        Fetch article details from PubMed using EFetch + History server.

        Args:
            count: Total number of records to fetch.
            webenv: WebEnv token from search_recent().
            query_key: QueryKey from search_recent().
            batch_size: Records per batch (recommended 200-500).

        Returns:
            List of Article objects with metadata and abstracts.
        """
        articles = []

        for start in range(0, count, batch_size):
            params = {
                **self._base_params(),
                "db": "pubmed",
                "query_key": query_key,
                "WebEnv": webenv,
                "retstart": start,
                "retmax": min(batch_size, count - start),
                "retmode": "xml",
            }

            logger.debug(f"Fetching batch: retstart={start}, retmax={params['retmax']}")
            self._throttle()
            resp = self.session.get(f"{BASE_URL}efetch.fcgi", params=params, timeout=60)
            resp.raise_for_status()

            batch = self._parse_xml(resp.text)
            articles.extend(batch)
            logger.info(f"Fetched {len(batch)} articles (total: {len(articles)}/{count})")

        return articles

    def search_and_fetch(
        self,
        query: str,
        days: int = 2,
        batch_size: int = 200,
        max_results: int = 500,
    ) -> list[Article]:
        """
        Convenience method: search + fetch in one call.

        Returns:
            List of Article objects.
        """
        count, webenv, query_key = self.search_recent(query, days, max_results)
        if count == 0:
            logger.info("No new articles found.")
            return []
        return self.fetch_articles(count, webenv, query_key, batch_size)

    @staticmethod
    def _parse_xml(xml_text: str) -> list[Article]:
        """Parse PubMed XML response into Article objects."""
        articles = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"Failed to parse PubMed XML: {e}")
            return articles

        for pub_article in root.findall(".//PubmedArticle"):
            try:
                citation = pub_article.find("MedlineCitation")
                if citation is None:
                    continue

                # PMID
                pmid = citation.findtext("PMID", default="").strip()
                if not pmid:
                    continue

                article_el = citation.find("Article")
                if article_el is None:
                    continue

                # Title
                title = article_el.findtext("ArticleTitle", default="").strip()

                # Abstract — may have multiple AbstractText elements (structured abstract)
                abstract_parts = []
                abstract_el = article_el.find("Abstract")
                if abstract_el is not None:
                    for abs_text in abstract_el.findall("AbstractText"):
                        label = abs_text.get("Label", "")
                        # Get all text content including nested elements
                        text = "".join(abs_text.itertext()).strip()
                        if label:
                            abstract_parts.append(f"{label}: {text}")
                        else:
                            abstract_parts.append(text)
                abstract = "\n".join(abstract_parts)

                # Authors
                authors = []
                author_list = article_el.find("AuthorList")
                if author_list is not None:
                    for author in author_list.findall("Author"):
                        last = author.findtext("LastName", default="")
                        fore = author.findtext("ForeName", default="")
                        if last:
                            authors.append(f"{last} {fore}".strip())

                # Journal
                journal_el = article_el.find("Journal")
                journal = ""
                if journal_el is not None:
                    journal = journal_el.findtext("Title", default="")

                # Publication date
                pub_date = ""
                if journal_el is not None:
                    pub_date_el = journal_el.find(".//PubDate")
                    if pub_date_el is not None:
                        year = pub_date_el.findtext("Year", default="")
                        month = pub_date_el.findtext("Month", default="")
                        day = pub_date_el.findtext("Day", default="")
                        pub_date = f"{year} {month} {day}".strip()

                # DOI
                doi = ""
                pubmed_data = pub_article.find("PubmedData")
                if pubmed_data is not None:
                    for aid in pubmed_data.findall(".//ArticleId"):
                        if aid.get("IdType") == "doi":
                            doi = (aid.text or "").strip()
                            break

                # MeSH / Keywords
                keywords = []
                mesh_list = citation.find("MeshHeadingList")
                if mesh_list is not None:
                    for mesh in mesh_list.findall("MeshHeading"):
                        desc = mesh.findtext("DescriptorName", default="").strip()
                        if desc:
                            keywords.append(desc)

                keyword_list = citation.find("KeywordList")
                if keyword_list is not None:
                    for kw in keyword_list.findall("Keyword"):
                        text = (kw.text or "").strip()
                        if text:
                            keywords.append(text)

                articles.append(Article(
                    pmid=pmid,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    journal=journal,
                    pub_date=pub_date,
                    doi=doi,
                    keywords=keywords,
                ))
            except Exception as e:
                logger.error(f"Error parsing article: {e}", exc_info=True)

        return articles
