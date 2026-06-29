"""Textual sentiment ingestion — NewsAPI headlines + multi-source RSS feeds.

Reddit has been removed due to API policy instability.
Mention velocity is now derived from RSS hit counts across major business
news wires and the per-ticker Yahoo Finance headline feed.

Rate limits
-----------
NewsAPI free tier: 100 req/day (no hard per-minute cap).
Each page fetch = 1 call. Pages per ticker are capped dynamically so the
total run stays within the daily budget regardless of ticker count.

RSS feeds (feedparser) use only HTTP GET — no API key or quota.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import feedparser
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ingestion.rate_limits import remaining, throttle_and_check

logger = logging.getLogger(__name__)

PROCESSED_PATH = Path("data/processed/sentiment.parquet")

# Yahoo Finance emits a pre-filtered RSS feed per ticker — highest signal source.
YAHOO_RSS_TEMPLATE = "https://finance.yahoo.com/rss/headline?s={ticker}"

# Broad business/wire feeds; entries are filtered by company name + layoff keyword.
BROAD_RSS_FEEDS: dict[str, str] = {
    "reuters_business":     "https://feeds.reuters.com/reuters/businessNews",
    "cnbc_top_news":        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "pr_newswire":          "https://www.prnewswire.com/rss/news-releases-list.rss",
    "businesswire":         "https://feed.businesswire.com/rss/home/?rss=G1&rssid=20",
    "globenewswire_employ": "https://www.globenewswire.com/RssFeed/subjectcode/17-Employment",
}

LAYOFF_KEYWORDS: frozenset[str] = frozenset([
    "layoff", "layoffs", "retrenchment", "restructuring", "downsizing",
    "workforce reduction", "job cut", "job cuts", "headcount reduction",
    "redundanc", "severance", "mass termination", "reduction in force", "rif",
])

_FINBERT_MAP = {"positive": 1, "negative": -1, "neutral": 0}


# ── Scorer factory ────────────────────────────────────────────────────────────

def _build_scorer():
    """Return a callable(texts: list[str]) -> list[int] that scores sentiment.

    Load order:
    1. FinBERT via HuggingFace transformers (high accuracy, requires download)
    2. VADER via vaderSentiment (rule-based, no download, always available)
    3. No-op fallback that returns an empty list (logs a warning)

    The returned callable maps each text to +1 (positive), -1 (negative), or
    0 (neutral), matching the scale used throughout the rest of the pipeline.
    """
    # Option 1: FinBERT
    try:
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True)
        logger.info("Sentiment scorer: FinBERT (ProsusAI/finbert)")

        def _finbert(texts: list[str]) -> list[int]:
            out = []
            for text in texts:
                if not text:
                    continue
                try:
                    result = pipe(text[:512])[0]
                    out.append(_FINBERT_MAP.get(result["label"].lower(), 0))
                except Exception:
                    pass
            return out

        return _finbert
    except Exception as exc:
        logger.info(f"FinBERT not available ({exc}) -- using VADER fallback")

    # Option 2: VADER
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
        logger.info("Sentiment scorer: VADER (vaderSentiment)")

        def _vader_score(texts: list[str]) -> list[int]:
            out = []
            for text in texts:
                if not text:
                    continue
                try:
                    c = _vader.polarity_scores(text)["compound"]
                    out.append(1 if c >= 0.05 else (-1 if c <= -0.05 else 0))
                except Exception:
                    pass
            return out

        return _vader_score
    except Exception as exc:
        logger.warning(f"VADER not available ({exc}) -- sentiment scoring disabled")
        return lambda texts: []


# ── Internal helpers ──────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=120))
def _fetch_news_page(query: str, from_date: str, api_key: str, page: int = 1) -> dict:
    """Pure HTTP call — throttle_and_check must be called by the caller."""
    from newsapi import NewsApiClient

    client = NewsApiClient(api_key=api_key)
    return client.get_everything(
        q=query,
        from_param=from_date,
        language="en",
        sort_by="relevancy",
        page=page,
        page_size=100,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=3, max=60))
def _parse_feed(url: str, timeout: int = 20) -> feedparser.FeedParserDict:
    resp = requests.get(
        url,
        headers={"User-Agent": "LayoffMonitorBot/2.0 (research pipeline; not for scraping)"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return feedparser.parse(resp.content)


def _entry_text(entry: dict) -> str:
    return f"{entry.get('title', '')} {entry.get('summary', '')}".strip()


def _contains_layoff_keyword(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in LAYOFF_KEYWORDS)


def _score_texts(texts: list[str], scorer) -> list[int]:
    return scorer(texts)


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_news_headlines(
    tickers: list[str],
    company_names: dict[str, str],
    lookback_days: int = 7,
    api_key: str | None = None,
    max_pages: int = 3,
) -> pd.DataFrame:
    """Pull NewsAPI headlines per ticker and compute a daily FinBERT sentiment score.

    Page count per ticker is determined dynamically:
        allowed_pages = min(max_pages, remaining_budget // n_tickers)
    This ensures the entire ticker list is covered before the budget runs out,
    with extra pages added only when there is comfortable headroom.

    Returns columns: ticker, date, sentiment_score, article_count
    """
    key = api_key or os.getenv("NEWS_API_KEY", "")
    from_date = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    scorer = _build_scorer()

    distress_terms = (
        "layoff OR retrenchment OR restructuring OR \"job cuts\" OR downsizing "
        "OR \"workforce reduction\" OR severance OR \"reduction in force\""
    )

    # ── Compute per-ticker page budget ────────────────────────────────────────
    # Reserve at least 1 call per ticker; add extra pages if budget allows.
    budget = remaining("news_api")
    if budget < len(tickers):
        raise RuntimeError(
            f"[sentiment] NewsAPI budget too low ({budget} calls left) "
            f"to cover {len(tickers)} tickers even at 1 page each."
        )
    pages_per_ticker = min(max_pages, budget // len(tickers))
    logger.info(
        f"[sentiment] NewsAPI budget: {budget} calls remaining — "
        f"using {pages_per_ticker} page(s) per ticker across {len(tickers)} tickers "
        f"({pages_per_ticker * len(tickers)} calls total)"
    )

    records: list[dict] = []
    for ticker in tickers:
        name = company_names.get(ticker, ticker)
        query = f'("{name}" OR "{ticker}") AND ({distress_terms})'
        all_articles: list[dict] = []

        for page in range(1, pages_per_ticker + 1):
            try:
                throttle_and_check("news_api")
            except RuntimeError as exc:
                logger.warning(str(exc))
                break  # budget gone mid-run; stop pagination

            try:
                resp = _fetch_news_page(query=query, from_date=from_date, api_key=key, page=page)
                batch = resp.get("articles", [])
                all_articles.extend(batch)
                if len(batch) < 100:
                    break  # last page — no point requesting more
            except Exception as exc:
                logger.warning(f"NewsAPI [{ticker}] page {page}: {exc}")
                break

        texts = [
            f"{a.get('title', '')} {a.get('description', '')}".strip()
            for a in all_articles
        ]
        scores = _score_texts(texts, scorer)
        records.append({
            "ticker": ticker,
            "date": datetime.utcnow().date(),
            "sentiment_score": sum(scores) / len(scores) if scores else 0.0,
            "article_count": len(all_articles),
        })
        logger.info(
            f"NewsAPI [{ticker}]: {len(all_articles)} articles, "
            f"score={records[-1]['sentiment_score']:.3f}"
        )

    return pd.DataFrame(records)


def fetch_rss_headlines(
    tickers: list[str],
    company_names: dict[str, str],
    lookback_days: int = 7,
) -> pd.DataFrame:
    """Aggregate mention velocity and sentiment from RSS feeds (no API quota).

    Two-pass strategy:
      1. Per-ticker Yahoo Finance RSS — pre-filtered, high precision.
      2. Broad business/wire feeds (Reuters, CNBC, PR Newswire, BusinessWire,
         GlobeNewswire) — fetched once then filtered by company name AND a
         layoff keyword. Catches press releases that NewsAPI may miss.

    `mention_velocity` counts distress-keyword RSS entries per company.

    Returns columns: ticker, date, rss_sentiment_score, mention_velocity, rss_article_count
    """
    scorer = _build_scorer()
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)

    # Pre-fetch broad feeds once (avoid re-downloading per ticker)
    broad_entries: list[dict] = []
    for feed_name, url in BROAD_RSS_FEEDS.items():
        try:
            parsed = _parse_feed(url)
            broad_entries.extend(parsed.get("entries", []))
            logger.debug(f"RSS [{feed_name}]: {len(parsed.get('entries', []))} entries fetched")
        except Exception as exc:
            logger.warning(f"RSS [{feed_name}]: failed — {exc}")

    records: list[dict] = []
    for ticker in tickers:
        name = company_names.get(ticker, ticker).lower()
        name_tokens = set(name.split())
        all_texts: list[str] = []

        # ── Pass 1: per-ticker Yahoo Finance RSS ──────────────────────────────
        try:
            yf_feed = _parse_feed(YAHOO_RSS_TEMPLATE.format(ticker=ticker))
            for entry in yf_feed.get("entries", []):
                published = entry.get("published_parsed")
                if published and datetime(*published[:6]) < cutoff:
                    continue
                text = _entry_text(entry)
                if _contains_layoff_keyword(text):
                    all_texts.append(text)
        except Exception as exc:
            logger.warning(f"RSS Yahoo [{ticker}]: {exc}")

        # ── Pass 2: filter broad feeds by company name + layoff keyword ───────
        for entry in broad_entries:
            published = entry.get("published_parsed")
            if published and datetime(*published[:6]) < cutoff:
                continue
            text = _entry_text(entry)
            text_lower = text.lower()
            if any(tok in text_lower for tok in name_tokens) and _contains_layoff_keyword(text):
                all_texts.append(text)

        scores = _score_texts(all_texts, scorer)
        records.append({
            "ticker": ticker,
            "date": datetime.utcnow().date(),
            "rss_sentiment_score": sum(scores) / len(scores) if scores else 0.0,
            "mention_velocity": len(all_texts),
            "rss_article_count": len(all_texts),
        })
        logger.info(
            f"RSS [{ticker}]: {len(all_texts)} distress entries, "
            f"score={records[-1]['rss_sentiment_score']:.3f}"
        )

    return pd.DataFrame(records)


def persist_to_parquet(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    if len(df) < before:
        logger.info(f"Sentiment dedup: dropped {before - len(df)} duplicate (ticker, date) rows")
    df.to_parquet(path, index=False, compression="snappy")
    logger.info(f"Sentiment → {path} ({df.shape})")
