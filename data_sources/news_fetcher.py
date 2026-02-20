"""Financial news fetching from public RSS feeds.

Fetches recent financial/economic news articles from multiple sources
using RSS feeds — no API keys required.
"""

import time
from datetime import datetime, timedelta
from typing import Optional

import feedparser
import requests

# Financial news RSS feed URLs
RSS_FEEDS = {
    "Google Finance": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB",
    "Google Business": "https://news.google.com/rss/topics/CAAqKggKIiRDQkFTRlFvSUwyMHZNRGx6TVdZU0JXVnVMVWRDR2dKVlV5Z0FQAQ",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "CNBC Top News": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "CNBC Finance": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "Reuters Business": "https://news.google.com/rss/search?q=reuters+business+finance&hl=en-US&gl=US&ceid=US:en",
    "Investing.com": "https://news.google.com/rss/search?q=investing+stock+market&hl=en-US&gl=US&ceid=US:en",
    "Bloomberg via Google": "https://news.google.com/rss/search?q=bloomberg+financial+markets&hl=en-US&gl=US&ceid=US:en",
    "WSJ via Google": "https://news.google.com/rss/search?q=wall+street+journal+economy&hl=en-US&gl=US&ceid=US:en",
}

# Request timeout and headers
REQUEST_TIMEOUT = 15
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _parse_date(entry) -> Optional[datetime]:
    """Extract publication date from a feed entry."""
    for attr in ("published_parsed", "updated_parsed"):
        parsed = getattr(entry, attr, None)
        if parsed:
            try:
                return datetime(*parsed[:6])
            except (TypeError, ValueError):
                continue
    return None


def _clean_html(text: str) -> str:
    """Remove HTML tags from text."""
    import re
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def fetch_news_from_feed(feed_url: str, source_name: str) -> list[dict]:
    """Fetch news articles from a single RSS feed.

    Returns list of dicts with keys: title, summary, url, source, published.
    """
    articles = []
    try:
        resp = requests.get(
            feed_url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)

        for entry in feed.entries:
            title = getattr(entry, "title", "").strip()
            if not title:
                continue

            summary = _clean_html(
                getattr(entry, "summary", "")
                or getattr(entry, "description", "")
            )
            url = getattr(entry, "link", "")
            published = _parse_date(entry)

            articles.append({
                "title": title,
                "summary": summary[:500] if summary else "",
                "url": url,
                "source": source_name,
                "published": published,
            })
    except Exception:
        # Silently skip feeds that fail (network issues, bad XML, etc.)
        pass

    return articles


def fetch_financial_news(
    n_articles: int = 100,
    progress_callback=None,
) -> list[dict]:
    """Fetch up to n_articles recent financial news from multiple RSS feeds.

    Parameters
    ----------
    n_articles : int
        Target number of articles to fetch (default 100).
    progress_callback : callable, optional
        Called with (current_count, total_feeds) for progress tracking.

    Returns
    -------
    list[dict]
        List of article dicts sorted by publication date (newest first).
        Each dict has keys: title, summary, url, source, published.
    """
    all_articles = []
    feed_items = list(RSS_FEEDS.items())
    total_feeds = len(feed_items)

    for idx, (name, url) in enumerate(feed_items):
        articles = fetch_news_from_feed(url, name)
        all_articles.extend(articles)

        if progress_callback:
            progress_callback(idx + 1, total_feeds)

        # Small delay to be polite to servers
        time.sleep(0.3)

        # Stop early if we already have more than enough
        if len(all_articles) >= n_articles * 2:
            break

    # Deduplicate by title (case-insensitive)
    seen_titles = set()
    unique = []
    for art in all_articles:
        key = art["title"].lower().strip()
        if key not in seen_titles:
            seen_titles.add(key)
            unique.append(art)

    # Sort by date (newest first), articles without date go last
    unique.sort(
        key=lambda a: a["published"] or datetime.min,
        reverse=True,
    )

    return unique[:n_articles]


def get_feed_sources() -> dict[str, str]:
    """Return the dict of available RSS feed sources."""
    return dict(RSS_FEEDS)
