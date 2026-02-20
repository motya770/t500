"""Financial news sentiment analysis.

Combines TextBlob general-purpose sentiment with a domain-specific
financial lexicon to produce sentiment scores tuned for market news.
All functions are stateless and operate on lists/DataFrames.
"""

import re
from collections import Counter

import numpy as np
import pandas as pd
from textblob import TextBlob

# ---------------------------------------------------------------------------
# Financial sentiment lexicon — words and their sentiment weight.
# Positive words score > 0, negative words score < 0.
# Weights are in the range [-1, 1].
# ---------------------------------------------------------------------------
_POSITIVE_TERMS = {
    # Strong positive
    "surge": 0.8, "soar": 0.8, "rally": 0.8, "boom": 0.8,
    "breakout": 0.7, "skyrocket": 0.9, "outperform": 0.7,
    # Moderate positive
    "gain": 0.5, "rise": 0.5, "climb": 0.5, "advance": 0.5,
    "growth": 0.6, "grow": 0.5, "profit": 0.6, "profitable": 0.6,
    "upgrade": 0.6, "upbeat": 0.6, "optimism": 0.6, "optimistic": 0.6,
    "bullish": 0.7, "bull": 0.5, "recovery": 0.6, "recover": 0.6,
    "strong": 0.4, "strength": 0.4, "positive": 0.4,
    "beat": 0.5, "exceed": 0.5, "record": 0.4, "high": 0.3,
    "boost": 0.5, "expand": 0.5, "expansion": 0.5,
    "dividend": 0.4, "buyback": 0.4, "innovation": 0.4,
    "breakthrough": 0.6, "opportunity": 0.4, "upside": 0.5,
    "success": 0.5, "successful": 0.5, "momentum": 0.4,
    "robust": 0.5, "resilient": 0.5, "accelerate": 0.5,
    "improve": 0.4, "improvement": 0.4, "upturn": 0.6,
    "earnings": 0.3, "revenue": 0.3, "outpace": 0.5,
}

_NEGATIVE_TERMS = {
    # Strong negative
    "crash": -0.9, "collapse": -0.9, "plunge": -0.8, "plummet": -0.8,
    "crisis": -0.8, "recession": -0.8, "depression": -0.7,
    "bankruptcy": -0.9, "bankrupt": -0.9, "default": -0.7,
    # Moderate negative
    "decline": -0.5, "drop": -0.5, "fall": -0.5, "slump": -0.6,
    "loss": -0.5, "lose": -0.5, "losses": -0.5,
    "bearish": -0.7, "bear": -0.4, "downturn": -0.6, "downgrade": -0.6,
    "selloff": -0.7, "sell-off": -0.7,
    "volatility": -0.3, "volatile": -0.3, "uncertainty": -0.4,
    "weak": -0.4, "weakness": -0.4, "negative": -0.4,
    "risk": -0.3, "risky": -0.4, "fear": -0.5, "fears": -0.5,
    "inflation": -0.3, "tariff": -0.4, "tariffs": -0.4,
    "sanction": -0.4, "sanctions": -0.4,
    "debt": -0.3, "deficit": -0.4, "layoff": -0.6, "layoffs": -0.6,
    "cut": -0.3, "cuts": -0.3, "slash": -0.5,
    "miss": -0.5, "missed": -0.5, "disappoint": -0.5,
    "warning": -0.5, "warn": -0.4, "concern": -0.3, "concerns": -0.3,
    "struggle": -0.4, "stagnation": -0.5, "stagnant": -0.5,
    "turmoil": -0.6, "panic": -0.7, "bubble": -0.4,
    "overvalued": -0.5, "downside": -0.5, "trouble": -0.4,
    "contraction": -0.5, "slowdown": -0.5, "slow": -0.3,
}

_FINANCIAL_LEXICON = {**_POSITIVE_TERMS, **_NEGATIVE_TERMS}


def _tokenize(text: str) -> list[str]:
    """Simple lowercase word tokenization."""
    return re.findall(r"[a-z]+(?:[-'][a-z]+)*", text.lower())


def _financial_lexicon_score(text: str) -> float:
    """Score text using the financial lexicon.

    Returns a score in [-1, 1] — the mean weight of all matched words.
    Returns 0 if no lexicon words are found.
    """
    tokens = _tokenize(text)
    matched_scores = [
        _FINANCIAL_LEXICON[tok] for tok in tokens if tok in _FINANCIAL_LEXICON
    ]
    if not matched_scores:
        return 0.0
    return float(np.clip(np.mean(matched_scores), -1.0, 1.0))


def _textblob_score(text: str) -> float:
    """Return TextBlob polarity in [-1, 1]."""
    return TextBlob(text).sentiment.polarity


def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of a single piece of text.

    Combines TextBlob polarity (general NLP) with a financial lexicon
    score (domain-specific). The final score is a weighted blend:
        final = 0.4 * textblob + 0.6 * lexicon
    giving more weight to domain-specific vocabulary.

    Returns dict with:
        textblob_score, lexicon_score, combined_score, label
    """
    tb = _textblob_score(text)
    lx = _financial_lexicon_score(text)
    combined = 0.4 * tb + 0.6 * lx

    if combined > 0.05:
        label = "Positive"
    elif combined < -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "textblob_score": round(tb, 4),
        "lexicon_score": round(lx, 4),
        "combined_score": round(combined, 4),
        "label": label,
    }


def analyze_articles(articles: list[dict]) -> pd.DataFrame:
    """Run sentiment analysis on a list of news articles.

    Each article dict should have at least 'title' and 'summary' keys.
    The analysis combines title and summary text.

    Returns a DataFrame with columns:
        title, summary, source, published, url,
        textblob_score, lexicon_score, combined_score, label
    """
    rows = []
    for art in articles:
        text = (art.get("title", "") + ". " + art.get("summary", "")).strip()
        scores = analyze_sentiment(text)
        rows.append({
            "title": art.get("title", ""),
            "summary": art.get("summary", "")[:300],
            "source": art.get("source", ""),
            "published": art.get("published"),
            "url": art.get("url", ""),
            **scores,
        })

    df = pd.DataFrame(rows)
    return df


def compute_market_signal(df: pd.DataFrame) -> dict:
    """Compute an aggregate market sentiment signal from analyzed articles.

    Parameters
    ----------
    df : pd.DataFrame
        Output of analyze_articles().

    Returns
    -------
    dict with keys:
        mean_score : float — average combined sentiment [-1, 1]
        median_score : float
        std_score : float — sentiment dispersion
        positive_pct : float — percentage of positive articles
        negative_pct : float — percentage of negative articles
        neutral_pct : float — percentage of neutral articles
        positive_count, negative_count, neutral_count : int
        total_articles : int
        signal : str — "BULLISH", "BEARISH", or "NEUTRAL"
        signal_strength : str — "Strong", "Moderate", or "Weak"
        advisory_note : str — human-readable interpretation
    """
    n = len(df)
    if n == 0:
        return {
            "mean_score": 0, "median_score": 0, "std_score": 0,
            "positive_pct": 0, "negative_pct": 0, "neutral_pct": 0,
            "positive_count": 0, "negative_count": 0, "neutral_count": 0,
            "total_articles": 0,
            "signal": "NEUTRAL", "signal_strength": "Weak",
            "advisory_note": "No articles to analyze.",
        }

    counts = Counter(df["label"])
    pos = counts.get("Positive", 0)
    neg = counts.get("Negative", 0)
    neu = counts.get("Neutral", 0)

    mean_s = float(df["combined_score"].mean())
    median_s = float(df["combined_score"].median())
    std_s = float(df["combined_score"].std())

    pos_pct = pos / n * 100
    neg_pct = neg / n * 100
    neu_pct = neu / n * 100

    # Determine signal
    abs_mean = abs(mean_s)
    if mean_s > 0.05:
        signal = "BULLISH"
    elif mean_s < -0.05:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    if abs_mean > 0.25:
        strength = "Strong"
    elif abs_mean > 0.10:
        strength = "Moderate"
    else:
        strength = "Weak"

    # Generate advisory note
    advisory = _generate_advisory(
        signal, strength, mean_s, pos_pct, neg_pct, std_s, n
    )

    return {
        "mean_score": round(mean_s, 4),
        "median_score": round(median_s, 4),
        "std_score": round(std_s, 4),
        "positive_pct": round(pos_pct, 1),
        "negative_pct": round(neg_pct, 1),
        "neutral_pct": round(neu_pct, 1),
        "positive_count": pos,
        "negative_count": neg,
        "neutral_count": neu,
        "total_articles": n,
        "signal": signal,
        "signal_strength": strength,
        "advisory_note": advisory,
    }


def _generate_advisory(
    signal: str,
    strength: str,
    mean_score: float,
    pos_pct: float,
    neg_pct: float,
    std_score: float,
    n_articles: int,
) -> str:
    """Generate a human-readable advisory interpretation."""
    parts = []

    parts.append(
        f"Based on analysis of {n_articles} recent financial news articles, "
        f"the overall market sentiment is **{signal}** "
        f"with **{strength.lower()}** conviction "
        f"(avg score: {mean_score:+.3f})."
    )

    parts.append(
        f"{pos_pct:.0f}% of articles are positive, "
        f"{neg_pct:.0f}% negative, "
        f"and {100 - pos_pct - neg_pct:.0f}% neutral."
    )

    if std_score > 0.3:
        parts.append(
            "Sentiment is highly dispersed — news sources disagree "
            "significantly, suggesting elevated uncertainty."
        )
    elif std_score > 0.15:
        parts.append(
            "There is moderate variance in sentiment across sources."
        )
    else:
        parts.append(
            "Sentiment is fairly consistent across sources."
        )

    # Advisory usability assessment
    if n_articles < 20:
        parts.append(
            "**Advisory reliability: LOW** — too few articles for a "
            "confident signal. Not recommended as a sole advisory input."
        )
    elif std_score > 0.35:
        parts.append(
            "**Advisory reliability: LOW** — high disagreement among "
            "sources reduces signal confidence. Use with caution."
        )
    elif strength == "Weak":
        parts.append(
            "**Advisory reliability: LOW** — the sentiment signal is too "
            "weak to serve as a meaningful advisory indicator."
        )
    elif strength == "Moderate":
        parts.append(
            "**Advisory reliability: MODERATE** — the sentiment leans "
            f"{'positive' if signal == 'BULLISH' else 'negative'} and "
            "can be used as one input among several in a broader analysis."
        )
    else:
        parts.append(
            "**Advisory reliability: MODERATE-HIGH** — there is a strong, "
            f"consistent {'positive' if signal == 'BULLISH' else 'negative'} "
            "sentiment across sources. This can supplement (but not replace) "
            "fundamental and technical analysis."
        )

    parts.append(
        "*Disclaimer: News sentiment is one factor among many. "
        "It should never be the sole basis for financial decisions.*"
    )

    return "\n\n".join(parts)


def get_top_positive(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the top-N most positive articles."""
    return (
        df.nlargest(n, "combined_score")[
            ["title", "source", "combined_score", "label"]
        ]
        .reset_index(drop=True)
    )


def get_top_negative(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the top-N most negative articles."""
    return (
        df.nsmallest(n, "combined_score")[
            ["title", "source", "combined_score", "label"]
        ]
        .reset_index(drop=True)
    )


def sentiment_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment statistics by news source."""
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby("source")["combined_score"].agg(
        ["mean", "median", "std", "count"]
    ).round(4)
    grouped.columns = ["Mean Score", "Median Score", "Std Dev", "Articles"]
    grouped = grouped.sort_values("Mean Score", ascending=False)
    return grouped
