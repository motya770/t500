"""Financial News Sentiment Analysis page.

Fetches ~100 recent financial news articles from public RSS feeds,
runs sentiment analysis (TextBlob + financial lexicon), and presents
an interactive dashboard with market signal advisory.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ui.theme import (
    apply_steam_style, CHART_COLORS, BRASS, COPPER, EMBER, CREAM,
    COAL, DARK_IRON, DARK_WOOD, STEEL, GOLD,
)


# Sentiment colors (themed to match steam palette)
_POS_COLOR = "#B87333"   # brass / copper for positive
_NEU_COLOR = "#DAA520"   # gold for neutral
_NEG_COLOR = "#8B2500"   # dark rust-red for negative


def render():
    """Render the News Sentiment Analysis page."""
    st.header("\U0001F4F0 Financial News Sentiment Analysis")
    st.markdown(
        "Analyze the latest financial news to gauge overall market sentiment "
        "and determine whether the news landscape leans **positive** or "
        "**negative** \u2014 and how useful it is as an advisory signal."
    )

    # --- Controls ---
    col1, col2 = st.columns([1, 1])
    with col1:
        n_articles = st.slider(
            "Number of articles to analyze",
            min_value=20,
            max_value=200,
            value=100,
            step=10,
        )
    with col2:
        st.markdown("")  # spacer
        run_btn = st.button("\U0001F682 Fetch & Analyze News", type="primary")

    if run_btn:
        _run_analysis(n_articles)

    # --- Display cached results if available ---
    if "news_sentiment_df" in st.session_state:
        _display_results(
            st.session_state["news_sentiment_df"],
            st.session_state["news_market_signal"],
        )


def _run_analysis(n_articles: int):
    """Fetch news and run sentiment analysis."""
    from data_sources.news_fetcher import fetch_financial_news
    from analysis.sentiment import analyze_articles, compute_market_signal

    # Fetch news with progress bar
    progress_bar = st.progress(0, text="Fetching financial news feeds...")

    def progress_cb(current, total):
        pct = current / total
        progress_bar.progress(pct, text=f"Fetching feed {current}/{total}...")

    with st.spinner("Fetching news from RSS feeds..."):
        articles = fetch_financial_news(
            n_articles=n_articles,
            progress_callback=progress_cb,
        )

    if not articles:
        progress_bar.empty()
        st.error(
            "Could not fetch any news articles. "
            "Please check your internet connection and try again."
        )
        return

    progress_bar.progress(0.8, text="Running sentiment analysis...")

    # Analyze sentiment
    df = analyze_articles(articles)
    signal = compute_market_signal(df)

    progress_bar.progress(1.0, text="Done!")
    progress_bar.empty()

    # Cache in session state
    st.session_state["news_sentiment_df"] = df
    st.session_state["news_market_signal"] = signal

    st.success(f"Analyzed **{len(df)}** financial news articles.")


def _display_results(df: pd.DataFrame, signal: dict):
    """Display the full sentiment analysis dashboard."""

    # --- 1. Market Signal Banner ---
    st.markdown("---")
    _render_signal_banner(signal)

    # --- 2. Sentiment Distribution ---
    st.markdown("---")
    st.subheader("Sentiment Distribution")
    _render_distribution(df, signal)

    # --- 3. Sentiment by Source ---
    st.markdown("---")
    st.subheader("Sentiment by News Source")
    _render_by_source(df)

    # --- 4. Score Distribution Histogram ---
    st.markdown("---")
    st.subheader("Sentiment Score Distribution")
    _render_histogram(df)

    # --- 5. Top Positive & Negative ---
    st.markdown("---")
    st.subheader("Most Positive vs Most Negative Articles")
    _render_top_articles(df)

    # --- 6. Full Article Table ---
    st.markdown("---")
    st.subheader("All Analyzed Articles")
    _render_article_table(df)

    # --- 7. Advisory Note ---
    st.markdown("---")
    st.subheader("Advisory Assessment")
    st.markdown(signal["advisory_note"])


def _render_signal_banner(signal: dict):
    """Render the main market signal banner with key metrics."""
    if signal["signal"] == "BULLISH":
        color = "green"
        icon = "arrow_up"
    elif signal["signal"] == "BEARISH":
        color = "red"
        icon = "arrow_down"
    else:
        color = "orange"
        icon = "left_right_arrow"

    st.markdown(
        f"### :{color}[Market Sentiment: {signal['signal']}] "
        f":{icon}: \u2014 {signal['signal_strength']} Signal"
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Avg Score", f"{signal['mean_score']:+.3f}")
    c2.metric("Median Score", f"{signal['median_score']:+.3f}")
    c3.metric("Std Dev", f"{signal['std_score']:.3f}")
    c4.metric("Positive", f"{signal['positive_pct']:.0f}%")
    c5.metric("Negative", f"{signal['negative_pct']:.0f}%")
    c6.metric("Articles", signal["total_articles"])


def _render_distribution(df: pd.DataFrame, signal: dict):
    """Render pie chart and bar chart of sentiment distribution."""
    col1, col2 = st.columns(2)

    with col1:
        labels = ["Positive", "Neutral", "Negative"]
        values = [
            signal["positive_count"],
            signal["neutral_count"],
            signal["negative_count"],
        ]
        colors = [_POS_COLOR, _NEU_COLOR, _NEG_COLOR]
        fig = go.Figure(
            data=[go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),
                hole=0.4,
                textinfo="label+percent+value",
                textfont=dict(color=CREAM),
            )]
        )
        fig.update_layout(
            title="Sentiment Breakdown",
            height=400,
        )
        apply_steam_style(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        color_map = {
            "Positive": _POS_COLOR,
            "Neutral": _NEU_COLOR,
            "Negative": _NEG_COLOR,
        }
        fig = px.bar(
            df,
            x=df.index,
            y="combined_score",
            color="label",
            color_discrete_map=color_map,
            labels={"combined_score": "Sentiment Score", "x": "Article #"},
            title="Per-Article Sentiment Scores",
        )
        fig.add_hline(y=0, line_dash="dash", line_color=STEEL)
        fig.update_layout(height=400, showlegend=True)
        apply_steam_style(fig)
        st.plotly_chart(fig, use_container_width=True)


def _render_by_source(df: pd.DataFrame):
    """Render sentiment breakdown by news source."""
    from analysis.sentiment import sentiment_by_source

    source_df = sentiment_by_source(df)
    if source_df.empty:
        st.info("No source data available.")
        return

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            source_df.reset_index(),
            x="source",
            y="Mean Score",
            color="Mean Score",
            color_continuous_scale=[_NEG_COLOR, _NEU_COLOR, _POS_COLOR],
            title="Average Sentiment by Source",
        )
        fig.add_hline(y=0, line_dash="dash", line_color=STEEL)
        fig.update_layout(height=400)
        apply_steam_style(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(source_df, use_container_width=True)


def _render_histogram(df: pd.DataFrame):
    """Render a histogram of sentiment scores."""
    fig = px.histogram(
        df,
        x="combined_score",
        nbins=30,
        color="label",
        color_discrete_map={
            "Positive": _POS_COLOR,
            "Neutral": _NEU_COLOR,
            "Negative": _NEG_COLOR,
        },
        marginal="box",
        title="Distribution of Sentiment Scores",
        labels={"combined_score": "Combined Sentiment Score"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color=STEEL)
    mean_val = df["combined_score"].mean()
    fig.add_vline(
        x=mean_val, line_dash="dot", line_color=BRASS,
        annotation_text=f"Mean: {mean_val:+.3f}",
        annotation_font_color=CREAM,
    )
    fig.update_layout(height=450)
    apply_steam_style(fig)
    st.plotly_chart(fig, use_container_width=True)


def _render_top_articles(df: pd.DataFrame):
    """Render top positive and negative articles side by side."""
    from analysis.sentiment import get_top_positive, get_top_negative

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top 10 Most Positive**")
        top_pos = get_top_positive(df, 10)
        for _, row in top_pos.iterrows():
            score = row["combined_score"]
            st.markdown(
                f"- **{score:+.3f}** | {row['title']} "
                f"*({row['source']})*"
            )

    with col2:
        st.markdown("**Top 10 Most Negative**")
        top_neg = get_top_negative(df, 10)
        for _, row in top_neg.iterrows():
            score = row["combined_score"]
            st.markdown(
                f"- **{score:+.3f}** | {row['title']} "
                f"*({row['source']})*"
            )


def _render_article_table(df: pd.DataFrame):
    """Render a sortable table of all analyzed articles."""
    display_df = df[[
        "title", "source", "combined_score", "label",
        "textblob_score", "lexicon_score", "published",
    ]].copy()

    display_df.columns = [
        "Title", "Source", "Score", "Sentiment",
        "TextBlob", "Lexicon", "Published",
    ]

    # Color-code the sentiment column
    st.dataframe(
        display_df.style.applymap(
            _color_sentiment, subset=["Sentiment"]
        ).format({
            "Score": "{:+.3f}",
            "TextBlob": "{:+.3f}",
            "Lexicon": "{:+.3f}",
        }),
        use_container_width=True,
        height=500,
    )


def _color_sentiment(val):
    """Return CSS color for sentiment label cells."""
    if val == "Positive":
        return f"color: {_POS_COLOR}; font-weight: bold"
    elif val == "Negative":
        return f"color: {_NEG_COLOR}; font-weight: bold"
    return f"color: {_NEU_COLOR}"
