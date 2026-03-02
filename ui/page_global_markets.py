"""Global Financial Markets download page.

FX pairs, commodities, and shipping indexes via yfinance.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from data_sources.global_markets import (
    FX_CATEGORIES,
    COMMODITY_CATEGORIES,
    SHIPPING_TICKERS,
    ALL_CATEGORIES,
    get_all_tickers,
    download_global_market_data,
    save_global_market_dataset,
)
from ui.theme import apply_steam_style, CHART_COLORS, BRASS


def render():
    st.header("Global Financial Markets")
    st.write(
        "Download FX pairs, commodity futures, and shipping indexes. "
        "Data sourced from Yahoo Finance."
    )

    all_tickers = get_all_tickers()

    # --- Download ALL shortcut ---
    st.subheader("Quick Download")
    st.write(
        f"Download **all {len(all_tickers)} tickers** at once (FX, commodities, shipping), "
        "or pick individual tickers below."
    )

    if st.button(
        f"Download ALL Global Markets ({len(all_tickers)} tickers)",
        type="primary",
        use_container_width=True,
        key="gm_download_all",
    ):
        _run_download_all(
            tickers=list(all_tickers.keys()),
            ticker_names=all_tickers,
            dataset_name="global_markets_all",
            start_year=2000,
            end_year=2025,
            interval="1mo",
        )
        return

    # --- Manual selection via tabs ---
    st.divider()
    st.subheader("Or Select Individual Tickers")

    tab_fx, tab_commodities, tab_shipping, tab_custom = st.tabs([
        "Forex",
        "Commodities",
        "Shipping",
        "Custom Tickers",
    ])

    selected_tickers = []
    ticker_names = {}

    with tab_fx:
        t, n = _render_category_selection(FX_CATEGORIES, "fx")
        selected_tickers.extend(t)
        ticker_names.update(n)

    with tab_commodities:
        t, n = _render_category_selection(COMMODITY_CATEGORIES, "cmd")
        selected_tickers.extend(t)
        ticker_names.update(n)

    with tab_shipping:
        st.caption("Baltic Dry Index and related shipping ETFs.")
        for ticker, name in SHIPPING_TICKERS.items():
            checked = st.checkbox(
                f"{ticker} - {name}",
                value=False,
                key=f"ship_{ticker}",
            )
            if checked:
                selected_tickers.append(ticker)
                ticker_names[ticker] = name

    with tab_custom:
        custom = _render_custom_tickers()
        selected_tickers.extend(custom)
        ticker_names.update({t: t for t in custom})

    # Deduplicate
    seen = set()
    unique = []
    for t in selected_tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    selected_tickers = unique

    # Summary
    st.divider()
    if selected_tickers:
        display_limit = 20
        display = ", ".join(selected_tickers[:display_limit])
        if len(selected_tickers) > display_limit:
            display += f", ... (+{len(selected_tickers) - display_limit} more)"
        st.success(f"**{len(selected_tickers)}** tickers selected: {display}")
    else:
        st.info("Select tickers from any tab above to proceed.")

    # Date range & interval
    st.subheader("Date Range & Interval")
    col1, col2, col3 = st.columns(3)
    with col1:
        start_year = st.number_input(
            "Start year", min_value=1990, max_value=2026, value=2000,
            key="gm_start",
        )
    with col2:
        end_year = st.number_input(
            "End year", min_value=1990, max_value=2026, value=2025,
            key="gm_end",
        )
    with col3:
        interval = st.selectbox(
            "Data interval",
            ["1mo", "1wk", "1d"],
            format_func=lambda x: {"1mo": "Monthly", "1wk": "Weekly", "1d": "Daily"}[x],
            key="gm_interval",
        )

    if start_year > end_year:
        st.error("Start year must be before end year.")
        return

    # Dataset name
    st.subheader("Name Your Dataset")
    dataset_name = st.text_input(
        "Dataset name",
        value="global_markets",
        key="gm_dataset_name",
    )

    st.divider()

    if not selected_tickers:
        st.warning("Please select at least one ticker from the tabs above.")
        return

    if st.button(
        f"Download {len(selected_tickers)} Tickers",
        type="primary",
        use_container_width=True,
        key="gm_download",
    ):
        _run_download_all(
            tickers=selected_tickers,
            ticker_names=ticker_names,
            dataset_name=dataset_name,
            start_year=start_year,
            end_year=end_year,
            interval=interval,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_download_all(
    tickers: list[str],
    ticker_names: dict[str, str],
    dataset_name: str,
    start_year: int,
    end_year: int,
    interval: str,
):
    """Execute download, save, and display results."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_cb(current, total, label):
        pct = current / total if total > 0 else 0
        progress_bar.progress(pct)
        status_text.text(f"Downloading {current + 1}/{total}: {label}")

    try:
        df, failed = download_global_market_data(
            tickers=tickers,
            start_year=start_year,
            end_year=end_year,
            interval=interval,
            progress_callback=progress_cb,
        )

        progress_bar.progress(1.0)
        status_text.text("Download complete!")

        if df.empty:
            st.error("No data returned. Try different tickers or date range.")
            if failed:
                st.warning(f"Failed tickers: {', '.join(failed[:30])}")
            return

        save_global_market_dataset(df, dataset_name)
        st.success(
            f"Saved **{dataset_name}** — {len(df)} rows, "
            f"{df['ticker'].nunique()} tickers"
        )

        if failed:
            st.warning(
                f"{len(failed)} tickers failed: {', '.join(failed[:30])}"
            )

        st.session_state["current_dataset"] = df
        st.session_state["current_dataset_name"] = dataset_name

        _show_results(df, ticker_names)

    except Exception as e:
        st.error(f"Download failed: {e}")


def _render_category_selection(
    categories: dict[str, dict[str, str]],
    prefix: str,
) -> tuple[list[str], dict[str, str]]:
    """Render expander-based checkbox selection for a group of categories."""
    selected = []
    names = {}

    cols = st.columns(3)
    for idx, (category, tickers) in enumerate(categories.items()):
        col = cols[idx % 3]
        with col:
            with st.expander(f"{category} ({len(tickers)})", expanded=idx == 0):
                select_all = st.checkbox(
                    "Select all", key=f"{prefix}_cat_{category}"
                )
                for ticker, name in tickers.items():
                    checked = st.checkbox(
                        f"{ticker} - {name}",
                        value=select_all,
                        key=f"{prefix}_{ticker}",
                    )
                    if checked:
                        selected.append(ticker)
                        names[ticker] = name

    return selected, names


def _render_custom_tickers() -> list[str]:
    st.caption(
        "Enter any Yahoo Finance ticker symbols (comma-separated). "
        "Works with FX pairs (e.g. EURUSD=X), futures (CL=F), indexes (^BDI)."
    )
    raw = st.text_area(
        "Ticker symbols",
        placeholder="e.g., EURUSD=X, CL=F, GC=F",
        key="gm_custom_tickers",
    )
    if not raw.strip():
        return []
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    if tickers:
        st.success(f"{len(tickers)} custom tickers: {', '.join(tickers)}")
    return tickers


def _show_results(df: pd.DataFrame, ticker_names: dict[str, str]):
    """Show price charts and data preview after download."""
    unique_tickers = df["ticker"].unique().tolist()

    st.subheader("Price History")

    if len(unique_tickers) <= 10:
        for ticker in unique_tickers:
            td = df[df["ticker"] == ticker]
            if td.empty:
                continue
            fig = px.line(
                td, x="date", y="close",
                title=f"{ticker} - {ticker_names.get(ticker, ticker)}",
                labels={"close": "Close", "date": "Date"},
                color_discrete_sequence=[BRASS],
            )
            fig.update_layout(hovermode="x unified")
            apply_steam_style(fig)
            st.plotly_chart(fig, use_container_width=True)
    else:
        chart_tickers = st.multiselect(
            "Select tickers to chart (max 20)",
            unique_tickers,
            default=unique_tickers[:10],
            max_selections=20,
            key="gm_chart_select",
        )
        if chart_tickers:
            chart_data = df[df["ticker"].isin(chart_tickers)].copy()
            normalized = []
            for ticker in chart_tickers:
                td = chart_data[chart_data["ticker"] == ticker].sort_values("date")
                if td.empty or td["close"].iloc[0] == 0:
                    continue
                td = td.copy()
                td["normalized"] = (td["close"] / td["close"].iloc[0]) * 100
                normalized.append(td)

            if normalized:
                norm_df = pd.concat(normalized, ignore_index=True)
                fig = px.line(
                    norm_df, x="date", y="normalized", color="ticker",
                    title="Normalized Price Comparison (Start = 100)",
                    labels={"normalized": "Normalized Price", "date": "Date"},
                    color_discrete_sequence=CHART_COLORS,
                )
                fig.update_layout(hovermode="x unified")
                apply_steam_style(fig)
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("Data Preview")
    st.dataframe(df.head(100), use_container_width=True)
