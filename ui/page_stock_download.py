"""Stock Data Download page for the Streamlit app.

Supports ETFs, S&P 500 individual stocks, and Nasdaq 100 individual stocks.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_sources.stock_data import (
    TICKER_CATEGORIES,
    get_all_tickers,
    get_sp500_tickers,
    get_nasdaq100_tickers,
    get_sp500_flat,
    get_nasdaq100_flat,
    download_stock_data,
    compute_annual_returns,
    merge_stock_with_economic,
    save_stock_dataset,
)
from data_sources.world_bank import list_saved_datasets, load_dataset
from ui.theme import apply_steam_style, CHART_COLORS, BRASS, COPPER, EMBER, CREAM


def render():
    st.header("Download Stock / ETF Data")
    st.write(
        "Download stock and ETF price data including individual S&P 500 and "
        "Nasdaq 100 component stocks."
    )

    # --- Mode selection via tabs ---
    tab_etf, tab_sp500, tab_nasdaq, tab_custom = st.tabs([
        "ETFs & Indices",
        "S&P 500 Stocks",
        "Nasdaq 100 Stocks",
        "Custom Tickers",
    ])

    selected_tickers = []
    ticker_names = {}  # ticker -> human name for display

    with tab_etf:
        selected_tickers, ticker_names = _render_etf_selection()

    with tab_sp500:
        sp_tickers, sp_names = _render_index_selection(
            "S&P 500",
            get_sp500_tickers,
            get_sp500_flat,
            key_prefix="sp500",
        )
        selected_tickers.extend(sp_tickers)
        ticker_names.update(sp_names)

    with tab_nasdaq:
        nq_tickers, nq_names = _render_index_selection(
            "Nasdaq 100",
            get_nasdaq100_tickers,
            get_nasdaq100_flat,
            key_prefix="nq100",
        )
        selected_tickers.extend(nq_tickers)
        ticker_names.update(nq_names)

    with tab_custom:
        custom_tickers = _render_custom_tickers()
        selected_tickers.extend(custom_tickers)
        ticker_names.update({t: t for t in custom_tickers})

    # Deduplicate while preserving order
    seen = set()
    unique_tickers = []
    for t in selected_tickers:
        if t not in seen:
            seen.add(t)
            unique_tickers.append(t)
    selected_tickers = unique_tickers

    # --- Summary ---
    st.divider()
    if selected_tickers:
        display_limit = 20
        ticker_display = ", ".join(selected_tickers[:display_limit])
        if len(selected_tickers) > display_limit:
            ticker_display += f", ... (+{len(selected_tickers) - display_limit} more)"
        st.success(f"**{len(selected_tickers)}** tickers selected: {ticker_display}")
    else:
        st.info("Select tickers from any tab above to proceed.")

    # --- Date range ---
    st.subheader("Date Range & Interval")
    col1, col2, col3 = st.columns(3)
    with col1:
        start_year = st.number_input(
            "Start year", min_value=1990, max_value=2026, value=2000, key="stock_start"
        )
    with col2:
        end_year = st.number_input(
            "End year", min_value=1990, max_value=2026, value=2025, key="stock_end"
        )
    with col3:
        interval = st.selectbox(
            "Data interval",
            ["1mo", "1wk", "1d"],
            format_func=lambda x: {"1mo": "Monthly", "1wk": "Weekly", "1d": "Daily"}[x],
        )

    if start_year > end_year:
        st.error("Start year must be before end year.")
        return

    # --- Dataset name ---
    st.subheader("Name Your Dataset")
    dataset_name = st.text_input(
        "Dataset name",
        value="stock_data",
        key="stock_dataset_name",
        help="This will be used as the filename for saving.",
    )

    # --- Download ---
    st.divider()

    if not selected_tickers:
        st.warning("Please select at least one ticker from the tabs above.")
        return

    if st.button(
        f"Download {len(selected_tickers)} Stocks",
        type="primary",
        use_container_width=True,
    ):
        progress_bar = st.progress(0)
        status_text = st.empty()
        total = len(selected_tickers)

        try:
            # Download in batches for large lists
            batch_size = 50
            all_data = []
            failed_tickers = []

            for batch_start in range(0, total, batch_size):
                batch = selected_tickers[batch_start : batch_start + batch_size]
                batch_num = batch_start // batch_size + 1
                total_batches = (total + batch_size - 1) // batch_size

                pct = batch_start / total
                progress_bar.progress(pct)
                status_text.text(
                    f"Batch {batch_num}/{total_batches}: "
                    f"downloading {', '.join(batch[:5])}{'...' if len(batch) > 5 else ''}"
                )

                try:
                    batch_df = download_stock_data(
                        tickers=batch,
                        start_year=start_year,
                        end_year=end_year,
                        interval=interval,
                    )
                    if not batch_df.empty:
                        all_data.append(batch_df)
                except RuntimeError as e:
                    # Individual ticker failures are caught here
                    failed_tickers.extend(batch)
                    st.warning(f"Batch failed: {e}")

            progress_bar.progress(1.0)
            status_text.text("Download complete!")

            if not all_data:
                st.error("No data returned. Try different tickers or date range.")
                return

            df = pd.concat(all_data, ignore_index=True)
            df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

            # Save raw data
            save_stock_dataset(df, dataset_name)
            st.success(f"Stock data saved: **{len(df)}** rows, **{df['ticker'].nunique()}** tickers")

            if failed_tickers:
                st.warning(f"{len(failed_tickers)} tickers failed: {', '.join(failed_tickers[:20])}")

            # Compute annual returns
            annual = compute_annual_returns(df)
            if not annual.empty:
                save_stock_dataset(annual, f"{dataset_name}_annual")
                st.success(f"Annual returns saved: **{len(annual)}** rows")

            # Store in session state
            st.session_state["stock_data"] = df
            st.session_state["stock_annual"] = annual
            st.session_state["stock_dataset_name"] = dataset_name

            # --- Visualizations ---
            _show_stock_results(df, annual, ticker_names)

            # --- Merge with economic data ---
            _offer_merge_with_economic(annual)

        except Exception as e:
            st.error(f"Download failed: {e}")


# ---------------------------------------------------------------------------
# Ticker selection renderers
# ---------------------------------------------------------------------------

def _render_etf_selection() -> tuple[list[str], dict[str, str]]:
    """Render ETF ticker selection. Returns (selected_tickers, names_dict)."""
    st.caption("Select ETFs and index funds.")
    selected = []
    names = {}

    cols = st.columns(3)
    for idx, (category, tickers) in enumerate(TICKER_CATEGORIES.items()):
        col = cols[idx % 3]
        with col:
            with st.expander(f"{category} ({len(tickers)})", expanded=idx == 0):
                select_all = st.checkbox(
                    f"Select all", key=f"etf_cat_{category}"
                )
                for ticker, name in tickers.items():
                    default = ticker in ("VOO", "QQQ") or select_all
                    checked = st.checkbox(
                        f"{ticker} - {name}",
                        value=default,
                        key=f"etf_{ticker}",
                    )
                    if checked:
                        selected.append(ticker)
                        names[ticker] = name

    return selected, names


def _render_index_selection(
    index_name: str,
    get_components_fn,
    get_flat_fn,
    key_prefix: str,
) -> tuple[list[str], dict[str, str]]:
    """Render S&P 500 or Nasdaq 100 stock selection.

    Returns (selected_tickers, names_dict).
    """
    st.caption(
        f"Select individual {index_name} component stocks. "
        f"Components are fetched from Wikipedia when available."
    )

    # Load components (cached)
    with st.spinner(f"Loading {index_name} components..."):
        components = get_components_fn()
        flat = get_flat_fn()

    total_count = sum(len(v) for v in components.values())
    st.write(f"**{total_count}** stocks available across **{len(components)}** sectors")

    selected = []
    names = {}

    # Quick actions
    col1, col2 = st.columns(2)
    with col1:
        select_all = st.checkbox(
            f"Select ALL {index_name} stocks ({total_count})",
            key=f"{key_prefix}_all",
        )
    with col2:
        use_search = st.checkbox(
            "Search by ticker/name",
            key=f"{key_prefix}_search_mode",
        )

    if select_all:
        selected = list(flat.keys())
        names = dict(flat)
        st.success(f"All {len(selected)} {index_name} stocks selected")
        return selected, names

    if use_search:
        # Multiselect search
        options = [f"{t} - {n}" for t, n in flat.items()]
        chosen = st.multiselect(
            f"Search and select {index_name} stocks",
            options=options,
            key=f"{key_prefix}_search",
            help="Type to search by ticker or company name",
        )
        for item in chosen:
            ticker = item.split(" - ")[0]
            selected.append(ticker)
            names[ticker] = flat.get(ticker, ticker)
        return selected, names

    # Sector-based selection with expanders
    sectors = list(components.items())
    cols_per_row = 3

    for row_start in range(0, len(sectors), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, (sector, tickers) in enumerate(
            sectors[row_start : row_start + cols_per_row]
        ):
            with cols[col_idx]:
                with st.expander(f"{sector} ({len(tickers)})", expanded=False):
                    sect_all = st.checkbox(
                        "Select all",
                        key=f"{key_prefix}_sect_{sector}",
                    )
                    if sect_all:
                        for ticker, name in tickers.items():
                            selected.append(ticker)
                            names[ticker] = name
                    else:
                        for ticker, name in tickers.items():
                            checked = st.checkbox(
                                f"{ticker} - {name}",
                                value=False,
                                key=f"{key_prefix}_{ticker}",
                            )
                            if checked:
                                selected.append(ticker)
                                names[ticker] = name

    return selected, names


def _render_custom_tickers() -> list[str]:
    """Render custom ticker input. Returns list of tickers."""
    st.caption(
        "Enter any ticker symbols (comma-separated). "
        "Works with any stock, ETF, or index available on Yahoo Finance."
    )
    custom_input = st.text_area(
        "Ticker symbols",
        placeholder="e.g., AAPL, MSFT, GOOGL, BRK-B, PLTR",
        key="custom_tickers_input",
    )
    if not custom_input.strip():
        return []

    tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
    if tickers:
        st.success(f"{len(tickers)} custom tickers: {', '.join(tickers)}")
    return tickers


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def _show_stock_results(df, annual, ticker_names):
    """Display stock data results and charts."""
    unique_tickers = df["ticker"].unique().tolist()

    st.subheader("Price History")

    # For many tickers, show a combined chart instead of individual ones
    if len(unique_tickers) <= 10:
        for ticker in unique_tickers:
            ticker_data = df[df["ticker"] == ticker]
            if ticker_data.empty:
                continue

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ticker_data["date"],
                y=ticker_data["close"],
                mode="lines",
                name=f"{ticker} Close",
                line=dict(color=BRASS),
            ))
            fig.update_layout(
                title=f"{ticker} - {ticker_names.get(ticker, ticker)}",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                hovermode="x unified",
            )
            apply_steam_style(fig)
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Combined normalized chart for many tickers
        st.write(f"Showing normalized price chart for **{len(unique_tickers)}** tickers")

        # Let user select which to chart
        chart_tickers = st.multiselect(
            "Select tickers to chart (max 20)",
            unique_tickers,
            default=unique_tickers[:10],
            max_selections=20,
            key="chart_ticker_select",
        )

        if chart_tickers:
            chart_data = df[df["ticker"].isin(chart_tickers)].copy()
            # Normalize to first value for comparison
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
                    norm_df,
                    x="date",
                    y="normalized",
                    color="ticker",
                    title="Normalized Price Comparison (Start = 100)",
                    labels={"normalized": "Normalized Price", "date": "Date"},
                    color_discrete_sequence=CHART_COLORS,
                )
                fig.update_layout(hovermode="x unified")
                apply_steam_style(fig)
                st.plotly_chart(fig, use_container_width=True)

    # Annual returns comparison (limit to manageable number)
    if not annual.empty:
        st.subheader("Annual Returns")

        if len(unique_tickers) <= 15:
            fig = px.bar(
                annual,
                x="year",
                y="annual_return_pct",
                color="ticker",
                barmode="group",
                title="Annual Returns (%)",
                labels={"annual_return_pct": "Return (%)", "year": "Year"},
                color_discrete_sequence=CHART_COLORS,
            )
            apply_steam_style(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Summary table for many tickers
            summary = annual.groupby("ticker").agg(
                avg_return=("annual_return_pct", "mean"),
                total_return=("annual_return_pct", "sum"),
                avg_volatility=("volatility", "mean"),
                years=("year", "count"),
            ).round(2).sort_values("avg_return", ascending=False)

            summary.columns = [
                "Avg Annual Return (%)",
                "Cumulative Return (%)",
                "Avg Volatility (%)",
                "Years of Data",
            ]

            st.dataframe(summary, use_container_width=True)

            # Top/bottom performers chart
            top10 = summary.head(10)
            bottom10 = summary.tail(10)

            fig = px.bar(
                x=top10.index.tolist() + bottom10.index.tolist(),
                y=top10["Avg Annual Return (%)"].tolist()
                + bottom10["Avg Annual Return (%)"].tolist(),
                title="Top 10 & Bottom 10 by Average Annual Return",
                labels={"x": "Ticker", "y": "Avg Annual Return (%)"},
                color_discrete_sequence=[BRASS],
            )
            apply_steam_style(fig)
            st.plotly_chart(fig, use_container_width=True)

    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(100), use_container_width=True)


def _offer_merge_with_economic(annual):
    """Offer to merge stock data with existing economic datasets."""
    st.divider()
    st.subheader("Merge with Economic Data")
    st.write(
        "Merge annual stock returns with economic indicator data to analyze "
        "correlations between market performance and economic indicators."
    )

    datasets = list_saved_datasets()
    economic_datasets = [d for d in datasets if not d.startswith("stock_")]

    if not economic_datasets:
        st.info(
            "No economic datasets found. Go to **Macro Data** or **USA Economy (FRED)** "
            "to download economic indicators first, then come back to merge."
        )
        return

    econ_dataset = st.selectbox(
        "Economic dataset to merge with", economic_datasets, key="merge_econ"
    )

    if st.button("Merge Datasets", key="merge_btn"):
        with st.spinner("Merging datasets..."):
            econ_df = load_dataset(econ_dataset)
            merged = merge_stock_with_economic(annual, econ_df, country="USA")

            if merged.empty:
                st.warning(
                    "No overlapping data found. Make sure the economic dataset "
                    "contains USA data with overlapping years."
                )
                return

            merge_name = "merged_stock_economic"
            save_stock_dataset(merged, merge_name)
            st.success(
                f"Merged dataset saved ({len(merged)} rows, "
                f"{len(merged.columns)} columns)"
            )

            # Store for correlation analysis
            st.session_state["current_dataset"] = merged
            st.session_state["current_dataset_name"] = merge_name

            st.dataframe(merged.head(20), use_container_width=True)
            st.info(
                "You can now go to **Correlation Analysis** to analyze "
                "relationships between stock returns and economic indicators."
            )
