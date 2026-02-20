"""Stock Data Download page for the Streamlit app."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_sources.stock_data import (
    TICKER_CATEGORIES,
    get_all_tickers,
    download_stock_data,
    compute_annual_returns,
    merge_stock_with_economic,
    save_stock_dataset,
)
from data_sources.world_bank import list_saved_datasets, load_dataset


def render():
    st.header("Download Stock / ETF Data")
    st.write(
        "Download stock and ETF price data to investigate how economic indicators "
        "are reflected in market performance. Focused on VOO (S&P 500) and QQQ (NASDAQ-100)."
    )

    # --- Ticker selection ---
    st.subheader("1. Select Tickers")

    all_tickers = get_all_tickers()
    selected_tickers = []

    cols = st.columns(3)
    for idx, (category, tickers) in enumerate(TICKER_CATEGORIES.items()):
        col = cols[idx % 3]
        with col:
            with st.expander(f"{category} ({len(tickers)} tickers)", expanded=idx == 0):
                select_all = st.checkbox(f"Select all {category}", key=f"stock_cat_{category}")
                for ticker, name in tickers.items():
                    default = ticker in ("VOO", "QQQ") or select_all
                    checked = st.checkbox(
                        f"{ticker} - {name}",
                        value=default,
                        key=f"stock_{ticker}",
                    )
                    if checked:
                        selected_tickers.append(ticker)

    if selected_tickers:
        st.success(f"Selected {len(selected_tickers)} tickers: {', '.join(selected_tickers)}")
    else:
        st.info("Select at least one ticker to proceed.")

    # --- Date range ---
    st.subheader("2. Select Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start year", min_value=1990, max_value=2025, value=2000, key="stock_start")
    with col2:
        end_year = st.number_input("End year", min_value=1990, max_value=2025, value=2024, key="stock_end")

    if start_year > end_year:
        st.error("Start year must be before end year.")
        return

    # --- Interval ---
    interval = st.selectbox(
        "Data interval",
        ["1mo", "1wk", "1d"],
        format_func=lambda x: {"1mo": "Monthly", "1wk": "Weekly", "1d": "Daily"}[x],
    )

    # --- Dataset name ---
    st.subheader("3. Name Your Dataset")
    dataset_name = st.text_input(
        "Dataset name",
        value="stock_data",
        key="stock_dataset_name",
        help="This will be used as the filename for saving.",
    )

    # --- Download ---
    st.divider()

    if not selected_tickers:
        st.warning("Please select at least one ticker.")
        return

    if st.button("Download Stock Data", type="primary", use_container_width=True):
        with st.spinner(f"Downloading data for {', '.join(selected_tickers)}..."):
            try:
                df = download_stock_data(
                    tickers=selected_tickers,
                    start_year=start_year,
                    end_year=end_year,
                    interval=interval,
                )

                if df.empty:
                    st.error("No data returned. Try different tickers or date range.")
                    return

                # Save raw data
                path = save_stock_dataset(df, dataset_name)
                st.success(f"Stock data saved to `{path}` ({len(df)} rows)")

                # Compute annual returns
                annual = compute_annual_returns(df)
                if not annual.empty:
                    annual_path = save_stock_dataset(annual, f"{dataset_name}_annual")
                    st.success(f"Annual returns saved to `{annual_path}`")

                # Store in session state
                st.session_state["stock_data"] = df
                st.session_state["stock_annual"] = annual
                st.session_state["stock_dataset_name"] = dataset_name

                # --- Visualizations ---
                _show_stock_results(df, annual, selected_tickers)

                # --- Merge with economic data ---
                _offer_merge_with_economic(annual)

            except Exception as e:
                st.error(f"Download failed: {e}")


def _show_stock_results(df, annual, tickers):
    """Display stock data results and charts."""
    st.subheader("Price History")

    for ticker in tickers:
        ticker_data = df[df["ticker"] == ticker]
        if ticker_data.empty:
            continue

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ticker_data["date"],
            y=ticker_data["close"],
            mode="lines",
            name=f"{ticker} Close",
        ))
        fig.update_layout(
            title=f"{ticker} - {get_all_tickers().get(ticker, ticker)}",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Annual returns comparison
    if not annual.empty and len(tickers) > 0:
        st.subheader("Annual Returns Comparison")
        fig = px.bar(
            annual,
            x="year",
            y="annual_return_pct",
            color="ticker",
            barmode="group",
            title="Annual Returns (%)",
            labels={"annual_return_pct": "Return (%)", "year": "Year"},
        )
        st.plotly_chart(fig, use_container_width=True)

        # Volatility comparison
        st.subheader("Volatility Comparison")
        fig = px.line(
            annual,
            x="year",
            y="volatility",
            color="ticker",
            title="Annual Volatility (Std Dev of Monthly Returns %)",
            labels={"volatility": "Volatility (%)", "year": "Year"},
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)


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
            "No economic datasets found. Go to **Download Data** to download "
            "World Bank economic indicators first, then come back to merge."
        )
        return

    econ_dataset = st.selectbox("Economic dataset to merge with", economic_datasets, key="merge_econ")

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

            merge_name = f"merged_stock_economic"
            save_stock_dataset(merged, merge_name)
            st.success(f"Merged dataset saved ({len(merged)} rows, {len(merged.columns)} columns)")

            # Store for correlation analysis
            st.session_state["current_dataset"] = merged
            st.session_state["current_dataset_name"] = merge_name

            st.dataframe(merged.head(20), use_container_width=True)
            st.info("You can now go to **Correlation Analysis** to analyze relationships between stock returns and economic indicators.")
