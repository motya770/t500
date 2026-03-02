"""Global Sentiment & Trade download page.

Consumer / business confidence (OECD via FRED), US ISM, OECD CLI,
and trade balance data (World Bank).
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
from data_sources.global_sentiment import (
    SENTIMENT_SERIES,
    TRADE_BALANCE_INDICATORS,
    get_all_sentiment_series,
    download_sentiment_series,
    download_trade_balance,
    save_sentiment_dataset,
)
from data_sources.world_bank import get_country_groups
from ui.theme import apply_steam_style, CHART_COLORS, BRASS


def _get_api_key() -> str:
    return os.environ.get("FRED_API_KEY", "")


def render():
    st.header("Global Sentiment & Trade")
    st.write(
        "Download consumer/business confidence (OECD), US ISM/PMI, "
        "OECD leading indicators, and trade balance data."
    )

    all_sentiment = get_all_sentiment_series()

    # --- Download ALL shortcut ---
    st.subheader("Quick Download")
    st.write(
        f"Download **all {len(all_sentiment)} FRED sentiment series** at once "
        "(consumer confidence, business confidence, ISM, leading indicators), "
        "or pick individual series in the tabs below. "
        "Trade Balance (World Bank) must be downloaded separately in its tab."
    )

    api_key = _get_api_key()

    if not api_key:
        st.warning(
            "FRED API key not found — set `FRED_API_KEY` to enable quick download. "
            "Trade Balance tab (World Bank) does not need an API key."
        )
    else:
        if st.button(
            f"Download ALL Sentiment Data ({len(all_sentiment)} FRED series)",
            type="primary",
            use_container_width=True,
            key="gsent_download_all",
        ):
            _run_download_all_sentiment(
                series_ids=list(all_sentiment.keys()),
                api_key=api_key,
                all_series=all_sentiment,
            )
            return

    # --- Tabs for individual selection ---
    st.divider()
    st.subheader("Or Select by Category")

    tab_consumer, tab_business, tab_leading, tab_trade = st.tabs([
        "Consumer Confidence",
        "Business Confidence (PMI)",
        "Leading Indicators",
        "Trade Balance",
    ])

    # -- Consumer Confidence (FRED) --
    with tab_consumer:
        _render_fred_tab(
            categories={
                k: v for k, v in SENTIMENT_SERIES.items()
                if "Consumer" in k
            },
            prefix="cons",
            tab_label="Consumer Confidence",
            default_dataset_name="consumer_confidence",
        )

    # -- Business Confidence (FRED) --
    with tab_business:
        _render_fred_tab(
            categories={
                k: v for k, v in SENTIMENT_SERIES.items()
                if "Business" in k or "ISM" in k
            },
            prefix="biz",
            tab_label="Business Confidence / PMI",
            default_dataset_name="business_confidence",
        )

    # -- Leading Indicators (FRED) --
    with tab_leading:
        _render_fred_tab(
            categories={
                k: v for k, v in SENTIMENT_SERIES.items()
                if "Leading" in k
            },
            prefix="cli",
            tab_label="OECD Leading Indicators",
            default_dataset_name="leading_indicators",
        )

    # -- Trade Balance (World Bank) --
    with tab_trade:
        _render_trade_tab()


# ---------------------------------------------------------------------------
# Download ALL helper
# ---------------------------------------------------------------------------

def _run_download_all_sentiment(
    series_ids: list[str],
    api_key: str,
    all_series: dict[str, str],
):
    """Download all FRED sentiment series at once, save, and display results."""
    dataset_name = "global_sentiment_all"

    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_cb(current, total, label):
        pct = current / total if total > 0 else 0
        progress_bar.progress(pct)
        status_text.text(f"Downloading {current + 1}/{total}: {label}")

    try:
        df, failed = download_sentiment_series(
            series_ids=series_ids,
            start_date="2000-01-01",
            end_date="2025-12-31",
            api_key=api_key,
            frequency=None,
            progress_callback=progress_cb,
        )

        progress_bar.progress(1.0)
        status_text.text("Download complete!")

        if df.empty:
            st.error("No data returned. Check your API key.")
            if failed:
                st.warning(f"Failed series: {', '.join(failed)}")
            return

        save_sentiment_dataset(df, dataset_name)
        st.success(
            f"Saved **{dataset_name}** ({len(df)} rows, "
            f"{len(df.columns)} columns)"
        )

        if failed:
            st.warning(
                f"{len(failed)} series failed to download: {', '.join(failed)}"
            )

        st.session_state["current_dataset"] = df
        st.session_state["current_dataset_name"] = dataset_name
        indicator_names = {
            sid: desc for sid, desc in all_series.items() if sid in df.columns
        }
        st.session_state["indicator_names"] = indicator_names

        _show_fred_results(df, indicator_names, "all")

    except Exception as e:
        st.error(f"Download failed: {e}")


# ---------------------------------------------------------------------------
# FRED-based tabs (consumer, business, leading)
# ---------------------------------------------------------------------------

def _render_fred_tab(
    categories: dict[str, dict[str, str]],
    prefix: str,
    tab_label: str,
    default_dataset_name: str,
):
    api_key = _get_api_key()
    if not api_key:
        st.error(
            "FRED API key not found. Set the `FRED_API_KEY` environment variable. "
            "Get a free key at [fred.stlouisfed.org/docs/api/api_key.html]"
            "(https://fred.stlouisfed.org/docs/api/api_key.html)."
        )
        return

    st.subheader(f"Select {tab_label} Series")

    selected_series: list[str] = []

    # Select all shortcut
    all_in_tab = {}
    for series in categories.values():
        all_in_tab.update(series)

    select_all_btn = st.checkbox(
        f"Select all {tab_label} ({len(all_in_tab)} series)",
        key=f"{prefix}_select_all",
    )

    if select_all_btn:
        selected_series = list(all_in_tab.keys())
    else:
        cols_per_row = min(3, len(categories))
        cat_list = list(categories.items())

        for row_start in range(0, len(cat_list), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, (category, series) in enumerate(
                cat_list[row_start : row_start + cols_per_row]
            ):
                with cols[col_idx]:
                    with st.expander(
                        f"{category} ({len(series)})", expanded=True
                    ):
                        cat_all = st.checkbox(
                            "Select all",
                            key=f"{prefix}_cat_{category}",
                        )
                        for sid, description in series.items():
                            checked = st.checkbox(
                                f"`{sid}` - {description}",
                                value=cat_all,
                                key=f"{prefix}_{sid}",
                            )
                            if checked:
                                selected_series.append(sid)

    if not selected_series:
        st.info("Select at least one series to proceed.")
        return

    st.success(f"Selected **{len(selected_series)}** series")

    # Date range
    col1, col2, col3 = st.columns(3)
    with col1:
        start_year = st.number_input(
            "Start year", min_value=1960, max_value=2026, value=2000,
            key=f"{prefix}_start",
        )
    with col2:
        end_year = st.number_input(
            "End year", min_value=1960, max_value=2026, value=2025,
            key=f"{prefix}_end",
        )
    with col3:
        freq_option = st.selectbox(
            "Frequency",
            ["Monthly (raw)", "Quarterly", "Annual"],
            key=f"{prefix}_freq",
        )

    freq_map = {"Monthly (raw)": None, "Quarterly": "q", "Annual": "a"}
    frequency = freq_map[freq_option]

    if start_year > end_year:
        st.error("Start year must be before end year.")
        return

    dataset_name = st.text_input(
        "Dataset name",
        value=default_dataset_name,
        key=f"{prefix}_dataset_name",
    )

    if st.button(
        f"Download {tab_label}",
        type="primary",
        use_container_width=True,
        key=f"{prefix}_download",
    ):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_cb(current, total, label):
            pct = current / total if total > 0 else 0
            progress_bar.progress(pct)
            status_text.text(f"Downloading {current + 1}/{total}: {label}")

        try:
            df, failed = download_sentiment_series(
                series_ids=selected_series,
                start_date=f"{start_year}-01-01",
                end_date=f"{end_year}-12-31",
                api_key=api_key,
                frequency=frequency,
                progress_callback=progress_cb,
            )

            progress_bar.progress(1.0)
            status_text.text("Download complete!")

            if df.empty:
                st.error("No data returned. Check your API key and series.")
                if failed:
                    st.warning(f"Failed: {', '.join(failed)}")
                return

            save_sentiment_dataset(df, dataset_name)
            st.success(
                f"Saved **{dataset_name}** ({len(df)} rows, "
                f"{len(df.columns)} columns)"
            )

            if failed:
                st.warning(f"{len(failed)} series failed: {', '.join(failed)}")

            st.session_state["current_dataset"] = df
            st.session_state["current_dataset_name"] = dataset_name
            indicator_names = {
                sid: desc
                for sid, desc in all_in_tab.items()
                if sid in df.columns
            }
            st.session_state["indicator_names"] = indicator_names

            _show_fred_results(df, indicator_names, prefix)

        except Exception as e:
            st.error(f"Download failed: {e}")


def _show_fred_results(
    df: pd.DataFrame,
    indicator_names: dict[str, str],
    prefix: str,
):
    st.subheader("Data Preview")
    st.dataframe(df.head(50), use_container_width=True)

    meta_cols = {"date", "year", "month", "country"}
    data_cols = [
        c for c in df.columns
        if c not in meta_cols and df[c].dtype in ("float64", "int64", "float32")
    ]

    if not data_cols:
        return

    st.subheader("Quick Charts")
    chart_series = st.selectbox(
        "Select series to visualize",
        data_cols,
        format_func=lambda x: f"{x} - {indicator_names.get(x, x)}",
        key=f"{prefix}_chart",
    )

    if chart_series and chart_series in df.columns:
        plot_df = df.dropna(subset=[chart_series])
        x_col = "date" if "date" in plot_df.columns else "year"

        fig = px.line(
            plot_df, x=x_col, y=chart_series,
            title=indicator_names.get(chart_series, chart_series),
            labels={chart_series: indicator_names.get(chart_series, chart_series)},
            color_discrete_sequence=[BRASS],
        )
        fig.update_layout(hovermode="x unified")
        apply_steam_style(fig)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Trade Balance tab (World Bank)
# ---------------------------------------------------------------------------

def _render_trade_tab():
    st.subheader("Trade Balance & External Sector (World Bank)")
    st.write(
        "Download trade balance indicators from the World Bank API. "
        "Covers exports, imports, current account, FDI, and merchandise trade."
    )

    # Indicator selection
    selected_indicators: list[str] = []

    select_all = st.checkbox(
        f"Select all trade indicators ({len(TRADE_BALANCE_INDICATORS)})",
        value=True,
        key="trade_select_all",
    )

    if select_all:
        selected_indicators = list(TRADE_BALANCE_INDICATORS.keys())
    else:
        for code, name in TRADE_BALANCE_INDICATORS.items():
            if st.checkbox(f"{name}", key=f"trade_{code}"):
                selected_indicators.append(code)

    if not selected_indicators:
        st.info("Select at least one trade indicator.")
        return

    st.success(f"Selected **{len(selected_indicators)}** indicators")

    # Country selection
    st.subheader("Select Countries")
    country_groups = get_country_groups()
    selection_mode = st.radio(
        "Selection mode",
        ["Predefined groups", "Manual entry"],
        horizontal=True,
        key="trade_country_mode",
    )

    selected_countries = []
    if selection_mode == "Predefined groups":
        chosen_groups = st.multiselect(
            "Choose country groups",
            options=list(country_groups.keys()),
            default=["G7"],
            key="trade_country_groups",
        )
        for group in chosen_groups:
            selected_countries.extend(country_groups[group])
        selected_countries = list(set(selected_countries))

        if selected_countries:
            st.write(f"Countries: {', '.join(sorted(selected_countries))}")
    else:
        manual_input = st.text_area(
            "Enter ISO3 country codes (comma-separated)",
            value="USA, GBR, DEU, FRA, JPN, CHN, IND, BRA",
            key="trade_manual_countries",
        )
        selected_countries = [
            c.strip() for c in manual_input.split(",") if c.strip()
        ]

    if not selected_countries:
        st.warning("Please select at least one country.")
        return

    # Year range
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input(
            "Start year", min_value=1960, max_value=2025, value=2000,
            key="trade_start",
        )
    with col2:
        end_year = st.number_input(
            "End year", min_value=1960, max_value=2025, value=2025,
            key="trade_end",
        )

    if start_year > end_year:
        st.error("Start year must be before end year.")
        return

    dataset_name = st.text_input(
        "Dataset name",
        value="trade_balance",
        key="trade_dataset_name",
    )

    st.write(
        f"**Summary:** {len(selected_indicators)} indicators x "
        f"{len(selected_countries)} countries x {end_year - start_year + 1} years"
    )

    if st.button(
        "Download Trade Data",
        type="primary",
        use_container_width=True,
        key="trade_download",
    ):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_cb(current, total, label):
            progress_bar.progress(current / total if total > 0 else 0)
            status_text.text(f"Downloading: {label} ({current + 1}/{total})")

        try:
            df, failed = download_trade_balance(
                indicator_codes=selected_indicators,
                countries=selected_countries,
                start_year=start_year,
                end_year=end_year,
                progress_callback=progress_cb,
            )

            progress_bar.progress(1.0)
            status_text.text("Download complete!")

            if df.empty:
                st.error("No data returned. Try different indicators or countries.")
                if failed:
                    st.warning(f"Failed: {', '.join(failed)}")
                return

            save_sentiment_dataset(df, dataset_name)
            st.success(
                f"Saved **{dataset_name}** ({len(df)} rows, "
                f"{len(df.columns)} columns)"
            )

            if failed:
                st.warning(f"{len(failed)} indicators failed: {', '.join(failed)}")

            indicator_names = {
                code: TRADE_BALANCE_INDICATORS.get(code, code)
                for code in selected_indicators
            }

            st.session_state["current_dataset"] = df
            st.session_state["current_dataset_name"] = dataset_name
            st.session_state["indicator_names"] = indicator_names

            st.subheader("Data Preview")
            st.dataframe(df.head(50), use_container_width=True)

            with st.expander("Dataset Summary"):
                st.write(f"**Rows:** {len(df)}")
                st.write(f"**Columns:** {len(df.columns)}")
                st.write(f"**Countries:** {df['country'].nunique()}")
                st.write(
                    f"**Year range:** {df['year'].min()} - {df['year'].max()}"
                )

        except Exception as e:
            st.error(f"Download failed: {e}")
