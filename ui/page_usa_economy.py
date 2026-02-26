"""USA Economy (FRED) data download page for the Streamlit app.

Provides access to detailed US economic data from FRED (Federal Reserve Economic
Data) including housing market, auto sales, retail/consumer spending,
manufacturing, employment, inflation, interest rates, GDP, and more.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
from data_sources.fred_data import (
    FRED_SERIES,
    get_all_fred_series,
    download_fred_series,
    download_fred_series_annual,
    save_fred_dataset,
)
from ui.theme import apply_steam_style, CHART_COLORS, BRASS


def _get_api_key() -> str:
    """Get FRED API key from environment variable."""
    return os.environ.get("FRED_API_KEY", "")


def render():
    st.header("USA Economy Data (FRED)")
    st.write(
        "Download detailed US economic data from **FRED** (Federal Reserve Economic Data). "
        "Covers housing, auto sales, retail, consumer spending, manufacturing, "
        "employment, inflation, interest rates, GDP, trade, and more."
    )

    # --- API Key from env ---
    api_key = _get_api_key()
    if not api_key:
        st.error(
            "FRED API key not found. Set the `FRED_API_KEY` environment variable. "
            "Get a free key at [fred.stlouisfed.org/docs/api/api_key.html]"
            "(https://fred.stlouisfed.org/docs/api/api_key.html)."
        )
        return

    all_series = get_all_fred_series()

    # --- Download ALL shortcut ---
    st.subheader("Quick Download")
    st.write(
        f"Download **all {len(all_series)} FRED series** at once, "
        "or pick individual series below."
    )

    col_all1, col_all2 = st.columns(2)
    with col_all1:
        download_all = st.button(
            f"Download ALL FRED Data ({len(all_series)} series)",
            type="primary",
            use_container_width=True,
            key="fred_download_all",
        )
    with col_all2:
        download_all_annual = st.button(
            f"Download ALL (Annual, World Bank compatible)",
            use_container_width=True,
            key="fred_download_all_annual",
        )

    if download_all or download_all_annual:
        _run_download(
            series_ids=list(all_series.keys()),
            api_key=api_key,
            dataset_name="usa_economy_fred_all",
            annual=download_all_annual,
            start_year=2000,
            end_year=2025,
            frequency=None,
            all_series=all_series,
        )
        return

    # --- Series Selection ---
    st.divider()
    st.subheader("1. Select Economic Series")
    st.caption("Choose which data series to download from FRED.")

    selected_series: list[str] = []

    # Quick-select presets
    preset = st.selectbox(
        "Quick presets",
        [
            "(Custom selection)",
            "Housing Market Overview",
            "Auto & Vehicle Sales",
            "Consumer Spending Deep Dive",
            "Full Employment Picture",
            "Inflation Dashboard",
            "Interest Rates & Fed Policy",
            "GDP & Income",
            "Everything (all series)",
        ],
        key="fred_preset",
    )

    preset_map = {
        "Housing Market Overview": list(FRED_SERIES["Housing Market"].keys()),
        "Auto & Vehicle Sales": list(FRED_SERIES["Auto & Vehicle Sales"].keys()),
        "Consumer Spending Deep Dive": (
            list(FRED_SERIES["Retail & Consumer Spending"].keys())
        ),
        "Full Employment Picture": list(FRED_SERIES["Employment & Labor"].keys()),
        "Inflation Dashboard": list(FRED_SERIES["Inflation & Prices"].keys()),
        "Interest Rates & Fed Policy": (
            list(FRED_SERIES["Interest Rates & Monetary Policy"].keys())
        ),
        "GDP & Income": list(FRED_SERIES["GDP & Income"].keys()),
        "Everything (all series)": list(all_series.keys()),
    }

    if preset != "(Custom selection)" and preset in preset_map:
        selected_series = preset_map[preset]
        st.success(
            f"Preset **{preset}** selected: {len(selected_series)} series"
        )
    else:
        # Manual category-based selection
        n_categories = len(FRED_SERIES)
        cols_per_row = 3
        categories = list(FRED_SERIES.items())

        for row_start in range(0, n_categories, cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, (category, series) in enumerate(
                categories[row_start : row_start + cols_per_row]
            ):
                with cols[col_idx]:
                    with st.expander(
                        f"{category} ({len(series)})", expanded=False
                    ):
                        select_all = st.checkbox(
                            f"Select all",
                            key=f"fred_cat_all_{category}",
                        )
                        for sid, description in series.items():
                            checked = st.checkbox(
                                f"`{sid}` - {description}",
                                value=select_all,
                                key=f"fred_{sid}",
                            )
                            if checked:
                                selected_series.append(sid)

    if not selected_series:
        st.warning("Select at least one data series to proceed.")
        return

    st.success(f"Selected **{len(selected_series)}** series")

    # --- Date Range ---
    st.subheader("2. Date Range & Frequency")
    col1, col2, col3 = st.columns(3)
    with col1:
        start_year = st.number_input(
            "Start year", min_value=1960, max_value=2026, value=2000, key="fred_start"
        )
    with col2:
        end_year = st.number_input(
            "End year", min_value=1960, max_value=2026, value=2025, key="fred_end"
        )
    with col3:
        freq_option = st.selectbox(
            "Frequency",
            ["Monthly (raw)", "Quarterly", "Annual"],
            key="fred_freq",
        )

    freq_map = {
        "Monthly (raw)": None,
        "Quarterly": "q",
        "Annual": "a",
    }
    frequency = freq_map[freq_option]

    if start_year > end_year:
        st.error("Start year must be before end year.")
        return

    # --- Dataset Name ---
    st.subheader("3. Name Your Dataset")
    dataset_name = st.text_input(
        "Dataset name",
        value="usa_economy_fred",
        key="fred_dataset_name",
    )

    # --- Download ---
    st.divider()

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        download_raw = st.button(
            "Download Data",
            type="primary",
            use_container_width=True,
            key="fred_download",
        )
    with col_dl2:
        download_annual = st.button(
            "Download (Annual, World Bank compatible)",
            use_container_width=True,
            key="fred_download_annual",
            help="Downloads with annual aggregation and adds a 'country' column "
            "for compatibility with World Bank datasets and correlation analysis.",
        )

    if download_raw or download_annual:
        _run_download(
            series_ids=selected_series,
            api_key=api_key,
            dataset_name=dataset_name,
            annual=download_annual,
            start_year=start_year,
            end_year=end_year,
            frequency=frequency,
            all_series=all_series,
        )


def _run_download(
    series_ids: list[str],
    api_key: str,
    dataset_name: str,
    annual: bool,
    start_year: int,
    end_year: int,
    frequency: str | None,
    all_series: dict[str, str],
):
    """Execute the FRED download, save, and display results."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_cb(current, total, label):
        pct = current / total if total > 0 else 0
        progress_bar.progress(pct)
        status_text.text(f"Downloading {current + 1}/{total}: {label}")

    try:
        if annual:
            df, failed = download_fred_series_annual(
                series_ids=series_ids,
                start_year=start_year,
                end_year=end_year,
                api_key=api_key,
                progress_callback=progress_cb,
            )
        else:
            df, failed = download_fred_series(
                series_ids=series_ids,
                start_date=f"{start_year}-01-01",
                end_date=f"{end_year}-12-31",
                api_key=api_key,
                frequency=frequency,
                progress_callback=progress_cb,
            )

        progress_bar.progress(1.0)
        status_text.text("Download complete!")

        if df.empty:
            st.error(
                "No data returned. Check your API key and series selection."
            )
            if failed:
                st.warning(f"Failed series: {', '.join(failed)}")
            return

        # Save
        save_fred_dataset(df, dataset_name)
        st.success(
            f"Saved **{dataset_name}** ({len(df)} rows, "
            f"{len(df.columns)} columns)"
        )

        if failed:
            st.warning(
                f"{len(failed)} series failed to download: {', '.join(failed)}"
            )

        # Store in session state
        st.session_state["current_dataset"] = df
        st.session_state["current_dataset_name"] = dataset_name
        indicator_names = {
            sid: desc
            for sid, desc in all_series.items()
            if sid in df.columns
        }
        st.session_state["indicator_names"] = indicator_names

        # --- Show results ---
        _show_fred_results(df, indicator_names)

    except Exception as e:
        st.error(f"Download failed: {e}")


def _show_fred_results(df: pd.DataFrame, indicator_names: dict[str, str]):
    """Display downloaded FRED data with charts."""
    st.subheader("Data Preview")
    st.dataframe(df.head(50), use_container_width=True)

    # Identify numeric data columns
    meta_cols = {"date", "year", "month", "country"}
    data_cols = [c for c in df.columns if c not in meta_cols and df[c].dtype in ("float64", "int64", "float32")]

    if not data_cols:
        return

    st.subheader("Quick Charts")

    # Let user pick a series to chart
    chart_series = st.selectbox(
        "Select series to visualize",
        data_cols,
        format_func=lambda x: f"{x} - {indicator_names.get(x, x)}",
        key="fred_chart_series",
    )

    if chart_series and chart_series in df.columns:
        plot_df = df.dropna(subset=[chart_series])

        if "date" in plot_df.columns:
            x_col = "date"
        elif "year" in plot_df.columns:
            x_col = "year"
        else:
            x_col = plot_df.columns[0]

        fig = px.line(
            plot_df,
            x=x_col,
            y=chart_series,
            title=indicator_names.get(chart_series, chart_series),
            labels={chart_series: indicator_names.get(chart_series, chart_series)},
            color_discrete_sequence=[BRASS],
        )
        fig.update_layout(hovermode="x unified")
        apply_steam_style(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Show summary statistics
    st.subheader("Summary Statistics")
    stats = df[data_cols].describe().T
    stats.index = [f"{idx} ({indicator_names.get(idx, '')})" for idx in stats.index]
    st.dataframe(stats, use_container_width=True)

    st.info(
        "Go to **Explore & Visualize** or **Correlation Analysis** "
        "to explore this data further."
    )
