"""Global Monetary & Bonds download page.

Bond yields, central bank policy rates, money supply (M2), and central
bank balance sheets — all via FRED.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
from data_sources.global_monetary import (
    GLOBAL_MONETARY_SERIES,
    get_all_series,
    download_monetary_series,
    save_monetary_dataset,
)
from ui.theme import apply_steam_style, CHART_COLORS, BRASS


def _get_api_key() -> str:
    return os.environ.get("FRED_API_KEY", "")


def render():
    st.header("Global Monetary & Bonds")
    st.write(
        "Download bond yields, central bank policy rates, money supply, "
        "and central bank balance sheet data from **FRED**. "
        "Covers US, Euro Area, Japan, UK, and more."
    )

    # API key check
    api_key = _get_api_key()
    if not api_key:
        st.error(
            "FRED API key not found. Set the `FRED_API_KEY` environment variable. "
            "Get a free key at [fred.stlouisfed.org/docs/api/api_key.html]"
            "(https://fred.stlouisfed.org/docs/api/api_key.html)."
        )
        return

    all_series = get_all_series()

    # --- Download ALL shortcut ---
    st.subheader("Quick Download")
    st.write(
        f"Download **all {len(all_series)} series** at once, "
        "or pick individual series below."
    )

    if st.button(
        f"Download ALL Monetary Data ({len(all_series)} series)",
        type="primary",
        use_container_width=True,
        key="gmon_download_all",
    ):
        _run_download(
            series_ids=list(all_series.keys()),
            api_key=api_key,
            dataset_name="global_monetary_all",
            start_year=2000,
            end_year=2025,
            frequency=None,
            all_series=all_series,
        )
        return

    # --- Presets / manual selection ---
    st.divider()
    st.subheader("Or Select Series")

    preset = st.selectbox(
        "Quick presets",
        [
            "(Custom selection)",
            "Bond Yields (10Y + 2Y)",
            "Central Bank Rates",
            "Money Supply (M2)",
            "Balance Sheets",
        ],
        key="gmon_preset",
    )

    preset_map = {
        "Bond Yields (10Y + 2Y)": (
            list(GLOBAL_MONETARY_SERIES["10-Year Bond Yields"].keys())
            + list(GLOBAL_MONETARY_SERIES["2-Year Bond Yields"].keys())
            + list(GLOBAL_MONETARY_SERIES["Yield Spreads"].keys())
        ),
        "Central Bank Rates": list(
            GLOBAL_MONETARY_SERIES["Central Bank Policy Rates"].keys()
        ),
        "Money Supply (M2)": list(
            GLOBAL_MONETARY_SERIES["Money Supply (M2)"].keys()
        ),
        "Balance Sheets": list(
            GLOBAL_MONETARY_SERIES["Central Bank Balance Sheets"].keys()
        ),
    }

    selected_series: list[str] = []

    if preset != "(Custom selection)" and preset in preset_map:
        selected_series = preset_map[preset]
        st.success(f"Preset **{preset}**: {len(selected_series)} series")
    else:
        # Manual category-based selection
        n_categories = len(GLOBAL_MONETARY_SERIES)
        cols_per_row = 3
        categories = list(GLOBAL_MONETARY_SERIES.items())

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
                            "Select all",
                            key=f"gmon_cat_all_{category}",
                        )
                        for sid, description in series.items():
                            checked = st.checkbox(
                                f"`{sid}` - {description}",
                                value=select_all,
                                key=f"gmon_{sid}",
                            )
                            if checked:
                                selected_series.append(sid)

    if not selected_series:
        st.warning("Select at least one data series to proceed.")
        return

    st.success(f"Selected **{len(selected_series)}** series")

    # Date range & frequency
    st.subheader("Date Range & Frequency")
    col1, col2, col3 = st.columns(3)
    with col1:
        start_year = st.number_input(
            "Start year", min_value=1960, max_value=2026, value=2000,
            key="gmon_start",
        )
    with col2:
        end_year = st.number_input(
            "End year", min_value=1960, max_value=2026, value=2025,
            key="gmon_end",
        )
    with col3:
        freq_option = st.selectbox(
            "Frequency",
            ["Monthly (raw)", "Quarterly", "Annual"],
            key="gmon_freq",
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

    # Dataset name
    st.subheader("Name Your Dataset")
    dataset_name = st.text_input(
        "Dataset name",
        value="global_monetary",
        key="gmon_dataset_name",
    )

    # Download
    st.divider()

    if st.button(
        "Download Data",
        type="primary",
        use_container_width=True,
        key="gmon_download",
    ):
        _run_download(
            series_ids=selected_series,
            api_key=api_key,
            dataset_name=dataset_name,
            start_year=start_year,
            end_year=end_year,
            frequency=frequency,
            all_series=all_series,
        )


def _run_download(
    series_ids: list[str],
    api_key: str,
    dataset_name: str,
    start_year: int,
    end_year: int,
    frequency: str | None,
    all_series: dict[str, str],
):
    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_cb(current, total, label):
        pct = current / total if total > 0 else 0
        progress_bar.progress(pct)
        status_text.text(f"Downloading {current + 1}/{total}: {label}")

    try:
        df, failed = download_monetary_series(
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
            st.error("No data returned. Check your API key and series selection.")
            if failed:
                st.warning(f"Failed series: {', '.join(failed)}")
            return

        save_monetary_dataset(df, dataset_name)
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

        _show_results(df, indicator_names)

    except Exception as e:
        st.error(f"Download failed: {e}")


def _show_results(df: pd.DataFrame, indicator_names: dict[str, str]):
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
        key="gmon_chart_series",
    )

    if chart_series and chart_series in df.columns:
        plot_df = df.dropna(subset=[chart_series])

        x_col = "date" if "date" in plot_df.columns else "year"

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

    st.subheader("Summary Statistics")
    stats = df[data_cols].describe().T
    stats.index = [
        f"{idx} ({indicator_names.get(idx, '')})" for idx in stats.index
    ]
    st.dataframe(stats, use_container_width=True)

    st.info(
        "Go to **Explore & Visualize** or **Correlation Analysis** "
        "to explore this data further."
    )
