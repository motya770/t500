"""Data Exploration & Visualization page for the Streamlit app."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_sources.world_bank import list_saved_datasets, load_dataset, get_all_indicators
from ui.theme import apply_steam_style, CHART_COLORS, HEATMAP_SCALE, BRASS, CREAM, EMBER


def _get_indicator_label(code: str) -> str:
    """Get human-readable label for an indicator code."""
    names = st.session_state.get("indicator_names", {})
    if code in names:
        return names[code]
    all_ind = get_all_indicators()
    return all_ind.get(code, code)


def _indicator_columns(df: pd.DataFrame) -> list[str]:
    """Get columns that are indicators (not country/year)."""
    return [c for c in df.columns if c not in ("country", "year")]


def render():
    st.header("Explore & Visualize Data")

    # --- Load dataset ---
    datasets = list_saved_datasets()
    if not datasets and "current_dataset" not in st.session_state:
        st.info("No datasets found. Go to the **Download Data** page first.")
        return

    source = st.radio("Data source", ["Current session", "Saved dataset"], horizontal=True)

    df = None
    if source == "Current session" and "current_dataset" in st.session_state:
        df = st.session_state["current_dataset"]
        st.write(f"Using: **{st.session_state.get('current_dataset_name', 'session data')}**")
    elif source == "Saved dataset" and datasets:
        chosen = st.selectbox("Select dataset", datasets)
        if chosen:
            df = load_dataset(chosen)
            st.session_state["current_dataset"] = df
            st.session_state["current_dataset_name"] = chosen
    else:
        st.warning("No data available. Download data first.")
        return

    if df is None or df.empty:
        st.warning("Dataset is empty.")
        return

    indicators = _indicator_columns(df)
    countries = sorted(df["country"].unique()) if "country" in df.columns else []

    # --- Data overview ---
    with st.expander("Dataset Overview", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", len(df))
        col2.metric("Indicators", len(indicators))
        col3.metric("Countries", len(countries))
        col4.metric("Year Range", f"{int(df['year'].min())}-{int(df['year'].max())}")

        st.dataframe(df.describe(), use_container_width=True)

    # --- Visualization type ---
    st.subheader("\U0001F4CA Visualizations")
    viz_type = st.selectbox(
        "Chart type",
        [
            "Time Series",
            "Country Comparison (Bar)",
            "Scatter Plot",
            "Distribution",
            "Heatmap (by country & year)",
            "Box Plot",
        ],
    )

    if viz_type == "Time Series":
        _render_time_series(df, indicators, countries)
    elif viz_type == "Country Comparison (Bar)":
        _render_country_comparison(df, indicators, countries)
    elif viz_type == "Scatter Plot":
        _render_scatter(df, indicators, countries)
    elif viz_type == "Distribution":
        _render_distribution(df, indicators)
    elif viz_type == "Heatmap (by country & year)":
        _render_heatmap(df, indicators, countries)
    elif viz_type == "Box Plot":
        _render_boxplot(df, indicators, countries)


def _render_time_series(df, indicators, countries):
    indicator = st.selectbox(
        "Indicator",
        indicators,
        format_func=_get_indicator_label,
        key="ts_indicator",
    )
    selected_countries = st.multiselect(
        "Countries",
        countries,
        default=countries[:5],
        key="ts_countries",
    )

    if not selected_countries or not indicator:
        return

    filtered = df[df["country"].isin(selected_countries)][["country", "year", indicator]].dropna()
    if filtered.empty:
        st.warning("No data for this selection.")
        return

    fig = px.line(
        filtered,
        x="year",
        y=indicator,
        color="country",
        title=_get_indicator_label(indicator),
        labels={indicator: _get_indicator_label(indicator), "year": "Year"},
        color_discrete_sequence=CHART_COLORS,
    )
    fig.update_layout(hovermode="x unified")
    apply_steam_style(fig)
    st.plotly_chart(fig, use_container_width=True)


def _render_country_comparison(df, indicators, countries):
    indicator = st.selectbox(
        "Indicator",
        indicators,
        format_func=_get_indicator_label,
        key="bar_indicator",
    )
    year = st.slider(
        "Year",
        int(df["year"].min()),
        int(df["year"].max()),
        int(df["year"].max()),
        key="bar_year",
    )

    filtered = df[df["year"] == year][["country", indicator]].dropna()
    if filtered.empty:
        st.warning("No data for this year.")
        return

    filtered = filtered.sort_values(indicator, ascending=False)
    fig = px.bar(
        filtered,
        x="country",
        y=indicator,
        title=f"{_get_indicator_label(indicator)} ({year})",
        labels={indicator: _get_indicator_label(indicator)},
        color=indicator,
        color_continuous_scale=HEATMAP_SCALE,
    )
    apply_steam_style(fig)
    st.plotly_chart(fig, use_container_width=True)


def _render_scatter(df, indicators, countries):
    col1, col2 = st.columns(2)
    with col1:
        x_ind = st.selectbox("X-axis", indicators, format_func=_get_indicator_label, key="sc_x")
    with col2:
        y_ind = st.selectbox(
            "Y-axis",
            indicators,
            index=min(1, len(indicators) - 1),
            format_func=_get_indicator_label,
            key="sc_y",
        )

    color_by = st.radio("Color by", ["Country", "Year"], horizontal=True, key="sc_color")
    filtered = df[["country", "year", x_ind, y_ind]].dropna()

    if filtered.empty:
        st.warning("No data for this selection.")
        return

    fig = px.scatter(
        filtered,
        x=x_ind,
        y=y_ind,
        color="country" if color_by == "Country" else "year",
        hover_data=["country", "year"],
        title=f"{_get_indicator_label(x_ind)} vs {_get_indicator_label(y_ind)}",
        labels={
            x_ind: _get_indicator_label(x_ind),
            y_ind: _get_indicator_label(y_ind),
        },
        trendline="ols",
        color_discrete_sequence=CHART_COLORS,
    )
    apply_steam_style(fig)
    st.plotly_chart(fig, use_container_width=True)


def _render_distribution(df, indicators):
    indicator = st.selectbox(
        "Indicator",
        indicators,
        format_func=_get_indicator_label,
        key="dist_indicator",
    )
    data = df[indicator].dropna()
    if data.empty:
        st.warning("No data.")
        return

    fig = px.histogram(
        data,
        nbins=30,
        title=f"Distribution: {_get_indicator_label(indicator)}",
        labels={"value": _get_indicator_label(indicator)},
        marginal="box",
        color_discrete_sequence=[BRASS],
    )
    apply_steam_style(fig)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean", f"{data.mean():.4f}")
    col2.metric("Median", f"{data.median():.4f}")
    col3.metric("Std Dev", f"{data.std():.4f}")


def _render_heatmap(df, indicators, countries):
    indicator = st.selectbox(
        "Indicator",
        indicators,
        format_func=_get_indicator_label,
        key="hm_indicator",
    )

    pivot = df.pivot_table(index="country", columns="year", values=indicator)
    if pivot.empty:
        st.warning("No data.")
        return

    fig = px.imshow(
        pivot,
        title=f"{_get_indicator_label(indicator)} by Country & Year",
        labels={"color": _get_indicator_label(indicator)},
        aspect="auto",
        color_continuous_scale=HEATMAP_SCALE,
    )
    apply_steam_style(fig)
    st.plotly_chart(fig, use_container_width=True)


def _render_boxplot(df, indicators, countries):
    indicator = st.selectbox(
        "Indicator",
        indicators,
        format_func=_get_indicator_label,
        key="bp_indicator",
    )

    filtered = df[["country", indicator]].dropna()
    if filtered.empty:
        st.warning("No data.")
        return

    fig = px.box(
        filtered,
        x="country",
        y=indicator,
        title=f"{_get_indicator_label(indicator)} by Country",
        labels={indicator: _get_indicator_label(indicator)},
        color_discrete_sequence=[BRASS],
    )
    apply_steam_style(fig)
    st.plotly_chart(fig, use_container_width=True)
