"""Cargo Plane Analysis page for the Streamlit app.

Provides specialized visualizations and analysis focused on air freight
and cargo transport indicators and their economic correlations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_sources.world_bank import (
    list_saved_datasets,
    load_dataset,
    get_all_indicators,
    get_country_groups,
    download_multiple_indicators,
    save_dataset,
    INDICATOR_CATEGORIES,
)
from analysis.cargo_analysis import (
    CARGO_FREIGHT_CODE,
    CARGO_PASSENGERS_CODE,
    CARGO_DEPARTURES_CODE,
    CONTAINER_PORT_CODE,
    CARGO_INDICATOR_CODES,
    ECONOMIC_CONTEXT_CODES,
    compute_cargo_trends,
    compute_cargo_rankings,
    compute_cargo_economic_correlation,
    compute_cargo_intensity,
    compute_cargo_growth_drivers,
    compute_yoy_growth,
)
from ui.theme import (
    apply_steam_style, CHART_COLORS, HEATMAP_SCALE, DIVERGING_SCALE,
    BRASS, COPPER, EMBER, CREAM, STEEL,
)


def _get_indicator_label(code: str) -> str:
    """Get human-readable label for an indicator code."""
    names = st.session_state.get("indicator_names", {})
    if code in names:
        return names[code]
    all_ind = get_all_indicators()
    return all_ind.get(code, code)


def _indicator_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("country", "year")]


def render():
    st.header("\u2708\uFE0F Cargo Plane Analysis")
    st.markdown(
        "Analyze global air freight and cargo transport patterns, "
        "trends, and their correlations with economic indicators."
    )

    # --- Data source selection ---
    st.subheader("Data Source")
    data_mode = st.radio(
        "How to load data",
        ["Download cargo indicators now", "Use current session data", "Load saved dataset"],
        horizontal=True,
        key="cargo_data_mode",
    )

    df = None

    if data_mode == "Download cargo indicators now":
        df = _download_cargo_data()
    elif data_mode == "Use current session data":
        if "current_dataset" in st.session_state:
            df = st.session_state["current_dataset"]
            st.write(f"Using: **{st.session_state.get('current_dataset_name', 'session data')}**")
        else:
            st.info("No data in current session. Download data first or choose another source.")
            return
    elif data_mode == "Load saved dataset":
        datasets = list_saved_datasets()
        if datasets:
            chosen = st.selectbox("Select dataset", datasets, key="cargo_saved")
            if chosen:
                df = load_dataset(chosen)
        else:
            st.info("No saved datasets found.")
            return

    if df is None or df.empty:
        return

    # Check which cargo indicators are available
    available_cargo = [c for c in CARGO_INDICATOR_CODES if c in df.columns]
    all_indicators = _indicator_columns(df)

    if not available_cargo:
        st.warning(
            "No air transport indicators found in this dataset. "
            "Use **Download cargo indicators now** to fetch the relevant data."
        )
        return

    # --- Dataset overview ---
    countries = sorted(df["country"].unique()) if "country" in df.columns else []
    with st.expander("Dataset Overview", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", len(df))
        col2.metric("Indicators", len(all_indicators))
        col3.metric("Countries", len(countries))
        col4.metric("Year Range", f"{int(df['year'].min())}-{int(df['year'].max())}")
        st.dataframe(df.describe(), use_container_width=True)

    # --- Analysis selection ---
    st.divider()
    st.subheader("\U0001F52C Analysis Type")

    analysis = st.selectbox(
        "Choose analysis",
        [
            "Freight Volume Trends",
            "Country Rankings",
            "Year-over-Year Growth",
            "Cargo Intensity (per GDP & per Capita)",
            "Economic Correlations",
            "Growth Drivers (ML)",
            "Multi-Indicator Comparison",
        ],
        key="cargo_analysis_type",
    )

    st.divider()

    if analysis == "Freight Volume Trends":
        _render_freight_trends(df, available_cargo, countries)
    elif analysis == "Country Rankings":
        _render_rankings(df, available_cargo, countries)
    elif analysis == "Year-over-Year Growth":
        _render_yoy_growth(df, available_cargo, countries)
    elif analysis == "Cargo Intensity (per GDP & per Capita)":
        _render_cargo_intensity(df, countries)
    elif analysis == "Economic Correlations":
        _render_economic_correlations(df, available_cargo)
    elif analysis == "Growth Drivers (ML)":
        _render_growth_drivers(df, available_cargo)
    elif analysis == "Multi-Indicator Comparison":
        _render_multi_indicator(df, available_cargo, all_indicators, countries)


def _download_cargo_data():
    """Handle downloading cargo-specific indicators."""
    st.markdown("Select countries and year range to download air transport data.")

    # Country selection
    groups = get_country_groups()
    group_names = list(groups.keys())
    selected_group = st.selectbox(
        "Country group",
        ["Custom"] + group_names,
        index=1,
        key="cargo_dl_group",
    )

    if selected_group == "Custom":
        country_input = st.text_input(
            "Country codes (comma-separated ISO3)",
            value="USA,CHN,DEU,JPN,GBR,KOR,SGP,ARE,IND,BRA",
            key="cargo_dl_countries",
        )
        country_list = [c.strip() for c in country_input.split(",") if c.strip()]
    else:
        country_list = groups[selected_group]
        st.write(f"Countries: {', '.join(country_list)}")

    # Year range
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start year", min_value=1970, max_value=2025, value=2000, key="cargo_dl_start")
    with col2:
        end_year = st.number_input("End year", min_value=1970, max_value=2025, value=2025, key="cargo_dl_end")

    # Indicator selection
    cargo_indicators = INDICATOR_CATEGORIES.get("Air Transport & Cargo", {})
    include_economic = st.checkbox("Include economic context indicators (GDP, trade, population)", value=True, key="cargo_dl_econ")

    indicators_to_download = list(cargo_indicators.keys())
    if include_economic:
        indicators_to_download += list(ECONOMIC_CONTEXT_CODES.keys())
    # Deduplicate while preserving order
    seen = set()
    unique_indicators = []
    for ind in indicators_to_download:
        if ind not in seen:
            seen.add(ind)
            unique_indicators.append(ind)
    indicators_to_download = unique_indicators

    st.write(f"Will download **{len(indicators_to_download)}** indicators for **{len(country_list)}** countries")

    dataset_name = st.text_input("Dataset name", value="cargo_analysis", key="cargo_dl_name")

    if st.button("Download Cargo Data", type="primary", key="cargo_dl_btn"):
        progress = st.progress(0)
        status = st.empty()

        def progress_cb(i, total, label):
            progress.progress(i / total)
            status.text(f"Downloading {i+1}/{total}: {label}")

        with st.spinner("Downloading data from World Bank..."):
            df, failed_indicators = download_multiple_indicators(
                indicators_to_download,
                country_list,
                int(start_year),
                int(end_year),
                progress_callback=progress_cb,
            )

        progress.progress(1.0)
        status.text("Download complete!")

        if failed_indicators:
            st.warning(
                f"**{len(failed_indicators)} indicator(s) failed to download** "
                f"(API errors) and were skipped:\n"
                + "\n".join(f"- {name}" for name in failed_indicators)
            )

        if df is not None and not df.empty:
            # Save and store in session
            save_dataset(df, dataset_name)
            st.session_state["current_dataset"] = df
            st.session_state["current_dataset_name"] = dataset_name

            all_ind = get_all_indicators()
            all_ind.update(ECONOMIC_CONTEXT_CODES)
            st.session_state["indicator_names"] = all_ind

            st.success(f"Downloaded {len(df)} rows with {len(_indicator_columns(df))} indicators. Saved as '{dataset_name}'.")
            return df
        else:
            st.error("Download failed or returned empty data.")
            return None

    return None


def _render_freight_trends(df, cargo_cols, countries):
    """Render freight volume trend analysis."""
    st.subheader("Air Freight Volume Trends")

    freight_col = st.selectbox(
        "Cargo indicator",
        cargo_cols,
        format_func=_get_indicator_label,
        key="trend_indicator",
    )

    result = compute_cargo_trends(df, freight_col)
    if "error" in result:
        st.error(result["error"])
        return

    # Global trend
    if "global_trend" in result:
        trend = result["global_trend"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Trend Direction", trend["direction"].title())
        col2.metric("R-squared", f"{trend['r_squared']:.4f}")
        col3.metric("P-value", f"{trend['p_value']:.6f}")

    # Yearly aggregate chart
    if "yearly_aggregates" in result:
        yearly = result["yearly_aggregates"]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=yearly["year"],
            y=yearly["total"],
            name="Total (all countries)",
            marker_color=BRASS,
        ))
        fig.add_trace(go.Scatter(
            x=yearly["year"],
            y=yearly["mean"],
            name="Mean per country",
            yaxis="y2",
            mode="lines+markers",
            marker_color=EMBER,
            line_color=EMBER,
        ))
        fig.update_layout(
            title=f"Global {_get_indicator_label(freight_col)} by Year",
            xaxis_title="Year",
            yaxis_title="Total",
            yaxis2=dict(title="Mean per Country", overlaying="y", side="right",
                        titlefont=dict(color=EMBER), tickfont=dict(color=EMBER)),
            hovermode="x unified",
        )
        apply_steam_style(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Per-country time series
    st.markdown("#### Per-Country Trends")
    selected = st.multiselect(
        "Select countries",
        countries,
        default=countries[:8] if len(countries) > 8 else countries,
        key="trend_countries",
    )

    if selected:
        filtered = df[df["country"].isin(selected)][["country", "year", freight_col]].dropna()
        if not filtered.empty:
            fig = px.line(
                filtered,
                x="year",
                y=freight_col,
                color="country",
                title=f"{_get_indicator_label(freight_col)} by Country",
                labels={freight_col: _get_indicator_label(freight_col), "year": "Year"},
                color_discrete_sequence=CHART_COLORS,
            )
            fig.update_layout(hovermode="x unified")
            apply_steam_style(fig)
            st.plotly_chart(fig, use_container_width=True)

    # Country growth table
    if "country_growth" in result and not result["country_growth"].empty:
        st.markdown("#### Country Growth Summary")
        growth = result["country_growth"].copy()
        growth["cagr_pct"] = growth["cagr"] * 100
        display_cols = ["country", "first_year", "last_year", "first_value", "last_value", "total_change_pct", "cagr_pct", "avg_annual"]
        display_cols = [c for c in display_cols if c in growth.columns]
        st.dataframe(
            growth[display_cols].rename(columns={
                "total_change_pct": "Total Change %",
                "cagr_pct": "CAGR %",
                "avg_annual": "Avg Annual Volume",
            }),
            use_container_width=True,
        )


def _render_rankings(df, cargo_cols, countries):
    """Render country rankings by cargo volume."""
    st.subheader("Country Rankings by Cargo Volume")

    freight_col = st.selectbox(
        "Cargo indicator",
        cargo_cols,
        format_func=_get_indicator_label,
        key="rank_indicator",
    )

    rankings = compute_cargo_rankings(df, freight_col)
    if rankings.empty:
        st.warning("No ranking data available.")
        return

    # Latest year ranking
    latest_year = int(rankings["year"].max())
    year = st.slider("Year", int(rankings["year"].min()), latest_year, latest_year, key="rank_year")

    year_data = rankings[rankings["year"] == year].sort_values("rank")
    if year_data.empty:
        st.warning(f"No data for year {year}.")
        return

    top_n = st.slider("Show top N countries", 5, min(30, len(year_data)), min(15, len(year_data)), key="rank_top_n")
    top = year_data.head(top_n)

    fig = px.bar(
        top,
        x="country",
        y=freight_col,
        color=freight_col,
        color_continuous_scale=HEATMAP_SCALE,
        title=f"Top {top_n} Countries by {_get_indicator_label(freight_col)} ({year})",
        labels={freight_col: _get_indicator_label(freight_col)},
    )
    fig.update_layout(xaxis_tickangle=-45)
    apply_steam_style(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Ranking table
    st.dataframe(
        top[["country", freight_col, "rank"]].rename(columns={
            freight_col: _get_indicator_label(freight_col),
            "rank": "Rank",
        }),
        use_container_width=True,
    )

    # Animated ranking over time (bubble chart)
    st.markdown("#### Rankings Over Time")
    if len(rankings["year"].unique()) > 1:
        top_countries = year_data.head(10)["country"].tolist()
        anim_data = rankings[rankings["country"].isin(top_countries)].copy()
        anim_data["label"] = anim_data["country"]

        fig = px.bar(
            anim_data.sort_values(["year", freight_col], ascending=[True, False]),
            x="country",
            y=freight_col,
            color="country",
            animation_frame="year",
            title=f"Top Countries {_get_indicator_label(freight_col)} Over Time",
            labels={freight_col: _get_indicator_label(freight_col)},
            color_discrete_sequence=CHART_COLORS,
        )
        fig.update_layout(showlegend=False)
        apply_steam_style(fig)
        st.plotly_chart(fig, use_container_width=True)


def _render_yoy_growth(df, cargo_cols, countries):
    """Render year-over-year growth analysis."""
    st.subheader("Year-over-Year Growth Rates")

    freight_col = st.selectbox(
        "Cargo indicator",
        cargo_cols,
        format_func=_get_indicator_label,
        key="yoy_indicator",
    )

    growth_df = compute_yoy_growth(df, freight_col)
    if growth_df.empty:
        st.warning("Not enough data to compute growth rates.")
        return

    # Global average YoY growth by year
    yearly_growth = growth_df.groupby("year")["yoy_growth_pct"].agg(["mean", "median"]).reset_index()
    yearly_growth.columns = ["year", "mean_growth", "median_growth"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yearly_growth["year"],
        y=yearly_growth["mean_growth"],
        name="Mean Growth %",
        marker_color=[BRASS if v >= 0 else EMBER for v in yearly_growth["mean_growth"]],
    ))
    fig.add_trace(go.Scatter(
        x=yearly_growth["year"],
        y=yearly_growth["median_growth"],
        name="Median Growth %",
        mode="lines+markers",
        marker_color=COPPER,
        line_color=COPPER,
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=STEEL)
    fig.update_layout(
        title=f"Year-over-Year Growth: {_get_indicator_label(freight_col)}",
        xaxis_title="Year",
        yaxis_title="Growth %",
        hovermode="x unified",
    )
    apply_steam_style(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Per-country growth
    st.markdown("#### Per-Country Growth")
    selected = st.multiselect(
        "Select countries",
        countries,
        default=countries[:6] if len(countries) > 6 else countries,
        key="yoy_countries",
    )

    if selected:
        filtered = growth_df[growth_df["country"].isin(selected)]
        if not filtered.empty:
            fig = px.line(
                filtered,
                x="year",
                y="yoy_growth_pct",
                color="country",
                title="Year-over-Year Growth by Country",
                labels={"yoy_growth_pct": "Growth %", "year": "Year"},
                color_discrete_sequence=CHART_COLORS,
            )
            fig.add_hline(y=0, line_dash="dash", line_color=STEEL)
            fig.update_layout(hovermode="x unified")
            apply_steam_style(fig)
            st.plotly_chart(fig, use_container_width=True)

    # Volatility summary
    st.markdown("#### Growth Volatility by Country")
    volatility = growth_df.groupby("country")["yoy_growth_pct"].agg(["mean", "std", "min", "max"]).reset_index()
    volatility.columns = ["Country", "Mean Growth %", "Std Dev %", "Min Growth %", "Max Growth %"]
    volatility = volatility.sort_values("Mean Growth %", ascending=False)
    st.dataframe(volatility, use_container_width=True)


def _render_cargo_intensity(df, countries):
    """Render cargo intensity analysis (freight per GDP, per capita)."""
    st.subheader("Cargo Intensity Analysis")

    intensity = compute_cargo_intensity(df)
    if intensity.empty:
        st.warning("Cannot compute intensity. Ensure dataset includes air freight, GDP, and population indicators.")
        return

    has_gdp = "freight_per_gdp" in intensity.columns
    has_pop = "freight_per_capita" in intensity.columns

    if not has_gdp and not has_pop:
        st.warning("Need GDP or Population data alongside freight data to compute intensity metrics.")
        return

    metric_options = []
    if has_gdp:
        metric_options.append("Freight per GDP")
    if has_pop:
        metric_options.append("Freight per Capita")

    metric = st.selectbox("Intensity metric", metric_options, key="intensity_metric")
    col_name = "freight_per_gdp" if metric == "Freight per GDP" else "freight_per_capita"

    # Time series of intensity
    selected = st.multiselect(
        "Countries",
        countries,
        default=countries[:8] if len(countries) > 8 else countries,
        key="intensity_countries",
    )

    if selected:
        filtered = intensity[intensity["country"].isin(selected)][["country", "year", col_name]].dropna()
        if not filtered.empty:
            fig = px.line(
                filtered,
                x="year",
                y=col_name,
                color="country",
                title=f"{metric} Over Time",
                labels={col_name: metric, "year": "Year"},
                color_discrete_sequence=CHART_COLORS,
            )
            fig.update_layout(hovermode="x unified")
            apply_steam_style(fig)
            st.plotly_chart(fig, use_container_width=True)

    # Latest year comparison
    latest = int(intensity["year"].max())
    latest_data = intensity[intensity["year"] == latest][["country", col_name]].dropna()
    latest_data = latest_data.sort_values(col_name, ascending=False)

    if not latest_data.empty:
        st.markdown(f"#### {metric} by Country ({latest})")
        fig = px.bar(
            latest_data,
            x="country",
            y=col_name,
            color=col_name,
            color_continuous_scale=HEATMAP_SCALE,
            title=f"{metric} ({latest})",
            labels={col_name: metric},
        )
        fig.update_layout(xaxis_tickangle=-45)
        apply_steam_style(fig)
        st.plotly_chart(fig, use_container_width=True)


def _render_economic_correlations(df, cargo_cols):
    """Render correlation analysis between cargo and economic indicators."""
    st.subheader("Cargo-Economic Correlations")

    freight_col = st.selectbox(
        "Cargo indicator",
        cargo_cols,
        format_func=_get_indicator_label,
        key="econcorr_indicator",
    )

    result = compute_cargo_economic_correlation(df, freight_col)
    if "error" in result:
        st.error(result["error"])
        return

    corr_df = result["correlations"]
    if corr_df.empty:
        st.warning("No correlations could be computed. Dataset may lack economic indicators.")
        return

    corr_df["indicator_name"] = corr_df["indicator"].map(_get_indicator_label)

    # Bar chart of Pearson correlations
    fig = px.bar(
        corr_df,
        x="indicator_name",
        y="pearson_r",
        color="pearson_r",
        color_continuous_scale=DIVERGING_SCALE,
        color_continuous_midpoint=0,
        title=f"Pearson Correlation with {_get_indicator_label(freight_col)}",
        labels={"pearson_r": "Pearson r", "indicator_name": "Indicator"},
    )
    fig.update_layout(xaxis_tickangle=-45)
    apply_steam_style(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation table
    st.markdown("#### Correlation Details")
    display = corr_df[["indicator_name", "pearson_r", "pearson_p", "spearman_r", "spearman_p", "n_observations"]].copy()
    display.columns = ["Indicator", "Pearson r", "Pearson p", "Spearman r", "Spearman p", "N"]
    display["Significant (5%)"] = display["Pearson p"].apply(lambda p: "Yes" if p < 0.05 else "No")
    st.dataframe(display, use_container_width=True)

    # Scatter plots for top correlations
    st.markdown("#### Scatter Plots (Top Correlations)")
    top_n = min(4, len(corr_df))
    top_indicators = corr_df.head(top_n)["indicator"].tolist()

    cols = st.columns(2)
    for idx, ind in enumerate(top_indicators):
        with cols[idx % 2]:
            scatter_data = df[[freight_col, ind, "country"]].dropna()
            if not scatter_data.empty:
                fig = px.scatter(
                    scatter_data,
                    x=freight_col,
                    y=ind,
                    color="country",
                    title=f"{_get_indicator_label(freight_col)} vs {_get_indicator_label(ind)}",
                    labels={
                        freight_col: _get_indicator_label(freight_col),
                        ind: _get_indicator_label(ind),
                    },
                    trendline="ols",
                    color_discrete_sequence=CHART_COLORS,
                )
                fig.update_layout(showlegend=False, height=400)
                apply_steam_style(fig)
                st.plotly_chart(fig, use_container_width=True)


def _render_growth_drivers(df, cargo_cols):
    """Render ML-based growth driver analysis."""
    st.subheader("Economic Drivers of Cargo Volume (Random Forest)")

    freight_col = st.selectbox(
        "Target cargo indicator",
        cargo_cols,
        format_func=_get_indicator_label,
        key="driver_indicator",
    )

    if st.button("Identify Growth Drivers", type="primary", key="driver_btn"):
        with st.spinner("Training Random Forest model..."):
            result = compute_cargo_growth_drivers(df, freight_col)

        if "error" in result:
            st.error(result["error"])
            return

        col1, col2, col3 = st.columns(3)
        col1.metric("Model R-squared", f"{result['r2_score']:.4f}")
        col2.metric("Samples", result["n_samples"])
        col3.metric("Features", result["n_features"])

        importances = result["importances"].copy()
        importances["indicator_name"] = importances["indicator"].map(_get_indicator_label)

        fig = px.bar(
            importances.head(15),
            x="importance",
            y="indicator_name",
            orientation="h",
            title=f"Top Drivers of {_get_indicator_label(freight_col)}",
            labels={"importance": "Feature Importance", "indicator_name": "Indicator"},
            color="importance",
            color_continuous_scale=HEATMAP_SCALE,
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        apply_steam_style(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            importances[["indicator_name", "importance"]].rename(columns={
                "indicator_name": "Indicator",
                "importance": "Feature Importance",
            }),
            use_container_width=True,
        )

        if result["r2_score"] > 0.8:
            st.success(f"Strong model fit (R-squared = {result['r2_score']:.3f}). The identified features explain most of the variance in cargo volumes.")
        elif result["r2_score"] > 0.5:
            st.info(f"Moderate model fit (R-squared = {result['r2_score']:.3f}).")
        else:
            st.warning(f"Weak model fit (R-squared = {result['r2_score']:.3f}). Cargo volumes may be driven by factors not in this dataset.")


def _render_multi_indicator(df, cargo_cols, all_indicators, countries):
    """Render multi-indicator comparison view."""
    st.subheader("Multi-Indicator Comparison")

    # Select two indicators to compare
    col1, col2 = st.columns(2)
    with col1:
        x_ind = st.selectbox(
            "X-axis indicator",
            all_indicators,
            index=all_indicators.index(cargo_cols[0]) if cargo_cols[0] in all_indicators else 0,
            format_func=_get_indicator_label,
            key="multi_x",
        )
    with col2:
        other_inds = [i for i in all_indicators if i != x_ind]
        y_ind = st.selectbox(
            "Y-axis indicator",
            other_inds,
            format_func=_get_indicator_label,
            key="multi_y",
        )

    color_by = st.radio("Color by", ["Country", "Year"], horizontal=True, key="multi_color")

    selected_countries = st.multiselect(
        "Filter countries",
        countries,
        default=countries,
        key="multi_countries",
    )

    filtered = df[df["country"].isin(selected_countries)][[x_ind, y_ind, "country", "year"]].dropna()

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

    # Heatmap: cargo indicator by country and year
    st.markdown("#### Heatmap View")
    heatmap_ind = st.selectbox(
        "Heatmap indicator",
        cargo_cols,
        format_func=_get_indicator_label,
        key="multi_heatmap",
    )

    hm_data = df[df["country"].isin(selected_countries)]
    pivot = hm_data.pivot_table(index="country", columns="year", values=heatmap_ind)
    if not pivot.empty:
        fig = px.imshow(
            pivot,
            title=f"{_get_indicator_label(heatmap_ind)} by Country & Year",
            labels={"color": _get_indicator_label(heatmap_ind)},
            aspect="auto",
            color_continuous_scale=HEATMAP_SCALE,
        )
        apply_steam_style(fig)
        st.plotly_chart(fig, use_container_width=True)
