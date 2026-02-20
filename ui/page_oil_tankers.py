"""Oil Tanker Analysis page for the Streamlit app.

Provides visualizations and analysis of global oil production,
tanker routes, trade flows between countries, and a deep-dive
into US oil metrics.
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
    save_dataset,
)
from data_sources.oil_data import (
    OIL_INDICATOR_CATEGORIES,
    OIL_RENTS_CODE,
    FUEL_IMPORTS_CODE,
    FUEL_EXPORTS_CODE,
    ENERGY_IMPORTS_CODE,
    ENERGY_USE_CODE,
    FOSSIL_FUEL_CODE,
    OIL_ELECTRICITY_CODE,
    GDP_CODE,
    POP_CODE,
    OIL_CORE_CODES,
    MAJOR_TANKER_ROUTES,
    get_oil_indicators,
    get_oil_country_groups,
    get_all_oil_indicators,
    download_oil_data,
)
from analysis.oil_analysis import (
    compute_oil_production_trends,
    compute_oil_trade_flows,
    compute_tanker_route_estimates,
    compute_us_oil_profile,
    compute_oil_dependency,
    compute_oil_economic_correlations,
    compute_oil_drivers,
    compute_oil_yoy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_indicator_label(code: str) -> str:
    """Resolve indicator code to human-readable label."""
    names = st.session_state.get("indicator_names", {})
    if code in names:
        return names[code]
    oil_ind = get_oil_indicators()
    if code in oil_ind:
        return oil_ind[code]
    all_ind = get_all_indicators()
    return all_ind.get(code, code)


def _indicator_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("country", "year")]


def _oil_indicator_columns(df: pd.DataFrame) -> list[str]:
    """Return indicator columns present in the dataframe that are oil-related."""
    oil_codes = set(get_all_oil_indicators())
    return [c for c in df.columns if c in oil_codes]


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    st.header("Oil Tanker & Petroleum Analysis")
    st.markdown(
        "Analyze global oil production trends, trade flows between countries, "
        "major tanker routes, oil dependency, and US-focused petroleum metrics."
    )

    # --- Data source ---
    st.subheader("Data Source")
    data_mode = st.radio(
        "How to load data",
        [
            "Download oil indicators now",
            "Use current session data",
            "Load saved dataset",
        ],
        horizontal=True,
        key="oil_data_mode",
    )

    df = None

    if data_mode == "Download oil indicators now":
        df = _download_oil_data()
    elif data_mode == "Use current session data":
        if "current_dataset" in st.session_state:
            df = st.session_state["current_dataset"]
            st.write(
                f"Using: **{st.session_state.get('current_dataset_name', 'session data')}**"
            )
        else:
            st.info("No data in current session. Download data first or choose another source.")
            return
    elif data_mode == "Load saved dataset":
        datasets = list_saved_datasets()
        if datasets:
            chosen = st.selectbox("Select dataset", datasets, key="oil_saved")
            if chosen:
                df = load_dataset(chosen)
        else:
            st.info("No saved datasets found.")
            return

    if df is None or df.empty:
        return

    # --- Dataset overview ---
    available_oil = [c for c in OIL_CORE_CODES if c in df.columns]
    all_indicators = _indicator_columns(df)
    countries = sorted(df["country"].unique()) if "country" in df.columns else []

    if not available_oil:
        st.warning(
            "No oil-related indicators found in this dataset. "
            "Use **Download oil indicators now** to fetch the relevant data."
        )
        return

    with st.expander("Dataset Overview", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", len(df))
        c2.metric("Indicators", len(all_indicators))
        c3.metric("Countries", len(countries))
        c4.metric(
            "Year Range",
            f"{int(df['year'].min())}-{int(df['year'].max())}",
        )
        st.dataframe(df.describe(), use_container_width=True)

    # --- Analysis selection ---
    st.divider()
    st.subheader("Analysis Type")

    analysis = st.selectbox(
        "Choose analysis",
        [
            "Oil Production Trends",
            "Oil Trade Flows",
            "Major Tanker Routes",
            "US Oil Deep Dive",
            "Oil Dependency Scores",
            "Oil-Economic Correlations",
            "Oil Production Drivers (ML)",
        ],
        key="oil_analysis_type",
    )

    st.divider()

    if analysis == "Oil Production Trends":
        _render_production_trends(df, available_oil, countries)
    elif analysis == "Oil Trade Flows":
        _render_trade_flows(df, countries)
    elif analysis == "Major Tanker Routes":
        _render_tanker_routes(df, countries)
    elif analysis == "US Oil Deep Dive":
        _render_us_deep_dive(df, countries)
    elif analysis == "Oil Dependency Scores":
        _render_dependency(df, countries)
    elif analysis == "Oil-Economic Correlations":
        _render_correlations(df, available_oil)
    elif analysis == "Oil Production Drivers (ML)":
        _render_drivers(df, available_oil)


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def _download_oil_data():
    """UI section for downloading oil-specific data."""
    st.markdown("Select countries and year range to download oil & energy data.")

    groups = get_oil_country_groups()
    group_names = list(groups.keys())
    selected_group = st.selectbox(
        "Country group",
        ["Custom"] + group_names,
        index=1,  # default: Top Oil Producers
        key="oil_dl_group",
    )

    if selected_group == "Custom":
        country_input = st.text_input(
            "Country codes (comma-separated ISO3)",
            value="USA,SAU,RUS,CAN,IRQ,CHN,ARE,BRA,KWT,NOR,IND,JPN,KOR,DEU,NGA",
            key="oil_dl_countries",
        )
        country_list = [c.strip() for c in country_input.split(",") if c.strip()]
    else:
        country_list = groups[selected_group]
        st.write(f"Countries: {', '.join(country_list)}")

    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input(
            "Start year", min_value=1970, max_value=2025, value=2000,
            key="oil_dl_start",
        )
    with col2:
        end_year = st.number_input(
            "End year", min_value=1970, max_value=2025, value=2025,
            key="oil_dl_end",
        )

    include_economic = st.checkbox(
        "Include economic context indicators (GDP, population, inflation)",
        value=True,
        key="oil_dl_econ",
    )

    # Show what will be downloaded
    codes = get_all_oil_indicators()
    st.write(
        f"Will download **{len(codes)}** indicators "
        f"for **{len(country_list)}** countries"
    )

    dataset_name = st.text_input(
        "Dataset name", value="oil_tanker_analysis", key="oil_dl_name"
    )

    if st.button("Download Oil Data", type="primary", key="oil_dl_btn"):
        progress = st.progress(0)
        status = st.empty()

        def progress_cb(i, total, label):
            progress.progress(i / total)
            status.text(f"Downloading {i + 1}/{total}: {label}")

        with st.spinner("Downloading oil data from World Bank..."):
            df, failed_indicators = download_oil_data(
                country_list,
                int(start_year),
                int(end_year),
                include_economic=include_economic,
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
            save_dataset(df, dataset_name)
            st.session_state["current_dataset"] = df
            st.session_state["current_dataset_name"] = dataset_name

            oil_ind = get_oil_indicators()
            existing = st.session_state.get("indicator_names", {})
            existing.update(oil_ind)
            st.session_state["indicator_names"] = existing

            st.success(
                f"Downloaded {len(df)} rows with "
                f"{len(_indicator_columns(df))} indicators. "
                f"Saved as '{dataset_name}'."
            )
            return df
        else:
            st.error("Download failed or returned empty data.")
            return None

    return None


# ---------------------------------------------------------------------------
# 1. Oil Production Trends
# ---------------------------------------------------------------------------

def _render_production_trends(df, oil_cols, countries):
    st.subheader("Oil Production & Rents Trends")

    result = compute_oil_production_trends(df)
    if "error" in result:
        st.error(result["error"])
        return

    # Global trend metrics
    if "global_trend" in result:
        trend = result["global_trend"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Average Oil Rents Trend", trend["direction"].title())
        c2.metric("R-squared", f"{trend['r_squared']:.4f}")
        c3.metric("P-value", f"{trend['p_value']:.6f}")

    # Yearly aggregate chart
    if "yearly_aggregates" in result:
        yearly = result["yearly_aggregates"]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=yearly["year"], y=yearly["mean"],
            name="Mean Oil Rents (% of GDP)",
            marker_color="darkgoldenrod",
        ))
        fig.add_trace(go.Scatter(
            x=yearly["year"], y=yearly["median"],
            name="Median Oil Rents",
            mode="lines+markers",
            marker_color="firebrick",
        ))
        fig.update_layout(
            title="Average Oil Rents (% of GDP) Across Countries by Year",
            xaxis_title="Year",
            yaxis_title="Oil Rents (% of GDP)",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Per-country time series
    st.markdown("#### Per-Country Oil Rents")
    selected = st.multiselect(
        "Select countries",
        countries,
        default=countries[:10] if len(countries) > 10 else countries,
        key="oil_trend_countries",
    )

    if selected and OIL_RENTS_CODE in df.columns:
        filtered = df[df["country"].isin(selected)][
            ["country", "year", OIL_RENTS_CODE]
        ].dropna()
        if not filtered.empty:
            fig = px.line(
                filtered,
                x="year", y=OIL_RENTS_CODE, color="country",
                title="Oil Rents (% of GDP) by Country",
                labels={
                    OIL_RENTS_CODE: "Oil Rents (% of GDP)",
                    "year": "Year",
                },
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

    # Country statistics table
    if "country_stats" in result and not result["country_stats"].empty:
        st.markdown("#### Country Oil Production Summary")
        st.dataframe(
            result["country_stats"].rename(columns={
                "avg_oil_rents_pct": "Avg Oil Rents %",
                "peak_oil_rents_pct": "Peak Oil Rents %",
                "peak_year": "Peak Year",
                "change_ppts": "Change (pp)",
            }),
            use_container_width=True,
        )

    # YoY changes
    st.markdown("#### Year-over-Year Changes")
    yoy = compute_oil_yoy(df, OIL_RENTS_CODE)
    if not yoy.empty:
        yearly_yoy = yoy.groupby("year")["yoy_change_pct"].agg(
            ["mean", "median"]
        ).reset_index()
        yearly_yoy.columns = ["year", "mean_change", "median_change"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=yearly_yoy["year"], y=yearly_yoy["mean_change"],
            name="Mean YoY Change %",
            marker_color=[
                "green" if v >= 0 else "red"
                for v in yearly_yoy["mean_change"]
            ],
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title="Year-over-Year Change in Oil Rents",
            xaxis_title="Year",
            yaxis_title="Change %",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# 2. Oil Trade Flows
# ---------------------------------------------------------------------------

def _render_trade_flows(df, countries):
    st.subheader("Oil Trade Flows Between Countries")

    result = compute_oil_trade_flows(df)
    if "error" in result:
        st.error(result["error"])
        return

    latest_year = result["latest_year"]
    snap = result["latest_snapshot"]

    # Net position chart
    st.markdown(f"#### Net Oil Trade Position ({latest_year})")
    fig = px.bar(
        snap.sort_values("net_fuel_position_usd"),
        x="net_fuel_position_usd",
        y="country",
        orientation="h",
        color="position",
        color_discrete_map={"Net Exporter": "green", "Net Importer": "red"},
        title=f"Net Fuel Trade Position by Country ({latest_year})",
        labels={
            "net_fuel_position_usd": "Estimated Net Fuel Position (US$)",
            "country": "Country",
        },
    )
    fig.update_layout(height=max(400, len(snap) * 28))
    st.plotly_chart(fig, use_container_width=True)

    # Export vs import scatter
    st.markdown("#### Fuel Export vs Import Intensity")
    if FUEL_EXPORTS_CODE in snap.columns and FUEL_IMPORTS_CODE in snap.columns:
        fig = px.scatter(
            snap,
            x=FUEL_IMPORTS_CODE,
            y=FUEL_EXPORTS_CODE,
            text="country",
            size="est_fuel_export_usd",
            color="position",
            color_discrete_map={"Net Exporter": "green", "Net Importer": "red"},
            title=f"Fuel Import vs Export Intensity ({latest_year})",
            labels={
                FUEL_IMPORTS_CODE: "Fuel Imports (% of merchandise imports)",
                FUEL_EXPORTS_CODE: "Fuel Exports (% of merchandise exports)",
            },
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Time series of fuel exports for top exporters
    st.markdown("#### Fuel Export Trends Over Time")
    exporters = result["exporter_profiles"]
    top_export_countries = (
        snap.sort_values("est_fuel_export_usd", ascending=False)
        .head(10)["country"]
        .tolist()
    )
    selected_exp = st.multiselect(
        "Select exporter countries",
        countries,
        default=[c for c in top_export_countries if c in countries],
        key="oil_flow_exporters",
    )

    if selected_exp and FUEL_EXPORTS_CODE in df.columns:
        filtered = df[df["country"].isin(selected_exp)][
            ["country", "year", FUEL_EXPORTS_CODE]
        ].dropna()
        if not filtered.empty:
            fig = px.line(
                filtered,
                x="year", y=FUEL_EXPORTS_CODE, color="country",
                title="Fuel Exports (% of merchandise exports) Over Time",
                labels={
                    FUEL_EXPORTS_CODE: "Fuel Exports (%)",
                    "year": "Year",
                },
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

    # Time series of fuel imports for top importers
    st.markdown("#### Fuel Import Trends Over Time")
    top_import_countries = (
        snap.sort_values("est_fuel_import_usd", ascending=False)
        .head(10)["country"]
        .tolist()
    )
    selected_imp = st.multiselect(
        "Select importer countries",
        countries,
        default=[c for c in top_import_countries if c in countries],
        key="oil_flow_importers",
    )

    if selected_imp and FUEL_IMPORTS_CODE in df.columns:
        filtered = df[df["country"].isin(selected_imp)][
            ["country", "year", FUEL_IMPORTS_CODE]
        ].dropna()
        if not filtered.empty:
            fig = px.line(
                filtered,
                x="year", y=FUEL_IMPORTS_CODE, color="country",
                title="Fuel Imports (% of merchandise imports) Over Time",
                labels={
                    FUEL_IMPORTS_CODE: "Fuel Imports (%)",
                    "year": "Year",
                },
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.markdown("#### Trade Flow Summary")
    display_cols = [
        "country", "position", FUEL_EXPORTS_CODE, FUEL_IMPORTS_CODE,
        "est_fuel_export_usd", "est_fuel_import_usd", "net_fuel_position_usd",
    ]
    display_cols = [c for c in display_cols if c in snap.columns]
    st.dataframe(
        snap[display_cols].rename(columns={
            FUEL_EXPORTS_CODE: "Fuel Exports (%)",
            FUEL_IMPORTS_CODE: "Fuel Imports (%)",
            "est_fuel_export_usd": "Est. Fuel Export (US$)",
            "est_fuel_import_usd": "Est. Fuel Import (US$)",
            "net_fuel_position_usd": "Net Position (US$)",
            "position": "Position",
        }),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# 3. Major Tanker Routes
# ---------------------------------------------------------------------------

def _render_tanker_routes(df, countries):
    st.subheader("Major Oil Tanker Routes")
    st.markdown(
        "Overview of the world's principal oil shipping routes, chokepoints, "
        "and estimated traffic based on trade data."
    )

    # Route estimates from data
    route_df = compute_tanker_route_estimates(df)

    # Route share chart
    if not route_df.empty:
        fig = px.bar(
            route_df,
            x="estimated_share_pct",
            y="route",
            orientation="h",
            color="estimated_share_pct",
            color_continuous_scale="YlOrRd",
            title="Estimated Share of Global Oil Tanker Traffic by Route",
            labels={
                "estimated_share_pct": "Estimated Share (%)",
                "route": "Route",
            },
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Data-driven activity scores
    if not route_df.empty and "composite_score" in route_df.columns:
        st.markdown("#### Route Activity Scores (Data-Driven)")
        st.markdown(
            "Composite scores based on fuel export intensity of origin "
            "countries and fuel import intensity of destination countries."
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=route_df["route"],
            x=route_df["origin_export_score"],
            name="Origin Export Score",
            orientation="h",
            marker_color="darkgreen",
        ))
        fig.add_trace(go.Bar(
            y=route_df["route"],
            x=route_df["dest_import_score"],
            name="Destination Import Score",
            orientation="h",
            marker_color="steelblue",
        ))
        fig.update_layout(
            barmode="group",
            title="Route Activity Scores",
            xaxis_title="Score",
            yaxis=dict(autorange="reversed"),
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Chokepoint analysis
    st.markdown("#### Key Chokepoints")
    chokepoints = {}
    for route in MAJOR_TANKER_ROUTES:
        for cp in route["chokepoint"].split(","):
            cp = cp.strip()
            if cp and cp != "None":
                if cp not in chokepoints:
                    chokepoints[cp] = {"routes": [], "total_share": 0}
                chokepoints[cp]["routes"].append(route["route"])
                chokepoints[cp]["total_share"] += route["estimated_share_pct"]

    if chokepoints:
        cp_rows = [
            {
                "Chokepoint": cp,
                "Routes Through": ", ".join(info["routes"]),
                "Combined Share (%)": info["total_share"],
            }
            for cp, info in sorted(
                chokepoints.items(), key=lambda x: -x[1]["total_share"]
            )
        ]
        st.dataframe(pd.DataFrame(cp_rows), use_container_width=True)

    # Route detail cards
    st.markdown("#### Route Details")
    for route in MAJOR_TANKER_ROUTES:
        with st.expander(
            f"{route['route']} — ~{route['estimated_share_pct']}% of traffic"
        ):
            c1, c2 = st.columns(2)
            c1.markdown(f"**From:** {route['from_region']}")
            c1.markdown(f"**Origins:** {', '.join(route['from_countries'])}")
            c2.markdown(f"**To:** {route['to_region']}")
            c2.markdown(f"**Destinations:** {', '.join(route['to_countries'])}")
            st.markdown(f"**Chokepoints:** {route['chokepoint']}")
            st.markdown(route["description"])


# ---------------------------------------------------------------------------
# 4. US Oil Deep Dive
# ---------------------------------------------------------------------------

def _render_us_deep_dive(df, countries):
    st.subheader("United States Oil Profile")

    if "USA" not in df["country"].values:
        st.warning(
            "USA not found in dataset. Download data including USA to use "
            "this analysis."
        )
        return

    result = compute_us_oil_profile(df)
    if "error" in result:
        st.error(result["error"])
        return

    # Key metrics
    trends = result.get("trends", {})
    if trends:
        st.markdown("#### Key US Oil Metrics & Trends")
        cols = st.columns(min(4, len(trends)))
        for i, (code, t) in enumerate(trends.items()):
            with cols[i % len(cols)]:
                direction_icon = "+" if t["direction"] == "increasing" else "-"
                st.metric(
                    _get_indicator_label(code),
                    f"{t['latest_value']:.2f}",
                    delta=f"{direction_icon} {abs(t['slope']):.3f}/yr",
                )

    # Energy independence
    if "energy_independence" in result:
        ei = result["energy_independence"]
        st.markdown("#### US Energy Independence")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Latest Net Energy Imports (%)",
            f"{ei['latest_net_import_pct']:.1f}%",
        )
        c2.metric(
            "Peak Net Imports",
            f"{ei['max_net_import_pct']:.1f}% ({ei['max_year']})",
        )
        c3.metric(
            "Lowest Net Imports",
            f"{ei['min_net_import_pct']:.1f}%",
        )

        ts = ei["timeseries"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts["year"],
            y=ts[ENERGY_IMPORTS_CODE],
            fill="tozeroy",
            fillcolor="rgba(255,165,0,0.3)",
            line=dict(color="darkorange", width=2),
            name="Net Energy Imports (% of use)",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="green")
        fig.update_layout(
            title="US Net Energy Imports Over Time (% of energy use)",
            xaxis_title="Year",
            yaxis_title="Net Energy Imports (%)",
            annotations=[
                dict(
                    text="Net Exporter",
                    x=0.02, y=-5, xref="paper", yref="y",
                    showarrow=False, font=dict(color="green"),
                ),
            ],
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "A **negative** value means the US is a net energy exporter. "
            "The US has significantly reduced its energy import dependence "
            "since the shale revolution."
        )

    # US time series for all available oil indicators
    us_ts = result.get("us_timeseries")
    available = result.get("available_indicators", [])
    if us_ts is not None and available:
        st.markdown("#### US Oil Indicator Time Series")
        selected_indicators = st.multiselect(
            "Select indicators",
            available,
            default=available[:4],
            format_func=_get_indicator_label,
            key="us_indicators",
        )

        if selected_indicators:
            for ind in selected_indicators:
                series = us_ts[["year", ind]].dropna()
                if series.empty:
                    continue
                fig = px.line(
                    series,
                    x="year", y=ind,
                    title=f"US — {_get_indicator_label(ind)}",
                    labels={ind: _get_indicator_label(ind), "year": "Year"},
                )
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

    # Peer comparison
    if "peer_comparison" in result:
        st.markdown(
            f"#### US vs. Top Oil Producers "
            f"({result.get('peer_comparison_year', '')})"
        )
        comp = result["peer_comparison"].copy()
        comp_display = comp.rename(
            columns={c: _get_indicator_label(c) for c in comp.columns if c != "country"}
        )
        st.dataframe(comp_display, use_container_width=True)

        # Bar comparison for each indicator
        compare_ind = st.selectbox(
            "Compare indicator",
            [c for c in comp.columns if c != "country"],
            format_func=_get_indicator_label,
            key="us_compare_ind",
        )

        if compare_ind:
            comp_data = comp[["country", compare_ind]].dropna()
            comp_data = comp_data.sort_values(compare_ind, ascending=False)
            colors = [
                "darkblue" if c == "USA" else "lightsteelblue"
                for c in comp_data["country"]
            ]
            fig = go.Figure(go.Bar(
                x=comp_data["country"],
                y=comp_data[compare_ind],
                marker_color=colors,
            ))
            fig.update_layout(
                title=f"{_get_indicator_label(compare_ind)} — US vs Peers",
                xaxis_title="Country",
                yaxis_title=_get_indicator_label(compare_ind),
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# 5. Oil Dependency Scores
# ---------------------------------------------------------------------------

def _render_dependency(df, countries):
    st.subheader("Oil Dependency Scores")
    st.markdown(
        "Composite score combining fuel import share, energy import share, "
        "fossil fuel consumption, and oil-based electricity production."
    )

    dep = compute_oil_dependency(df)
    if dep.empty:
        st.warning(
            "Cannot compute dependency scores. Ensure dataset includes "
            "fuel imports, energy imports, fossil fuel consumption, and "
            "oil electricity indicators."
        )
        return

    # Bar chart
    fig = px.bar(
        dep.head(25),
        x="country",
        y="oil_dependency_score",
        color="oil_dependency_score",
        color_continuous_scale="Reds",
        title="Oil Dependency Score by Country (Top 25)",
        labels={
            "oil_dependency_score": "Dependency Score",
            "country": "Country",
        },
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Highlight US
    us_row = dep[dep["country"] == "USA"]
    if not us_row.empty:
        us_score = float(us_row["oil_dependency_score"].iloc[0])
        us_rank = int((dep["oil_dependency_score"] >= us_score).sum())
        st.info(
            f"**US Oil Dependency Score:** {us_score:.1f} "
            f"(Rank {us_rank} of {len(dep)} countries)"
        )

    # Full table
    display_cols = ["country", "oil_dependency_score"]
    for col in dep.columns:
        if col.endswith("_norm"):
            display_cols.append(col)

    available_display = [c for c in display_cols if c in dep.columns]
    st.dataframe(
        dep[available_display].rename(columns={
            "oil_dependency_score": "Dependency Score",
        }),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# 6. Oil-Economic Correlations
# ---------------------------------------------------------------------------

def _render_correlations(df, oil_cols):
    st.subheader("Oil-Economic Correlations")

    oil_col = st.selectbox(
        "Oil indicator",
        oil_cols,
        format_func=_get_indicator_label,
        key="oil_corr_indicator",
    )

    result = compute_oil_economic_correlations(df, oil_col)
    if "error" in result:
        st.error(result["error"])
        return

    corr_df = result["correlations"]
    if corr_df.empty:
        st.warning("No correlations could be computed.")
        return

    corr_df = corr_df.copy()
    corr_df["indicator_name"] = corr_df["indicator"].map(_get_indicator_label)

    # Bar chart
    fig = px.bar(
        corr_df,
        x="indicator_name",
        y="pearson_r",
        color="pearson_r",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        title=f"Pearson Correlation with {_get_indicator_label(oil_col)}",
        labels={"pearson_r": "Pearson r", "indicator_name": "Indicator"},
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.markdown("#### Correlation Details")
    display = corr_df[[
        "indicator_name", "pearson_r", "pearson_p",
        "spearman_r", "spearman_p", "n_observations",
    ]].copy()
    display.columns = [
        "Indicator", "Pearson r", "Pearson p",
        "Spearman r", "Spearman p", "N",
    ]
    display["Significant (5%)"] = display["Pearson p"].apply(
        lambda p: "Yes" if p < 0.05 else "No"
    )
    st.dataframe(display, use_container_width=True)

    # Scatter plots for top correlations
    st.markdown("#### Scatter Plots (Top Correlations)")
    top_n = min(4, len(corr_df))
    top_indicators = corr_df.head(top_n)["indicator"].tolist()

    cols = st.columns(2)
    for idx, ind in enumerate(top_indicators):
        with cols[idx % 2]:
            scatter_data = df[[oil_col, ind, "country"]].dropna()
            if not scatter_data.empty:
                fig = px.scatter(
                    scatter_data,
                    x=oil_col, y=ind, color="country",
                    title=(
                        f"{_get_indicator_label(oil_col)} vs "
                        f"{_get_indicator_label(ind)}"
                    ),
                    labels={
                        oil_col: _get_indicator_label(oil_col),
                        ind: _get_indicator_label(ind),
                    },
                    trendline="ols",
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# 7. Oil Production Drivers (ML)
# ---------------------------------------------------------------------------

def _render_drivers(df, oil_cols):
    st.subheader("Economic Drivers of Oil Production (Random Forest)")

    target = st.selectbox(
        "Target oil indicator",
        oil_cols,
        format_func=_get_indicator_label,
        key="oil_driver_target",
    )

    if st.button("Identify Drivers", type="primary", key="oil_driver_btn"):
        with st.spinner("Training Random Forest model..."):
            result = compute_oil_drivers(df, target)

        if "error" in result:
            st.error(result["error"])
            return

        c1, c2, c3 = st.columns(3)
        c1.metric("Model R-squared", f"{result['r2_score']:.4f}")
        c2.metric("Samples", result["n_samples"])
        c3.metric("Features", result["n_features"])

        importances = result["importances"].copy()
        importances["indicator_name"] = importances["indicator"].map(
            _get_indicator_label
        )

        fig = px.bar(
            importances.head(15),
            x="importance",
            y="indicator_name",
            orientation="h",
            title=f"Top Drivers of {_get_indicator_label(target)}",
            labels={
                "importance": "Feature Importance",
                "indicator_name": "Indicator",
            },
            color="importance",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            importances[["indicator_name", "importance"]].rename(columns={
                "indicator_name": "Indicator",
                "importance": "Feature Importance",
            }),
            use_container_width=True,
        )

        if result["r2_score"] > 0.8:
            st.success(
                f"Strong model fit (R-squared = {result['r2_score']:.3f}). "
                "The identified features explain most of the variance."
            )
        elif result["r2_score"] > 0.5:
            st.info(
                f"Moderate model fit (R-squared = {result['r2_score']:.3f})."
            )
        else:
            st.warning(
                f"Weak model fit (R-squared = {result['r2_score']:.3f}). "
                "Oil production may be driven by factors not in this dataset."
            )
