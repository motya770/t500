"""Oil tanker and petroleum analysis module.

Stateless analysis functions for oil production trends, trade flow
estimation, tanker route analysis, US-focused metrics, and oil
dependency scoring.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from data_sources.oil_data import (
    OIL_RENTS_CODE,
    FUEL_IMPORTS_CODE,
    FUEL_EXPORTS_CODE,
    ENERGY_IMPORTS_CODE,
    ENERGY_USE_CODE,
    FOSSIL_FUEL_CODE,
    OIL_ELECTRICITY_CODE,
    GDP_CODE,
    POP_CODE,
    MAJOR_TANKER_ROUTES,
)


# ---------------------------------------------------------------------------
# Oil production / rent trends
# ---------------------------------------------------------------------------

def compute_oil_production_trends(df: pd.DataFrame) -> dict:
    """Analyse oil-rent and energy production trends across countries.

    Uses oil rents (% of GDP) as a proxy for production intensity.

    Returns dict with global trend, yearly aggregates, and per-country
    growth statistics.
    """
    rent_col = OIL_RENTS_CODE
    if rent_col not in df.columns:
        return {"error": f"Column {rent_col} (Oil rents) not found in data"}

    result = {}

    # --- global yearly aggregates ---
    yearly = (
        df.groupby("year")[rent_col]
        .agg(["mean", "median", "sum", "count"])
        .reset_index()
    )
    yearly.columns = ["year", "mean", "median", "total", "num_countries"]

    if len(yearly) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            yearly["year"], yearly["mean"]
        )
        result["global_trend"] = {
            "slope_per_year": float(slope),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "direction": "increasing" if slope > 0 else "decreasing",
        }

    result["yearly_aggregates"] = yearly

    # --- per-country statistics ---
    country_stats = []
    for country in df["country"].unique():
        cdata = (
            df[df["country"] == country][["year", rent_col]]
            .dropna()
            .sort_values("year")
        )
        if len(cdata) < 2:
            continue

        first_val = cdata[rent_col].iloc[0]
        last_val = cdata[rent_col].iloc[-1]
        avg_val = cdata[rent_col].mean()
        max_val = cdata[rent_col].max()
        max_year = int(cdata.loc[cdata[rent_col].idxmax(), "year"])

        country_stats.append({
            "country": country,
            "first_year": int(cdata["year"].iloc[0]),
            "last_year": int(cdata["year"].iloc[-1]),
            "first_value": float(first_val),
            "last_value": float(last_val),
            "avg_oil_rents_pct": float(avg_val),
            "peak_oil_rents_pct": float(max_val),
            "peak_year": max_year,
            "change_ppts": float(last_val - first_val),
        })

    result["country_stats"] = pd.DataFrame(country_stats)
    if not result["country_stats"].empty:
        result["country_stats"] = result["country_stats"].sort_values(
            "avg_oil_rents_pct", ascending=False
        )

    return result


# ---------------------------------------------------------------------------
# Trade flow estimation
# ---------------------------------------------------------------------------

def compute_oil_trade_flows(df: pd.DataFrame) -> dict:
    """Estimate oil trade flow intensity between countries.

    Uses fuel import/export shares and GDP to approximate the relative
    magnitude of cross-border oil movements.

    Returns dict with:
      - exporter_profiles: DataFrame of export-side metrics per country/year
      - importer_profiles: DataFrame of import-side metrics per country/year
      - latest_snapshot: summarised latest-year data
    """
    needed = {FUEL_IMPORTS_CODE, FUEL_EXPORTS_CODE, GDP_CODE}
    missing = needed - set(df.columns)
    if missing:
        return {"error": f"Missing columns: {missing}"}

    base = df[["country", "year", FUEL_EXPORTS_CODE, FUEL_IMPORTS_CODE, GDP_CODE]].dropna()
    if base.empty:
        return {"error": "No data after dropping NaNs"}

    base = base.copy()

    # Estimated absolute fuel export/import value (US$)
    # fuel_exports_pct is % of merchandise exports; approximate via GDP share
    base["est_fuel_export_usd"] = base[GDP_CODE] * base[FUEL_EXPORTS_CODE] / 100
    base["est_fuel_import_usd"] = base[GDP_CODE] * base[FUEL_IMPORTS_CODE] / 100

    # Exporter profiles
    exporters = base.copy()
    exporters["export_intensity"] = exporters[FUEL_EXPORTS_CODE]
    exporters = exporters.sort_values(
        ["year", "est_fuel_export_usd"], ascending=[True, False]
    )

    # Importer profiles
    importers = base.copy()
    importers["import_intensity"] = importers[FUEL_IMPORTS_CODE]
    importers = importers.sort_values(
        ["year", "est_fuel_import_usd"], ascending=[True, False]
    )

    # Latest-year snapshot
    latest_year = int(base["year"].max())
    snap = base[base["year"] == latest_year].copy()
    snap["net_fuel_position_usd"] = snap["est_fuel_export_usd"] - snap["est_fuel_import_usd"]
    snap["position"] = snap["net_fuel_position_usd"].apply(
        lambda v: "Net Exporter" if v > 0 else "Net Importer"
    )
    snap = snap.sort_values("net_fuel_position_usd", ascending=False)

    return {
        "exporter_profiles": exporters,
        "importer_profiles": importers,
        "latest_snapshot": snap,
        "latest_year": latest_year,
    }


# ---------------------------------------------------------------------------
# Tanker route volume estimation
# ---------------------------------------------------------------------------

def compute_tanker_route_estimates(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate relative tanker traffic on major routes using country trade data.

    For each predefined route, sums the fuel-export intensity of origin
    countries and fuel-import intensity of destination countries to create
    a composite route activity score.

    Returns DataFrame with route metadata and estimated scores.
    """
    export_col = FUEL_EXPORTS_CODE
    import_col = FUEL_IMPORTS_CODE

    if export_col not in df.columns or import_col not in df.columns:
        return pd.DataFrame()

    latest_year = int(df["year"].max())
    snap = df[df["year"] == latest_year].set_index("country")

    rows = []
    for route in MAJOR_TANKER_ROUTES:
        # Sum export intensity of origin countries present in data
        origin_vals = []
        for c in route["from_countries"]:
            if c in snap.index and pd.notna(snap.loc[c, export_col]):
                origin_vals.append(float(snap.loc[c, export_col]))

        # Sum import intensity of destination countries present in data
        dest_vals = []
        for c in route["to_countries"]:
            if c in snap.index and pd.notna(snap.loc[c, import_col]):
                dest_vals.append(float(snap.loc[c, import_col]))

        origin_score = np.mean(origin_vals) if origin_vals else 0
        dest_score = np.mean(dest_vals) if dest_vals else 0
        composite = (origin_score + dest_score) / 2

        rows.append({
            "route": route["route"],
            "from_region": route["from_region"],
            "to_region": route["to_region"],
            "chokepoint": route["chokepoint"],
            "estimated_share_pct": route["estimated_share_pct"],
            "origin_export_score": float(origin_score),
            "dest_import_score": float(dest_score),
            "composite_score": float(composite),
            "description": route["description"],
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("estimated_share_pct", ascending=False)
    return result


# ---------------------------------------------------------------------------
# US-focused oil profile
# ---------------------------------------------------------------------------

def compute_us_oil_profile(df: pd.DataFrame) -> dict:
    """Build a comprehensive oil profile for the United States.

    Returns dict with time series, trend statistics, and comparison
    against peer countries.
    """
    if "country" not in df.columns:
        return {"error": "No 'country' column in data"}

    us_data = df[df["country"] == "USA"].sort_values("year").copy()
    if us_data.empty:
        return {"error": "No data for USA"}

    result = {"us_timeseries": us_data}

    # Key oil metrics over time
    oil_cols = [
        c for c in [
            OIL_RENTS_CODE, FUEL_IMPORTS_CODE, FUEL_EXPORTS_CODE,
            ENERGY_IMPORTS_CODE, ENERGY_USE_CODE, FOSSIL_FUEL_CODE,
            OIL_ELECTRICITY_CODE,
        ]
        if c in us_data.columns
    ]
    result["available_indicators"] = oil_cols

    # Trend for each available indicator
    trends = {}
    for col in oil_cols:
        series = us_data[["year", col]].dropna()
        if len(series) >= 3:
            slope, intercept, r_value, p_value, _ = stats.linregress(
                series["year"], series[col]
            )
            trends[col] = {
                "slope": float(slope),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "direction": "increasing" if slope > 0 else "decreasing",
                "latest_value": float(series[col].iloc[-1]),
                "latest_year": int(series["year"].iloc[-1]),
            }
    result["trends"] = trends

    # Energy independence proxy: net energy imports trend
    if ENERGY_IMPORTS_CODE in us_data.columns:
        ei = us_data[["year", ENERGY_IMPORTS_CODE]].dropna()
        if not ei.empty:
            result["energy_independence"] = {
                "latest_net_import_pct": float(ei[ENERGY_IMPORTS_CODE].iloc[-1]),
                "max_net_import_pct": float(ei[ENERGY_IMPORTS_CODE].max()),
                "max_year": int(
                    ei.loc[ei[ENERGY_IMPORTS_CODE].idxmax(), "year"]
                ),
                "min_net_import_pct": float(ei[ENERGY_IMPORTS_CODE].min()),
                "timeseries": ei,
            }

    # Comparison with top producers
    top_producers = ["SAU", "RUS", "CAN", "IRQ", "CHN", "ARE", "BRA", "NOR"]
    compare_countries = ["USA"] + [
        c for c in top_producers if c in df["country"].unique()
    ]
    comparison_cols = [c for c in oil_cols if c in df.columns]
    if comparison_cols:
        latest_year = int(df["year"].max())
        comp = df[
            (df["country"].isin(compare_countries)) & (df["year"] == latest_year)
        ][["country"] + comparison_cols].dropna(how="all")
        result["peer_comparison"] = comp
        result["peer_comparison_year"] = latest_year

    return result


# ---------------------------------------------------------------------------
# Oil dependency scores
# ---------------------------------------------------------------------------

def compute_oil_dependency(df: pd.DataFrame) -> pd.DataFrame:
    """Compute oil dependency score for each country in the latest year.

    Combines fuel import share, energy import share, fossil fuel
    consumption share, and oil electricity share into a composite score.

    Returns DataFrame sorted by dependency score descending.
    """
    cols_used = []
    weights = {}

    for col, w in [
        (FUEL_IMPORTS_CODE, 0.30),
        (ENERGY_IMPORTS_CODE, 0.25),
        (FOSSIL_FUEL_CODE, 0.25),
        (OIL_ELECTRICITY_CODE, 0.20),
    ]:
        if col in df.columns:
            cols_used.append(col)
            weights[col] = w

    if not cols_used:
        return pd.DataFrame()

    # Normalise weights to sum to 1
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    latest_year = int(df["year"].max())
    snap = df[df["year"] == latest_year][["country"] + cols_used].dropna(how="all")
    if snap.empty:
        return pd.DataFrame()

    snap = snap.copy()

    # Normalise each indicator to 0-100 range across countries
    for col in cols_used:
        vals = snap[col]
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            snap[f"{col}_norm"] = (vals - vmin) / (vmax - vmin) * 100
        else:
            snap[f"{col}_norm"] = 50.0

    # Weighted composite
    snap["oil_dependency_score"] = sum(
        snap[f"{col}_norm"] * weights[col] for col in cols_used
    )

    snap = snap.sort_values("oil_dependency_score", ascending=False)
    return snap


# ---------------------------------------------------------------------------
# Oil-economic correlation
# ---------------------------------------------------------------------------

def compute_oil_economic_correlations(
    df: pd.DataFrame,
    oil_col: str = OIL_RENTS_CODE,
) -> dict:
    """Compute Pearson and Spearman correlations between an oil indicator
    and every other numeric column.

    Returns dict with correlations DataFrame.
    """
    if oil_col not in df.columns:
        return {"error": f"Column {oil_col} not found"}

    other_cols = [
        c for c in df.columns
        if c not in ("country", "year") and c != oil_col
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    results = []
    for col in other_cols:
        valid = df[[oil_col, col]].dropna()
        if len(valid) < 5:
            continue

        pearson_r, pearson_p = stats.pearsonr(valid[oil_col], valid[col])
        spearman_r, spearman_p = stats.spearmanr(valid[oil_col], valid[col])

        results.append({
            "indicator": col,
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "abs_pearson": abs(float(pearson_r)),
            "n_observations": len(valid),
        })

    corr_df = pd.DataFrame(results)
    if not corr_df.empty:
        corr_df = corr_df.sort_values("abs_pearson", ascending=False)

    return {"correlations": corr_df}


# ---------------------------------------------------------------------------
# Oil production drivers (ML)
# ---------------------------------------------------------------------------

def compute_oil_drivers(
    df: pd.DataFrame,
    target_col: str = OIL_RENTS_CODE,
) -> dict:
    """Use Random Forest to identify economic drivers of oil production.

    Returns feature importances and model R-squared.
    """
    if target_col not in df.columns:
        return {"error": f"Column {target_col} not found"}

    other_cols = [
        c for c in df.columns
        if c not in ("country", "year") and c != target_col
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not other_cols:
        return {"error": "No numeric predictor columns found"}

    analysis_df = df[[target_col] + other_cols].dropna()
    if len(analysis_df) < 20:
        return {"error": f"Not enough data ({len(analysis_df)} rows, need 20+)"}

    X = analysis_df[other_cols]
    y = analysis_df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)
    score = model.score(X_scaled, y)

    importances = pd.DataFrame({
        "indicator": other_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "importances": importances,
        "r2_score": float(score),
        "n_samples": len(analysis_df),
        "n_features": len(other_cols),
    }


# ---------------------------------------------------------------------------
# Year-over-year change
# ---------------------------------------------------------------------------

def compute_oil_yoy(
    df: pd.DataFrame,
    col: str = OIL_RENTS_CODE,
) -> pd.DataFrame:
    """Compute year-over-year change for an oil indicator by country."""
    if col not in df.columns:
        return pd.DataFrame()

    subset = df[["country", "year", col]].dropna().sort_values(["country", "year"])
    subset = subset.copy()
    subset["prev_value"] = subset.groupby("country")[col].shift(1)
    subset["yoy_change"] = subset[col] - subset["prev_value"]
    subset["yoy_change_pct"] = (
        (subset[col] - subset["prev_value"]) / subset["prev_value"].replace(0, np.nan)
    ) * 100
    subset = subset.dropna(subset=["yoy_change_pct"])

    return subset[["country", "year", col, "yoy_change", "yoy_change_pct"]]
