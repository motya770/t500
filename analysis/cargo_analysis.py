"""Cargo plane analysis module.

Provides specialized analysis functions for air freight and cargo transport
data, including trend analysis, economic correlations, and ranking computations.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Core air transport indicator codes
CARGO_FREIGHT_CODE = "IS.AIR.GOOD.MT.K1"
CARGO_PASSENGERS_CODE = "IS.AIR.PSGR"
CARGO_DEPARTURES_CODE = "IS.AIR.DPRT"
CONTAINER_PORT_CODE = "IS.SHP.GOOD.TU"

CARGO_INDICATOR_CODES = [
    CARGO_FREIGHT_CODE,
    CARGO_PASSENGERS_CODE,
    CARGO_DEPARTURES_CODE,
    CONTAINER_PORT_CODE,
]

# Economic indicators commonly correlated with air cargo
ECONOMIC_CONTEXT_CODES = {
    "NY.GDP.MKTP.CD": "GDP (current US$)",
    "NY.GDP.PCAP.CD": "GDP per capita (current US$)",
    "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
    "NE.EXP.GNFS.ZS": "Exports of goods and services (% of GDP)",
    "NE.IMP.GNFS.ZS": "Imports of goods and services (% of GDP)",
    "TG.VAL.TOTL.GD.ZS": "Merchandise trade (% of GDP)",
    "BX.KLT.DINV.WD.GD.ZS": "Foreign direct investment, net inflows (% of GDP)",
    "SP.POP.TOTL": "Population, total",
    "IT.NET.USER.ZS": "Individuals using the Internet (% of population)",
    "SP.URB.TOTL.IN.ZS": "Urban population (% of total population)",
}


def compute_cargo_trends(df: pd.DataFrame, freight_col: str = CARGO_FREIGHT_CODE) -> dict:
    """Compute trend statistics for air freight data across countries.

    Returns dict with global trend, per-country trends, and growth rates.
    """
    if freight_col not in df.columns:
        return {"error": f"Column {freight_col} not found in data"}

    result = {}

    # Global trend (aggregate across all countries per year)
    yearly = df.groupby("year")[freight_col].agg(["sum", "mean", "count"]).reset_index()
    yearly.columns = ["year", "total", "mean", "num_countries"]

    if len(yearly) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(yearly["year"], yearly["total"])
        result["global_trend"] = {
            "slope_per_year": float(slope),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "direction": "increasing" if slope > 0 else "decreasing",
        }

    result["yearly_aggregates"] = yearly

    # Per-country growth rates
    country_growth = []
    for country in df["country"].unique():
        cdata = df[df["country"] == country][["year", freight_col]].dropna().sort_values("year")
        if len(cdata) < 2:
            continue

        first_val = cdata[freight_col].iloc[0]
        last_val = cdata[freight_col].iloc[-1]
        n_years = cdata["year"].iloc[-1] - cdata["year"].iloc[0]

        if first_val > 0 and n_years > 0:
            cagr = (last_val / first_val) ** (1 / n_years) - 1
        else:
            cagr = 0.0

        total_change_pct = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0.0

        country_growth.append({
            "country": country,
            "first_year": int(cdata["year"].iloc[0]),
            "last_year": int(cdata["year"].iloc[-1]),
            "first_value": float(first_val),
            "last_value": float(last_val),
            "total_change_pct": float(total_change_pct),
            "cagr": float(cagr),
            "avg_annual": float(cdata[freight_col].mean()),
        })

    result["country_growth"] = pd.DataFrame(country_growth)
    if not result["country_growth"].empty:
        result["country_growth"] = result["country_growth"].sort_values("avg_annual", ascending=False)

    return result


def compute_cargo_rankings(df: pd.DataFrame, freight_col: str = CARGO_FREIGHT_CODE) -> pd.DataFrame:
    """Rank countries by air freight volume for each year.

    Returns DataFrame with country, year, value, and rank columns.
    """
    if freight_col not in df.columns:
        return pd.DataFrame()

    subset = df[["country", "year", freight_col]].dropna()
    subset = subset.copy()
    subset["rank"] = subset.groupby("year")[freight_col].rank(ascending=False, method="min")
    subset = subset.sort_values(["year", "rank"])
    return subset


def compute_cargo_economic_correlation(df: pd.DataFrame, freight_col: str = CARGO_FREIGHT_CODE) -> dict:
    """Compute correlations between air freight and all other economic indicators.

    Returns dict with Pearson, Spearman correlations and scatter data.
    """
    if freight_col not in df.columns:
        return {"error": f"Column {freight_col} not found"}

    other_cols = [c for c in df.columns if c not in ("country", "year") and c != freight_col]
    numeric_cols = [c for c in other_cols if pd.api.types.is_numeric_dtype(df[c])]

    results = []
    for col in numeric_cols:
        valid = df[[freight_col, col]].dropna()
        if len(valid) < 5:
            continue

        pearson_r, pearson_p = stats.pearsonr(valid[freight_col], valid[col])
        spearman_r, spearman_p = stats.spearmanr(valid[freight_col], valid[col])

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


def compute_cargo_intensity(df: pd.DataFrame, freight_col: str = CARGO_FREIGHT_CODE) -> pd.DataFrame:
    """Compute cargo intensity metrics: freight per GDP, freight per capita.

    Returns DataFrame with derived intensity columns.
    """
    if freight_col not in df.columns:
        return pd.DataFrame()

    result = df[["country", "year", freight_col]].dropna().copy()

    # Freight per GDP (if GDP available)
    gdp_col = "NY.GDP.MKTP.CD"
    if gdp_col in df.columns:
        merged = result.merge(df[["country", "year", gdp_col]].dropna(), on=["country", "year"], how="inner")
        merged["freight_per_gdp"] = merged[freight_col] / merged[gdp_col].replace(0, np.nan)
        result = merged

    # Freight per capita (if population available)
    pop_col = "SP.POP.TOTL"
    if pop_col in df.columns:
        merged = result.merge(df[["country", "year", pop_col]].dropna(), on=["country", "year"], how="inner")
        merged["freight_per_capita"] = merged[freight_col] / merged[pop_col].replace(0, np.nan)
        result = merged

    return result


def compute_cargo_growth_drivers(df: pd.DataFrame, freight_col: str = CARGO_FREIGHT_CODE) -> dict:
    """Use Random Forest to identify economic drivers of cargo growth.

    Returns feature importances and model score.
    """
    if freight_col not in df.columns:
        return {"error": f"Column {freight_col} not found"}

    other_cols = [c for c in df.columns if c not in ("country", "year") and c != freight_col]
    numeric_cols = [c for c in other_cols if pd.api.types.is_numeric_dtype(df[c])]

    if not numeric_cols:
        return {"error": "No numeric predictor columns found"}

    analysis_df = df[[freight_col] + numeric_cols].dropna()
    if len(analysis_df) < 20:
        return {"error": f"Not enough data ({len(analysis_df)} rows, need 20+)"}

    X = analysis_df[numeric_cols]
    y = analysis_df[freight_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)
    score = model.score(X_scaled, y)

    importances = pd.DataFrame({
        "indicator": numeric_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "importances": importances,
        "r2_score": float(score),
        "n_samples": len(analysis_df),
        "n_features": len(numeric_cols),
    }


def compute_yoy_growth(df: pd.DataFrame, freight_col: str = CARGO_FREIGHT_CODE) -> pd.DataFrame:
    """Compute year-over-year growth rates for air freight by country.

    Returns DataFrame with country, year, value, and yoy_growth_pct.
    """
    if freight_col not in df.columns:
        return pd.DataFrame()

    subset = df[["country", "year", freight_col]].dropna().sort_values(["country", "year"])
    subset = subset.copy()
    subset["prev_value"] = subset.groupby("country")[freight_col].shift(1)
    subset["yoy_growth_pct"] = ((subset[freight_col] - subset["prev_value"]) / subset["prev_value"]) * 100
    subset = subset.dropna(subset=["yoy_growth_pct"])

    return subset[["country", "year", freight_col, "yoy_growth_pct"]]
