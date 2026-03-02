"""Global sentiment & trade data source module.

Downloads consumer / business confidence (OECD via FRED), US ISM / PMI,
OECD leading indicators (CLI), and trade balance data (World Bank).

FRED API key is read from the FRED_API_KEY environment variable.
"""

import pandas as pd
from pathlib import Path

from data_sources.database import save_dataset_smart, load_dataset_smart

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# FRED series for sentiment / confidence
# ---------------------------------------------------------------------------

SENTIMENT_SERIES = {
    "Consumer Confidence (OECD)": {
        "CSCICP03USM665S": "USA Consumer Confidence (OECD)",
        "CSCICP03EZM665S": "Euro Area Consumer Confidence (OECD)",
        "CSCICP03DEM665S": "Germany Consumer Confidence (OECD)",
        "CSCICP03FRM665S": "France Consumer Confidence (OECD)",
        "CSCICP03GBM665S": "UK Consumer Confidence (OECD)",
        "CSCICP03JPM665S": "Japan Consumer Confidence (OECD)",
        "CSCICP03CNM665S": "China Consumer Confidence (OECD)",
        "CSCICP03CAM665S": "Canada Consumer Confidence (OECD)",
        "CSCICP03AUM665S": "Australia Consumer Confidence (OECD)",
        "CSCICP03KRM665S": "South Korea Consumer Confidence (OECD)",
        "CSCICP03BRM665S": "Brazil Consumer Confidence (OECD)",
        "CSCICP03INM665S": "India Consumer Confidence (OECD)",
    },
    "US Consumer Confidence (Detailed)": {
        "UMCSENT": "Univ. of Michigan Consumer Sentiment Index",
        "CONCCONF": "Conference Board Consumer Confidence Index",
    },
    "Business Confidence (OECD)": {
        "BSCICP03USM665S": "USA Business Confidence (OECD)",
        "BSCICP03EZM665S": "Euro Area Business Confidence (OECD)",
        "BSCICP03DEM665S": "Germany Business Confidence (OECD)",
        "BSCICP03FRM665S": "France Business Confidence (OECD)",
        "BSCICP03GBM665S": "UK Business Confidence (OECD)",
        "BSCICP03JPM665S": "Japan Business Confidence (OECD)",
        "BSCICP03CNM665S": "China Business Confidence (OECD)",
        "BSCICP03CAM665S": "Canada Business Confidence (OECD)",
        "BSCICP03AUM665S": "Australia Business Confidence (OECD)",
        "BSCICP03KRM665S": "South Korea Business Confidence (OECD)",
        "BSCICP03BRM665S": "Brazil Business Confidence (OECD)",
        "BSCICP03INM665S": "India Business Confidence (OECD)",
    },
    "US ISM Manufacturing": {
        "NAPM": "ISM Manufacturing: PMI Composite Index",
        "NAPMNOI": "ISM Manufacturing: New Orders Index",
        "NAPMSI": "ISM Manufacturing: Supplier Deliveries Index",
    },
    "OECD Leading Indicators (CLI)": {
        "USALOLITONOSTSAM": "USA Composite Leading Indicator",
        "DEULOLIT02IXOBSAM": "Germany Composite Leading Indicator",
        "JPNLOLITONOSTSAM": "Japan Composite Leading Indicator",
        "GBRLOLITONOSTSAM": "UK Composite Leading Indicator",
        "CHNLOLITONOSTSAM": "China Composite Leading Indicator",
        "G7LOLITONOSTSAM": "G7 Composite Leading Indicator",
    },
}

# ---------------------------------------------------------------------------
# World Bank indicators for trade balance
# ---------------------------------------------------------------------------

TRADE_BALANCE_INDICATORS = {
    "NE.EXP.GNFS.ZS": "Exports of goods and services (% of GDP)",
    "NE.IMP.GNFS.ZS": "Imports of goods and services (% of GDP)",
    "BN.CAB.XOKA.GD.ZS": "Current account balance (% of GDP)",
    "BX.KLT.DINV.WD.GD.ZS": "Foreign direct investment, net inflows (% of GDP)",
    "TG.VAL.TOTL.GD.ZS": "Merchandise trade (% of GDP)",
    "NE.TRD.GNFS.ZS": "Trade (% of GDP)",
    "BN.GSR.GNFS.CD": "Net trade in goods and services (current US$)",
}


def get_all_sentiment_series() -> dict[str, str]:
    """Return a flat dict of all sentiment FRED series IDs to descriptions."""
    result = {}
    for series in SENTIMENT_SERIES.values():
        result.update(series)
    return result


def download_sentiment_series(
    series_ids: list[str],
    start_date: str = "2000-01-01",
    end_date: str = "2025-12-31",
    api_key: str = "",
    frequency: str | None = None,
    progress_callback=None,
) -> tuple[pd.DataFrame, list[str]]:
    """Download FRED sentiment series and merge into a single DataFrame.

    Returns:
        Tuple of (DataFrame with date/year/month + series columns, list of failed series).
    """
    from fredapi import Fred

    fred = Fred(api_key=api_key)
    all_series = get_all_sentiment_series()

    merged = None
    failed = []

    for i, sid in enumerate(series_ids):
        if progress_callback:
            label = all_series.get(sid, sid)
            progress_callback(i, len(series_ids), label)

        try:
            kwargs = {
                "observation_start": start_date,
                "observation_end": end_date,
            }
            if frequency:
                kwargs["frequency"] = frequency

            data = fred.get_series(sid, **kwargs)

            if data is None or data.empty:
                failed.append(all_series.get(sid, sid))
                continue

            df = pd.DataFrame({
                "date": data.index,
                sid: data.values,
            })
            df["date"] = pd.to_datetime(df["date"])

            if merged is None:
                merged = df
            else:
                merged = merged.merge(df, on="date", how="outer")

        except Exception:
            failed.append(all_series.get(sid, sid))
            continue

    if merged is None:
        return pd.DataFrame(), failed

    merged = merged.sort_values("date").reset_index(drop=True)
    merged["year"] = merged["date"].dt.year
    merged["month"] = merged["date"].dt.month

    return merged, failed


def download_trade_balance(
    indicator_codes: list[str],
    countries: list[str],
    start_year: int = 2000,
    end_year: int = 2025,
    progress_callback=None,
) -> tuple[pd.DataFrame, list[str]]:
    """Download World Bank trade balance indicators.

    Returns:
        Tuple of (DataFrame with country/year + indicator columns, failed indicator names).
    """
    from data_sources.world_bank import download_multiple_indicators

    return download_multiple_indicators(
        indicator_codes=indicator_codes,
        countries=countries,
        start_year=start_year,
        end_year=end_year,
        progress_callback=progress_callback,
    )


def save_sentiment_dataset(df: pd.DataFrame, name: str) -> Path:
    """Save dataset to the SQLite database."""
    save_dataset_smart(df, name)
    return DATA_DIR / f"{name}.db"


def load_sentiment_dataset(name: str) -> pd.DataFrame:
    """Load a dataset from the database, with CSV fallback."""
    try:
        return load_dataset_smart(name)
    except FileNotFoundError:
        csv_path = DATA_DIR / f"{name}.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise FileNotFoundError(
            f"Sentiment dataset '{name}' not found in database or CSV files"
        )
