"""Global monetary & bond data source module.

Downloads bond yields, central bank rates, money supply (M2), and central
bank balance sheet data from FRED.  Covers multiple countries via FRED's
mirrored international series.

API key is read from the FRED_API_KEY environment variable.
"""

import pandas as pd
from pathlib import Path

from data_sources.database import save_dataset_smart, load_dataset_smart

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# FRED series organised by category
# ---------------------------------------------------------------------------

GLOBAL_MONETARY_SERIES = {
    "10-Year Bond Yields": {
        "DGS10": "US 10-Year Treasury Yield (%)",
        "IRLTLT01DEM156N": "Germany 10-Year Govt Bond Yield (%)",
        "IRLTLT01JPM156N": "Japan 10-Year Govt Bond Yield (%)",
        "IRLTLT01GBM156N": "UK 10-Year Govt Bond Yield (%)",
        "IRLTLT01FRM156N": "France 10-Year Govt Bond Yield (%)",
        "IRLTLT01ITM156N": "Italy 10-Year Govt Bond Yield (%)",
        "IRLTLT01CAM156N": "Canada 10-Year Govt Bond Yield (%)",
        "IRLTLT01AUM156N": "Australia 10-Year Govt Bond Yield (%)",
        "IRLTLT01KRM156N": "South Korea 10-Year Govt Bond Yield (%)",
        "IRLTLT01EZM156N": "Euro Area 10-Year Govt Bond Yield (%)",
    },
    "2-Year Bond Yields": {
        "DGS2": "US 2-Year Treasury Yield (%)",
        "IRLTST01DEM156N": "Germany 2-Year Govt Bond Yield (%)",
        "IRLTST01JPM156N": "Japan 2-Year Govt Bond Yield (%)",
        "IRLTST01GBM156N": "UK 2-Year Govt Bond Yield (%)",
    },
    "Yield Spreads": {
        "T10Y2Y": "US 10Y-2Y Treasury Spread (%)",
        "T10YFF": "US 10Y minus Fed Funds Rate (%)",
    },
    "Central Bank Policy Rates": {
        "FEDFUNDS": "US Federal Funds Rate (%)",
        "ECBMLFR": "ECB Main Refinancing Rate (%)",
        "IUDSRT": "Bank of England Official Bank Rate (%)",
        "IRSTCB01JPM156N": "Bank of Japan Policy Rate (%)",
        "IRSTCB01CAM156N": "Bank of Canada Policy Rate (%)",
        "IRSTCB01AUM156N": "Reserve Bank of Australia Cash Rate (%)",
        "IRSTCB01KRM156N": "Bank of Korea Base Rate (%)",
    },
    "Money Supply (M2)": {
        "M2SL": "US M2 Money Stock (Billions USD)",
        "MYAGM2EZM196N": "Euro Area M2 (National Currency)",
        "MYAGM2JPM189N": "Japan M2 (National Currency)",
        "MYAGM2GBM189S": "UK M2 (National Currency)",
        "MYAGM2CAM189N": "Canada M2 (National Currency)",
        "MYAGM2AUM189N": "Australia M2 (National Currency)",
        "MYAGM2KRM189N": "South Korea M2 (National Currency)",
        "MYAGM2CNM189N": "China M2 (National Currency)",
    },
    "Central Bank Balance Sheets": {
        "WALCL": "Fed Total Assets (Millions USD)",
        "ECBASSETSW": "ECB Total Assets (Millions EUR)",
        "JPNASSETS": "BOJ Total Assets (Hundreds of Millions JPY)",
    },
}


def get_all_series() -> dict[str, str]:
    """Return a flat dict of all FRED series IDs to descriptions."""
    result = {}
    for series in GLOBAL_MONETARY_SERIES.values():
        result.update(series)
    return result


def download_monetary_series(
    series_ids: list[str],
    start_date: str = "2000-01-01",
    end_date: str = "2025-12-31",
    api_key: str = "",
    frequency: str | None = None,
    progress_callback=None,
) -> tuple[pd.DataFrame, list[str]]:
    """Download FRED series and merge into a single DataFrame.

    Args:
        series_ids: List of FRED series IDs.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        api_key: FRED API key.
        frequency: Optional aggregation ('a' annual, 'q' quarterly, 'm' monthly).
        progress_callback: Optional callback(current, total, label).

    Returns:
        Tuple of (DataFrame with date/year/month + series columns, list of failed series).
    """
    from fredapi import Fred

    fred = Fred(api_key=api_key)
    all_series = get_all_series()

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


def save_monetary_dataset(df: pd.DataFrame, name: str) -> Path:
    """Save dataset to the SQLite database."""
    save_dataset_smart(df, name)
    return DATA_DIR / f"{name}.db"


def load_monetary_dataset(name: str) -> pd.DataFrame:
    """Load a dataset from the database, with CSV fallback."""
    try:
        return load_dataset_smart(name)
    except FileNotFoundError:
        csv_path = DATA_DIR / f"{name}.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise FileNotFoundError(
            f"Monetary dataset '{name}' not found in database or CSV files"
        )
