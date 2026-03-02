"""Global financial markets data source module.

Downloads FX pairs, commodities, and shipping index data using yfinance.
Reuses the same yf.Ticker().history() pattern as stock_data.py.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

from data_sources.database import save_dataset_smart, load_dataset_smart

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Ticker categories
# ---------------------------------------------------------------------------

FX_CATEGORIES = {
    "Major Pairs": {
        "EURUSD=X": "EUR/USD",
        "USDJPY=X": "USD/JPY",
        "GBPUSD=X": "GBP/USD",
        "USDCHF=X": "USD/CHF",
        "AUDUSD=X": "AUD/USD",
        "USDCAD=X": "USD/CAD",
        "NZDUSD=X": "NZD/USD",
    },
    "Emerging Market Pairs": {
        "USDCNY=X": "USD/CNY (Chinese Yuan)",
        "USDINR=X": "USD/INR (Indian Rupee)",
        "USDBRL=X": "USD/BRL (Brazilian Real)",
        "USDMXN=X": "USD/MXN (Mexican Peso)",
        "USDZAR=X": "USD/ZAR (South African Rand)",
        "USDTRY=X": "USD/TRY (Turkish Lira)",
        "USDKRW=X": "USD/KRW (Korean Won)",
        "USDRUB=X": "USD/RUB (Russian Ruble)",
        "USDPLN=X": "USD/PLN (Polish Zloty)",
        "USDTHB=X": "USD/THB (Thai Baht)",
    },
    "Cross Rates": {
        "EURGBP=X": "EUR/GBP",
        "EURJPY=X": "EUR/JPY",
        "GBPJPY=X": "GBP/JPY",
        "EURCHF=X": "EUR/CHF",
        "AUDNZD=X": "AUD/NZD",
        "EURAUD=X": "EUR/AUD",
    },
}

COMMODITY_CATEGORIES = {
    "Energy": {
        "CL=F": "WTI Crude Oil",
        "BZ=F": "Brent Crude Oil",
        "NG=F": "Natural Gas",
        "HO=F": "Heating Oil",
        "RB=F": "RBOB Gasoline",
    },
    "Precious Metals": {
        "GC=F": "Gold",
        "SI=F": "Silver",
        "PL=F": "Platinum",
        "PA=F": "Palladium",
    },
    "Industrial Metals": {
        "HG=F": "Copper",
        "ALI=F": "Aluminum",
    },
    "Agriculture": {
        "ZW=F": "Wheat",
        "ZC=F": "Corn",
        "ZS=F": "Soybeans",
        "KC=F": "Coffee",
        "SB=F": "Sugar",
        "CT=F": "Cotton",
        "CC=F": "Cocoa",
    },
}

SHIPPING_TICKERS = {
    "^BDI": "Baltic Dry Index",
    "BDRY": "Breakwave Dry Bulk Shipping ETF",
}

# Convenience: all categories combined
ALL_CATEGORIES = {
    **{f"FX: {k}": v for k, v in FX_CATEGORIES.items()},
    **{f"Commodities: {k}": v for k, v in COMMODITY_CATEGORIES.items()},
    "Shipping Indexes": SHIPPING_TICKERS,
}


def get_all_tickers() -> dict[str, str]:
    """Return a flat dict of every ticker to its description."""
    result = {}
    for cat in ALL_CATEGORIES.values():
        result.update(cat)
    return result


def download_global_market_data(
    tickers: list[str],
    start_year: int = 2000,
    end_year: int = 2025,
    interval: str = "1mo",
    progress_callback=None,
) -> tuple[pd.DataFrame, list[str]]:
    """Download price data for FX / commodity / shipping tickers.

    Args:
        tickers: List of Yahoo Finance ticker symbols.
        start_year: Start year.
        end_year: End year.
        interval: "1d", "1wk", or "1mo".
        progress_callback: Optional callback(current, total, label).

    Returns:
        Tuple of (DataFrame, list of failed ticker symbols).
        Individual ticker failures are skipped so the rest succeeds.
    """
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    all_tickers = get_all_tickers()

    all_data = []
    failed = []
    for i, ticker in enumerate(tickers):
        if progress_callback:
            label = all_tickers.get(ticker, ticker)
            progress_callback(i, len(tickers), label)

        try:
            t = yf.Ticker(ticker)
            hist = t.history(start=start_date, end=end_date, interval=interval)

            if hist.empty:
                failed.append(ticker)
                continue

            hist = hist.reset_index()
            hist["ticker"] = ticker

            # Normalize column names
            hist.columns = [c.lower().replace(" ", "_") for c in hist.columns]

            if "date" not in hist.columns and "datetime" in hist.columns:
                hist = hist.rename(columns={"datetime": "date"})

            hist["date"] = pd.to_datetime(hist["date"])
            hist["year"] = hist["date"].dt.year
            hist["month"] = hist["date"].dt.month

            cols = ["ticker", "date", "year", "month"]
            for col in ["open", "high", "low", "close", "volume"]:
                if col in hist.columns:
                    cols.append(col)

            hist = hist[cols]
            all_data.append(hist)

        except Exception:
            failed.append(ticker)
            continue

    if not all_data:
        return pd.DataFrame(), failed

    result = pd.concat(all_data, ignore_index=True)
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    return result, failed


def save_global_market_dataset(df: pd.DataFrame, name: str) -> Path:
    """Save dataset to the SQLite database."""
    save_dataset_smart(df, name)
    return DATA_DIR / f"{name}.db"


def load_global_market_dataset(name: str) -> pd.DataFrame:
    """Load a dataset from the database, with CSV fallback."""
    try:
        return load_dataset_smart(name)
    except FileNotFoundError:
        csv_path = DATA_DIR / f"{name}.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise FileNotFoundError(
            f"Global market dataset '{name}' not found in database or CSV files"
        )
