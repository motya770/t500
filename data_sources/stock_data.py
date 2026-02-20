"""Stock market data fetcher.

Downloads historical stock/ETF price data using yfinance and provides
annual aggregation for merging with World Bank economic indicators.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Popular tickers for correlation studies
STOCK_PRESETS = {
    "VOO": "Vanguard S&P 500 ETF",
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust (Nasdaq-100)",
    "DIA": "SPDR Dow Jones Industrial Average ETF",
    "IWM": "iShares Russell 2000 ETF",
    "VTI": "Vanguard Total Stock Market ETF",
    "EFA": "iShares MSCI EAFE ETF (International)",
    "EEM": "iShares MSCI Emerging Markets ETF",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "GLD": "SPDR Gold Shares ETF",
}


def download_stock_data(
    ticker: str = "VOO",
    start_year: int = 2000,
    end_year: int = 2024,
) -> pd.DataFrame:
    """Download daily stock data and aggregate to annual metrics.

    Returns a DataFrame with columns:
        year, avg_price, end_price, annual_return_pct, volatility
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for stock data. "
            "Install it with: pip install yfinance"
        )

    start = f"{start_year}-01-01"
    end = f"{end_year}-12-31"

    raw = yf.download(ticker, start=start, end=end, progress=False)

    if raw.empty:
        return pd.DataFrame()

    raw = raw.reset_index()

    # Handle multi-level columns from yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] if col[1] == "" else col[0] for col in raw.columns]

    raw["year"] = pd.to_datetime(raw["Date"]).dt.year
    raw["daily_return"] = raw["Close"].pct_change()

    annual = raw.groupby("year").agg(
        avg_price=("Close", "mean"),
        end_price=("Close", "last"),
        high=("High", "max"),
        low=("Low", "min"),
        avg_volume=("Volume", "mean"),
        volatility=("daily_return", "std"),
    ).reset_index()

    # Annual return (year-over-year % change of end-of-year price)
    annual["annual_return_pct"] = annual["end_price"].pct_change() * 100

    return annual


def merge_stock_with_economic(
    stock_df: pd.DataFrame,
    economic_df: pd.DataFrame,
    stock_col: str = "annual_return_pct",
    stock_col_name: str = "stock_return",
) -> pd.DataFrame:
    """Merge annual stock data with an economic indicator DataFrame.

    The economic DataFrame should have a ``year`` column (plus country and
    indicator columns in the standard app format).  A single ``stock_col``
    from *stock_df* is joined on year.

    Returns a new DataFrame with the stock column appended.
    """
    subset = stock_df[["year", stock_col]].rename(columns={stock_col: stock_col_name})
    return economic_df.merge(subset, on="year", how="inner")


def save_stock_data(df: pd.DataFrame, name: str) -> Path:
    """Persist stock data to the shared data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"stock_{name}.csv"
    df.to_csv(path, index=False)
    return path


def load_stock_data(name: str) -> pd.DataFrame:
    """Load previously saved stock data."""
    path = DATA_DIR / f"stock_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Stock data '{name}' not found at {path}")
    return pd.read_csv(path)
