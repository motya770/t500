"""Stock market data source module.

Downloads stock/ETF price and volume data using yfinance.
Focused on ETFs like VOO (S&P 500) and QQQ (NASDAQ-100) to investigate
how well economic indicators are reflected in stock market performance.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

from data_sources.database import save_dataset_smart, load_dataset_smart

DATA_DIR = Path(__file__).parent.parent / "data"

# Predefined tickers of interest
STOCK_TICKERS = {
    "VOO": "Vanguard S&P 500 ETF",
    "QQQ": "Invesco QQQ Trust (NASDAQ-100)",
    "SPY": "SPDR S&P 500 ETF Trust",
    "DIA": "SPDR Dow Jones Industrial Average ETF",
    "IWM": "iShares Russell 2000 ETF",
    "VTI": "Vanguard Total Stock Market ETF",
    "EFA": "iShares MSCI EAFE ETF (International)",
    "EEM": "iShares MSCI Emerging Markets ETF",
    "GLD": "SPDR Gold Shares",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
}

TICKER_CATEGORIES = {
    "US Broad Market": {
        "VOO": "Vanguard S&P 500 ETF",
        "QQQ": "Invesco QQQ Trust (NASDAQ-100)",
        "SPY": "SPDR S&P 500 ETF Trust",
        "DIA": "SPDR Dow Jones Industrial Average ETF",
        "IWM": "iShares Russell 2000 ETF",
        "VTI": "Vanguard Total Stock Market ETF",
    },
    "International": {
        "EFA": "iShares MSCI EAFE ETF",
        "EEM": "iShares MSCI Emerging Markets ETF",
    },
    "Commodities & Bonds": {
        "GLD": "SPDR Gold Shares",
        "TLT": "iShares 20+ Year Treasury Bond ETF",
    },
}


def get_all_tickers() -> dict[str, str]:
    """Return a flat dict of all ticker symbols to their descriptions."""
    result = {}
    for tickers in TICKER_CATEGORIES.values():
        result.update(tickers)
    return result


def download_stock_data(
    tickers: list[str],
    start_year: int = 2000,
    end_year: int = 2024,
    interval: str = "1mo",
) -> pd.DataFrame:
    """Download stock/ETF data for given tickers.

    Args:
        tickers: List of ticker symbols (e.g., ["VOO", "QQQ"]).
        start_year: Start year for data range.
        end_year: End year for data range.
        interval: Data interval - "1d" (daily), "1wk" (weekly), "1mo" (monthly).

    Returns:
        DataFrame with columns: ticker, date, year, month, open, high, low,
        close, adj_close, volume.
    """
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    all_data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date, interval=interval)

            if hist.empty:
                continue

            hist = hist.reset_index()
            hist["ticker"] = ticker

            # Normalize column names
            hist.columns = [c.lower().replace(" ", "_") for c in hist.columns]

            # Ensure we have a date column
            if "date" not in hist.columns and "datetime" in hist.columns:
                hist = hist.rename(columns={"datetime": "date"})

            hist["date"] = pd.to_datetime(hist["date"])
            hist["year"] = hist["date"].dt.year
            hist["month"] = hist["date"].dt.month

            # Select relevant columns
            cols = ["ticker", "date", "year", "month"]
            for col in ["open", "high", "low", "close", "volume"]:
                if col in hist.columns:
                    cols.append(col)

            hist = hist[cols]
            all_data.append(hist)

        except Exception as e:
            raise RuntimeError(f"Failed to download data for {ticker}: {e}")

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    return result


def compute_annual_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute annual returns from stock data.

    Takes monthly or daily stock data and computes year-over-year returns.

    Returns:
        DataFrame with columns: ticker, year, annual_return_pct, avg_close,
        total_volume, volatility (std of monthly returns).
    """
    if df.empty:
        return pd.DataFrame()

    results = []
    for ticker in df["ticker"].unique():
        ticker_data = df[df["ticker"] == ticker].copy()

        # Group by year
        for year, year_data in ticker_data.groupby("year"):
            if len(year_data) < 2:
                continue

            year_data = year_data.sort_values("date")
            first_close = year_data["close"].iloc[0]
            last_close = year_data["close"].iloc[-1]

            if first_close > 0:
                annual_return = ((last_close - first_close) / first_close) * 100
            else:
                annual_return = 0.0

            # Monthly returns for volatility
            monthly_returns = year_data["close"].pct_change().dropna()
            volatility = monthly_returns.std() * 100 if len(monthly_returns) > 1 else 0.0

            results.append({
                "ticker": ticker,
                "year": int(year),
                "annual_return_pct": round(annual_return, 2),
                "avg_close": round(year_data["close"].mean(), 2),
                "total_volume": int(year_data["volume"].sum()),
                "volatility": round(volatility, 2),
            })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values(["ticker", "year"]).reset_index(drop=True)


def merge_stock_with_economic(
    stock_annual: pd.DataFrame,
    economic_df: pd.DataFrame,
    country: str = "USA",
) -> pd.DataFrame:
    """Merge annual stock returns with economic indicator data.

    Args:
        stock_annual: DataFrame from compute_annual_returns().
        economic_df: Economic data with columns: country, year, <indicators...>.
        country: Country to filter economic data for (default USA).

    Returns:
        Merged DataFrame with both stock and economic data by year.
    """
    if stock_annual.empty or economic_df.empty:
        return pd.DataFrame()

    # Filter economic data for the specified country
    econ = economic_df[economic_df["country"] == country].copy()
    if econ.empty:
        return pd.DataFrame()

    # Pivot stock data: one column per ticker metric
    stock_wide = stock_annual.pivot(index="year", columns="ticker")
    stock_wide.columns = [f"{col[1]}_{col[0]}" for col in stock_wide.columns]
    stock_wide = stock_wide.reset_index()

    # Merge on year
    merged = econ.merge(stock_wide, on="year", how="inner")
    return merged.sort_values("year").reset_index(drop=True)


def save_stock_dataset(df: pd.DataFrame, name: str) -> Path:
    """Save stock dataset to the SQLite database."""
    save_dataset_smart(df, name)
    return DATA_DIR / f"{name}.db"


def load_stock_dataset(name: str) -> pd.DataFrame:
    """Load a stock dataset from the database, with CSV fallback."""
    try:
        return load_dataset_smart(name)
    except FileNotFoundError:
        csv_path = DATA_DIR / f"{name}.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise FileNotFoundError(
            f"Stock dataset '{name}' not found in database or CSV files"
        )
