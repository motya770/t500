"""FRED (Federal Reserve Economic Data) source module.

Downloads detailed US economic data from FRED using the fredapi package.
Provides comprehensive data on housing, auto sales, retail, consumer spending,
manufacturing, employment, interest rates, and more.

API key is read from the FRED_API_KEY environment variable.
Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html
"""

import pandas as pd
from pathlib import Path

from data_sources.database import save_dataset_smart, load_dataset_smart

DATA_DIR = Path(__file__).parent.parent / "data"

# Comprehensive FRED series organized by category
FRED_SERIES = {
    "Housing Market": {
        "HOUST": "Housing Starts (Thousands of Units)",
        "PERMIT": "Building Permits (Thousands of Units)",
        "HSN1F": "New One Family Houses Sold (Thousands)",
        "EXHOSLUSM495S": "Existing Home Sales (Millions)",
        "MSPUS": "Median Sales Price of Houses Sold (USD)",
        "ASPUS": "Average Sales Price of Houses Sold (USD)",
        "CSUSHPISA": "S&P/Case-Shiller U.S. National Home Price Index",
        "MORTGAGE30US": "30-Year Fixed Rate Mortgage Average (%)",
        "MSACSR": "Monthly Supply of New Houses (Months)",
        "USSTHPI": "All-Transactions House Price Index",
    },
    "Auto & Vehicle Sales": {
        "TOTALSA": "Total Vehicle Sales (Millions of Units)",
        "LAUTOSA": "Light Vehicle Sales: Autos (Millions)",
        "LTRUCKSA": "Light Vehicle Sales: Trucks and SUVs (Millions)",
        "HTRUCKSSA": "Heavy Truck Sales (Thousands)",
        "AISRSA": "Auto Inventory to Sales Ratio",
    },
    "Retail & Consumer Spending": {
        "RSAFS": "Advance Retail Sales: Total (Millions USD)",
        "RRSFS": "Real Retail and Food Services Sales (Millions 2017 USD)",
        "PCEC96": "Real Personal Consumption Expenditures (Billions 2017 USD)",
        "PCEDG": "Personal Consumption: Durable Goods (Billions USD)",
        "PCEND": "Personal Consumption: Nondurable Goods (Billions USD)",
        "PCESV": "Personal Consumption: Services (Billions USD)",
        "UMCSENT": "Univ. of Michigan Consumer Sentiment Index",
        "DGORDER": "Durable Goods New Orders (Millions USD)",
        "RETAILSMNSA": "Retail Sales: Clothing Stores (Millions USD)",
        "RSFSDP": "Retail Sales: Food Services & Drinking Places (Millions USD)",
        "MRTSSM44X72USS": "Retail Sales: Retail Trade (Millions USD)",
    },
    "Manufacturing & Industry": {
        "INDPRO": "Industrial Production Index (2017=100)",
        "IPMAN": "Industrial Production: Manufacturing (2017=100)",
        "NEWORDER": "Mfrs New Orders: Nondefense Capital Goods ex Aircraft (Millions USD)",
        "AMTMNO": "Mfrs New Orders: Total Manufacturing (Millions USD)",
        "CMRMTSPL": "Real Manufacturing & Trade Sales (Millions 2017 USD)",
        "NAPM": "ISM Manufacturing: PMI Composite Index",
        "ACDGNO": "Mfrs New Orders: Consumer Durable Goods (Millions USD)",
    },
    "Employment & Labor": {
        "UNRATE": "Unemployment Rate (%)",
        "PAYEMS": "Total Nonfarm Payrolls (Thousands)",
        "ICSA": "Initial Unemployment Claims (Thousands)",
        "JTSJOL": "Job Openings: Total Nonfarm (Thousands)",
        "JTSQUR": "Quits Rate: Total Nonfarm (%)",
        "CES0500000003": "Average Hourly Earnings (USD)",
        "AWHMAN": "Average Weekly Hours: Manufacturing",
        "LNS11300000": "Labor Force Participation Rate (%)",
        "CIVPART": "Civilian Labor Force Participation Rate (%)",
        "MANEMP": "Manufacturing Employees (Thousands)",
    },
    "Inflation & Prices": {
        "CPIAUCSL": "Consumer Price Index: All Items (1982-84=100)",
        "CPILFESL": "Core CPI ex Food & Energy (1982-84=100)",
        "PCEPI": "PCE Price Index (2017=100)",
        "PCEPILFE": "Core PCE Price Index (2017=100)",
        "PPIFIS": "Producer Price Index: Final Demand (2009=100)",
        "CUUR0000SAF11": "CPI: Food at Home",
        "CUSR0000SETB01": "CPI: Gasoline (All Types)",
        "CUSR0000SETA02": "CPI: Used Cars and Trucks",
        "CUSR0000SAH1": "CPI: Shelter",
        "CUSR0000SAM": "CPI: Medical Care",
    },
    "Interest Rates & Monetary Policy": {
        "FEDFUNDS": "Federal Funds Effective Rate (%)",
        "DGS10": "10-Year Treasury Rate (%)",
        "DGS2": "2-Year Treasury Rate (%)",
        "DGS30": "30-Year Treasury Rate (%)",
        "T10Y2Y": "10-Year minus 2-Year Treasury Spread (%)",
        "T10YFF": "10-Year Treasury minus Fed Funds Rate (%)",
        "M2SL": "M2 Money Stock (Billions USD)",
        "BOGMBASE": "Monetary Base (Millions USD)",
        "WALCL": "Fed Total Assets (Millions USD)",
        "DPRIME": "Bank Prime Loan Rate (%)",
    },
    "GDP & Income": {
        "GDP": "Gross Domestic Product (Billions USD)",
        "GDPC1": "Real GDP (Billions 2017 USD)",
        "A191RL1Q225SBEA": "Real GDP Growth Rate (Quarterly Annualized %)",
        "DSPIC96": "Real Disposable Personal Income (Billions 2017 USD)",
        "PI": "Personal Income (Billions USD)",
        "W068RCQ027SBEA": "Govt Total Expenditures (Billions USD)",
        "FGEXPND": "Federal Govt Current Expenditures (Billions USD)",
        "A939RC0Q052SBEA": "Corporate Profits After Tax (Billions USD)",
        "CP": "Corporate Profits Before Tax (Billions USD)",
    },
    "Trade & Dollar": {
        "BOPGSTB": "Trade Balance: Goods & Services (Millions USD)",
        "BOPGTB": "Trade Balance: Goods (Millions USD)",
        "IMPGS": "Imports of Goods & Services (Billions USD)",
        "EXPGS": "Exports of Goods & Services (Billions USD)",
        "DTWEXBGS": "Nominal Broad US Dollar Index",
        "RTWEXBGS": "Real Broad US Dollar Index",
    },
    "Financial Markets": {
        "SP500": "S&P 500 Index",
        "NASDAQCOM": "NASDAQ Composite Index",
        "DJIA": "Dow Jones Industrial Average",
        "VIXCLS": "CBOE Volatility Index (VIX)",
        "BAMLH0A0HYM2": "High Yield Bond Spread (%)",
        "DCOILWTICO": "Crude Oil Price: WTI (USD/Barrel)",
        "GOLDAMGBD228NLBM": "Gold Price (USD/Troy Ounce)",
    },
    "Debt & Credit": {
        "GFDEBTN": "Federal Debt: Total Public Debt (Millions USD)",
        "GFDEGDQ188S": "Federal Debt to GDP Ratio (%)",
        "TOTALSL": "Total Consumer Credit (Billions USD)",
        "REVOLSL": "Revolving Consumer Credit (Billions USD)",
        "NONREVSL": "Nonrevolving Consumer Credit (Billions USD)",
        "DRSFRMACBS": "Delinquency Rate on Single-Family Mortgages (%)",
        "DRCCLACBS": "Delinquency Rate on Credit Card Loans (%)",
        "BUSLOANS": "Commercial and Industrial Loans (Billions USD)",
    },
}


def get_all_fred_series() -> dict[str, str]:
    """Return a flat dict of all FRED series IDs to their descriptions."""
    result = {}
    for series in FRED_SERIES.values():
        result.update(series)
    return result


def download_fred_series(
    series_ids: list[str],
    start_date: str = "2000-01-01",
    end_date: str = "2025-12-31",
    api_key: str = "",
    frequency: str | None = None,
    progress_callback=None,
) -> tuple[pd.DataFrame, list[str]]:
    """Download multiple FRED series and merge into a single DataFrame.

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
    all_series = get_all_fred_series()

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


def download_fred_series_annual(
    series_ids: list[str],
    start_year: int = 2000,
    end_year: int = 2025,
    api_key: str = "",
    progress_callback=None,
) -> tuple[pd.DataFrame, list[str]]:
    """Download FRED series with annual aggregation for compatibility with World Bank data.

    Returns a DataFrame with columns: country, year, <series_id>, ...
    The 'country' column is always 'USA' for FRED data.
    """
    df, failed = download_fred_series(
        series_ids=series_ids,
        start_date=f"{start_year}-01-01",
        end_date=f"{end_year}-12-31",
        api_key=api_key,
        frequency="a",
        progress_callback=progress_callback,
    )

    if df.empty:
        return df, failed

    # Add country column for compatibility with World Bank data format
    df["country"] = "USA"

    # Keep year and series columns, drop date/month
    cols = ["country", "year"] + [c for c in df.columns if c not in ("date", "year", "month", "country")]
    df = df[cols]

    return df, failed


def save_fred_dataset(df: pd.DataFrame, name: str) -> Path:
    """Save FRED dataset to the SQLite database."""
    save_dataset_smart(df, name)
    return DATA_DIR / f"{name}.db"


def load_fred_dataset(name: str) -> pd.DataFrame:
    """Load a FRED dataset from the database."""
    try:
        return load_dataset_smart(name)
    except FileNotFoundError:
        csv_path = DATA_DIR / f"{name}.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise FileNotFoundError(f"FRED dataset '{name}' not found")
