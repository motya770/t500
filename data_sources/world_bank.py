"""World Bank data source module.

Downloads economic indicators from the World Bank Open Data API using wbgapi.
"""

import wbgapi as wb
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Common economic indicators grouped by category
INDICATOR_CATEGORIES = {
    "GDP & Growth": {
        "NY.GDP.MKTP.CD": "GDP (current US$)",
        "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
        "NY.GDP.PCAP.CD": "GDP per capita (current US$)",
        "NY.GDP.PCAP.KD.ZG": "GDP per capita growth (annual %)",
        "NE.GDI.TOTL.ZS": "Gross capital formation (% of GDP)",
    },
    "Trade": {
        "NE.EXP.GNFS.ZS": "Exports of goods and services (% of GDP)",
        "NE.IMP.GNFS.ZS": "Imports of goods and services (% of GDP)",
        "TG.VAL.TOTL.GD.ZS": "Merchandise trade (% of GDP)",
        "BN.CAB.XOKA.GD.ZS": "Current account balance (% of GDP)",
        "BX.KLT.DINV.WD.GD.ZS": "Foreign direct investment, net inflows (% of GDP)",
    },
    "Inflation & Prices": {
        "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
        "FP.CPI.TOTL": "Consumer price index (2010 = 100)",
        "NY.GDP.DEFL.KD.ZG": "GDP deflator (annual %)",
    },
    "Employment": {
        "SL.UEM.TOTL.ZS": "Unemployment, total (% of total labor force)",
        "SL.TLF.CACT.ZS": "Labor force participation rate (% of total population ages 15+)",
        "SL.EMP.TOTL.SP.ZS": "Employment to population ratio, 15+ (%, total)",
        "SL.AGR.EMPL.ZS": "Employment in agriculture (% of total employment)",
        "SL.IND.EMPL.ZS": "Employment in industry (% of total employment)",
        "SL.SRV.EMPL.ZS": "Employment in services (% of total employment)",
    },
    "Government & Debt": {
        "GC.DOD.TOTL.GD.ZS": "Central government debt, total (% of GDP)",
        "GC.REV.XGRT.GD.ZS": "Revenue, excluding grants (% of GDP)",
        "GC.XPN.TOTL.GD.ZS": "Expense (% of GDP)",
        "GC.BAL.CASH.GD.ZS": "Cash surplus/deficit (% of GDP)",
    },
    "Population & Demographics": {
        "SP.POP.TOTL": "Population, total",
        "SP.POP.GROW": "Population growth (annual %)",
        "SP.URB.TOTL.IN.ZS": "Urban population (% of total population)",
        "SP.DYN.LE00.IN": "Life expectancy at birth (years)",
        "SP.DYN.TFRT.IN": "Fertility rate, total (births per woman)",
    },
    "Financial": {
        "FR.INR.RINR": "Real interest rate (%)",
        "FR.INR.LEND": "Lending interest rate (%)",
        "FR.INR.DPST": "Deposit interest rate (%)",
        "FM.LBL.BMNY.GD.ZS": "Broad money (% of GDP)",
        "PA.NUS.FCRF": "Official exchange rate (LCU per US$)",
    },
    "Education & Technology": {
        "SE.XPD.TOTL.GD.ZS": "Government expenditure on education (% of GDP)",
        "GB.XPD.RSDV.GD.ZS": "Research and development expenditure (% of GDP)",
        "IT.NET.USER.ZS": "Individuals using the Internet (% of population)",
        "IP.PAT.RESD": "Patent applications, residents",
    },
    "Energy & Environment": {
        "EG.USE.PCAP.KG.OE": "Energy use (kg of oil equivalent per capita)",
        "EN.ATM.CO2E.PC": "CO2 emissions (metric tons per capita)",
        "EG.FEC.RNEW.ZS": "Renewable energy consumption (% of total)",
        "EG.USE.ELEC.KH.PC": "Electric power consumption (kWh per capita)",
    },
    "Health": {
        "SH.XPD.CHEX.GD.ZS": "Current health expenditure (% of GDP)",
        "SH.MED.PHYS.ZS": "Physicians (per 1,000 people)",
        "SH.DYN.MORT": "Mortality rate, under-5 (per 1,000 live births)",
        "SH.STA.MMRT": "Maternal mortality ratio (per 100,000 live births)",
    },
    "Air Transport & Cargo": {
        "IS.AIR.GOOD.MT.K1": "Air transport, freight (million ton-km)",
        "IS.AIR.PSGR": "Air transport, passengers carried",
        "IS.AIR.DPRT": "Air transport, registered carrier departures worldwide",
        "IS.SHP.GOOD.TU": "Container port traffic (TEU: 20 foot equivalent units)",
    },
}


def get_all_indicators() -> dict[str, str]:
    """Return a flat dict of all indicator codes to their descriptions."""
    result = {}
    for indicators in INDICATOR_CATEGORIES.values():
        result.update(indicators)
    return result


def get_countries() -> pd.DataFrame:
    """Fetch list of countries from World Bank."""
    countries = []
    for c in wb.economy.list():
        countries.append({
            "id": c["id"],
            "name": c["value"],
            "region": c.get("region", {}).get("value", ""),
            "income_level": c.get("incomeLevel", {}).get("value", ""),
        })
    return pd.DataFrame(countries)


def get_country_groups() -> dict[str, list[str]]:
    """Return common country groupings."""
    return {
        "G7": ["USA", "GBR", "FRA", "DEU", "ITA", "JPN", "CAN"],
        "BRICS": ["BRA", "RUS", "IND", "CHN", "ZAF"],
        "EU Major": ["DEU", "FRA", "ITA", "ESP", "NLD", "BEL", "POL"],
        "East Asia": ["CHN", "JPN", "KOR", "TWN", "HKG", "SGP"],
        "Latin America": ["BRA", "MEX", "ARG", "COL", "CHL", "PER"],
        "Middle East": ["SAU", "ARE", "ISR", "TUR", "IRN", "EGY"],
    }


def download_indicator(
    indicator_code: str,
    countries: list[str],
    start_year: int = 2000,
    end_year: int = 2023,
) -> pd.DataFrame:
    """Download a single indicator for given countries and year range.

    Returns a DataFrame with columns: country, year, value
    """
    try:
        data = wb.data.DataFrame(
            indicator_code,
            economy=countries,
            time=range(start_year, end_year + 1),
        )

        if data.empty:
            return pd.DataFrame(columns=["country", "year", "value"])

        # wbgapi returns data with economies as rows and years as columns
        # Column names are like 'YR2000', 'YR2001', etc.
        data = data.reset_index()
        id_col = data.columns[0]  # economy column

        melted = data.melt(
            id_vars=[id_col],
            var_name="year_str",
            value_name="value",
        )
        melted["year"] = melted["year_str"].str.replace("YR", "").astype(int)
        melted = melted.rename(columns={id_col: "country"})
        melted = melted[["country", "year", "value"]].dropna(subset=["value"])
        melted = melted.sort_values(["country", "year"]).reset_index(drop=True)

        return melted

    except Exception as e:
        raise RuntimeError(f"Failed to download indicator {indicator_code}: {e}")


def download_multiple_indicators(
    indicator_codes: list[str],
    countries: list[str],
    start_year: int = 2000,
    end_year: int = 2023,
    progress_callback=None,
) -> pd.DataFrame:
    """Download multiple indicators and merge into a wide-format DataFrame.

    Returns DataFrame with columns: country, year, <indicator1>, <indicator2>, ...
    """
    all_indicators = get_all_indicators()
    merged = None

    for i, code in enumerate(indicator_codes):
        if progress_callback:
            label = all_indicators.get(code, code)
            progress_callback(i, len(indicator_codes), label)

        df = download_indicator(code, countries, start_year, end_year)
        if df.empty:
            continue

        df = df.rename(columns={"value": code})

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["country", "year"], how="outer")

    if merged is None:
        return pd.DataFrame()

    merged = merged.sort_values(["country", "year"]).reset_index(drop=True)
    return merged


def save_dataset(df: pd.DataFrame, name: str) -> Path:
    """Save a dataset to the data directory as CSV."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def load_dataset(name: str) -> pd.DataFrame:
    """Load a dataset from the data directory."""
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset '{name}' not found at {path}")
    return pd.read_csv(path)


def list_saved_datasets() -> list[str]:
    """List all saved datasets in the data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return [p.stem for p in DATA_DIR.glob("*.csv")]
