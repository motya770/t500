"""Oil tanker and petroleum data source module.

Provides oil production, trade, and energy indicator definitions for
World Bank API fetching, along with predefined country groupings and
major tanker route metadata.
"""

import pandas as pd
from data_sources.world_bank import (
    download_multiple_indicators,
    get_all_indicators,
    save_dataset,
    load_dataset,
    list_saved_datasets,
)

# ---------------------------------------------------------------------------
# Oil / energy World Bank indicator codes
# ---------------------------------------------------------------------------

OIL_INDICATOR_CATEGORIES = {
    "Oil & Energy Production": {
        "NY.GDP.PETR.RT.ZS": "Oil rents (% of GDP)",
        "EG.ELC.PETR.ZS": "Electricity production from oil sources (% of total)",
        "EG.USE.COMM.FO.ZS": "Fossil fuel energy consumption (% of total)",
        "EG.FEC.RNEW.ZS": "Renewable energy consumption (% of total)",
        "EG.USE.PCAP.KG.OE": "Energy use (kg of oil equivalent per capita)",
    },
    "Oil Trade": {
        "TM.VAL.FUEL.ZS.UN": "Fuel imports (% of merchandise imports)",
        "TX.VAL.FUEL.ZS.UN": "Fuel exports (% of merchandise exports)",
        "EG.IMP.CONS.ZS": "Energy imports, net (% of energy use)",
        "NE.EXP.GNFS.ZS": "Exports of goods and services (% of GDP)",
        "NE.IMP.GNFS.ZS": "Imports of goods and services (% of GDP)",
        "TG.VAL.TOTL.GD.ZS": "Merchandise trade (% of GDP)",
    },
    "Oil Prices & Consumption": {
        "EP.PMP.SGAS.CD": "Pump price for gasoline (US$ per liter)",
        "EP.PMP.DESL.CD": "Pump price for diesel fuel (US$ per liter)",
        "EG.USE.ELEC.KH.PC": "Electric power consumption (kWh per capita)",
    },
    "Economic Context": {
        "NY.GDP.MKTP.CD": "GDP (current US$)",
        "NY.GDP.PCAP.CD": "GDP per capita (current US$)",
        "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
        "SP.POP.TOTL": "Population, total",
        "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
        "BN.CAB.XOKA.GD.ZS": "Current account balance (% of GDP)",
    },
    "Maritime & Transport": {
        "IS.SHP.GOOD.TU": "Container port traffic (TEU: 20 foot equivalent units)",
        "TG.VAL.TOTL.GD.ZS": "Merchandise trade (% of GDP)",
    },
}

# Core oil indicator codes (flat list for quick checks)
OIL_RENTS_CODE = "NY.GDP.PETR.RT.ZS"
FUEL_IMPORTS_CODE = "TM.VAL.FUEL.ZS.UN"
FUEL_EXPORTS_CODE = "TX.VAL.FUEL.ZS.UN"
ENERGY_IMPORTS_CODE = "EG.IMP.CONS.ZS"
ENERGY_USE_CODE = "EG.USE.PCAP.KG.OE"
FOSSIL_FUEL_CODE = "EG.USE.COMM.FO.ZS"
OIL_ELECTRICITY_CODE = "EG.ELC.PETR.ZS"
GDP_CODE = "NY.GDP.MKTP.CD"
POP_CODE = "SP.POP.TOTL"
GASOLINE_PRICE_CODE = "EP.PMP.SGAS.CD"
DIESEL_PRICE_CODE = "EP.PMP.DESL.CD"

OIL_CORE_CODES = [
    OIL_RENTS_CODE,
    FUEL_IMPORTS_CODE,
    FUEL_EXPORTS_CODE,
    ENERGY_IMPORTS_CODE,
    ENERGY_USE_CODE,
    FOSSIL_FUEL_CODE,
    OIL_ELECTRICITY_CODE,
]


def get_oil_indicators() -> dict[str, str]:
    """Return a flat dict of all oil-related indicator codes to descriptions."""
    result = {}
    for indicators in OIL_INDICATOR_CATEGORIES.values():
        result.update(indicators)
    return result


# ---------------------------------------------------------------------------
# Country groupings relevant to oil analysis
# ---------------------------------------------------------------------------

def get_oil_country_groups() -> dict[str, list[str]]:
    """Return country groups relevant to oil production and trade."""
    return {
        "Top Oil Producers": [
            "USA", "SAU", "RUS", "CAN", "IRQ",
            "CHN", "ARE", "BRA", "KWT", "NOR",
        ],
        "Top Oil Consumers": [
            "USA", "CHN", "IND", "JPN", "SAU",
            "RUS", "KOR", "BRA", "CAN", "DEU",
        ],
        "OPEC Members": [
            "SAU", "IRQ", "ARE", "KWT", "IRN",
            "VEN", "NGA", "AGO", "LBY", "DZA",
            "GAB", "GNQ", "COG",
        ],
        "Major Oil Importers": [
            "USA", "CHN", "IND", "JPN", "KOR",
            "DEU", "ESP", "ITA", "FRA", "NLD",
        ],
        "Major Oil Exporters": [
            "SAU", "RUS", "IRQ", "CAN", "ARE",
            "KWT", "NGA", "NOR", "AGO", "KAZ",
        ],
        "US & Neighbours": [
            "USA", "CAN", "MEX",
        ],
        "Middle East Producers": [
            "SAU", "IRQ", "ARE", "KWT", "IRN",
            "OMN", "QAT", "BHR", "YEM",
        ],
    }


# ---------------------------------------------------------------------------
# Major tanker route metadata
# ---------------------------------------------------------------------------

MAJOR_TANKER_ROUTES = [
    {
        "route": "Persian Gulf → East Asia",
        "from_region": "Middle East",
        "to_region": "East Asia",
        "chokepoint": "Strait of Hormuz, Strait of Malacca",
        "from_countries": ["SAU", "IRQ", "ARE", "KWT", "IRN"],
        "to_countries": ["CHN", "JPN", "KOR", "IND"],
        "estimated_share_pct": 27.0,
        "description": "Largest oil trade route by volume. Crude oil from "
                       "Persian Gulf producers to refineries in China, Japan, "
                       "South Korea, and India.",
    },
    {
        "route": "Persian Gulf → Europe",
        "from_region": "Middle East",
        "to_region": "Europe",
        "chokepoint": "Strait of Hormuz, Suez Canal / Cape of Good Hope",
        "from_countries": ["SAU", "IRQ", "ARE", "KWT"],
        "to_countries": ["DEU", "FRA", "ITA", "ESP", "NLD", "GBR"],
        "estimated_share_pct": 10.0,
        "description": "Crude oil from the Persian Gulf to European refineries "
                       "via the Suez Canal or around the Cape of Good Hope.",
    },
    {
        "route": "West Africa → Europe & Americas",
        "from_region": "West Africa",
        "to_region": "Europe / Americas",
        "chokepoint": "Cape of Good Hope (for eastern routes)",
        "from_countries": ["NGA", "AGO", "GNQ", "GAB", "COG"],
        "to_countries": ["USA", "CHN", "IND", "ESP", "FRA"],
        "estimated_share_pct": 8.0,
        "description": "Nigerian and Angolan crude shipped to European and "
                       "American markets. Light sweet crude preferred by "
                       "US Gulf Coast refineries.",
    },
    {
        "route": "Russia → Europe",
        "from_region": "Russia / Caspian",
        "to_region": "Europe",
        "chokepoint": "Turkish Straits (Bosphorus), Danish Straits",
        "from_countries": ["RUS", "KAZ"],
        "to_countries": ["DEU", "NLD", "POL", "ITA", "FRA"],
        "estimated_share_pct": 9.0,
        "description": "Pipeline and seaborne crude from Russia to European "
                       "markets. Volume declined post-2022 sanctions.",
    },
    {
        "route": "Russia → East Asia",
        "from_region": "Russia / Caspian",
        "to_region": "East Asia",
        "chokepoint": "None (Pacific route)",
        "from_countries": ["RUS"],
        "to_countries": ["CHN", "IND", "KOR"],
        "estimated_share_pct": 8.0,
        "description": "Russian crude redirected to Asian buyers, primarily "
                       "China and India, via Pacific ports (Kozmino) and "
                       "the ESPO pipeline.",
    },
    {
        "route": "Canada → USA",
        "from_region": "North America",
        "to_region": "North America",
        "chokepoint": "None (pipeline + tanker)",
        "from_countries": ["CAN"],
        "to_countries": ["USA"],
        "estimated_share_pct": 7.0,
        "description": "Canadian oil sands crude shipped via pipeline and "
                       "tanker to US Gulf Coast and Midwest refineries. "
                       "Largest single-country source of US oil imports.",
    },
    {
        "route": "Latin America → USA",
        "from_region": "Latin America",
        "to_region": "North America",
        "chokepoint": "Panama Canal (for Pacific-origin)",
        "from_countries": ["MEX", "VEN", "COL", "BRA", "ECU"],
        "to_countries": ["USA"],
        "estimated_share_pct": 6.0,
        "description": "Crude oil from Mexico, Colombia, Brazil, and "
                       "Ecuador to US Gulf Coast refineries.",
    },
    {
        "route": "North Sea → Europe",
        "from_region": "North Sea",
        "to_region": "Europe",
        "chokepoint": "None",
        "from_countries": ["NOR", "GBR"],
        "to_countries": ["DEU", "NLD", "FRA", "GBR"],
        "estimated_share_pct": 5.0,
        "description": "Brent-benchmark crude from Norwegian and UK North "
                       "Sea fields to northwest European refineries.",
    },
    {
        "route": "Persian Gulf → USA",
        "from_region": "Middle East",
        "to_region": "North America",
        "chokepoint": "Strait of Hormuz",
        "from_countries": ["SAU", "IRQ", "KWT"],
        "to_countries": ["USA"],
        "estimated_share_pct": 4.0,
        "description": "Saudi and Iraqi crude to US Gulf Coast refineries. "
                       "Volume has decreased as US domestic production rose.",
    },
    {
        "route": "Intra-Asia",
        "from_region": "Southeast Asia / Australia",
        "to_region": "East Asia",
        "chokepoint": "Strait of Malacca",
        "from_countries": ["MYS", "IDN", "AUS", "BRN"],
        "to_countries": ["CHN", "JPN", "KOR", "SGP"],
        "estimated_share_pct": 4.0,
        "description": "Crude and condensate from Southeast Asian producers "
                       "and Australia to Northeast Asian refineries.",
    },
]


def get_all_oil_indicators() -> list[str]:
    """Return deduplicated list of all oil indicator codes."""
    seen = set()
    codes = []
    for indicators in OIL_INDICATOR_CATEGORIES.values():
        for code in indicators:
            if code not in seen:
                seen.add(code)
                codes.append(code)
    return codes


def download_oil_data(
    countries: list[str],
    start_year: int = 2000,
    end_year: int = 2025,
    include_economic: bool = True,
    progress_callback=None,
) -> tuple[pd.DataFrame, list[str]]:
    """Download oil-related indicators from World Bank.

    Parameters
    ----------
    countries : list[str]
        ISO3 country codes.
    start_year, end_year : int
        Year range.
    include_economic : bool
        If True, also download GDP, population, and inflation indicators.
    progress_callback : callable, optional
        Called with (current, total, label).

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        (DataFrame with country/year/indicator columns, list of failed indicator names).
    """
    codes = []
    seen = set()

    for cat_name, indicators in OIL_INDICATOR_CATEGORIES.items():
        if cat_name == "Economic Context" and not include_economic:
            continue
        for code in indicators:
            if code not in seen:
                seen.add(code)
                codes.append(code)

    return download_multiple_indicators(
        codes, countries, start_year, end_year, progress_callback
    )
