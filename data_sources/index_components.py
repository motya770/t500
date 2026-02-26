"""S&P 500 and Nasdaq 100 index component ticker lists.

Provides functions to fetch current index components from Wikipedia,
with comprehensive hardcoded fallback lists organized by GICS sector.
"""

import pandas as pd
from functools import lru_cache

# ---------------------------------------------------------------------------
# Wikipedia fetching
# ---------------------------------------------------------------------------

def _fetch_sp500_from_wikipedia() -> dict[str, dict[str, str]]:
    """Fetch S&P 500 components from Wikipedia, organized by GICS sector.

    Returns:
        Dict of {sector: {ticker: company_name}}.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    result: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        sector = str(row.get("GICS Sector", "Other"))
        symbol = str(row["Symbol"]).replace(".", "-")  # BRK.B -> BRK-B for yfinance
        name = str(row["Security"])
        if sector not in result:
            result[sector] = {}
        result[sector][symbol] = name
    return result


def _fetch_nasdaq100_from_wikipedia() -> dict[str, dict[str, str]]:
    """Fetch Nasdaq 100 components from Wikipedia, organized by sector.

    Returns:
        Dict of {sector: {ticker: company_name}}.
    """
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    # The Nasdaq-100 table is usually the 4th or 5th table on the page
    target = None
    for tbl in tables:
        cols_lower = [str(c).lower() for c in tbl.columns]
        if "ticker" in cols_lower or "symbol" in cols_lower:
            target = tbl
            break
    if target is None and len(tables) >= 4:
        target = tables[3]
    if target is None:
        return {}

    cols_lower = [str(c).lower() for c in target.columns]
    ticker_col = None
    name_col = None
    sector_col = None
    for c in target.columns:
        cl = str(c).lower()
        if cl in ("ticker", "symbol"):
            ticker_col = c
        elif cl in ("company", "security", "name"):
            name_col = c
        elif "sector" in cl or "industry" in cl:
            sector_col = c

    if ticker_col is None:
        return {}

    result: dict[str, dict[str, str]] = {}
    for _, row in target.iterrows():
        symbol = str(row[ticker_col]).strip()
        name = str(row[name_col]).strip() if name_col else symbol
        sector = str(row[sector_col]).strip() if sector_col else "Technology"
        if sector not in result:
            result[sector] = {}
        result[sector][symbol] = name

    return result


# ---------------------------------------------------------------------------
# Public API (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_sp500_components() -> dict[str, dict[str, str]]:
    """Get S&P 500 components organized by GICS sector.

    Tries Wikipedia first, falls back to hardcoded list.

    Returns:
        Dict of {sector: {ticker: company_name}}.
    """
    try:
        result = _fetch_sp500_from_wikipedia()
        if result and sum(len(v) for v in result.values()) > 400:
            return result
    except Exception:
        pass
    return _FALLBACK_SP500


@lru_cache(maxsize=1)
def get_nasdaq100_components() -> dict[str, dict[str, str]]:
    """Get Nasdaq 100 components organized by sector.

    Tries Wikipedia first, falls back to hardcoded list.

    Returns:
        Dict of {sector: {ticker: company_name}}.
    """
    try:
        result = _fetch_nasdaq100_from_wikipedia()
        if result and sum(len(v) for v in result.values()) > 80:
            return result
    except Exception:
        pass
    return _FALLBACK_NASDAQ100


def get_sp500_flat() -> dict[str, str]:
    """Get flat dict of all S&P 500 tickers to company names."""
    result = {}
    for tickers in get_sp500_components().values():
        result.update(tickers)
    return result


def get_nasdaq100_flat() -> dict[str, str]:
    """Get flat dict of all Nasdaq 100 tickers to company names."""
    result = {}
    for tickers in get_nasdaq100_components().values():
        result.update(tickers)
    return result


# ---------------------------------------------------------------------------
# Hardcoded fallback: S&P 500 by GICS sector (as of early 2025)
# ---------------------------------------------------------------------------
_FALLBACK_SP500: dict[str, dict[str, str]] = {
    "Information Technology": {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corp.",
        "NVDA": "NVIDIA Corp.",
        "AVGO": "Broadcom Inc.",
        "ORCL": "Oracle Corp.",
        "CRM": "Salesforce Inc.",
        "CSCO": "Cisco Systems",
        "AMD": "Advanced Micro Devices",
        "ACN": "Accenture plc",
        "ADBE": "Adobe Inc.",
        "IBM": "International Business Machines",
        "INTU": "Intuit Inc.",
        "TXN": "Texas Instruments",
        "QCOM": "Qualcomm Inc.",
        "NOW": "ServiceNow Inc.",
        "AMAT": "Applied Materials",
        "SNPS": "Synopsys Inc.",
        "CDNS": "Cadence Design Systems",
        "ADI": "Analog Devices",
        "PANW": "Palo Alto Networks",
        "LRCX": "Lam Research",
        "KLAC": "KLA Corp.",
        "CRWD": "CrowdStrike Holdings",
        "FTNT": "Fortinet Inc.",
        "ROP": "Roper Technologies",
        "MCHP": "Microchip Technology",
        "NXPI": "NXP Semiconductors",
        "MPWR": "Monolithic Power Systems",
        "FSLR": "First Solar Inc.",
        "ANSS": "ANSYS Inc.",
        "CDW": "CDW Corp.",
        "KEYS": "Keysight Technologies",
        "ON": "ON Semiconductor",
        "TDY": "Teledyne Technologies",
        "PTC": "PTC Inc.",
        "TRMB": "Trimble Inc.",
        "GEN": "Gen Digital Inc.",
        "VRSN": "VeriSign Inc.",
        "IT": "Gartner Inc.",
        "FFIV": "F5 Inc.",
        "JNPR": "Juniper Networks",
        "SWKS": "Skyworks Solutions",
        "AKAM": "Akamai Technologies",
        "ENPH": "Enphase Energy",
        "TER": "Teradyne Inc.",
        "EPAM": "EPAM Systems",
        "CTSH": "Cognizant Technology Solutions",
        "HPE": "Hewlett Packard Enterprise",
        "HPQ": "HP Inc.",
        "WDC": "Western Digital",
        "STX": "Seagate Technology",
        "NTAP": "NetApp Inc.",
        "ZBRA": "Zebra Technologies",
    },
    "Health Care": {
        "LLY": "Eli Lilly & Co.",
        "UNH": "UnitedHealth Group",
        "JNJ": "Johnson & Johnson",
        "ABBV": "AbbVie Inc.",
        "MRK": "Merck & Co.",
        "TMO": "Thermo Fisher Scientific",
        "ABT": "Abbott Laboratories",
        "AMGN": "Amgen Inc.",
        "DHR": "Danaher Corp.",
        "PFE": "Pfizer Inc.",
        "ISRG": "Intuitive Surgical",
        "ELV": "Elevance Health",
        "SYK": "Stryker Corp.",
        "BSX": "Boston Scientific",
        "VRTX": "Vertex Pharmaceuticals",
        "MDT": "Medtronic plc",
        "REGN": "Regeneron Pharmaceuticals",
        "GILD": "Gilead Sciences",
        "BMY": "Bristol-Myers Squibb",
        "ZTS": "Zoetis Inc.",
        "CI": "The Cigna Group",
        "BDX": "Becton Dickinson",
        "HCA": "HCA Healthcare",
        "MCK": "McKesson Corp.",
        "COR": "Cencora Inc.",
        "A": "Agilent Technologies",
        "IDXX": "IDEXX Laboratories",
        "EW": "Edwards Lifesciences",
        "IQV": "IQVIA Holdings",
        "RMD": "ResMed Inc.",
        "DXCM": "DexCom Inc.",
        "MTD": "Mettler-Toledo International",
        "HOLX": "Hologic Inc.",
        "WAT": "Waters Corp.",
        "WST": "West Pharmaceutical Services",
        "BAX": "Baxter International",
        "ALGN": "Align Technology",
        "TECH": "Bio-Techne Corp.",
        "HSIC": "Henry Schein Inc.",
        "BIO": "Bio-Rad Laboratories",
        "INCY": "Incyte Corp.",
        "VTRS": "Viatris Inc.",
        "CRL": "Charles River Laboratories",
        "OGN": "Organon & Co.",
        "DVA": "DaVita Inc.",
        "PODD": "Insulet Corp.",
        "RVTY": "Revvity Inc.",
    },
    "Financials": {
        "BRK-B": "Berkshire Hathaway (B)",
        "JPM": "JPMorgan Chase & Co.",
        "V": "Visa Inc.",
        "MA": "Mastercard Inc.",
        "BAC": "Bank of America",
        "WFC": "Wells Fargo & Co.",
        "GS": "Goldman Sachs Group",
        "MS": "Morgan Stanley",
        "SPGI": "S&P Global Inc.",
        "BLK": "BlackRock Inc.",
        "AXP": "American Express",
        "PGR": "Progressive Corp.",
        "C": "Citigroup Inc.",
        "SCHW": "Charles Schwab Corp.",
        "MMC": "Marsh & McLennan",
        "ICE": "Intercontinental Exchange",
        "CB": "Chubb Ltd.",
        "CME": "CME Group",
        "AON": "Aon plc",
        "PNC": "PNC Financial Services",
        "TFC": "Truist Financial",
        "USB": "U.S. Bancorp",
        "MET": "MetLife Inc.",
        "AIG": "American International Group",
        "AFL": "Aflac Inc.",
        "TROW": "T. Rowe Price Group",
        "TRV": "The Travelers Companies",
        "AJG": "Arthur J. Gallagher",
        "ALL": "Allstate Corp.",
        "MTB": "M&T Bank Corp.",
        "FI": "Fiserv Inc.",
        "WTW": "Willis Towers Watson",
        "CPAY": "Corpay Inc.",
        "HBAN": "Huntington Bancshares",
        "DFS": "Discover Financial Services",
        "KEY": "KeyCorp",
        "CFG": "Citizens Financial Group",
        "RJF": "Raymond James Financial",
        "FDS": "FactSet Research Systems",
        "CINF": "Cincinnati Financial",
        "L": "Loews Corp.",
        "RE": "Everest Group",
        "BEN": "Franklin Resources",
        "IVZ": "Invesco Ltd.",
        "GL": "Globe Life Inc.",
        "MKTX": "MarketAxess Holdings",
        "NDAQ": "Nasdaq Inc.",
        "NTRS": "Northern Trust Corp.",
        "STT": "State Street Corp.",
        "SYF": "Synchrony Financial",
        "BRO": "Brown & Brown Inc.",
        "FITB": "Fifth Third Bancorp",
        "RF": "Regions Financial",
        "ZION": "Zions Bancorporation",
        "CMA": "Comerica Inc.",
        "WRB": "W. R. Berkley Corp.",
        "ERIE": "Erie Indemnity Co.",
    },
    "Consumer Discretionary": {
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "HD": "Home Depot Inc.",
        "MCD": "McDonald's Corp.",
        "NKE": "NIKE Inc.",
        "LOW": "Lowe's Companies",
        "BKNG": "Booking Holdings",
        "SBUX": "Starbucks Corp.",
        "TJX": "TJX Companies",
        "ORLY": "O'Reilly Automotive",
        "AZO": "AutoZone Inc.",
        "CMG": "Chipotle Mexican Grill",
        "ROST": "Ross Stores Inc.",
        "DHI": "D.R. Horton Inc.",
        "LEN": "Lennar Corp.",
        "GM": "General Motors",
        "F": "Ford Motor Co.",
        "GPC": "Genuine Parts Co.",
        "YUM": "Yum! Brands Inc.",
        "DRI": "Darden Restaurants",
        "POOL": "Pool Corp.",
        "KMX": "CarMax Inc.",
        "EBAY": "eBay Inc.",
        "GRMN": "Garmin Ltd.",
        "ULTA": "Ulta Beauty Inc.",
        "BWA": "BorgWarner Inc.",
        "BBY": "Best Buy Co.",
        "MGM": "MGM Resorts International",
        "LKQ": "LKQ Corp.",
        "APTV": "Aptiv plc",
        "TPR": "Tapestry Inc.",
        "NVR": "NVR Inc.",
        "PHM": "PulteGroup Inc.",
        "CCL": "Carnival Corp.",
        "RCL": "Royal Caribbean Cruises",
        "NCLH": "Norwegian Cruise Line",
        "HAS": "Hasbro Inc.",
        "MHK": "Mohawk Industries",
        "RL": "Ralph Lauren Corp.",
        "PVH": "PVH Corp.",
        "WYNN": "Wynn Resorts",
        "CZR": "Caesars Entertainment",
        "EXPE": "Expedia Group",
        "MAR": "Marriott International",
        "HLT": "Hilton Worldwide",
        "LVS": "Las Vegas Sands",
        "WHR": "Whirlpool Corp.",
        "DECK": "Deckers Outdoor",
    },
    "Communication Services": {
        "META": "Meta Platforms Inc.",
        "GOOGL": "Alphabet Inc. (A)",
        "GOOG": "Alphabet Inc. (C)",
        "NFLX": "Netflix Inc.",
        "DIS": "Walt Disney Co.",
        "CMCSA": "Comcast Corp.",
        "T": "AT&T Inc.",
        "VZ": "Verizon Communications",
        "TMUS": "T-Mobile US Inc.",
        "CHTR": "Charter Communications",
        "EA": "Electronic Arts",
        "TTWO": "Take-Two Interactive",
        "MTCH": "Match Group Inc.",
        "OMC": "Omnicom Group",
        "IPG": "Interpublic Group",
        "FOXA": "Fox Corp. (A)",
        "FOX": "Fox Corp. (B)",
        "PARA": "Paramount Global",
        "WBD": "Warner Bros. Discovery",
        "LYV": "Live Nation Entertainment",
        "NWSA": "News Corp. (A)",
        "NWS": "News Corp. (B)",
    },
    "Industrials": {
        "GE": "GE Aerospace",
        "CAT": "Caterpillar Inc.",
        "UNP": "Union Pacific Corp.",
        "HON": "Honeywell International",
        "RTX": "RTX Corp.",
        "UPS": "United Parcel Service",
        "DE": "Deere & Co.",
        "BA": "Boeing Co.",
        "LMT": "Lockheed Martin",
        "ADP": "Automatic Data Processing",
        "ETN": "Eaton Corp.",
        "GD": "General Dynamics",
        "ITW": "Illinois Tool Works",
        "NOC": "Northrop Grumman",
        "TT": "Trane Technologies",
        "WM": "Waste Management",
        "PH": "Parker-Hannifin",
        "EMR": "Emerson Electric",
        "CTAS": "Cintas Corp.",
        "CSX": "CSX Corp.",
        "NSC": "Norfolk Southern",
        "FAST": "Fastenal Co.",
        "CARR": "Carrier Global",
        "VRSK": "Verisk Analytics",
        "OTIS": "Otis Worldwide",
        "GWW": "W.W. Grainger",
        "PCAR": "PACCAR Inc.",
        "FDX": "FedEx Corp.",
        "RSG": "Republic Services",
        "AME": "AMETEK Inc.",
        "EFX": "Equifax Inc.",
        "LDOS": "Leidos Holdings",
        "DAL": "Delta Air Lines",
        "SWK": "Stanley Black & Decker",
        "ROK": "Rockwell Automation",
        "DOV": "Dover Corp.",
        "XYL": "Xylem Inc.",
        "HUBB": "Hubbell Inc.",
        "BAH": "Booz Allen Hamilton",
        "TDG": "TransDigm Group",
        "AXON": "Axon Enterprise",
        "IR": "Ingersoll Rand",
        "WAB": "Westinghouse Air Brake",
        "SNA": "Snap-on Inc.",
        "TXT": "Textron Inc.",
        "LHX": "L3Harris Technologies",
        "HII": "Huntington Ingalls Industries",
        "J": "Jacobs Solutions",
        "MAS": "Masco Corp.",
        "IEX": "IDEX Corp.",
        "NDSN": "Nordson Corp.",
        "UAL": "United Airlines Holdings",
        "AAL": "American Airlines Group",
        "LUV": "Southwest Airlines",
        "GNRC": "Generac Holdings",
        "PWR": "Quanta Services",
        "CPRT": "Copart Inc.",
        "PAYX": "Paychex Inc.",
    },
    "Consumer Staples": {
        "PG": "Procter & Gamble",
        "COST": "Costco Wholesale",
        "WMT": "Walmart Inc.",
        "KO": "Coca-Cola Co.",
        "PEP": "PepsiCo Inc.",
        "PM": "Philip Morris International",
        "MO": "Altria Group",
        "MDLZ": "Mondelez International",
        "CL": "Colgate-Palmolive",
        "KMB": "Kimberly-Clark",
        "GIS": "General Mills",
        "SYY": "Sysco Corp.",
        "HSY": "Hershey Co.",
        "K": "Kellanova",
        "KHC": "Kraft Heinz Co.",
        "ADM": "Archer-Daniels-Midland",
        "MKC": "McCormick & Co.",
        "STZ": "Constellation Brands",
        "SJM": "J.M. Smucker Co.",
        "CLX": "Clorox Co.",
        "CAG": "Conagra Brands",
        "CHD": "Church & Dwight",
        "HRL": "Hormel Foods",
        "TAP": "Molson Coors Beverage",
        "BG": "Bunge Global SA",
        "CPB": "Campbell Soup Co.",
        "LW": "Lamb Weston Holdings",
        "TSN": "Tyson Foods Inc.",
        "KR": "Kroger Co.",
        "WBA": "Walgreens Boots Alliance",
        "EL": "Estee Lauder Companies",
        "MNST": "Monster Beverage",
        "KDP": "Keurig Dr Pepper",
    },
    "Energy": {
        "XOM": "Exxon Mobil Corp.",
        "CVX": "Chevron Corp.",
        "COP": "ConocoPhillips",
        "SLB": "Schlumberger Ltd.",
        "EOG": "EOG Resources",
        "MPC": "Marathon Petroleum",
        "PSX": "Phillips 66",
        "VLO": "Valero Energy",
        "OXY": "Occidental Petroleum",
        "HAL": "Halliburton Co.",
        "HES": "Hess Corp.",
        "DVN": "Devon Energy",
        "FANG": "Diamondback Energy",
        "BKR": "Baker Hughes Co.",
        "CTRA": "Coterra Energy",
        "MRO": "Marathon Oil",
        "APA": "APA Corp.",
        "TRGP": "Targa Resources",
        "OKE": "ONEOK Inc.",
        "WMB": "Williams Companies",
        "KMI": "Kinder Morgan",
        "EQT": "EQT Corp.",
    },
    "Utilities": {
        "NEE": "NextEra Energy",
        "SO": "Southern Co.",
        "DUK": "Duke Energy",
        "D": "Dominion Energy",
        "SRE": "Sempra",
        "AEP": "American Electric Power",
        "CEG": "Constellation Energy",
        "EXC": "Exelon Corp.",
        "XEL": "Xcel Energy",
        "ED": "Consolidated Edison",
        "PCG": "PG&E Corp.",
        "WEC": "WEC Energy Group",
        "AWK": "American Water Works",
        "EIX": "Edison International",
        "DTE": "DTE Energy",
        "ES": "Eversource Energy",
        "AEE": "Ameren Corp.",
        "CMS": "CMS Energy",
        "FE": "FirstEnergy Corp.",
        "ETR": "Entergy Corp.",
        "ATO": "Atmos Energy",
        "CNP": "CenterPoint Energy",
        "NI": "NiSource Inc.",
        "EVRG": "Evergy Inc.",
        "PPL": "PPL Corp.",
        "PNW": "Pinnacle West Capital",
        "LNT": "Alliant Energy",
        "NRG": "NRG Energy",
    },
    "Real Estate": {
        "PLD": "Prologis Inc.",
        "AMT": "American Tower",
        "EQIX": "Equinix Inc.",
        "CCI": "Crown Castle Inc.",
        "PSA": "Public Storage",
        "O": "Realty Income Corp.",
        "SPG": "Simon Property Group",
        "WELL": "Welltower Inc.",
        "DLR": "Digital Realty Trust",
        "VICI": "VICI Properties",
        "IRM": "Iron Mountain Inc.",
        "ARE": "Alexandria Real Estate",
        "AVB": "AvalonBay Communities",
        "EQR": "Equity Residential",
        "ESS": "Essex Property Trust",
        "MAA": "Mid-America Apartment",
        "UDR": "UDR Inc.",
        "KIM": "Kimco Realty",
        "REG": "Regency Centers",
        "FRT": "Federal Realty Trust",
        "CPT": "Camden Property Trust",
        "HST": "Host Hotels & Resorts",
        "BXP": "BXP Inc.",
        "VTR": "Ventas Inc.",
        "INVH": "Invitation Homes",
        "SUI": "Sun Communities",
    },
    "Materials": {
        "LIN": "Linde plc",
        "SHW": "Sherwin-Williams",
        "APD": "Air Products & Chemicals",
        "ECL": "Ecolab Inc.",
        "FCX": "Freeport-McMoRan",
        "NEM": "Newmont Corp.",
        "NUE": "Nucor Corp.",
        "VMC": "Vulcan Materials",
        "MLM": "Martin Marietta Materials",
        "DOW": "Dow Inc.",
        "DD": "DuPont de Nemours",
        "PPG": "PPG Industries",
        "IFF": "International Flavors & Fragrances",
        "CE": "Celanese Corp.",
        "BALL": "Ball Corp.",
        "PKG": "Packaging Corp. of America",
        "IP": "International Paper",
        "CF": "CF Industries Holdings",
        "MOS": "Mosaic Co.",
        "EMN": "Eastman Chemical",
        "ALB": "Albemarle Corp.",
        "AMCR": "Amcor plc",
        "SEE": "Sealed Air Corp.",
        "FMC": "FMC Corp.",
        "CTVA": "Corteva Inc.",
        "STLD": "Steel Dynamics",
    },
}

# ---------------------------------------------------------------------------
# Hardcoded fallback: Nasdaq 100 (as of early 2025)
# ---------------------------------------------------------------------------
_FALLBACK_NASDAQ100: dict[str, dict[str, str]] = {
    "Technology": {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corp.",
        "NVDA": "NVIDIA Corp.",
        "AVGO": "Broadcom Inc.",
        "ADBE": "Adobe Inc.",
        "CRM": "Salesforce Inc.",
        "AMD": "Advanced Micro Devices",
        "CSCO": "Cisco Systems",
        "INTU": "Intuit Inc.",
        "QCOM": "Qualcomm Inc.",
        "TXN": "Texas Instruments",
        "AMAT": "Applied Materials",
        "SNPS": "Synopsys Inc.",
        "CDNS": "Cadence Design Systems",
        "ADI": "Analog Devices",
        "PANW": "Palo Alto Networks",
        "LRCX": "Lam Research",
        "KLAC": "KLA Corp.",
        "CRWD": "CrowdStrike Holdings",
        "FTNT": "Fortinet Inc.",
        "MCHP": "Microchip Technology",
        "NXPI": "NXP Semiconductors",
        "ON": "ON Semiconductor",
        "MRVL": "Marvell Technology",
        "TEAM": "Atlassian Corp.",
        "WDAY": "Workday Inc.",
        "SPLK": "Splunk Inc.",
        "ZS": "Zscaler Inc.",
        "ANSS": "ANSYS Inc.",
        "CDW": "CDW Corp.",
        "TTWO": "Take-Two Interactive",
        "GEN": "Gen Digital Inc.",
    },
    "Communication Services": {
        "META": "Meta Platforms Inc.",
        "GOOGL": "Alphabet Inc. (A)",
        "GOOG": "Alphabet Inc. (C)",
        "NFLX": "Netflix Inc.",
        "CMCSA": "Comcast Corp.",
        "TMUS": "T-Mobile US Inc.",
        "CHTR": "Charter Communications",
        "EA": "Electronic Arts",
    },
    "Consumer Discretionary": {
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "BKNG": "Booking Holdings",
        "SBUX": "Starbucks Corp.",
        "ORLY": "O'Reilly Automotive",
        "AZO": "AutoZone Inc.",
        "ROST": "Ross Stores Inc.",
        "LULU": "Lululemon Athletica",
        "EBAY": "eBay Inc.",
        "RIVN": "Rivian Automotive",
        "LCID": "Lucid Group",
        "JD": "JD.com Inc.",
        "PDD": "PDD Holdings",
        "CPRT": "Copart Inc.",
        "PAYX": "Paychex Inc.",
        "FAST": "Fastenal Co.",
    },
    "Consumer Staples": {
        "COST": "Costco Wholesale",
        "PEP": "PepsiCo Inc.",
        "MDLZ": "Mondelez International",
        "KHC": "Kraft Heinz Co.",
        "KDP": "Keurig Dr Pepper",
        "MNST": "Monster Beverage",
        "WBA": "Walgreens Boots Alliance",
    },
    "Health Care": {
        "AMGN": "Amgen Inc.",
        "GILD": "Gilead Sciences",
        "ISRG": "Intuitive Surgical",
        "VRTX": "Vertex Pharmaceuticals",
        "REGN": "Regeneron Pharmaceuticals",
        "MRNA": "Moderna Inc.",
        "DXCM": "DexCom Inc.",
        "ILMN": "Illumina Inc.",
        "IDXX": "IDEXX Laboratories",
        "BIIB": "Biogen Inc.",
        "SGEN": "Seagen Inc.",
        "AZN": "AstraZeneca plc",
    },
    "Industrials": {
        "HON": "Honeywell International",
        "CSX": "CSX Corp.",
        "VRSK": "Verisk Analytics",
        "ODFL": "Old Dominion Freight Line",
        "CTAS": "Cintas Corp.",
        "PCAR": "PACCAR Inc.",
        "ADP": "Automatic Data Processing",
        "GWW": "W.W. Grainger",
        "AXON": "Axon Enterprise",
    },
    "Utilities & Energy": {
        "AEP": "American Electric Power",
        "XEL": "Xcel Energy",
        "EXC": "Exelon Corp.",
        "CEG": "Constellation Energy",
        "OKE": "ONEOK Inc.",
        "FANG": "Diamondback Energy",
    },
    "Financials": {
        "PYPL": "PayPal Holdings",
        "ABNB": "Airbnb Inc.",
        "MAR": "Marriott International",
        "COIN": "Coinbase Global",
    },
}
