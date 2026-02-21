"""HuggingFace World Development Indicators data source.

Downloads the datonic/world_development_indicators dataset (1,400+ indicators,
9M+ rows) from HuggingFace Hub as a cached Parquet file and provides filtering
functions compatible with the World Bank data source interface.
"""

import json

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

from data_sources.world_bank import (
    INDICATOR_CATEGORIES,
    get_all_indicators,
    get_country_groups,
    save_dataset,
)

HF_DATASET_URL = (
    "hf://datasets/datonic/world_development_indicators/data/world_development_indicators.parquet"
)

CACHE_DIR = Path(__file__).parent.parent / "data" / "hf_cache"
CACHE_FILE = CACHE_DIR / "wdi.parquet"
CATALOG_FILE = CACHE_DIR / "indicator_catalog.json"


def is_cache_available() -> bool:
    """Check whether the HuggingFace dataset cache exists locally."""
    return CACHE_FILE.exists()


def get_cache_info() -> dict:
    """Return metadata about the cached dataset file."""
    if not CACHE_FILE.exists():
        return {"available": False}

    stat = CACHE_FILE.stat()
    info = {
        "available": True,
        "size_mb": round(stat.st_size / (1024 * 1024), 1),
        "modified": stat.st_mtime,
    }

    if CATALOG_FILE.exists():
        catalog = _load_catalog()
        info["indicator_count"] = len(catalog)

    return info


def download_hf_dataset(force: bool = False) -> Path:
    """Download the full WDI Parquet from HuggingFace and cache locally.

    Also extracts an indicator catalog JSON for fast UI lookups.
    Returns the path to the cached Parquet file.
    """
    if CACHE_FILE.exists() and not force:
        return CACHE_FILE

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(HF_DATASET_URL)
    df.to_parquet(CACHE_FILE, index=False)

    _build_catalog(df)

    return CACHE_FILE


def _build_catalog(df: pd.DataFrame) -> None:
    """Extract unique indicator code/name pairs and save as JSON."""
    pairs = (
        df[["indicator_code", "indicator_name"]]
        .drop_duplicates()
        .sort_values("indicator_code")
    )
    catalog = dict(zip(pairs["indicator_code"], pairs["indicator_name"]))
    CATALOG_FILE.write_text(json.dumps(catalog, indent=2))


def _load_catalog() -> dict[str, str]:
    """Load the indicator catalog from the cached JSON file."""
    return json.loads(CATALOG_FILE.read_text())


def get_hf_indicator_catalog() -> dict[str, str]:
    """Return {indicator_code: indicator_name} for all indicators in the cache.

    If the catalog JSON doesn't exist but the Parquet does, rebuilds it.
    """
    if CATALOG_FILE.exists():
        return _load_catalog()

    if not CACHE_FILE.exists():
        return {}

    # Rebuild catalog from Parquet (reads only 2 columns)
    table = pq.read_table(CACHE_FILE, columns=["indicator_code", "indicator_name"])
    _build_catalog(table.to_pandas())
    return _load_catalog()


def download_from_hf(
    indicator_codes: list[str],
    countries: list[str],
    start_year: int = 2000,
    end_year: int = 2020,
    progress_callback=None,
) -> tuple[pd.DataFrame, list[str]]:
    """Filter the cached HuggingFace dataset and return a wide-format DataFrame.

    Uses pyarrow predicate pushdown for memory-efficient reads.

    Returns (DataFrame, failed_indicators) matching the signature of
    download_multiple_indicators in world_bank.py.
    """
    if not CACHE_FILE.exists():
        raise RuntimeError(
            "HuggingFace dataset cache not found. "
            "Please download it first from the Download page."
        )

    if progress_callback:
        progress_callback(0, 1, "Reading cached dataset...")

    # Predicate pushdown: only read matching rows and needed columns
    filters = [
        ("indicator_code", "in", indicator_codes),
        ("country_code", "in", countries),
        ("year", ">=", start_year),
        ("year", "<=", end_year),
    ]
    table = pq.read_table(
        CACHE_FILE,
        columns=["country_code", "indicator_code", "year", "indicator_value"],
        filters=filters,
    )
    filtered = table.to_pandas()

    # Identify which requested indicators were not found
    found_codes = set(filtered["indicator_code"].unique()) if not filtered.empty else set()
    catalog = get_hf_indicator_catalog()
    all_wb = get_all_indicators()
    failed = []
    for code in indicator_codes:
        if code not in found_codes:
            label = catalog.get(code) or all_wb.get(code, code)
            failed.append(label)

    if filtered.empty:
        if failed and len(failed) == len(indicator_codes):
            raise RuntimeError(
                f"All indicators failed: {', '.join(failed)}"
            )
        return pd.DataFrame(columns=["country", "year"]), failed

    # Pivot from long to wide format
    wide = filtered.pivot_table(
        index=["country_code", "year"],
        columns="indicator_code",
        values="indicator_value",
    ).reset_index()
    wide = wide.rename(columns={"country_code": "country"})
    wide.columns.name = None
    wide = wide.sort_values(["country", "year"]).reset_index(drop=True)

    if progress_callback:
        progress_callback(1, 1, "Complete")

    return wide, failed
