"""SQLite database storage backend.

Replaces CSV file persistence with a local SQLite database.
The database file is stored at data/econ_dashboard.db.
"""

import sqlite3
import json
import pandas as pd
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime

DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "econ_dashboard.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- Dataset metadata registry
CREATE TABLE IF NOT EXISTS datasets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    data_type   TEXT NOT NULL DEFAULT 'economic',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    row_count   INTEGER,
    col_count   INTEGER,
    description TEXT
);

-- World Bank indicators in normalized long format
CREATE TABLE IF NOT EXISTS economic_data (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id      INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    country         TEXT NOT NULL,
    year            INTEGER NOT NULL,
    indicator_code  TEXT NOT NULL,
    value           REAL,
    UNIQUE(dataset_id, country, year, indicator_code)
);

CREATE INDEX IF NOT EXISTS idx_econ_dataset ON economic_data(dataset_id);
CREATE INDEX IF NOT EXISTS idx_econ_country_year ON economic_data(dataset_id, country, year);

-- Stock/ETF price data
CREATE TABLE IF NOT EXISTS stock_data (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id  INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,
    year        INTEGER NOT NULL,
    month       INTEGER NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      REAL,
    UNIQUE(dataset_id, ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_stock_dataset ON stock_data(dataset_id);
CREATE INDEX IF NOT EXISTS idx_stock_ticker_date ON stock_data(dataset_id, ticker, date);

-- Catch-all for merged/arbitrary DataFrames (JSON-serialized rows)
CREATE TABLE IF NOT EXISTS generic_data (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id  INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    row_index   INTEGER NOT NULL,
    row_data    TEXT NOT NULL,
    UNIQUE(dataset_id, row_index)
);

CREATE INDEX IF NOT EXISTS idx_generic_dataset ON generic_data(dataset_id);

-- Indicator code lookup table
CREATE TABLE IF NOT EXISTS indicator_metadata (
    code        TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    category    TEXT
);

-- News article persistence
CREATE TABLE IF NOT EXISTS news_articles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT NOT NULL,
    summary         TEXT,
    url             TEXT,
    source          TEXT,
    published       TEXT,
    fetched_at      TEXT NOT NULL DEFAULT (datetime('now')),
    combined_score  REAL,
    label           TEXT,
    textblob_score  REAL,
    lexicon_score   REAL
);

CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles(published);
"""

# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------


@contextmanager
def get_connection():
    """Context manager for database connections with WAL mode and foreign keys."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create all tables if they don't exist. Idempotent."""
    with get_connection() as conn:
        conn.executescript(_SCHEMA_SQL)


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------


def _register_dataset(conn, name, data_type, row_count, col_count,
                      description=None):
    """Insert or replace a dataset record. Returns the dataset_id."""
    conn.execute(
        """INSERT INTO datasets (name, data_type, row_count, col_count,
                                 description, updated_at)
           VALUES (?, ?, ?, ?, ?, datetime('now'))
           ON CONFLICT(name) DO UPDATE SET
               data_type=excluded.data_type,
               row_count=excluded.row_count,
               col_count=excluded.col_count,
               description=excluded.description,
               updated_at=datetime('now')""",
        (name, data_type, row_count, col_count, description),
    )
    cursor = conn.execute("SELECT id FROM datasets WHERE name=?", (name,))
    return cursor.fetchone()[0]


def list_datasets(data_type=None):
    """List all saved dataset names, optionally filtered by type."""
    with get_connection() as conn:
        if data_type:
            rows = conn.execute(
                "SELECT name FROM datasets WHERE data_type=? ORDER BY updated_at DESC",
                (data_type,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT name FROM datasets ORDER BY updated_at DESC"
            ).fetchall()
    return [r[0] for r in rows]


def delete_dataset(name):
    """Delete a dataset and all associated data (CASCADE)."""
    with get_connection() as conn:
        conn.execute("DELETE FROM datasets WHERE name=?", (name,))


# ---------------------------------------------------------------------------
# Economic data save/load
# ---------------------------------------------------------------------------


def save_economic_dataset(df, name):
    """Save a wide-format economic DataFrame to the database.

    Converts from wide format (country, year, IND1, IND2, ...)
    to long format for storage.
    """
    indicator_cols = [c for c in df.columns if c not in ("country", "year")]

    with get_connection() as conn:
        dataset_id = _register_dataset(
            conn, name, "economic", len(df), len(df.columns)
        )
        conn.execute(
            "DELETE FROM economic_data WHERE dataset_id=?", (dataset_id,)
        )

        # Rename any existing 'value' column to avoid pandas melt conflict
        df_to_melt = df.copy()
        if "value" in indicator_cols:
            df_to_melt = df_to_melt.rename(columns={"value": "_value_col"})
            indicator_cols = [
                "_value_col" if c == "value" else c for c in indicator_cols
            ]

        long_df = df_to_melt.melt(
            id_vars=["country", "year"],
            value_vars=indicator_cols,
            var_name="indicator_code",
            value_name="indicator_value",
        ).dropna(subset=["indicator_value"])

        # Restore original indicator code name
        long_df["indicator_code"] = long_df["indicator_code"].replace(
            "_value_col", "value"
        )

        conn.executemany(
            """INSERT INTO economic_data
               (dataset_id, country, year, indicator_code, value)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (dataset_id, row.country, int(row.year),
                 row.indicator_code, float(row.indicator_value))
                for row in long_df.itertuples(index=False)
            ],
        )


def load_economic_dataset(name):
    """Load an economic dataset and return it in wide format."""
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT id FROM datasets WHERE name=?", (name,)
        )
        row = cursor.fetchone()
        if row is None:
            raise FileNotFoundError(f"Dataset '{name}' not found")
        dataset_id = row[0]

        df = pd.read_sql_query(
            "SELECT country, year, indicator_code, value "
            "FROM economic_data WHERE dataset_id=?",
            conn,
            params=(dataset_id,),
        )

    if df.empty:
        return pd.DataFrame()

    wide = df.pivot_table(
        index=["country", "year"],
        columns="indicator_code",
        values="value",
    ).reset_index()
    wide.columns.name = None
    return wide.sort_values(["country", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stock data save/load
# ---------------------------------------------------------------------------


def save_stock_db(df, name):
    """Save stock price data to the database."""
    with get_connection() as conn:
        dataset_id = _register_dataset(
            conn, name, "stock", len(df), len(df.columns)
        )
        conn.execute(
            "DELETE FROM stock_data WHERE dataset_id=?", (dataset_id,)
        )

        conn.executemany(
            """INSERT INTO stock_data
               (dataset_id, ticker, date, year, month,
                open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    dataset_id,
                    row.get("ticker", ""),
                    str(row.get("date", "")),
                    int(row.get("year", 0)),
                    int(row.get("month", 0)),
                    row.get("open"),
                    row.get("high"),
                    row.get("low"),
                    row.get("close"),
                    row.get("volume"),
                )
                for row in df.to_dict("records")
            ],
        )


def load_stock_db(name):
    """Load stock data from the database."""
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT id FROM datasets WHERE name=?", (name,)
        )
        row = cursor.fetchone()
        if row is None:
            raise FileNotFoundError(f"Stock dataset '{name}' not found")
        dataset_id = row[0]

        df = pd.read_sql_query(
            "SELECT ticker, date, year, month, open, high, low, close, volume "
            "FROM stock_data WHERE dataset_id=?",
            conn,
            params=(dataset_id,),
        )

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Generic data save/load (for merged/arbitrary DataFrames)
# ---------------------------------------------------------------------------


def save_generic_dataset(df, name, data_type="merged"):
    """Save an arbitrary DataFrame as JSON rows."""
    with get_connection() as conn:
        dataset_id = _register_dataset(
            conn, name, data_type, len(df), len(df.columns)
        )
        conn.execute(
            "DELETE FROM generic_data WHERE dataset_id=?", (dataset_id,)
        )

        # Store column names as row_index -1 for schema preservation
        conn.execute(
            "INSERT INTO generic_data (dataset_id, row_index, row_data) "
            "VALUES (?, ?, ?)",
            (dataset_id, -1, json.dumps(list(df.columns))),
        )

        conn.executemany(
            "INSERT INTO generic_data (dataset_id, row_index, row_data) "
            "VALUES (?, ?, ?)",
            [
                (dataset_id, idx, json.dumps(row, default=_json_serializer))
                for idx, row in enumerate(df.to_dict("records"))
            ],
        )


def load_generic_dataset(name):
    """Load a generic dataset from JSON rows."""
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT id FROM datasets WHERE name=?", (name,)
        )
        row = cursor.fetchone()
        if row is None:
            raise FileNotFoundError(f"Dataset '{name}' not found")
        dataset_id = row[0]

        rows = conn.execute(
            "SELECT row_index, row_data FROM generic_data "
            "WHERE dataset_id=? ORDER BY row_index",
            (dataset_id,),
        ).fetchall()

    if not rows:
        return pd.DataFrame()

    columns = None
    data_rows = []
    for row_index, row_data in rows:
        if row_index == -1:
            columns = json.loads(row_data)
        else:
            data_rows.append(json.loads(row_data))

    df = pd.DataFrame(data_rows)
    if columns and not df.empty and len(df.columns) == len(columns):
        df.columns = columns
    return df


# ---------------------------------------------------------------------------
# Smart save/load (auto-detect data type)
# ---------------------------------------------------------------------------


def save_dataset_smart(df, name):
    """Auto-detect the data type and save to the appropriate table."""
    if _is_stock_data(df):
        save_stock_db(df, name)
    elif _is_economic_data(df):
        save_economic_dataset(df, name)
    else:
        save_generic_dataset(df, name)


def load_dataset_smart(name):
    """Load a dataset, routing to the correct loader by data_type."""
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT data_type FROM datasets WHERE name=?", (name,)
        )
        row = cursor.fetchone()
        if row is None:
            raise FileNotFoundError(f"Dataset '{name}' not found")
        data_type = row[0]

    if data_type == "economic":
        return load_economic_dataset(name)
    elif data_type == "stock":
        return load_stock_db(name)
    else:
        return load_generic_dataset(name)


# ---------------------------------------------------------------------------
# News article persistence
# ---------------------------------------------------------------------------


def save_news_articles(articles):
    """Save a list of news article dicts to the database.

    Each dict should have keys: title, summary, url, source, published.
    Optional sentiment keys: combined_score, label, textblob_score, lexicon_score.
    """
    with get_connection() as conn:
        conn.executemany(
            """INSERT INTO news_articles
               (title, summary, url, source, published,
                combined_score, label, textblob_score, lexicon_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    a.get("title", ""),
                    a.get("summary", ""),
                    a.get("url", ""),
                    a.get("source", ""),
                    a["published"].isoformat() if isinstance(
                        a.get("published"), datetime
                    ) else str(a.get("published", "")),
                    a.get("combined_score"),
                    a.get("label"),
                    a.get("textblob_score"),
                    a.get("lexicon_score"),
                )
                for a in articles
            ],
        )


def load_news_articles(limit=200):
    """Load recent news articles from the database."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM news_articles ORDER BY published DESC LIMIT ?",
            conn,
            params=(limit,),
        )
    return df


# ---------------------------------------------------------------------------
# Indicator metadata
# ---------------------------------------------------------------------------


def save_indicator_metadata(indicators_by_category):
    """Populate the indicator_metadata table from INDICATOR_CATEGORIES dict."""
    with get_connection() as conn:
        for category, indicators in indicators_by_category.items():
            conn.executemany(
                """INSERT INTO indicator_metadata (code, name, category)
                   VALUES (?, ?, ?)
                   ON CONFLICT(code) DO UPDATE SET
                       name=excluded.name, category=excluded.category""",
                [(code, name, category) for code, name in indicators.items()],
            )


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def migrate_csv_to_db():
    """Migrate all existing CSV files in data/ to the database.

    Returns the number of datasets migrated.
    """
    csv_dir = Path(__file__).parent.parent / "data"
    if not csv_dir.exists():
        return 0

    migrated = 0
    for csv_path in csv_dir.glob("*.csv"):
        name = csv_path.stem
        # Skip if already in database
        existing = list_datasets()
        if name in existing:
            continue
        try:
            df = pd.read_csv(csv_path)
            save_dataset_smart(df, name)
            migrated += 1
        except Exception:
            pass

    return migrated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_economic_data(df):
    """Heuristic: has 'country' and 'year' columns, no 'ticker' column."""
    cols = set(df.columns)
    return "country" in cols and "year" in cols and "ticker" not in cols


def _is_stock_data(df):
    """Heuristic: has 'ticker' and 'date' columns."""
    cols = set(df.columns)
    return "ticker" in cols and "date" in cols


def _json_serializer(obj):
    """JSON serializer for numpy/pandas types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if pd.isna(obj):
        return None
    return str(obj)
