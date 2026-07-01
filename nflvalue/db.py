"""Thin SQLite helper for the prop-shortlister warehouse.

One file (``data/nfl.db``), a handful of tables, and three functions:
``connect``, ``upsert``, ``query_df``. Raw play-by-play stays on disk as
parquet (it's 400 columns and huge); this DB holds the aggregates the
projection engine and backtest actually read: ``player_week``, ``opp_pos_def``,
``projections``, ``prop_backtest``.

Schema intent follows ``docs/RAG_PIPELINE_PLAN.md`` §3.2, adapted for the
player-prop layer described in ``PROP_SHORTLISTER_SPEC.md`` and
``PHASE1_HANDSOFF_DESIGN.md``.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Iterable, List, Mapping, Sequence

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# NOTE: named nfl_props.db rather than nfl.db purely because of a leftover,
# undeletable stale journal file from local debugging in this environment
# (some mounted/synced project folders don't allow removing a written file);
# there was never a design reason for the literal filename "nfl.db".
DEFAULT_DB_PATH = os.path.join(ROOT, "data", "nfl_props.db")


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
SCHEMA = {
    "player_week": """
        CREATE TABLE IF NOT EXISTS player_week (
            season INTEGER, week INTEGER,
            player_id TEXT, player_name TEXT, team TEXT, role TEXT,
            -- actuals for this (season, week) -- the realized result
            targets REAL, receptions REAL, rec_yards REAL, air_yards_sum REAL,
            yac_sum REAL, carries REAL, rush_yards REAL,
            pass_attempts REAL, completions REAL, pass_yards REAL,
            pass_tds REAL, rush_tds REAL, rec_tds REAL,
            team_pass_att REAL, team_rush_att REAL, team_plays REAL,
            -- rolling, PRIOR-WEEKS-ONLY features (walk-forward; see features.py)
            roll_games INTEGER,
            roll_targets REAL, roll_target_share REAL, roll_air_yards REAL, roll_adot REAL,
            roll_carries REAL, roll_carry_share REAL,
            roll_pass_attempts REAL, roll_completions REAL,
            roll_ypt REAL, roll_catch_rate REAL, roll_ypc REAL, roll_ypa REAL,
            roll_pass_td_rate REAL, roll_rush_td_rate REAL, roll_rec_td_rate REAL,
            PRIMARY KEY (season, week, player_id)
        );
    """,
    "opp_pos_def": """
        CREATE TABLE IF NOT EXISTS opp_pos_def (
            season INTEGER, week INTEGER, defteam TEXT, role TEXT,
            -- actuals allowed this week (for rolling calc; not for direct use)
            targets_allowed REAL, rec_yards_allowed REAL, carries_allowed REAL,
            rush_yards_allowed REAL, pass_yards_allowed REAL, epa_allowed_sum REAL, plays_faced REAL,
            -- rolling, PRIOR-WEEKS-ONLY, relative to league mean (1.0 = average)
            roll_games INTEGER,
            roll_ypt_allowed_factor REAL, roll_ypc_allowed_factor REAL,
            roll_ypa_allowed_factor REAL, roll_epa_allowed_factor REAL,
            PRIMARY KEY (season, week, defteam, role)
        );
    """,
    "projections": """
        CREATE TABLE IF NOT EXISTS projections (
            season INTEGER, week INTEGER, player_id TEXT, name TEXT, pos TEXT,
            market TEXT, mean REAL, sd REAL, dist TEXT, line REAL,
            p_over REAL, p_under REAL, components_json TEXT, low_confidence INTEGER,
            created_at TEXT,
            PRIMARY KEY (season, week, player_id, market)
        );
    """,
    "prop_backtest": """
        CREATE TABLE IF NOT EXISTS prop_backtest (
            run_at TEXT, market TEXT, sample_bucket TEXT,
            n INTEGER, mae REAL, rmse REAL, corr REAL,
            calibration_bucket TEXT, calibration_p_over REAL, calibration_actual_over_rate REAL,
            PRIMARY KEY (run_at, market, sample_bucket, calibration_bucket)
        );
    """,
}


def connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open (and initialize) the warehouse DB, creating tables if needed."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    # WAL needs shared-memory mmap, which some mounted/synced project folders
    # don't support (raises "disk I/O error" on the first write); try modes in
    # order from most to least durable and keep whichever actually works.
    for mode in ("DELETE", "MEMORY", "OFF"):
        try:
            conn.execute(f"PRAGMA journal_mode={mode};")
            conn.execute("CREATE TABLE IF NOT EXISTS _mode_probe (x INTEGER);")
            conn.execute("DROP TABLE IF EXISTS _mode_probe;")
            conn.commit()
            break
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            continue
    for ddl in SCHEMA.values():
        conn.execute(ddl)
    conn.commit()
    return conn


def upsert(conn: sqlite3.Connection, table: str, rows: Iterable[Mapping], key_cols: Sequence[str]) -> int:
    """Insert-or-replace a batch of dict-like rows into ``table``.

    Uses SQLite's ``INSERT OR REPLACE`` keyed on the table's declared primary
    key (``key_cols`` must match). Column set is taken from the first row;
    all rows must share the same keys. Returns the number of rows written.
    """
    rows = list(rows)
    if not rows:
        return 0
    cols = list(rows[0].keys())
    missing_key = [k for k in key_cols if k not in cols]
    if missing_key:
        raise ValueError(f"key_cols {missing_key} not present in row columns {cols}")
    placeholders = ", ".join(["?"] * len(cols))
    col_list = ", ".join(cols)
    sql = f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({placeholders})"
    values = [[r.get(c) for c in cols] for r in rows]
    conn.executemany(sql, values)
    conn.commit()
    return len(rows)


def upsert_df(conn: sqlite3.Connection, table: str, df: pd.DataFrame, key_cols: Sequence[str]) -> int:
    """Convenience wrapper: upsert a DataFrame (NaN -> NULL)."""
    if df.empty:
        return 0
    clean = df.where(pd.notnull(df), None)
    return upsert(conn, table, clean.to_dict("records"), key_cols)


def query_df(conn: sqlite3.Connection, sql: str, params: Sequence = ()) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame."""
    return pd.read_sql_query(sql, conn, params=params)


def table_names(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r[0] for r in rows]
