"""Frame contracts at stage boundaries.

A stage that accepts whatever it is handed will produce a plausible-looking
number from a frame that lost half its rows to a bad merge. These checks are
cheap relative to a week of pipeline time and they name the offending keys,
because "duplicate player-weeks" without the keys is a bug report you cannot
act on.

Fail closed: a violated contract raises. There is no "warn and continue" mode,
because that is how the silent-empty-frame defects Phase B is removing got in.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd

from .failures import SchemaViolation

# How many offending keys to name before truncating the message.
MAX_REPORTED = 10


def _describe(values: Sequence) -> str:
    shown = list(values[:MAX_REPORTED])
    more = len(values) - len(shown)
    text = ", ".join(repr(v) for v in shown)
    return text + (f", ... (+{more} more)" if more > 0 else "")


def require_columns(frame: pd.DataFrame, name: str, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise SchemaViolation(
            f"{name}: missing required column(s) {missing}; "
            f"present: {sorted(frame.columns)[:20]}")


def require_dtypes(frame: pd.DataFrame, name: str, dtypes: Mapping[str, str]) -> None:
    """``dtypes`` maps column -> pandas dtype *kind* family.

    Families ('numeric', 'integer', 'float', 'string', 'bool') rather than exact
    dtypes: int32 vs int64 is a platform detail, not a contract violation, and
    pinning exact dtypes would make this fail on an ARM runner for no reason.
    """
    bad = []
    for column, family in dtypes.items():
        if column not in frame.columns:
            raise SchemaViolation(f"{name}: dtype contract names absent column {column!r}")
        series = frame[column]
        ok = {
            "numeric": pd.api.types.is_numeric_dtype,
            "integer": pd.api.types.is_integer_dtype,
            "float": pd.api.types.is_float_dtype,
            "bool": pd.api.types.is_bool_dtype,
            "string": lambda s: pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s),
        }[family](series)
        if not ok:
            bad.append(f"{column}: expected {family}, got {series.dtype}")
    if bad:
        raise SchemaViolation(f"{name}: dtype contract violated -- " + "; ".join(bad))


def require_unique(frame: pd.DataFrame, name: str, keys: Sequence[str]) -> None:
    """Reject duplicate rows on ``keys``, naming the duplicated keys."""
    require_columns(frame, name, keys)
    duplicated = frame.duplicated(subset=list(keys), keep=False)
    if not duplicated.any():
        return
    offenders = (frame.loc[duplicated, list(keys)]
                 .drop_duplicates()
                 .itertuples(index=False, name=None))
    offenders = list(offenders)
    raise SchemaViolation(
        f"{name}: {int(duplicated.sum())} row(s) violate uniqueness on {list(keys)}; "
        f"duplicated key(s): {_describe(offenders)}")


def require_non_null(frame: pd.DataFrame, name: str, columns: Sequence[str]) -> None:
    require_columns(frame, name, columns)
    bad = {c: int(frame[c].isna().sum()) for c in columns if frame[c].isna().any()}
    if bad:
        raise SchemaViolation(f"{name}: null values in non-nullable column(s) {bad}")


def check_frame(frame: pd.DataFrame, name: str, *,
                columns: Optional[Iterable[str]] = None,
                dtypes: Optional[Mapping[str, str]] = None,
                unique_keys: Optional[Sequence[str]] = None,
                non_null: Optional[Sequence[str]] = None,
                allow_empty: bool = True) -> pd.DataFrame:
    """Validate ``frame`` and return it unchanged, so it can wrap an expression.

    Returning the frame keeps call sites honest: ``pw = check_frame(build(...))``
    cannot be accidentally left un-validated the way a bare assertion can be
    deleted without the value disappearing.
    """
    if not isinstance(frame, pd.DataFrame):
        raise SchemaViolation(f"{name}: expected a DataFrame, got {type(frame).__name__}")
    if not allow_empty and frame.empty:
        raise SchemaViolation(
            f"{name}: frame is empty, but this stage requires rows. An empty "
            "frame here means an upstream fetch or filter dropped everything.")
    if columns:
        require_columns(frame, name, columns)
    if dtypes:
        require_dtypes(frame, name, dtypes)
    if unique_keys:
        require_unique(frame, name, unique_keys)
    if non_null:
        require_non_null(frame, name, non_null)
    return frame


# --------------------------------------------------------------------------- #
# The contracts themselves -- one place, so a stage cannot invent its own.
# --------------------------------------------------------------------------- #
PLAYER_WEEK: Dict = {
    "columns": ("season", "week", "player_id", "team", "role"),
    "dtypes": {"season": "numeric", "week": "numeric", "player_id": "string"},
    "unique_keys": ("season", "week", "player_id"),
    "non_null": ("season", "week", "player_id"),
}

OPP_POS_DEF: Dict = {
    "columns": ("season", "week", "defteam", "role"),
    "dtypes": {"season": "numeric", "week": "numeric"},
    "unique_keys": ("season", "week", "defteam", "role"),
}

TEAM_WEEK: Dict = {
    "columns": ("season", "week", "team"),
    "dtypes": {"season": "numeric", "week": "numeric"},
    "unique_keys": ("season", "week", "team"),
}

CANDIDATES: Dict = {
    "columns": ("season", "week", "game_id", "player_id", "market"),
    "unique_keys": ("season", "week", "game_id", "player_id", "market"),
    "non_null": ("player_id", "market"),
}

LINES: Dict = {
    "columns": ("ts", "game_id", "book", "market", "player_name", "side", "price"),
    "unique_keys": ("ts", "game_id", "book", "market", "player_name", "side"),
}
