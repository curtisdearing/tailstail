"""Phase A: immutable decision snapshots, closing capture, and forward CLV.

The unit of measurement is a DECISION, not a bet. The moment a lean exists we
persist exactly what the market was offering at that instant; whether money is
ever staked is irrelevant to whether the model beat the price. Staking adds
bankroll noise on top of the thing we are actually trying to measure.

Three rules hold the measurement together:

1. A decision row is written once. ``decision_id`` IS its canonical content
   hash, so mutating any hashed field would have to change the primary key,
   and SQLite triggers (``db.TRIGGERS``) refuse UPDATE and DELETE outright.
2. Closing information lives in ``closing_snapshots``, keyed BY ``decision_id``
   and never merged into the decision row. Writing a close back into an earlier
   projection is information travelling backwards in time; every downstream
   number computed from that row is then unfalsifiable. This has bitten this
   codebase before, so it is structural here rather than a convention.
3. Below the precommitted resolved-pair minimum the aggregator emits NO point
   estimate at all -- not a flagged one, not a "trending" one. The keys are
   absent from the payload, so a caller cannot render a number that does not
   deserve to exist.

Probabilities are de-vigged (``prob_kind='devig'``) except for one-sided
markets such as ``anytime_td``, where the quote is Yes-only and the stored
probability is raw implied -- vig included at BOTH decision and close, so the
DIFFERENCE remains meaningful. ``prob_kind`` travels with every row so the two
are never silently pooled.
"""

from __future__ import annotations

import hashlib
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from . import clv as clvmod
from . import db as dbmod
from .reproducibility import _cell

# Bump ONLY with a migration: changing the encoding changes every decision_id.
HASH_VERSION = 1

# Ordered: the hash is over this exact field sequence. ``recorded_at`` is
# deliberately excluded -- when we wrote the row is not part of the decision.
DECISION_HASH_FIELDS: Sequence[str] = (
    "decided_at_utc", "season", "week", "game_id", "player_id", "player_name",
    "book", "market", "side", "price", "point", "model_prob", "devig_prob",
    "prob_kind", "edge", "model_version", "commit_sha",
)

CLOSING_HASH_FIELDS: Sequence[str] = (
    "decision_id", "close_ts", "kickoff_ts", "cutoff_ts", "close_point",
    "close_devig_prob", "prob_kind", "n_books",
)

DEFAULT_CLOSE_BUFFER_MINUTES = 15


class AppendOnlyViolation(RuntimeError):
    """Raised when a caller tries to rewrite recorded history."""


class AsOfViolation(ValueError):
    """Raised when a capture would use information from the wrong side of the clock."""


# --------------------------------------------------------------------------- #
# Canonical hashing
# --------------------------------------------------------------------------- #
def canonical_record_sha256(record: Dict, fields: Sequence[str],
                            *, hash_version: int = HASH_VERSION) -> str:
    """Type-tagged, order-fixed digest of ``record`` over ``fields``.

    Reuses ``reproducibility._cell`` so that None, "", 0, and 0.0 cannot
    collide -- an untagged join would hash ``point=0`` and ``point=None``
    identically and quietly merge two different decisions.
    """
    missing = [f for f in fields if f not in record]
    if missing:
        raise ValueError(f"cannot hash record: missing fields {missing}")
    payload = "\x1e".join(
        [f"v{hash_version}"] + [f"{f}\x1f{_cell(record[f])}" for f in fields]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def current_commit_sha(default: str = "unknown") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            cwd=dbmod.ROOT, text=True).strip()
    except Exception:
        return default


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #
def _parse_ts(ts: str) -> datetime:
    text = str(ts).strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _fmt_ts(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# A.1 -- immutable decision snapshot
# --------------------------------------------------------------------------- #
def record_decision(conn: sqlite3.Connection, *, decided_at_utc: str,
                    season: int, week: int, game_id: str, player_id: str,
                    market: str, side: str, book: str, price: float,
                    model_prob: float, devig_prob: float,
                    model_version: str, commit_sha: Optional[str] = None,
                    point: Optional[float] = None,
                    player_name: Optional[str] = None,
                    prob_kind: str = "devig") -> Dict:
    """Append one decision. Returns the stored row (including ``decision_id``).

    Re-appending a byte-identical decision is a no-op (the job retried);
    a decision whose id already exists with DIFFERENT content raises.
    """
    if prob_kind not in ("devig", "raw_implied"):
        raise ValueError(f"prob_kind must be 'devig' or 'raw_implied', got {prob_kind!r}")
    row = {
        "decided_at_utc": _fmt_ts(_parse_ts(decided_at_utc)),
        "season": int(season), "week": int(week), "game_id": game_id,
        "player_id": player_id, "player_name": player_name,
        "book": book, "market": market, "side": side,
        "price": float(price), "point": None if point is None else float(point),
        "model_prob": float(model_prob), "devig_prob": float(devig_prob),
        "prob_kind": prob_kind,
        "edge": float(model_prob) - float(devig_prob),
        "model_version": model_version,
        "commit_sha": commit_sha if commit_sha is not None else current_commit_sha(),
    }
    decision_id = canonical_record_sha256(row, DECISION_HASH_FIELDS)
    stored = {"decision_id": decision_id, **row, "hash_version": HASH_VERSION,
              "recorded_at": _fmt_ts(datetime.now(timezone.utc))}

    existing = conn.execute(
        "SELECT * FROM decision_snapshots WHERE decision_id=?", (decision_id,)).fetchone()
    if existing is not None:
        cols = [d[0] for d in conn.execute(
            "SELECT * FROM decision_snapshots LIMIT 0").description]
        prior = dict(zip(cols, existing))
        if any(prior[f] != stored[f] for f in DECISION_HASH_FIELDS):
            raise AppendOnlyViolation(
                f"decision_id {decision_id} already recorded with different content")
        return prior  # idempotent replay of the same decision

    cols = list(stored.keys())
    conn.execute(
        f"INSERT INTO decision_snapshots ({', '.join(cols)}) "
        f"VALUES ({', '.join(['?'] * len(cols))})",
        [stored[c] for c in cols])
    conn.commit()
    return stored


def load_decision(conn: sqlite3.Connection, decision_id: str) -> Optional[Dict]:
    """Read a decision. Reads ``decision_snapshots`` ONLY -- by construction
    there is no join here through which closing data could reach a caller
    reasoning about what was known at decision time."""
    cur = conn.execute("SELECT * FROM decision_snapshots WHERE decision_id=?", (decision_id,))
    row = cur.fetchone()
    if row is None:
        return None
    return dict(zip([d[0] for d in cur.description], row))


def verify_decision(conn: sqlite3.Connection, decision_id: str) -> bool:
    """Recompute the content hash and confirm it still equals the primary key."""
    row = load_decision(conn, decision_id)
    if row is None:
        return False
    return canonical_record_sha256(
        row, DECISION_HASH_FIELDS, hash_version=int(row["hash_version"])) == decision_id


def decisions(conn: sqlite3.Connection, season: Optional[int] = None,
              week: Optional[int] = None) -> pd.DataFrame:
    sql, params = "SELECT * FROM decision_snapshots", []
    clauses = []
    if season is not None:
        clauses.append("season=?"); params.append(season)
    if week is not None:
        clauses.append("week=?"); params.append(week)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    return dbmod.query_df(conn, sql + " ORDER BY decided_at_utc", params)


# --------------------------------------------------------------------------- #
# A.2 -- closing-line capture (separate row, keyed to the decision)
# --------------------------------------------------------------------------- #
def _default_close_fetcher(conn: sqlite3.Connection, decision: Dict,
                           cutoff_ts: str) -> Optional[Dict]:
    """Read the last market snapshot at or before ``cutoff_ts`` from ``lines``.

    Injectable so the capture job does not require a live book feed; a caller
    with its own price source passes ``close_fetcher``.
    """
    return clvmod.snapshot_prob(conn, decision["game_id"], decision["market"],
                                decision["player_id"], decision["side"],
                                at_or_before_ts=cutoff_ts)


def capture_close(conn: sqlite3.Connection, decision_id: str, *, kickoff_ts: str,
                  buffer_minutes: int = DEFAULT_CLOSE_BUFFER_MINUTES,
                  close_fetcher: Optional[Callable] = None,
                  captured_at_utc: Optional[str] = None) -> Optional[Dict]:
    """Append a closing row for ``decision_id``. Never touches the decision row.

    Returns None (unresolved, and counted as such) when there is no usable
    snapshot in the window. Raises ``AsOfViolation`` if the candidate close
    predates the decision -- that is not a close, it is the entry again.
    """
    decision = load_decision(conn, decision_id)
    if decision is None:
        raise KeyError(f"unknown decision_id {decision_id}")

    kickoff = _parse_ts(kickoff_ts)
    cutoff = kickoff - timedelta(minutes=buffer_minutes)
    decided_at = _parse_ts(decision["decided_at_utc"])
    if cutoff <= decided_at:
        raise AsOfViolation(
            f"close cutoff {_fmt_ts(cutoff)} is not after decision "
            f"{decision['decided_at_utc']}: no closing window exists")

    fetcher = close_fetcher or _default_close_fetcher
    snap = fetcher(conn, decision, _fmt_ts(cutoff))
    if snap is None:
        return None
    close_ts = _parse_ts(snap["ts"])
    if close_ts <= decided_at:
        return None          # only the entry snapshot exists; nothing to say yet
    if close_ts > cutoff:
        raise AsOfViolation(
            f"close snapshot {_fmt_ts(close_ts)} is after cutoff {_fmt_ts(cutoff)}")
    if snap.get("prob_kind", "devig") != decision["prob_kind"]:
        raise AsOfViolation(
            f"prob_kind mismatch: decision {decision['prob_kind']!r} vs "
            f"close {snap.get('prob_kind')!r}; these are not comparable")

    row = {
        "decision_id": decision_id,
        "close_ts": _fmt_ts(close_ts),
        "kickoff_ts": _fmt_ts(kickoff),
        "cutoff_ts": _fmt_ts(cutoff),
        "close_point": None if snap.get("point") is None else float(snap["point"]),
        "close_devig_prob": float(snap["prob"]),
        "prob_kind": snap.get("prob_kind", "devig"),
        "n_books": None if snap.get("n_books") is None else int(snap["n_books"]),
    }
    stored = {
        **row,
        "captured_at_utc": _fmt_ts(_parse_ts(captured_at_utc) if captured_at_utc
                                   else datetime.now(timezone.utc)),
        "hash_version": HASH_VERSION,
        "content_hash": canonical_record_sha256(row, CLOSING_HASH_FIELDS),
    }
    already = conn.execute(
        "SELECT content_hash FROM closing_snapshots WHERE decision_id=? AND content_hash=?",
        (decision_id, stored["content_hash"])).fetchone()
    if already is not None:
        return stored        # identical capture already appended
    cols = list(stored.keys())
    conn.execute(
        f"INSERT INTO closing_snapshots ({', '.join(cols)}) "
        f"VALUES ({', '.join(['?'] * len(cols))})",
        [stored[c] for c in cols])
    conn.commit()
    return stored


def capture_closes_for_week(conn: sqlite3.Connection, season: int, week: int,
                            kickoffs: Dict[str, str], **kwargs) -> Dict[str, int]:
    """Scheduled-job entry point: attempt a close for every decision of a week."""
    frame = decisions(conn, season=season, week=week)
    captured = skipped = unresolved = 0
    for record in frame.to_dict("records"):
        kickoff = kickoffs.get(record["game_id"])
        if not kickoff:
            skipped += 1
            continue
        try:
            result = capture_close(conn, record["decision_id"], kickoff_ts=kickoff, **kwargs)
        except AsOfViolation:
            skipped += 1
            continue
        captured += result is not None
        unresolved += result is None
    return {"decisions": int(len(frame)), "captured": captured,
            "unresolved": unresolved, "skipped": skipped}


# --------------------------------------------------------------------------- #
# A.3 -- CLV computation
# --------------------------------------------------------------------------- #
def resolved_pairs(conn: sqlite3.Connection) -> pd.DataFrame:
    """Matched (decision, close) pairs with CLV, point movement, beat-close.

    Where a decision has several captures (a re-run of the job), the EARLIEST
    capture wins: the first honest observation, not the most flattering one.
    """
    frame = dbmod.query_df(conn, """
        SELECT d.*, c.close_ts, c.close_point, c.close_devig_prob,
               c.n_books, c.captured_at_utc, c.prob_kind AS close_prob_kind
        FROM decision_snapshots d
        JOIN closing_snapshots c ON c.decision_id = d.decision_id
        ORDER BY d.decision_id, c.captured_at_utc
    """)
    if frame.empty:
        return frame
    frame = frame.groupby("decision_id", as_index=False).first()
    frame["clv_prob"] = frame["close_devig_prob"] - frame["devig_prob"]
    frame["point_move"] = frame["close_point"] - frame["point"]
    frame["beat_close"] = frame["clv_prob"] > 0
    return frame


def unresolved_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("""
        SELECT COUNT(*) FROM decision_snapshots d
        WHERE NOT EXISTS (SELECT 1 FROM closing_snapshots c
                          WHERE c.decision_id = d.decision_id)
    """).fetchone()[0])


def paired_week_bootstrap_clv(pairs: pd.DataFrame, *, iterations: int = 20_000,
                              random_seed: int = 6102026) -> Dict:
    """Paired block bootstrap over season-week blocks.

    Mirrors ``fantasy.audit.paired_week_bootstrap``: a 200-selection week is
    correlated through shared weather, injuries, and a shared market state, so
    the resampling unit is the WEEK. Treating selections as independent shrinks
    the interval by roughly the square root of the block size and manufactures
    significance that is not there.
    """
    required = {"season", "week", "clv_prob"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"bootstrap frame missing columns: {missing}")
    valid = pairs[["season", "week", "clv_prob"]].dropna()
    if valid.empty:
        raise ValueError("no resolved pairs to bootstrap")
    blocks = valid.groupby(["season", "week"])["clv_prob"].agg(["size", "sum"]).reset_index()
    rng = np.random.default_rng(random_seed)
    sampled = rng.integers(0, len(blocks), size=(iterations, len(blocks)))
    counts = blocks["size"].to_numpy()[sampled].sum(axis=1)
    means = blocks["sum"].to_numpy()[sampled].sum(axis=1) / counts
    observed = float(blocks["sum"].sum() / blocks["size"].sum())
    return {
        "n": int(len(valid)),
        "weeks": int(len(blocks)),
        "mean_clv_prob": observed,
        "ci95": [float(v) for v in np.quantile(means, [0.025, 0.975])],
        "probability_nonpositive": float(np.mean(means <= 0)),
        "iterations": int(iterations),
        "random_seed": int(random_seed),
    }
