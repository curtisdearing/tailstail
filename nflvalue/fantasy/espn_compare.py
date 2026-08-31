"""ESPN-vs-model weekly comparison: display and grading ONLY.

The 2026 season freeze registers the market-shrinkage blend as its own lever
(PROTOCOL_FREEZE_2026 §6, lever 3).  Nothing in this module feeds ESPN numbers
into the model; it snapshots both sides before kickoff, shows the
discrepancies, and — after games — grades who was closer.

Prospective-grading invariants (docs/ACCURACY_PROTOCOL.md):

* A comparison row exists only if BOTH the ESPN snapshot retrieval timestamp
  and the model generation timestamp predate that game's kickoff.
* A row whose game has kicked off is locked: a later refresh cannot touch it,
  because the refresh's own timestamps fail the pre-kickoff check.
* Grading never rewrites stored projections.  Graded results live in a
  separate block and the stored rows are hash-verified before and after.
* Never backfill: a week with no pre-kickoff ledger entry is never graded.

Identity: ESPN ids map to model (gsis) ids through the nflverse weekly-roster
crosswalk.  Unmatched players are REPORTED, never silently dropped, and the
match coverage is part of every comparison payload.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

LEDGER_SCHEMA_VERSION = 1
LEDGER_KIND = "espn_comparison_ledger"
TIE_TOLERANCE = 1e-9
EASTERN = ZoneInfo("America/New_York")

DISCLAIMER = (
    "External challenger display: ESPN numbers are shown and graded, never "
    "fed into the model (market blend = separately registered 2026 lever). "
    "This is a projection-accuracy scoreboard, not a betting edge."
)


# ---------------------------------------------------------------------------
# Identity mapping
# ---------------------------------------------------------------------------

def build_identity_map(rosters: pd.DataFrame, season: int) -> pd.DataFrame:
    """gsis_id <-> espn_id crosswalk from the nflverse weekly rosters.

    Fail-loud: a rosters frame without an ``espn_id`` column means the vendor
    schema changed; that must break, not degrade into an empty comparison.
    """
    if "espn_id" not in rosters.columns:
        raise ValueError(
            "weekly rosters have no espn_id column; the nflverse schema changed "
            "and the ESPN identity map cannot be built (fail closed)"
        )
    if "gsis_id" not in rosters.columns:
        raise ValueError("weekly rosters have no gsis_id column (fail closed)")
    frame = rosters[pd.to_numeric(rosters["season"], errors="coerce").eq(season)]
    frame = frame[["gsis_id", "espn_id"]].copy()
    frame["gsis_id"] = frame["gsis_id"].fillna("").astype(str)
    frame["espn_id"] = pd.to_numeric(frame["espn_id"], errors="coerce")
    frame = frame[frame["gsis_id"].ne("") & frame["espn_id"].notna()]
    frame["espn_id"] = frame["espn_id"].astype(int)
    # keep="last": latest roster row wins if a player changed teams mid-season.
    return frame.drop_duplicates("espn_id", keep="last").reset_index(drop=True)


def match_players(
    espn_players: list[dict[str, Any]],
    identity: pd.DataFrame,
    model_ids: set[str],
) -> tuple[dict[int, str], dict[str, Any]]:
    """Map ESPN ids to model gsis ids; report every failure mode explicitly.

    Returns ``(espn_id -> gsis_id for players the model also projected,
    identity_report)``.  Nothing is silently dropped: ESPN players without a
    roster crosswalk row and crosswalked players the model did not project are
    both counted and named in the report.
    """
    crosswalk = dict(zip(identity["espn_id"], identity["gsis_id"]))
    matched: dict[int, str] = {}
    unmatched_no_crosswalk: list[str] = []
    unmatched_not_projected: list[str] = []
    for record in espn_players:
        gsis = crosswalk.get(int(record["espn_id"]))
        if gsis is None:
            unmatched_no_crosswalk.append(record["player_name"])
        elif gsis not in model_ids:
            unmatched_not_projected.append(record["player_name"])
        else:
            matched[int(record["espn_id"])] = gsis
    total = len(espn_players)
    report = {
        "espn_players": total,
        "matched": len(matched),
        "coverage_pct": round(100.0 * len(matched) / total, 1) if total else 0.0,
        "unmatched_no_crosswalk_count": len(unmatched_no_crosswalk),
        "unmatched_no_crosswalk_names": sorted(unmatched_no_crosswalk)[:25],
        "unmatched_model_not_projected_count": len(unmatched_not_projected),
        "unmatched_model_not_projected_names": sorted(unmatched_not_projected)[:25],
        "note": "unmatched players are reported, never silently dropped",
    }
    return matched, report


# ---------------------------------------------------------------------------
# Kickoffs
# ---------------------------------------------------------------------------

def game_kickoffs_utc(schedules: pd.DataFrame, season: int, week: int) -> dict[str, str]:
    """{game_id: kickoff ISO-8601 UTC} from nflverse schedules (ET local times)."""
    games = schedules[
        pd.to_numeric(schedules["season"], errors="coerce").eq(season)
        & pd.to_numeric(schedules["week"], errors="coerce").eq(week)
    ]
    kickoffs: dict[str, str] = {}
    for row in games.to_dict("records"):
        gameday = str(row.get("gameday", "") or "")
        gametime = str(row.get("gametime", "") or "") or "13:00"
        try:
            local = datetime.strptime(f"{gameday} {gametime}", "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        kickoffs[str(row["game_id"])] = (
            local.replace(tzinfo=EASTERN).astimezone(timezone.utc).isoformat()
        )
    return kickoffs


def is_prospective(snapshot_time: str, kickoff_utc: str) -> bool:
    """True iff the snapshot strictly predates kickoff (both ISO-8601)."""
    snap = datetime.fromisoformat(snapshot_time)
    kick = datetime.fromisoformat(kickoff_utc)
    if snap.tzinfo is None:
        snap = snap.replace(tzinfo=timezone.utc)
    if kick.tzinfo is None:
        kick = kick.replace(tzinfo=timezone.utc)
    return snap < kick


# ---------------------------------------------------------------------------
# Ledger (immutable prospective record + separate grading block)
# ---------------------------------------------------------------------------

def _rows_sha256(rows: list[dict[str, Any]]) -> str:
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def new_ledger(season: int) -> dict[str, Any]:
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "kind": LEDGER_KIND,
        "season": int(season),
        "scoring_basis": "full_ppr",
        "weeks": {},
    }


def load_ledger(path: str | Path, season: int) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return new_ledger(season)
    ledger = json.loads(path.read_text())
    if ledger.get("kind") != LEDGER_KIND:
        raise ValueError(f"{path} is not an ESPN comparison ledger")
    if int(ledger.get("season", -1)) != int(season):
        # A new season starts a fresh ledger; the old one is archived by the
        # caller if wanted. Never mix seasons in one prospective record.
        return new_ledger(season)
    verify_ledger_integrity(ledger)
    return ledger


def verify_ledger_integrity(ledger: dict[str, Any]) -> None:
    """Every recorded week's rows still hash to what was stored, or raise.

    Split out of `load_ledger` so the private-state boundary can run the same
    check on a ledger arriving from the private store. A second implementation
    of "are these the rows we recorded" would be a second answer to it.
    """
    for week_key, entry in (ledger.get("weeks") or {}).items():
        stored = entry.get("projections_sha256")
        actual = _rows_sha256(entry.get("rows", []))
        if stored != actual:
            raise ValueError(
                f"ledger week {week_key} failed its projections hash "
                f"(stored {stored}, recomputed {actual}); stored projections are immutable"
            )


def save_ledger(ledger: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")


def record_week(
    ledger: dict[str, Any],
    *,
    week: int,
    espn_players: list[dict[str, Any]],
    espn_retrieved_at: str,
    espn_snapshot_sha256: str,
    matched: dict[int, str],
    model_points: dict[str, float],
    model_meta: dict[str, dict[str, str]],
    model_generated_at: str,
    player_games: dict[str, str],
    kickoffs_utc: dict[str, str],
) -> dict[str, Any]:
    """Insert or prospectively refresh one week's comparison rows.

    Row update rule (the immutability core): a row may be written only when
    BOTH sides' timestamps predate its game's kickoff.  Rows whose games have
    started are therefore locked automatically — a post-kickoff refresh fails
    the check and the earlier pre-kickoff row survives.  A graded week is
    fully immutable.
    """
    week_key = str(int(week))
    entry = ledger["weeks"].get(week_key)
    if entry is None:
        entry = {"rows": [], "grading": None, "sources": []}
        ledger["weeks"][week_key] = entry
    if entry.get("grading") is not None:
        raise ValueError(f"week {week_key} is already graded; its rows are immutable")

    existing = {row["player_id"]: row for row in entry["rows"]}
    skipped_post_kickoff = 0
    skipped_no_kickoff = 0
    for record in espn_players:
        gsis = matched.get(int(record["espn_id"]))
        if gsis is None or gsis not in model_points:
            continue
        game_id = player_games.get(gsis)
        kickoff = kickoffs_utc.get(game_id or "")
        if kickoff is None:
            skipped_no_kickoff += 1
            continue
        if not (
            is_prospective(espn_retrieved_at, kickoff)
            and is_prospective(model_generated_at, kickoff)
        ):
            skipped_post_kickoff += 1
            continue  # locked: the earlier pre-kickoff row (if any) stands
        meta = model_meta.get(gsis, {})
        existing[gsis] = {
            "player_id": gsis,
            "espn_id": int(record["espn_id"]),
            "player_name": record["player_name"],
            "position": record["position"],
            "team": meta.get("team", record["team"]),
            "game_id": game_id,
            "kickoff_utc": kickoff,
            "espn_pts": round(float(record["espn_ppr_points"]), 3),
            "espn_points_basis": record["points_basis"],
            "model_pts": round(float(model_points[gsis]), 3),
            "espn_retrieved_at": espn_retrieved_at,
            "model_generated_at": model_generated_at,
        }
    rows = sorted(existing.values(), key=lambda row: row["player_id"])
    entry["rows"] = rows
    entry["projections_sha256"] = _rows_sha256(rows)
    entry["sources"].append({
        "espn_retrieved_at": espn_retrieved_at,
        "espn_snapshot_sha256": espn_snapshot_sha256,
        "model_generated_at": model_generated_at,
        "rows_written_or_refreshed": len(rows),
        "skipped_post_kickoff": skipped_post_kickoff,
        "skipped_no_kickoff": skipped_no_kickoff,
    })
    return entry


def who_was_closer(espn_pts: float, model_pts: float, actual: float) -> str:
    espn_err = abs(float(espn_pts) - float(actual))
    model_err = abs(float(model_pts) - float(actual))
    if abs(espn_err - model_err) <= TIE_TOLERANCE:
        return "tie"
    return "model" if model_err < espn_err else "espn"


def _aggregate(graded_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mae(rows: list[dict[str, Any]], key: str) -> float | None:
        return round(sum(row[key] for row in rows) / len(rows), 3) if rows else None

    played = [row for row in graded_rows if row["played"]]
    by_position: dict[str, Any] = {}
    for position in sorted({row["position"] for row in graded_rows}):
        rows = [row for row in played if row["position"] == position]
        by_position[position] = {
            "n": len(rows),
            "mae_espn": mae(rows, "abs_err_espn"),
            "mae_model": mae(rows, "abs_err_model"),
        }
    closer = [row["who_was_closer"] for row in played]
    return {
        "n": len(graded_rows),
        "n_played": len(played),
        "n_dnp": len(graded_rows) - len(played),
        "mae_espn": mae(played, "abs_err_espn"),
        "mae_model": mae(played, "abs_err_model"),
        "mae_espn_incl_dnp": mae(graded_rows, "abs_err_espn"),
        "mae_model_incl_dnp": mae(graded_rows, "abs_err_model"),
        "model_closer": closer.count("model"),
        "espn_closer": closer.count("espn"),
        "ties": closer.count("tie"),
        "by_position": by_position,
    }


def grade_week(
    ledger: dict[str, Any],
    *,
    week: int,
    actual_points: dict[str, float],
    graded_at: str | None = None,
) -> dict[str, Any]:
    """Grade one recorded week against actual full-PPR points.

    ``actual_points`` maps gsis player_id -> actual points scored with the
    SAME ScoringRules as both projections; a player-week absent from the map
    is a DNP (actual 0.0, flagged, and reported separately in aggregates).

    Stored projections are hash-verified before and after: grading writes a
    separate block and can never alter what was predicted.  Re-grading a
    graded week is a no-op returning the existing block.
    """
    week_key = str(int(week))
    entry = ledger["weeks"].get(week_key)
    if entry is None:
        raise ValueError(
            f"week {week_key} has no prospective ledger entry; grading cannot backfill"
        )
    if entry.get("grading") is not None:
        return entry["grading"]
    hash_before = _rows_sha256(entry["rows"])
    if hash_before != entry["projections_sha256"]:
        raise ValueError(f"week {week_key} rows do not match their stored hash; refusing to grade")
    graded_rows = []
    for row in entry["rows"]:
        played = row["player_id"] in actual_points
        actual = float(actual_points.get(row["player_id"], 0.0))
        graded_rows.append({
            "player_id": row["player_id"],
            "player_name": row["player_name"],
            "position": row["position"],
            "espn_pts": row["espn_pts"],
            "model_pts": row["model_pts"],
            "actual_pts": round(actual, 3),
            "played": played,
            "abs_err_espn": round(abs(row["espn_pts"] - actual), 3),
            "abs_err_model": round(abs(row["model_pts"] - actual), 3),
            "who_was_closer": who_was_closer(row["espn_pts"], row["model_pts"], actual),
        })
    grading = {
        "graded_at": graded_at or datetime.now(timezone.utc).isoformat(),
        "rows": graded_rows,
        "aggregate": _aggregate(graded_rows),
    }
    if _rows_sha256(entry["rows"]) != hash_before:
        raise AssertionError("grading altered stored projections; this must be impossible")
    entry["grading"] = grading
    return grading


# ---------------------------------------------------------------------------
# Public grading history — aggregates only, and durable on its own
# ---------------------------------------------------------------------------
# The season series used to be re-derived from the ledger on every run. The
# ledger holds one row per player per week with both sides' projections and the
# actual points, so it is exactly the file that may not be published -- which
# left the published history depending on a private file staying reachable from
# a public job. This contract breaks that dependency: one immutable aggregate
# per graded week, carrying nothing that could be a player, loaded and validated
# independently of the raw ledger and carried between runs in the checksummed
# public state archive.
#
# The guard is a POSITIVE allow-list. A blocklist would have to anticipate the
# next field somebody adds to `_aggregate`; an allow-list drops it until it is
# deliberately named here, which is the right default for a file that is
# published.
HISTORY_SCHEMA_VERSION = 1
HISTORY_KIND = "espn-comparison-history/1"

#: Aggregate scalars a published week may carry.
HISTORY_AGGREGATE_FIELDS = (
    "n", "n_played", "n_dnp",
    "mae_espn", "mae_model", "mae_espn_incl_dnp", "mae_model_incl_dnp",
    "model_closer", "espn_closer", "ties",
)
#: Per-position aggregate scalars.
HISTORY_POSITION_FIELDS = ("n", "mae_espn", "mae_model")
#: Everything else a week entry may carry: its index, when it was graded, the
#: by-position block, and the ledger's own projections digest as non-reversible
#: audit linkage back to the private rows.
HISTORY_WEEK_FIELDS = HISTORY_AGGREGATE_FIELDS + (
    "week", "graded_at", "by_position", "projections_sha256",
)
HISTORY_TOP_FIELDS = ("schema_version", "kind", "season", "weeks")

#: A position label, not a player. Restricting the key shape is what stops a
#: name being smuggled in as a grouping key.
_POSITION_KEY_RE = re.compile(r"^[A-Z][A-Z/]{0,4}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class HistoryRejected(ValueError):
    """The public grading history broke its contract. Never repaired, refused."""


class HistoryConflict(HistoryRejected):
    """A graded week was rewritten with different numbers. Immutable means no."""


def public_aggregate(aggregate: dict[str, Any]) -> dict[str, Any]:
    """The allow-listed projection of `_aggregate` output -- and only that."""
    out: dict[str, Any] = {key: aggregate.get(key) for key in HISTORY_AGGREGATE_FIELDS}
    by_position = aggregate.get("by_position") or {}
    out["by_position"] = {
        str(position): {key: values.get(key) for key in HISTORY_POSITION_FIELDS}
        for position, values in sorted(by_position.items())
    }
    return out


def new_history(season: int) -> dict[str, Any]:
    return {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "kind": HISTORY_KIND,
        "season": int(season),
        "weeks": {},
    }


def _reject(message: str) -> None:
    raise HistoryRejected(message)


def validate_history(history: Any) -> None:
    """Refuse anything outside the allow-list, by key name and by value shape.

    Error messages name the offending KEY and never its value: this contract
    exists because the values on the other side of it are private, and a
    validator that quotes what it rejected is a validator that leaks.
    """
    if not isinstance(history, dict):
        _reject(f"grading history is not an object ({type(history).__name__})")
    unknown = sorted(set(history) - set(HISTORY_TOP_FIELDS))
    if unknown:
        _reject(f"grading history carries unexpected top-level keys: {unknown}")
    if history.get("kind") != HISTORY_KIND:
        _reject(f"grading history claims kind {history.get('kind')!r}")
    if history.get("schema_version") != HISTORY_SCHEMA_VERSION:
        _reject(f"grading history claims schema {history.get('schema_version')!r}")
    if not isinstance(history.get("season"), int) or isinstance(history.get("season"), bool):
        _reject("grading history carries no integer season")
    weeks = history.get("weeks")
    if not isinstance(weeks, dict):
        _reject("grading history carries no weeks object")

    for week_key, entry in weeks.items():
        where = f"grading history week {week_key!r}"
        if not str(week_key).isdigit():
            _reject(f"{where} is not a week number")
        if not isinstance(entry, dict):
            _reject(f"{where} is not an object")
        extra = sorted(set(entry) - set(HISTORY_WEEK_FIELDS))
        if extra:
            _reject(f"{where} carries unexpected keys: {extra}")
        missing = sorted(set(HISTORY_WEEK_FIELDS) - set(entry))
        if missing:
            _reject(f"{where} is missing {missing}")
        if entry.get("week") != int(week_key):
            _reject(f"{where} disagrees with its own week number")
        if not isinstance(entry.get("graded_at"), str) or not entry["graded_at"].strip():
            _reject(f"{where} carries no graded_at")
        digest = entry.get("projections_sha256")
        if not isinstance(digest, str) or not _SHA256_RE.match(digest):
            _reject(f"{where} carries no sha256 audit digest")
        for key in ("n", "n_played", "n_dnp", "model_closer", "espn_closer", "ties"):
            value = entry.get(key)
            if not isinstance(value, int) or isinstance(value, bool):
                _reject(f"{where} field {key!r} is not a count")
        for key in ("mae_espn", "mae_model", "mae_espn_incl_dnp", "mae_model_incl_dnp"):
            value = entry.get(key)
            if value is not None and (isinstance(value, bool)
                                      or not isinstance(value, (int, float))):
                _reject(f"{where} field {key!r} is not a number")
        by_position = entry.get("by_position")
        if not isinstance(by_position, dict):
            _reject(f"{where} carries no by_position object")
        for position, values in by_position.items():
            if not _POSITION_KEY_RE.match(str(position)):
                _reject(f"{where} groups by something that is not a position label")
            if not isinstance(values, dict):
                _reject(f"{where} position group is not an object")
            odd = sorted(set(values) - set(HISTORY_POSITION_FIELDS))
            if odd:
                _reject(f"{where} position group carries unexpected keys: {odd}")
            # The key names were checked above; without this the VALUES are
            # unconstrained, and an arbitrary string under a legal key passes
            # both this validator and the public boundary guard.
            count = values.get("n")
            if not isinstance(count, int) or isinstance(count, bool):
                _reject(f"{where} position group field 'n' is not a count")
            for key in ("mae_espn", "mae_model"):
                value = values.get(key)
                if value is not None and (isinstance(value, bool)
                                          or not isinstance(value, (int, float))):
                    _reject(f"{where} position group field {key!r} is not a number")


def load_history(path: str | Path, season: int) -> dict[str, Any]:
    """The published grading history, validated, or a fresh one.

    Absent means a first run: an empty history. Malformed does NOT mean a first
    run -- it fails closed, because quietly starting over would erase a season
    of published grading and read on the site as a model that had never been
    checked.

    This function never touches the raw ledger. That independence is the point:
    a run whose private raw state is unreachable still loads, publishes and
    re-saves every week already graded.
    """
    path = Path(path)
    if not path.exists():
        return new_history(season)
    try:
        history = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise HistoryRejected(f"{path} is not readable JSON: {exc}") from exc
    validate_history(history)
    stored_season = int(history.get("season", -1))
    if stored_season != int(season):
        # A new season starts a fresh series -- but this file is gitignored and
        # its only durable copy is the public state release, so returning an
        # empty history and letting the caller write it back is how a whole
        # prior season disappears at the first run of the next one. Archive it
        # beside itself first; the archive travels in the same release.
        archive = path.with_name(f"{path.stem}.{stored_season}{path.suffix}")
        if not archive.exists():
            archive.write_text(path.read_text())
        return new_history(season)
    return history


def save_history(history: dict[str, Any], path: str | Path) -> None:
    """Validate, refuse to shrink a published season, then write atomically.

    The floor is the important half. This file is gitignored and travels in the
    public state release, so if a restore fails, the run loads an empty history,
    saves it, and the release pointer moves to the empty archive -- one
    transient network error, and a season of published grading is gone. A save
    that would publish fewer weeks than the file already holds is refused.
    """
    validate_history(history)
    path = Path(path)
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = None
        if isinstance(existing, dict) and int(existing.get("season", -1)) == int(
                history.get("season", -2)):
            lost = sorted(set(existing.get("weeks") or {}) - set(history.get("weeks") or {}),
                          key=int)
            if lost:
                raise HistoryConflict(
                    f"refusing to publish a {history['season']} grading history that drops "
                    f"week(s) {lost} already on disk; a published week is never withdrawn")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def record_graded_week(history: dict[str, Any], *, week: int, aggregate: dict[str, Any],
                       graded_at: str, projections_sha256: str) -> dict[str, Any]:
    """Append one graded week's aggregate. A published week is immutable.

    Re-recording the identical week is a no-op, so a rerun of the same data is
    safe. Re-recording it with *different* numbers raises: a published grading
    that can be rewritten is not a record of what the model did, it is a record
    of what the last run said it did. The numbers and the audit digest are what
    is frozen; `graded_at` is metadata and the first one is kept.
    """
    week_key = str(int(week))
    entry = {
        **public_aggregate(aggregate),
        "week": int(week),
        "graded_at": str(graded_at),
        "projections_sha256": str(projections_sha256),
    }
    existing = history["weeks"].get(week_key)
    if existing is not None:
        frozen = {key: value for key, value in existing.items() if key != "graded_at"}
        proposed = {key: value for key, value in entry.items() if key != "graded_at"}
        if frozen != proposed:
            # The message names the week and nothing else on purpose: the values
            # either side of this comparison are the private ones.
            raise HistoryConflict(
                f"grading history week {week_key} is already published with different "
                "numbers; a graded week is immutable and is not rewritten")
        return existing
    # Validated as a candidate, then assigned. Inserting first and validating
    # after leaves the rejected week in the caller's object when the raise is
    # caught, which is a poisoned history that only fails later.
    candidate = {**history, "weeks": {**history["weeks"], week_key: entry}}
    validate_history(candidate)
    history["weeks"][week_key] = entry
    return entry


def sync_history_from_ledger(history: dict[str, Any], ledger: dict[str, Any]) -> list[int]:
    """Copy every graded ledger week that is not already public. Returns the weeks added.

    One direction only. The ledger is the private record and the history is the
    published one; nothing here reads the history back into the ledger, and an
    ungraded week simply has nothing to copy yet.
    """
    added: list[int] = []
    for week_key in sorted(ledger.get("weeks", {}), key=int):
        entry = ledger["weeks"][week_key]
        grading = entry.get("grading")
        if grading is None:
            continue
        if not isinstance(grading.get("aggregate"), dict) or not grading.get("graded_at"):
            raise HistoryRejected(
                f"ledger week {week_key} is marked graded but carries no usable aggregate")
        already_published = week_key in history["weeks"]
        # A week already published is still passed through `record_graded_week`
        # rather than skipped, so a ledger that has been re-graded into
        # different numbers raises instead of being quietly ignored.
        record_graded_week(
            history, week=int(week_key), aggregate=grading["aggregate"],
            graded_at=grading["graded_at"],
            projections_sha256=entry.get("projections_sha256", ""))
        if not already_published:
            added.append(int(week_key))
    return added


def history_series(history: dict[str, Any]) -> list[dict[str, Any]]:
    """Ordered per-week aggregates -- the published 'watch it improve' view."""
    return [dict(history["weeks"][key]) for key in sorted(history.get("weeks", {}), key=int)]


# ---------------------------------------------------------------------------
# Dashboard payload
# ---------------------------------------------------------------------------

def comparison_rows(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Current-week display rows: delta and abs-delta rank, largest first."""
    rows = []
    for row in entry.get("rows", []):
        delta = round(row["model_pts"] - row["espn_pts"], 3)
        rows.append({**row, "delta": delta, "abs_delta": abs(delta)})
    rows.sort(key=lambda row: (-row["abs_delta"], row["player_id"]))
    for rank, row in enumerate(rows, start=1):
        row["abs_delta_rank"] = rank
        row.pop("abs_delta")
    return rows


def season_series(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    """Ordered per-week aggregates read out of the PRIVATE ledger.

    Kept for diagnostics and for seeding the public history; it is no longer
    what gets published. `history_series` is.
    """
    series = []
    for week_key in sorted(ledger.get("weeks", {}), key=int):
        entry = ledger["weeks"][week_key]
        grading = entry.get("grading")
        if grading is not None:
            series.append({"week": int(week_key), **grading["aggregate"]})
    return series


def build_payload(
    ledger: dict[str, Any],
    *,
    current_week: int,
    espn_provenance: dict[str, Any] | None,
    identity_report: dict[str, Any] | None,
    history: dict[str, Any],
    status: str = "ok",
    error: str | None = None,
) -> dict[str, Any]:
    """The ``espn_comparison`` object embedded in data/fantasy_latest.json.

    *history* is required and is where the published season series comes from.
    Defaulting it to ``season_series(ledger)`` would restore exactly the
    coupling this contract exists to break: the published history would once
    again be readable only from the private row-level file, and a run that
    could not reach that file would publish an empty series as though the model
    had never been graded.
    """
    entry = ledger.get("weeks", {}).get(str(int(current_week)), {})
    series = history_series(history)
    latest_graded = series[-1] if series else None
    return {
        "status": status,
        "error": error,
        "season": ledger.get("season"),
        "current_week": int(current_week),
        "disclaimer": DISCLAIMER,
        "espn_provenance": espn_provenance,
        "identity": identity_report,
        "current_week_rows": comparison_rows(entry) if entry else [],
        "latest_graded_week": latest_graded,
        "season_series": series,
        "prospective_rule": (
            "rows exist only when both projections were snapshotted before "
            "kickoff; post-kickoff refreshes cannot touch a locked row; graded "
            "weeks are immutable; no backfill"
        ),
    }
