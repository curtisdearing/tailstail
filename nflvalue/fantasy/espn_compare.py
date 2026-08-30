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
    for week_key, entry in ledger.get("weeks", {}).items():
        stored = entry.get("projections_sha256")
        actual = _rows_sha256(entry.get("rows", []))
        if stored != actual:
            raise ValueError(
                f"ledger week {week_key} failed its projections hash "
                f"(stored {stored}, recomputed {actual}); stored projections are immutable"
            )
    return ledger


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
    """Ordered per-week aggregates — the 'watch it improve' view."""
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
    status: str = "ok",
    error: str | None = None,
) -> dict[str, Any]:
    """The ``espn_comparison`` object embedded in data/fantasy_latest.json."""
    entry = ledger.get("weeks", {}).get(str(int(current_week)), {})
    series = season_series(ledger)
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
