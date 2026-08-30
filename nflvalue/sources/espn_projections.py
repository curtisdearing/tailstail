"""Weekly ESPN fantasy-point projections as an EXTERNAL CHALLENGER snapshot.

Protocol position (docs/ACCURACY_PROTOCOL.md): "Public consensus is an external
challenger until source, retrieval timestamp, scoring rules, player coverage,
and redistribution rights are recorded."  Every snapshot this module writes
records all five, carries a content hash, and is immutable once written.

These numbers are for DISPLAY AND GRADING ONLY.  Feeding them into the model
(the market-shrinkage blend) is a separately registered lever under the 2026
season freeze (PROTOCOL_FREEZE_2026 §6, lever 3) and is deliberately not
implemented here.

Endpoint: the public league-agnostic fantasy read API
``lm-api-reads.fantasy.espn.com`` with the ``kona_player_info`` view against
``leaguedefaults/3`` (ESPN's full-PPR default scoring).  No authentication is
required.  If ESPN ever gates the endpoint, ``ESPN_S2`` / ``ESPN_SWID``
environment variables are attached as cookies when present — they are never
hardcoded and everything degrades cleanly without them.

Scoring basis (hard boundary): the league is full PPR.  ESPN's
``appliedTotal`` is ESPN's own full-PPR application, but we do not trust it
blindly: each player's raw projected stat lines are re-scored through the
repo's own ``ScoringRules`` (the exact function that scores the model and the
historical targets), and the applied-vs-rescored difference is recorded
explicitly in the snapshot so a scoring mismatch can never silently fabricate
discrepancies.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..fantasy.config import ScoringRules
from ..fantasy.scoring import score_components
from ._http import get_json

SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_KIND = "espn_projection_snapshot"

API_HOST = "https://lm-api-reads.fantasy.espn.com"
API_TEMPLATE = (
    API_HOST + "/apis/v3/games/ffl/seasons/{season}/segments/0/leaguedefaults/3"
    "?view=kona_player_info&scoringPeriodId={week}"
)
LEAGUE_SCORING_BASIS = "leaguedefaults/3 (ESPN default full-PPR scoring)"

# Verified live against the 2025 season payloads: the raw projected stat line
# for Jahmyr Gibbs week 14 re-scores under these ids to ESPN's appliedTotal
# within displayed rounding.
ESPN_STAT_IDS = {
    "0": "attempts",
    "1": "completions",
    "3": "passing_yards",
    "4": "passing_tds",
    "19": "passing_2pt_conversions",
    "20": "passing_interceptions",
    "23": "carries",
    "24": "rushing_yards",
    "25": "rushing_tds",
    "26": "rushing_2pt_conversions",
    "42": "receiving_yards",
    "43": "receiving_tds",
    "44": "receiving_2pt_conversions",
    "53": "receptions",
    "58": "targets",
    "72": "fumbles_lost",
}

POSITION_IDS = {1: "QB", 2: "RB", 3: "WR", 4: "TE"}
# Offensive lineup slot ids used to filter the pull server-side.
SLOT_IDS_OFFENSE = [0, 2, 4, 6]

PRO_TEAM_ABBREV = {
    0: "FA", 1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL",
    7: "DEN", 8: "DET", 9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV",
    14: "LA", 15: "MIA", 16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ",
    21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC", 25: "SF", 26: "SEA",
    27: "TB", 28: "WAS", 29: "CAR", 30: "JAX", 33: "BAL", 34: "HOU",
}

REDISTRIBUTION_RIGHTS = (
    "ESPN Fan Access terms: personal, non-commercial use; no bulk "
    "redistribution right is granted or claimed. Published output is limited "
    "to the weekly per-player projection comparison for personal-league "
    "model grading; raw snapshots are retained for audit, not republication."
)

# Fail-loud floor: a real weekly pull of QB/RB/WR/TE projections is hundreds
# of players. Anything under this is a broken or empty pull, not a thin week.
MIN_EXPECTED_PLAYERS = 150


class EspnProjectionsError(RuntimeError):
    """The ESPN pull violated the schema contract (fail loud, never empty)."""


def _auth_headers() -> tuple[dict[str, str], str]:
    """Cookie header from env vars if present. Never hardcoded, never required."""
    espn_s2 = os.environ.get("ESPN_S2", "").strip()
    swid = os.environ.get("ESPN_SWID", "").strip()
    if espn_s2 and swid:
        return {"Cookie": f"espn_s2={espn_s2}; SWID={swid}"}, "espn_s2+SWID (from env)"
    return {}, "public (no auth)"


def fetch_week_raw(
    season: int,
    week: int,
    *,
    limit: int = 800,
    get: Callable[..., Any] = get_json,
) -> tuple[dict[str, Any], str, str]:
    """Fetch the raw kona_player_info payload. Returns (payload, url, auth_mode)."""
    url = API_TEMPLATE.format(season=int(season), week=int(week))
    fantasy_filter = {
        "players": {
            "filterSlotIds": {"value": SLOT_IDS_OFFENSE},
            "limit": int(limit),
            "sortPercOwned": {"sortAsc": False, "sortPriority": 1},
        }
    }
    headers, auth_mode = _auth_headers()
    headers["x-fantasy-filter"] = json.dumps(fantasy_filter)
    payload = get(url, headers=headers, source="espn_projections", timeout=30)
    if not isinstance(payload, dict):
        raise EspnProjectionsError(f"unexpected payload type {type(payload).__name__} from {url}")
    return payload, url, auth_mode


def _projection_entry(player: dict[str, Any], season: int, week: int) -> dict[str, Any] | None:
    for entry in player.get("stats", []) or []:
        if (
            entry.get("statSourceId") == 1          # 1 = projection, 0 = actual
            and entry.get("statSplitTypeId") == 1   # 1 = weekly split
            and entry.get("scoringPeriodId") == week
            and entry.get("seasonId") == season
        ):
            return entry
    return None


def raw_stats_to_components(raw_stats: dict[str, Any]) -> dict[str, float]:
    """Map ESPN stat-id keys onto the repo's scoring component names."""
    components: dict[str, float] = {}
    for stat_id, value in (raw_stats or {}).items():
        name = ESPN_STAT_IDS.get(str(stat_id))
        if name is not None:
            components[name] = float(value)
    return components


def rescore_full_ppr(raw_stats: dict[str, Any], rules: ScoringRules | None = None) -> float:
    """Score an ESPN raw projected stat line with the repo's own scorer."""
    components = raw_stats_to_components(raw_stats)
    return float(score_components(components, rules or ScoringRules.preset("ppr")))


def parse_players(
    payload: dict[str, Any],
    *,
    season: int,
    week: int,
    rules: ScoringRules | None = None,
) -> list[dict[str, Any]]:
    """Extract one comparison-ready record per QB/RB/WR/TE projection.

    Fail-loud contract: an empty or implausibly thin result raises rather than
    returning quietly, so a broken pull can never masquerade as a real week.
    """
    rules = rules or ScoringRules.preset("ppr")
    entries = payload.get("players")
    if not isinstance(entries, list) or not entries:
        raise EspnProjectionsError(
            f"ESPN payload for {season} week {week} contains no players key/rows"
        )
    records: list[dict[str, Any]] = []
    missing_projection = 0
    for wrapper in entries:
        player = (wrapper or {}).get("player") or {}
        position = POSITION_IDS.get(player.get("defaultPositionId"))
        if position is None:
            continue
        espn_id = player.get("id")
        full_name = player.get("fullName")
        if espn_id is None or not full_name:
            raise EspnProjectionsError(
                f"ESPN player entry missing id/fullName: {json.dumps(wrapper)[:200]}"
            )
        entry = _projection_entry(player, season, week)
        if entry is None:
            missing_projection += 1
            continue
        applied_total = entry.get("appliedTotal")
        raw_stats = entry.get("stats") or {}
        components = raw_stats_to_components(raw_stats)
        if components:
            points = float(score_components(components, rules))
            basis = "full_ppr_rescored"
        elif applied_total is not None:
            points = float(applied_total)
            basis = "espn_applied_total"
        else:
            missing_projection += 1
            continue
        records.append({
            "espn_id": int(espn_id),
            "player_name": str(full_name),
            "position": position,
            "team": PRO_TEAM_ABBREV.get(player.get("proTeamId"), "UNK"),
            "espn_applied_total": None if applied_total is None else float(applied_total),
            "espn_ppr_points": points,
            "points_basis": basis,
            "raw_stat_components": components,
        })
    if len(records) < MIN_EXPECTED_PLAYERS:
        raise EspnProjectionsError(
            f"ESPN pull for {season} week {week} parsed only {len(records)} projections "
            f"({missing_projection} entries lacked a week-{week} projection); "
            f"a real weekly slate is >= {MIN_EXPECTED_PLAYERS}. Refusing a thin/empty pull."
        )
    return records


def players_sha256(players: list[dict[str, Any]]) -> str:
    canonical = json.dumps(players, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_snapshot(
    players: list[dict[str, Any]],
    *,
    season: int,
    week: int,
    endpoint: str,
    auth_mode: str,
    retrieved_at: str | None = None,
    rules: ScoringRules | None = None,
) -> dict[str, Any]:
    """Assemble the immutable provenance-complete snapshot document.

    Records all five protocol-required fields: source, retrieval timestamp,
    scoring rules, player coverage, and redistribution rights.
    """
    rules = rules or ScoringRules.preset("ppr")
    retrieved = retrieved_at or datetime.now(timezone.utc).isoformat()
    by_position: dict[str, int] = {}
    rescored = 0
    deltas: list[float] = []
    for record in players:
        by_position[record["position"]] = by_position.get(record["position"], 0) + 1
        if record["points_basis"] == "full_ppr_rescored":
            rescored += 1
            if record["espn_applied_total"] is not None:
                deltas.append(abs(record["espn_ppr_points"] - record["espn_applied_total"]))
    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "kind": SNAPSHOT_KIND,
        "season": int(season),
        "week": int(week),
        "retrieved_at": retrieved,
        "source": {
            "name": "ESPN Fantasy API (lm-api-reads.fantasy.espn.com)",
            "endpoint": endpoint,
            "view": "kona_player_info",
            "league_scoring_basis": LEAGUE_SCORING_BASIS,
            "auth": auth_mode,
        },
        "scoring": {
            "comparison_basis": "full PPR, re-scored from ESPN raw projected stat lines "
                                "with nflvalue.fantasy.scoring.score_components (the model's scorer)",
            "rules": rules.to_dict(),
            "rescored_players": rescored,
            "applied_total_only_players": len(players) - rescored,
            "rescored_vs_applied_mean_abs_delta": (
                float(sum(deltas) / len(deltas)) if deltas else None
            ),
        },
        "coverage": {"players": len(players), "by_position": by_position},
        "redistribution_rights": REDISTRIBUTION_RIGHTS,
        "players": players,
        "players_sha256": players_sha256(players),
    }
    return snapshot


def snapshot_path(directory: str | Path, snapshot: dict[str, Any]) -> Path:
    stamp = (
        snapshot["retrieved_at"].replace("-", "").replace(":", "").split(".")[0].replace("+0000", "")
    )
    name = f"espn_projections_{snapshot['season']}_wk{int(snapshot['week']):02d}_{stamp}.json"
    return Path(directory) / name


def write_snapshot(snapshot: dict[str, Any], directory: str | Path) -> Path:
    """Persist a snapshot. Immutable: an existing file is never overwritten."""
    path = snapshot_path(directory, snapshot)
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable ESPN snapshot {path}; "
            "a new retrieval must produce a new timestamped file"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    return path


def load_snapshot(path: str | Path) -> dict[str, Any]:
    """Load and integrity-check a snapshot; a hash mismatch is tampering."""
    snapshot = json.loads(Path(path).read_text())
    if snapshot.get("kind") != SNAPSHOT_KIND:
        raise EspnProjectionsError(f"{path} is not an ESPN projection snapshot")
    expected = snapshot.get("players_sha256")
    actual = players_sha256(snapshot.get("players", []))
    if expected != actual:
        raise EspnProjectionsError(
            f"ESPN snapshot {path} failed its content hash "
            f"(stored {expected}, recomputed {actual}); snapshots are immutable"
        )
    return snapshot


def fetch_week_snapshot(
    season: int,
    week: int,
    *,
    rules: ScoringRules | None = None,
    get: Callable[..., Any] = get_json,
) -> dict[str, Any]:
    """Fetch + parse + assemble in one call (the weekly pipeline entrypoint)."""
    payload, url, auth_mode = fetch_week_raw(season, week, get=get)
    players = parse_players(payload, season=season, week=week, rules=rules)
    return build_snapshot(
        players, season=season, week=week, endpoint=url, auth_mode=auth_mode, rules=rules
    )
