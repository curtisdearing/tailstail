"""Sleeper projections as an independent CROSS-CHECK -- never a target.

PHASE1_HANDSOFF_DESIGN.md H5: fantasy consensus is already baked into soft
prop lines, so regressing our model toward it would quietly delete any edge.
This module therefore exposes exactly one judgment: a **divergence flag** --
"our number is far from Sleeper's, look closer / lower confidence" -- and no
code path that returns an altered projection.

Endpoints (free, no auth, keep well under ~1000 calls/min):
  * https://api.sleeper.com/projections/nfl/{season}/{week}?season_type=regular
        -> list of per-player projection objects ({player_id, player{...},
           stats{rec_yd, rec, rush_yd, pass_yd, pass_att, rush_att, ...}})
  * https://api.sleeper.app/v1/players/nfl
        -> the full player dump (fetched rarely; cached to parquet) -- the
           only place Sleeper exposes ``gsis_id``, which is how we join to
           nflverse player_ids without name-matching guesswork.

Both payloads are schema-validated at parse time (H3: undocumented endpoints
break silently; fail loud instead). Recorded fixtures for offline tests live
in tests/fixtures/.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import pandas as pd

from ..freshness import stamp_now
from ._http import get_json

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLAYER_MAP_CACHE = os.path.join(ROOT, "historical", "sleeper_players.parquet")

PROJECTIONS_URL = "https://api.sleeper.com/projections/nfl/{season}/{week}"
PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"

# our market -> Sleeper stat key(s). anytime_td compares expected-TD lambdas.
MARKET_TO_SLEEPER = {
    "receiving_yards": ("rec_yd",),
    "receptions": ("rec",),
    "rushing_yards": ("rush_yd",),
    "passing_yards": ("pass_yd",),
    "pass_attempts": ("pass_att",),
    "rush_attempts": ("rush_att",),
    "anytime_td": ("rush_td", "rec_td"),
}

# Divergence thresholds: flag when |model - sleeper| exceeds
# max(abs_floor[market], rel_frac * sleeper). Tuned loose on purpose --
# the flag means "meaningfully different opinion", not "tiny disagreement".
DEFAULT_REL_FRAC = 0.35
DEFAULT_ABS_FLOOR = {
    "receiving_yards": 15.0, "receptions": 1.5, "rushing_yards": 15.0,
    "passing_yards": 45.0, "pass_attempts": 5.0, "rush_attempts": 4.0,
    "anytime_td": 0.25,
}


class SleeperSchemaError(RuntimeError):
    """Raised when a Sleeper payload doesn't look like what we recorded (H3)."""


# --------------------------------------------------------------------------- #
# Parsing (pure -- fixture-testable offline)
# --------------------------------------------------------------------------- #
def parse_projections(raw: List[Dict]) -> pd.DataFrame:
    """Sleeper projections payload -> tidy frame, one row per (player, market).

    Output columns: sleeper_id, name, team, pos, market, sleeper_proj.
    Schema-validated: raises SleeperSchemaError on an unrecognizable payload
    rather than returning something silently empty/garbled.
    """
    if not isinstance(raw, list):
        raise SleeperSchemaError(f"projections payload is {type(raw).__name__}, expected list")
    rows = []
    n_shaped = 0
    for item in raw:
        if not isinstance(item, dict) or "stats" not in item or "player_id" not in item:
            continue
        n_shaped += 1
        stats = item.get("stats") or {}
        player = item.get("player") or {}
        name = " ".join(x for x in (player.get("first_name"), player.get("last_name")) if x) or None
        team = item.get("team") or player.get("team")
        pos = player.get("position")
        for market, keys in MARKET_TO_SLEEPER.items():
            vals = [stats.get(k) for k in keys]
            if all(v is None for v in vals):
                continue
            proj = float(sum(float(v) for v in vals if v is not None))
            rows.append({
                "sleeper_id": str(item["player_id"]), "name": name, "team": team,
                "pos": pos, "market": market, "sleeper_proj": proj,
            })
    if raw and n_shaped == 0:
        raise SleeperSchemaError("no projection item had the recorded shape "
                                 "({player_id, stats, ...}) -- endpoint schema changed?")
    return pd.DataFrame(rows, columns=["sleeper_id", "name", "team", "pos", "market", "sleeper_proj"])


def parse_player_map(raw: Dict) -> pd.DataFrame:
    """Sleeper full player dump -> {sleeper_id, gsis_id, full_name, team, position}.

    Only rows with a gsis_id are kept -- that's the join key to nflverse
    player_ids; players without one (mostly non-skill/practice bodies) can't
    be matched reliably and are dropped rather than name-guessed.
    """
    if not isinstance(raw, dict):
        raise SleeperSchemaError(f"players payload is {type(raw).__name__}, expected dict")
    rows = []
    for sleeper_id, p in raw.items():
        if not isinstance(p, dict):
            continue
        gsis = p.get("gsis_id")
        if not gsis:
            continue
        rows.append({
            "sleeper_id": str(sleeper_id),
            "gsis_id": str(gsis).strip(),
            "full_name": p.get("full_name") or " ".join(
                x for x in (p.get("first_name"), p.get("last_name")) if x),
            "team": p.get("team"), "position": p.get("position"),
        })
    if raw and not rows:
        raise SleeperSchemaError("players dump contained no gsis_id fields -- schema changed?")
    return pd.DataFrame(rows, columns=["sleeper_id", "gsis_id", "full_name", "team", "position"])


# --------------------------------------------------------------------------- #
# Fetch (thin, timestamped; cache only for the big, rarely-changing dump)
# --------------------------------------------------------------------------- #
def fetch_projections(season: int, week: int, season_type: str = "regular") -> Dict:
    """Live pull -> {"df": DataFrame, "fetched_at": iso, "n_raw": int}."""
    raw = get_json(PROJECTIONS_URL.format(season=season, week=week),
                   params={"season_type": season_type})
    df = parse_projections(raw)
    return {"df": df, "fetched_at": stamp_now(), "n_raw": len(raw)}


def fetch_player_map(cache_path: str = PLAYER_MAP_CACHE, refresh: bool = False) -> pd.DataFrame:
    """sleeper_id <-> gsis_id mapping, cached to parquet (the dump is ~5MB
    and changes rarely; don't hammer it)."""
    if not refresh and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    raw = get_json(PLAYERS_URL, timeout=60.0)
    df = parse_player_map(raw)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def attach_gsis(proj_df: pd.DataFrame, player_map: pd.DataFrame) -> pd.DataFrame:
    """Join projections to nflverse gsis ids. Unmatched rows are KEPT with
    gsis_id=None (visible, flaggable) -- never name-guessed."""
    out = proj_df.merge(player_map[["sleeper_id", "gsis_id"]], on="sleeper_id", how="left")
    return out


# --------------------------------------------------------------------------- #
# The one judgment this module is allowed to make (H5)
# --------------------------------------------------------------------------- #
def divergence(model_mean: float, sleeper_proj: Optional[float], market: str,
               rel_frac: float = DEFAULT_REL_FRAC,
               abs_floor: Optional[Dict[str, float]] = None) -> Dict:
    """Compare our deterministic mean to Sleeper's projection.

    Returns {divergence_flag, abs_diff, rel_diff, threshold, note} and NOTHING
    that could be used as a replacement number. If Sleeper has no opinion,
    the flag is False with note "no_fantasy_ref" (absence of a cross-check is
    a freshness concern, not a divergence).
    """
    floors = dict(DEFAULT_ABS_FLOOR)
    if abs_floor:
        floors.update(abs_floor)
    if sleeper_proj is None or (isinstance(sleeper_proj, float) and math.isnan(sleeper_proj)):
        return {"divergence_flag": False, "abs_diff": None, "rel_diff": None,
                "threshold": None, "note": "no_fantasy_ref"}
    sp = float(sleeper_proj)
    diff = abs(float(model_mean) - sp)
    threshold = max(floors.get(market, 10.0), rel_frac * abs(sp))
    flag = diff > threshold
    rel = diff / abs(sp) if sp else None
    note = (f"model {model_mean:g} vs sleeper {sp:g} (|diff| {diff:.1f} "
            f"{'>' if flag else '<='} threshold {threshold:.1f})")
    return {"divergence_flag": bool(flag), "abs_diff": round(diff, 3),
            "rel_diff": round(rel, 4) if rel is not None else None,
            "threshold": round(threshold, 3), "note": note}
