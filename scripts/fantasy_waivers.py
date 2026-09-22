#!/usr/bin/env python3
"""Drive ``nflvalue.fantasy.waivers.plan()`` from the artifacts a weekly run leaves behind.

`my_team` has always had a waivers section and it has always said the same
thing: "no waiver plan was supplied for this snapshot". The planner was
written, tested and never called — nothing in the repo turned a league
snapshot, an ESPN free-agent pool and a simulation into the three arguments
`plan()` wants. This script is that wiring, and nothing else: it reads files,
calls the planner once, and prints/serialises what comes back. It performs no
ESPN write of any kind and holds no ESPN session.

Three joins do the work, and each one fails visibly rather than quietly:

* **Identity.** ESPN ids become model ids through
  :func:`nflvalue.fantasy.identity.build_crosswalk` (nflverse weekly rosters).
  A pool player with no crosswalk row is *not* priced at zero — he is reported
  in ``unpriced`` with the reason, because "we cannot value him" and "he is
  worth nothing" are different claims and only one of them is true.

* **Draws.** ``plan()`` values a move from paired simulation rows. The weekly
  run publishes the event samples (``samples.parquet``) and the *calibrated*
  per-player summary, but not the calibrated draw matrix. This script
  reconstructs the calibrated draws by mapping each player's raw scored
  samples onto his published quantiles, preserving the simulation's own
  cross-player rank structure. The reconstruction error against every
  published statistic is measured and reported (``draws.reconstruction``); it
  is not assumed.

* **Seats.** ``--skill-only`` narrows the contract to the QB/RB/WR/TE seats.
  K and D/ST are shadow-only in this repo (no promoted projection model), so a
  contract that still demands those seats cannot solve a lineup at all. The
  narrowed roster limit is the real one for the skill positions: total roster
  size minus the K and D/ST slots.

Usage (all paths required — nothing is discovered or guessed)::

    python scripts/fantasy_waivers.py \
        --league-snapshot PATH.json --pool PATH.json \
        --fantasy-latest PATH.json --samples PATH.parquet \
        --team-id 1 --skill-only --out PATH.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflvalue.fantasy import identity, waiver_rules, waivers
from nflvalue.fantasy.config import ScoringRules
from nflvalue.fantasy.scoring import score_components

UTC = timezone.utc

#: ESPN's `defaultPositionId`, only for the positions this script prices.
POSITION_BY_ID = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "D/ST"}
SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
SHADOW_SLOTS = ("K", "D/ST")

#: The quantiles the weekly summary publishes, and the keys they live under.
PUBLISHED_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
PUBLISHED_KEYS = ("p10", "p25", "median", "p75", "p90")

#: How far the affine reconstruction of a player's calibrated draws may miss
#: his published mean before this script falls back to the slower, exact
#: quantile remap for that player.
AFFINE_MEAN_TOLERANCE = 0.25


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _json(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def pool_position(entry: dict) -> str | None:
    return POSITION_BY_ID.get(entry.get("player", {}).get("defaultPositionId"))


def espn_week_projection(entry: dict, *, season: int, week: int) -> float | None:
    """ESPN's own projection for one scoring period, or None if it has none.

    `statSourceId` 1 is a projection (0 is an actual) and `statSplitTypeId` 1
    is the single-week split — the season-to-date projection lives under the
    same source id and is a different number entirely.
    """
    for stat in entry.get("player", {}).get("stats", []):
        if (stat.get("statSourceId") == 1 and stat.get("statSplitTypeId") == 1
                and stat.get("scoringPeriodId") == week
                and stat.get("seasonId") == season):
            total = stat.get("appliedTotal")
            return None if total is None else float(total)
    return None


# --------------------------------------------------------------------------- #
# Calibrated draws
# --------------------------------------------------------------------------- #
def _affine_draws(raw: np.ndarray, summary: dict) -> np.ndarray:
    """Least-squares location/scale fit of the raw draws onto the published quantiles."""
    source = np.quantile(raw, PUBLISHED_QUANTILES)
    target = np.array([float(summary[key]) for key in PUBLISHED_KEYS], dtype=float)
    design = np.vstack([source, np.ones_like(source)]).T
    (scale, shift), *_ = np.linalg.lstsq(design, target, rcond=None)
    return np.maximum(scale * raw + shift, min(0.0, float(target[0])))


def _remapped_draws(raw: np.ndarray, summary: dict) -> np.ndarray:
    """Rank-preserving remap onto the published quantiles, with linear tails."""
    count = raw.size
    order = np.argsort(raw, kind="stable")
    uniform = np.empty(count, dtype=float)
    uniform[order] = (np.arange(count) + 0.5) / count
    target = np.maximum.accumulate(
        np.array([float(summary[key]) for key in PUBLISHED_KEYS], dtype=float))
    grid = np.asarray(PUBLISHED_QUANTILES, dtype=float)
    out = np.interp(uniform, grid, target)
    low_slope = (target[1] - target[0]) / (grid[1] - grid[0])
    high_slope = (target[4] - target[3]) / (grid[4] - grid[3])
    low, high = uniform < grid[0], uniform > grid[-1]
    out[low] = target[0] + (uniform[low] - grid[0]) * low_slope
    out[high] = target[4] + (uniform[high] - grid[4]) * high_slope
    return np.maximum(out, min(0.0, float(target[0])))


def calibrated_draws(samples_path: str, summaries: dict, rules: ScoringRules
                     ) -> tuple[dict[str, np.ndarray], dict]:
    """``{model_id: draws}`` on the *published* scale, plus a measured error report.

    The weekly run publishes raw event samples and calibrated summaries; the
    calibrated draw matrix itself is not written out. Reconstructing it is the
    only way to give `plan()` paired rows that agree with the numbers on the
    card — feeding it the raw event samples instead would value every move on
    a distribution the card never showed anyone.
    """
    import polars as pl

    frame = pl.read_parquet(samples_path).sort(["player_id", "simulation"])
    component_columns = [c for c in frame.columns if c not in ("simulation", "player_id")]
    draws: dict[str, np.ndarray] = {}
    method_counts = {"affine": 0, "quantile_remap": 0}
    errors: dict[str, list[float]] = {}

    for key, group in frame.group_by("player_id", maintain_order=True):
        model_id = str(key[0] if isinstance(key, tuple) else key)
        summary = summaries.get(model_id)
        if summary is None:
            continue
        raw = np.asarray(score_components(
            {c: group[c].to_numpy() for c in component_columns}, rules), dtype=float)
        candidate = _affine_draws(raw, summary)
        method = "affine"
        if abs(float(candidate.mean()) - float(summary["mean"])) > AFFINE_MEAN_TOLERANCE:
            candidate = _remapped_draws(raw, summary)
            method = "quantile_remap"
        method_counts[method] += 1
        draws[model_id] = candidate
        errors.setdefault("mean", []).append(float(candidate.mean()) - float(summary["mean"]))
        errors.setdefault("median", []).append(
            float(np.median(candidate)) - float(summary["median"]))
        for quantile, published in zip(PUBLISHED_QUANTILES, PUBLISHED_KEYS):
            if published == "median":
                continue
            errors.setdefault(published, []).append(
                float(np.quantile(candidate, quantile)) - float(summary[published]))
        for threshold in (10, 15, 20, 25):
            errors.setdefault(f"prob_{threshold}_plus", []).append(
                float((candidate >= threshold).mean())
                - float(summary[f"prob_{threshold}_plus"]))

    report = {
        "basis": ("raw event samples mapped onto each player's published quantiles; "
                  "cross-player rank structure preserved"),
        "players": len(draws),
        "methods": method_counts,
        "error_vs_published": {
            key: {"mean_abs": round(float(np.abs(values).mean()), 5),
                  "max_abs": round(float(np.abs(values).max()), 5)}
            for key, values in sorted(errors.items())},
    }
    return draws, report


# --------------------------------------------------------------------------- #
# Contract / roster / pool
# --------------------------------------------------------------------------- #
def skill_only(contract: waiver_rules.WaiverRules) -> waiver_rules.WaiverRules:
    """The same contract with the shadow seats removed, limits adjusted.

    K and D/ST have no promoted projection model here, so they carry no draws.
    A lineup solve that still has to fill those seats fails for every roster,
    which reads as "nothing can be valued" when the truth is "two seats are
    out of scope". Dropping the seats *and* the roster spots they occupy keeps
    the remaining arithmetic honest: 16 spots minus one K and one D/ST is a
    14-player skill roster, and a full one still forces a drop.
    """
    kept = tuple(slot for slot in contract.slots if slot.label not in SHADOW_SLOTS)
    removed = sum(slot.count for slot in contract.slots if slot.label in SHADOW_SLOTS)
    return replace(
        contract, slots=kept, roster_limit=contract.roster_limit - removed,
        position_limits={k: v for k, v in contract.position_limits.items()
                         if k not in SHADOW_SLOTS},
        notes=contract.notes + (
            f"narrowed to the skill seats: {', '.join(SHADOW_SLOTS)} removed, "
            f"roster limit {contract.roster_limit} -> {contract.roster_limit - removed}",))


def roster_entries(snapshot: dict, pool_by_id: dict, team_id: int, *,
                   skill: bool, ir_ids: frozenset[int] = frozenset()
                   ) -> list[waivers.RosterEntry]:
    """Roster state, with lock/droppable/injury read from the *pool* payload.

    The league snapshot's `injury_status` is whatever ESPN wrote when the
    snapshot was taken; the player pool is the fresher read and is the one
    that decides IR eligibility and droppability. Where they disagree the
    fresher one wins, and the disagreement is worth printing.
    """
    out = []
    for row in snapshot["rosters"][str(team_id)]:
        position = str(row["default_position"])
        if skill and position not in SKILL_POSITIONS:
            continue
        espn_id = int(row["player_id"])
        entry = pool_by_id.get(espn_id) or {}
        player = entry.get("player") or {}
        out.append(waivers.RosterEntry(
            espn_id=espn_id, name=str(row["full_name"]), position=position,
            slot=str(row.get("lineup_slot") or "BE"),
            locked=bool(entry.get("lineupLocked") or entry.get("rosterLocked")),
            undroppable=not bool(player.get("droppable", True)),
            injury_status=str(player.get("injuryStatus")
                              or row.get("injury_status") or "ACTIVE").upper(),
            on_ir=(espn_id in ir_ids or str(row.get("lineup_slot") or "") == "IR"),
        ))
    return out


def pool_entries(pool: list, *, skill: bool, as_of: datetime
                 ) -> tuple[list[waivers.PoolEntry], list[dict]]:
    """Available players as `PoolEntry`, and the rows that could not become one."""
    entries, refused = [], []
    for row in pool:
        status = str(row.get("status") or "").upper()
        if status == "ONTEAM":
            continue
        position = pool_position(row)
        if position is None:
            refused.append({"espn_player_id": row.get("id"),
                            "name": (row.get("player") or {}).get("fullName"),
                            "reason": "ESPN position id is not one this script prices"})
            continue
        if skill and position not in SKILL_POSITIONS:
            continue
        entries.append(waivers.PoolEntry(
            espn_id=int(row["id"]),
            name=str((row.get("player") or {}).get("fullName") or row["id"]),
            position=position,
            availability=("waivers" if status == "WAIVERS" else "freeagent"),
            waiver_process_time=None, as_of=as_of))
    return entries, refused


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build(args) -> dict:
    snapshot = _json(args.league_snapshot)
    pool_payload = _json(args.pool)["players"]
    latest = _json(args.fantasy_latest)
    now = (datetime.fromisoformat(args.now) if args.now
           else datetime.now(tz=UTC)).astimezone(UTC)

    contract = waiver_rules.from_snapshot(snapshot)
    if args.skill_only:
        contract = skill_only(contract)

    season, week = int(latest["season"]), int(latest["week"])
    rules = ScoringRules(**{k: v for k, v in latest["simulation"]["scoring"].items()
                            if k in ScoringRules.__dataclass_fields__})
    summaries = {str(p["player_id"]): p for p in latest["players"]}

    import polars as pl
    rosters = pl.read_parquet(args.weekly_rosters).to_pandas()
    crosswalk = identity.build_crosswalk(rosters, season)

    draws, reconstruction = calibrated_draws(args.samples, summaries, rules)

    pool_by_id = {int(row["id"]): row for row in pool_payload}
    ir_ids = frozenset(int(v) for v in (args.ir or []))
    roster = roster_entries(snapshot, pool_by_id, args.team_id,
                            skill=args.skill_only, ir_ids=ir_ids)
    available, refused = pool_entries(pool_payload, skill=args.skill_only,
                                      as_of=now)

    # Every player the planner will see, keyed by ESPN id, on the published
    # scale. A player with no model id, or with a model id the run did not
    # project, is simply absent — never a zero.
    distributions: dict[int, np.ndarray] = {}
    unpriced: list[dict] = []
    for entry in list(roster) + list(available):
        model_id = crosswalk.get(int(entry.espn_id))
        if model_id is not None and model_id in draws:
            distributions[int(entry.espn_id)] = draws[model_id]
            continue
        reason = ("no nflverse crosswalk row for this ESPN id"
                  if model_id is None else
                  f"crosswalk resolves to {model_id}, which this week's run did not project")
        unpriced.append({"espn_player_id": int(entry.espn_id), "name": entry.name,
                         "position": entry.position, "model_player_id": model_id,
                         "reason": reason})

    # A rostered player the run gated Out is published at zero on the card, so
    # he is carried at zero here too — explicitly, and only for the roster.
    for entry in roster:
        if int(entry.espn_id) in distributions:
            continue
        model_id = crosswalk.get(int(entry.espn_id))
        card = next((r for r in (latest.get("my_team") or {}).get("roster") or []
                     if int(r.get("espn_player_id") or -1) == int(entry.espn_id)), None)
        projection = (card or {}).get("projection") or {}
        if projection.get("mean") == 0.0 and projection.get("availability_probability") == 0.0:
            distributions[int(entry.espn_id)] = np.zeros(
                int(latest["simulation"]["simulations"]), dtype=float)
            for row in unpriced:
                if row["espn_player_id"] == int(entry.espn_id):
                    row["reason"] = ("gated Out by the weekly availability gate; "
                                     "carried at the card's published zero")
                    row["model_player_id"] = model_id

    records = waivers.plan(contract, roster=roster, pool=available, now=now,
                           distributions=distributions, my_team_id=args.team_id)

    return {
        "recommendation_only": True,
        "generated_at": now.isoformat(),
        "season": season, "week": week,
        "contract": contract.to_dict(),
        "roster": [{"espn_player_id": e.espn_id, "name": e.name, "position": e.position,
                    "slot": e.slot, "injury_status": e.injury_status, "on_ir": e.on_ir,
                    "locked": e.locked, "undroppable": e.undroppable,
                    "priced": int(e.espn_id) in distributions} for e in roster],
        "pool": {"considered": len(available), "priced": sum(
            1 for e in available if int(e.espn_id) in distributions)},
        "draws": {"reconstruction": reconstruction,
                  "simulations": int(latest["simulation"]["simulations"]),
                  "random_seed": latest["simulation"]["random_seed"]},
        "espn_projection_week": week,
        "unpriced": unpriced,
        "refused_pool_rows": refused,
        "records": [r.to_dict() for r in records],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league-snapshot", required=True)
    parser.add_argument("--pool", required=True, help="ESPN kona_player_info payload")
    parser.add_argument("--fantasy-latest", required=True)
    parser.add_argument("--samples", required=True, help="samples.parquet from the same run")
    parser.add_argument("--weekly-rosters", required=True,
                        help="nflverse weekly_rosters parquet, for the ESPN id crosswalk")
    parser.add_argument("--team-id", type=int, required=True)
    parser.add_argument("--skill-only", action="store_true",
                        help="QB/RB/WR/TE seats only; K and D/ST are shadow-only here")
    parser.add_argument("--ir", type=int, action="append",
                        help="ESPN id to treat as parked on IR (repeatable)")
    parser.add_argument("--now", help="ISO timestamp; defaults to the wall clock")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    payload = build(args)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, default=str)
    picked = [r for r in payload["records"] if r["status"] == "recommendation"]
    print(f"[waivers] {len(payload['records'])} candidate(s) considered, "
          f"{payload['pool']['priced']} priced by the model, "
          f"{len(picked)} clear the declared gate -> {args.out}")
    for record in picked[:10]:
        delta = record["lineup_delta"]
        print(f"  ADD {record['add_name']} ({record['add_position']}) "
              f"DROP {record['drop_name']}  "
              f"{delta['own_optimal_lineup_delta']:+.2f} pts "
              f"[p10 {delta['p10']:+.2f}, p90 {delta['p90']:+.2f}] "
              f"helps in {delta['model_relative_prob_improves']:.0%} of simulated weeks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
