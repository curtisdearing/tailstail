"""Rest-of-season, lineup, waiver, and trade translation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import LineupRules
from .simulation import SimulationResult


@dataclass
class SeasonSimulation:
    summaries: pd.DataFrame
    points: pd.DataFrame
    player_meta: pd.DataFrame
    metadata: dict[str, object]


def aggregate_rest_of_season(
    weekly: Mapping[int, SimulationResult], *, random_seed: int = 6102026
) -> SeasonSimulation:
    """Combine weekly samples without introducing same-seed week correlation."""

    if not weekly:
        raise ValueError("at least one simulated week is required")
    counts = {result.metadata["simulations"] for result in weekly.values()}
    if len(counts) != 1:
        raise ValueError("all weekly simulations must use the same sample count")
    simulations = int(next(iter(counts)))
    player_ids = sorted({column for result in weekly.values() for column in result.points.columns})
    total = pd.DataFrame(0.0, index=np.arange(simulations), columns=player_ids)
    meta_rows = []
    seen: set[str] = set()
    for week, result in sorted(weekly.items()):
        # If callers reused a seed, this permutation prevents simulation row 7
        # from representing the same latent good/bad draw every NFL week.
        rng = np.random.default_rng(random_seed + int(week) * 7919)
        permutation = rng.permutation(simulations)
        aligned = result.points.iloc[permutation].reset_index(drop=True)
        total.loc[:, aligned.columns] += aligned.to_numpy()
        for row in result.summaries.to_dict("records"):
            player_id = str(row["player_id"])
            if player_id not in seen:
                meta_rows.append({
                    "player_id": player_id,
                    "player_name": row["player_name"],
                    "position": row["position"],
                    "team": row["team"],
                })
                seen.add(player_id)
    meta = pd.DataFrame(meta_rows).drop_duplicates("player_id").set_index("player_id")
    summaries = []
    for player_id in player_ids:
        values = total[player_id].to_numpy(dtype=float)
        info = meta.loc[player_id].to_dict() if player_id in meta.index else {}
        summaries.append({
            "player_id": player_id,
            **info,
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "sd": float(values.std()),
            "p10": float(np.quantile(values, 0.10)),
            "p90": float(np.quantile(values, 0.90)),
        })
    return SeasonSimulation(
        summaries=pd.DataFrame(summaries).sort_values("mean", ascending=False).reset_index(drop=True),
        points=total,
        player_meta=meta.reset_index(),
        metadata={
            "weeks": sorted(map(int, weekly)),
            "simulations": simulations,
            "random_seed": random_seed,
        },
    )


def lineup_points(
    season: SeasonSimulation,
    roster: Sequence[str],
    rules: LineupRules | None = None,
) -> np.ndarray:
    """Optimize a fantasy lineup independently inside every simulation."""

    lineup = rules or LineupRules()
    roster = [str(player_id) for player_id in roster]
    missing = sorted(set(roster) - set(season.points.columns))
    if missing:
        raise ValueError(f"roster players missing from season simulation: {missing}")
    meta = season.player_meta.set_index("player_id")
    simulations = len(season.points)
    output = np.zeros(simulations, dtype=float)
    base_slots = {position: count for position, count in lineup.starters.items() if position != "FLEX"}
    flex_count = int(lineup.starters.get("FLEX", 0))
    for simulation in range(simulations):
        selected: set[str] = set()
        row = season.points.iloc[simulation]
        for position, count in base_slots.items():
            candidates = [pid for pid in roster if meta.loc[pid, "position"] == position]
            best = sorted(candidates, key=lambda pid: float(row[pid]), reverse=True)[: int(count)]
            selected.update(best)
        flex_candidates = [
            pid for pid in roster
            if pid not in selected and meta.loc[pid, "position"] in lineup.flex_positions
        ]
        selected.update(sorted(flex_candidates, key=lambda pid: float(row[pid]), reverse=True)[:flex_count])
        output[simulation] = float(row[list(selected)].sum()) if selected else 0.0
    return output


def evaluate_trade(
    season: SeasonSimulation,
    roster: Sequence[str],
    send: Sequence[str],
    receive: Sequence[str],
    *,
    rules: LineupRules | None = None,
) -> dict[str, object]:
    """Evaluate a package by lineup points, not by adding player projections."""

    current = [str(player_id) for player_id in roster]
    send_set = set(map(str, send))
    if not send_set.issubset(current):
        raise ValueError("cannot send a player who is not on the current roster")
    after = [player_id for player_id in current if player_id not in send_set]
    after.extend(player_id for player_id in map(str, receive) if player_id not in after)
    before_points = lineup_points(season, current, rules)
    after_points = lineup_points(season, after, rules)
    delta = after_points - before_points
    return {
        "send": sorted(send_set),
        "receive": list(map(str, receive)),
        "before_mean_lineup_points": float(before_points.mean()),
        "after_mean_lineup_points": float(after_points.mean()),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "p10_delta": float(np.quantile(delta, 0.10)),
        "p90_delta": float(np.quantile(delta, 0.90)),
        "probability_trade_improves_lineup": float(np.mean(delta > 0)),
    }


def value_over_replacement(
    season: SeasonSimulation,
    *,
    league_teams: int = 12,
    rules: LineupRules | None = None,
    bench_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Normalize season value against a league-size replacement baseline."""

    lineup = rules or LineupRules()
    summary = season.summaries.copy()
    output = []
    for position, group in summary.groupby("position"):
        direct = int(lineup.starters.get(position, 0))
        flex_share = int(lineup.starters.get("FLEX", 0)) if position in lineup.flex_positions else 0
        replacement_rank = max(int(np.ceil(league_teams * (direct + flex_share / max(len(lineup.flex_positions), 1)) * bench_multiplier)), 1)
        ordered = group.sort_values("mean", ascending=False)
        replacement = float(ordered.iloc[min(replacement_rank - 1, len(ordered) - 1)]["mean"])
        assigned = ordered.assign(
            replacement_points=replacement,
            value_over_replacement=ordered["mean"] - replacement,
            replacement_rank=replacement_rank,
        )
        output.append(assigned)
    return pd.concat(output, ignore_index=True).sort_values("value_over_replacement", ascending=False)
