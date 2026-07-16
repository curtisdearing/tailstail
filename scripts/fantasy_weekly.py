#!/usr/bin/env python3
"""Fetch → snapshot → fit → project → simulate the next fantasy week."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue.fantasy.config import ModelConfig, ScoringRules, SimulationConfig
from nflvalue.fantasy.dashboard import render_fantasy_dashboard
from nflvalue.fantasy.data import HistoricalData, fetch_historical, materialize_projection_week
from nflvalue.fantasy.features import build_feature_frame, frame_quality_report
from nflvalue.fantasy.models import fit_ensemble
from nflvalue.fantasy.simulation import simulate_week


def current_nfl_season() -> int:
    today = date.today()
    return today.year if today.month >= 3 else today.year - 1


def select_week(schedules: pd.DataFrame, season: int | None, week: int | None) -> tuple[int, int]:
    if (season is None) != (week is None):
        raise ValueError("season and week overrides must be provided together")
    games = schedules.copy()
    if "game_type" in games:
        games = games[games["game_type"].fillna("REG").eq("REG")]
    games["gameday_value"] = pd.to_datetime(games["gameday"], errors="coerce").dt.date
    if season is not None and week is not None:
        if games[pd.to_numeric(games["season"], errors="coerce").eq(season)
                 & pd.to_numeric(games["week"], errors="coerce").eq(week)].empty:
            raise ValueError(f"schedule has no {season} week {week}")
        return int(season), int(week)
    cutoff = date.today() - timedelta(days=2)
    future = games[games["gameday_value"].ge(cutoff)].sort_values("gameday_value")
    if future.empty:
        latest = games.sort_values(["season", "week"]).iloc[-1]
        return int(latest["season"]), int(latest["week"])
    next_game = future.iloc[0]
    return int(next_game["season"]), int(next_game["week"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="historical/fantasy")
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument("--start-season", type=int, default=2019)
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--scoring", choices=["ppr", "half_ppr", "standard"], default="ppr")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--output", default="data/fantasy_latest.json")
    parser.add_argument("--dashboard", default="fantasy.html")
    parser.add_argument("--model", default="data/fantasy_model.joblib")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    if not args.no_fetch:
        end = max(args.season or current_nfl_season(), current_nfl_season())
        fetch_historical(range(args.start_season, end + 1), data_dir)
    data = HistoricalData.load(data_dir)
    season, week = select_week(data.schedules, args.season, args.week)
    data = materialize_projection_week(data, season, week)
    rules = ScoringRules.preset(args.scoring)
    frame = build_feature_frame(data, rules)
    before = (frame["season"].astype(int) < season) | (
        frame["season"].astype(int).eq(season) & frame["week"].astype(int).lt(week)
    )
    artifact = fit_ensemble(
        frame[before],
        config=ModelConfig(fast=args.fast, stack_validation_seasons=2 if args.fast else 3),
        scoring=rules,
    )
    target = frame[
        frame["season"].astype(int).eq(season) & frame["week"].astype(int).eq(week)
    ].copy()
    projected = artifact.predict(target)
    projected = projected[
        projected["projection_mean"].notna()
        & projected["model_eligible"].fillna(False)
    ].copy()
    result = simulate_week(
        projected,
        config=SimulationConfig(simulations=args.simulations, random_seed=6102026 + season * 100 + week),
        scoring=rules,
    )
    generated = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at": generated,
        "season": season,
        "week": week,
        "data_quality": frame_quality_report(frame),
        "model_card": artifact.model_card(),
        "simulation": result.metadata,
        "players": result.summaries.to_dict("records"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    artifact.save(args.model)
    artifact.write_model_card("reports/fantasy_model_card.json")
    render_fantasy_dashboard(
        result.summaries, args.dashboard, season=season, week=week, generated_at=generated
    )
    print(f"projected {len(result.summaries)} players for {season} week {week}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
