"""Command-line entrypoint for the complete fantasy projection lifecycle."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from .audit import red_team_report
from .config import ModelConfig, ScoringRules, SimulationConfig
from .data import HistoricalData, fetch_historical
from .features import build_feature_frame, frame_quality_report
from .monte_carlo_audit import historical_monte_carlo_replay, render_monte_carlo_markdown
from .models import FantasyEnsemble, fit_ensemble, season_forward_backtest
from .simulation import simulate_week


def _seasons(value: str) -> list[int]:
    if ":" in value:
        start, end = map(int, value.split(":", 1))
        if end < start:
            raise argparse.ArgumentTypeError("season range end must not precede start")
        return list(range(start, end + 1))
    return sorted({int(item) for item in value.split(",") if item.strip()})


def _rules(name: str) -> ScoringRules:
    return ScoringRules.preset(name)


def _default_seasons() -> list[int]:
    today = date.today()
    nfl_season = today.year if today.month >= 3 else today.year - 1
    return list(range(2019, nfl_season + 1))


def _json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download official historical tables")
    fetch.add_argument("--seasons", type=_seasons, default=_default_seasons())
    fetch.add_argument("--data-dir", default="historical/fantasy")
    fetch.add_argument("--force", action="store_true")

    build = sub.add_parser("build", help="build the roster-first feature frame")
    build.add_argument("--data-dir", default="historical/fantasy")
    build.add_argument("--output", default="historical/fantasy/feature_frame.parquet")
    build.add_argument("--quality", default="reports/fantasy_data_quality.json")
    build.add_argument("--scoring", choices=["ppr", "half_ppr", "standard"], default="ppr")

    train = sub.add_parser("train", help="fit the production ensemble")
    train.add_argument("--frame", default="historical/fantasy/feature_frame.parquet")
    train.add_argument("--output", default="data/fantasy_model.joblib")
    train.add_argument("--model-card", default="reports/fantasy_model_card.json")
    train.add_argument("--scoring", choices=["ppr", "half_ppr", "standard"], default="ppr")
    train.add_argument("--fast", action="store_true")

    backtest = sub.add_parser("backtest", help="run untouched season-forward evaluation")
    backtest.add_argument("--frame", default="historical/fantasy/feature_frame.parquet")
    backtest.add_argument("--test-seasons", type=_seasons, default=_seasons("2023:2025"))
    backtest.add_argument("--predictions", default="reports/fantasy_outer_predictions.parquet")
    backtest.add_argument("--report", default="reports/fantasy_red_team.json")
    backtest.add_argument("--scoring", choices=["ppr", "half_ppr", "standard"], default="ppr")
    backtest.add_argument("--full", action="store_true", help="use production-size learners")

    replay = sub.add_parser(
        "audit-monte-carlo",
        help="replay frozen outer predictions through the historical event simulator",
    )
    replay.add_argument("--frame", default="historical/fantasy/feature_frame.parquet")
    replay.add_argument("--predictions", default="reports/fantasy_outer_predictions.parquet")
    replay.add_argument("--simulations", type=int, default=1_000)
    replay.add_argument("--seed", type=int, default=6102026)
    replay.add_argument("--bootstrap-iterations", type=int, default=20_000)
    replay.add_argument("--scoring", choices=["ppr", "half_ppr", "standard"], default="ppr")
    replay.add_argument("--output", default="reports/fantasy_monte_carlo_history.parquet")
    replay.add_argument("--report", default="reports/fantasy_monte_carlo_history.json")
    replay.add_argument("--markdown", default="reports/fantasy_monte_carlo_history.md")

    project = sub.add_parser("project", help="project one season/week snapshot")
    project.add_argument("--frame", default="historical/fantasy/feature_frame.parquet")
    project.add_argument("--model", default="data/fantasy_model.joblib")
    project.add_argument("--season", type=int, required=True)
    project.add_argument("--week", type=int, required=True)
    project.add_argument("--output", default="data/fantasy_weekly_projections.json")

    simulate = sub.add_parser("simulate", help="run correlated event Monte Carlo for one week")
    simulate.add_argument("--frame", default="historical/fantasy/feature_frame.parquet")
    simulate.add_argument("--model", default="data/fantasy_model.joblib")
    simulate.add_argument("--season", type=int, required=True)
    simulate.add_argument("--week", type=int, required=True)
    simulate.add_argument("--simulations", type=int, default=10_000)
    simulate.add_argument("--seed", type=int, default=6102026)
    simulate.add_argument("--scoring", choices=["ppr", "half_ppr", "standard"], default="ppr")
    simulate.add_argument("--output", default="data/fantasy_weekly_simulation.json")
    simulate.add_argument("--samples", default=None, help="optional Parquet path for player sample matrix")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fetch":
        result = fetch_historical(args.seasons, args.data_dir, force=args.force)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "build":
        data = HistoricalData.load(args.data_dir)
        frame = build_feature_frame(data, _rules(args.scoring))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output, index=False)
        quality = frame_quality_report(frame)
        _json(args.quality, quality)
        print(json.dumps(quality, indent=2, sort_keys=True))
        return 0

    frame = pd.read_parquet(args.frame)
    if args.command == "audit-monte-carlo":
        predictions = pd.read_parquet(args.predictions)
        replayed, report = historical_monte_carlo_replay(
            frame,
            predictions,
            simulations=args.simulations,
            random_seed=args.seed,
            scoring=_rules(args.scoring),
            bootstrap_iterations=args.bootstrap_iterations,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        replayed.to_parquet(output, index=False)
        _json(args.report, report)
        markdown = Path(args.markdown)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(render_monte_carlo_markdown(report))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.command == "train":
        config = ModelConfig(fast=args.fast)
        artifact = fit_ensemble(frame, config=config, scoring=_rules(args.scoring))
        artifact.save(args.output)
        artifact.write_model_card(args.model_card)
        print(json.dumps(artifact.model_card(), indent=2, sort_keys=True))
        return 0

    if args.command == "backtest":
        config = ModelConfig(
            fast=not args.full,
            stack_validation_seasons=2 if not args.full else 3,
        )
        predictions, _ = season_forward_backtest(
            frame, args.test_seasons, config=config, scoring=_rules(args.scoring)
        )
        output = Path(args.predictions)
        output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(output, index=False)
        report = red_team_report(predictions)
        _json(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    artifact = FantasyEnsemble.load(args.model)
    week = frame[
        frame["season"].astype(int).eq(args.season)
        & frame["week"].astype(int).eq(args.week)
    ].copy()
    if week.empty:
        raise SystemExit(f"no rows found for {args.season} week {args.week}")
    projected = artifact.predict(week)
    projected = projected[projected["projection_mean"].notna()].copy()
    if args.command == "project":
        columns = [
            "season", "week", "player_id", "player_name", "position", "team",
            "projection_mean", "projection_lower80", "projection_upper80",
            "projection_model_sd", "model_eligible", "status_inactive",
        ]
        _json(args.output, projected[columns].sort_values("projection_mean", ascending=False).to_dict("records"))
        print(f"wrote {len(projected)} projections to {args.output}")
        return 0

    requested_rules = _rules(args.scoring)
    if artifact.scoring != requested_rules.to_dict():
        raise SystemExit(
            "simulation scoring differs from the fitted model; rebuild and retrain for this league scoring"
        )
    simulation = simulate_week(
        projected,
        config=SimulationConfig(simulations=args.simulations, random_seed=args.seed),
        scoring=requested_rules,
    )
    payload = {"metadata": simulation.metadata, "players": simulation.summaries.to_dict("records")}
    _json(args.output, payload)
    if args.samples:
        sample_path = Path(args.samples)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        simulation.points.to_parquet(sample_path, index=False)
    print(f"wrote {len(simulation.summaries)} simulated players to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
