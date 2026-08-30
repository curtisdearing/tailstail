#!/usr/bin/env python3
"""Fetch → snapshot → fit → project → simulate the next fantasy week."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue.fantasy import espn_compare
from nflvalue.fantasy import my_team as my_team_mod
from nflvalue.fantasy.config import ModelConfig, ScoringRules, SimulationConfig
from nflvalue.fantasy.dashboard import render_fantasy_dashboard
from nflvalue.fantasy.data import HistoricalData, fetch_historical, materialize_projection_week
from nflvalue.fantasy.features import build_feature_frame, frame_quality_report
from nflvalue.fantasy.models import fit_ensemble
from nflvalue.fantasy.scoring import add_fantasy_points
from nflvalue.fantasy.simulation import simulate_week
from nflvalue.projection_snapshot import (
    build_projection_snapshot,
    write_component_samples,
    write_projection_snapshot,
)
from nflvalue.sources import espn_projections


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


def _actual_ppr_points(stats: pd.DataFrame, season: int, week: int, rules: ScoringRules) -> dict:
    """gsis player_id -> actual points, scored with the SAME rules as both projections."""
    rows = stats[
        pd.to_numeric(stats["season"], errors="coerce").eq(season)
        & pd.to_numeric(stats["week"], errors="coerce").eq(week)
    ].copy()
    if rows.empty:
        return {}
    scored = add_fantasy_points(rows, rules, output="_actual_ppr")
    scored["player_id"] = scored["player_id"].astype(str)
    return dict(zip(scored["player_id"], scored["_actual_ppr"].astype(float)))


def _week_is_complete(schedules: pd.DataFrame, season: int, week: int) -> bool:
    """Grade only finished weeks: every kickoff at least 12 hours in the past."""
    kickoffs = espn_compare.game_kickoffs_utc(schedules, season, week)
    if not kickoffs:
        return False
    now = datetime.now(timezone.utc)
    latest = max(datetime.fromisoformat(value) for value in kickoffs.values())
    return (now - latest) >= timedelta(hours=12)


def run_espn_comparison(
    data: HistoricalData,
    summaries: pd.DataFrame,
    player_games: dict,
    *,
    season: int,
    week: int,
    generated_at: str,
    rules: ScoringRules,
    ledger_path: str = "data/espn_comparison_ledger.json",
    snapshot_dir: str = "data/espn_snapshots",
) -> dict:
    """Snapshot ESPN, refresh the prospective ledger, grade finished weeks.

    Display and grading ONLY (2026 freeze: the market blend is a separately
    registered lever). ESPN being unreachable degrades to an explicit,
    labelled failure — never a silent empty comparison, never a crash of the
    model publish.
    """
    ledger = espn_compare.load_ledger(ledger_path, season)

    # 1) Grade any recorded, ungraded, finished weeks (prospective rows only).
    for week_key in sorted(ledger["weeks"], key=int):
        entry = ledger["weeks"][week_key]
        if entry.get("grading") is None and _week_is_complete(data.schedules, season, int(week_key)):
            actuals = _actual_ppr_points(data.stats, season, int(week_key), rules)
            if actuals:
                espn_compare.grade_week(ledger, week=int(week_key), actual_points=actuals)
                print(f"[espn-compare] graded week {week_key} against actual PPR points")

    # 2) Snapshot ESPN for the upcoming week and refresh pre-kickoff rows.
    status, error = "ok", None
    provenance = None
    identity_report = None
    try:
        snapshot = espn_projections.fetch_week_snapshot(season, week, rules=rules)
        espn_projections.write_snapshot(snapshot, snapshot_dir)
        identity = espn_compare.build_identity_map(data.rosters, season)
        model_points = dict(
            zip(summaries["player_id"].astype(str), summaries["mean"].astype(float))
        )
        model_meta = {
            str(row["player_id"]): {"team": str(row["team"])}
            for row in summaries.to_dict("records")
        }
        matched, identity_report = espn_compare.match_players(
            snapshot["players"], identity, set(model_points)
        )
        espn_compare.record_week(
            ledger,
            week=week,
            espn_players=snapshot["players"],
            espn_retrieved_at=snapshot["retrieved_at"],
            espn_snapshot_sha256=snapshot["players_sha256"],
            matched=matched,
            model_points=model_points,
            model_meta=model_meta,
            model_generated_at=generated_at,
            player_games=player_games,
            kickoffs_utc=espn_compare.game_kickoffs_utc(data.schedules, season, week),
        )
        provenance = {key: snapshot[key] for key in (
            "retrieved_at", "source", "scoring", "coverage",
            "redistribution_rights", "players_sha256",
        )}
    except Exception as exc:
        status, error = "espn_unavailable", f"{type(exc).__name__}: {exc}"
        print(f"[espn-compare] ESPN comparison unavailable this run: {error}")

    espn_compare.save_ledger(ledger, ledger_path)
    return espn_compare.build_payload(
        ledger,
        current_week=week,
        espn_provenance=provenance,
        identity_report=identity_report,
        status=status,
        error=error,
    )


def run_my_team(
    summaries: pd.DataFrame,
    *,
    generated_at: str,
    snapshot_dir: str = "data/espn_league",
    contract=None,
    waiver_plan=None,
    espn_crosswalk: dict | None = None,
) -> dict:
    """Build the Curtis-specific Monitor contract from the read-only snapshot.

    Never raises and never fabricates: a missing, unreadable or unusable
    snapshot produces a contract whose sections all say NO CURRENT PICK with the
    reason, exactly as a stale one does.  ESPN is read-only here — this function
    performs no write of any kind.
    """
    # Scoring/roster identity comes from espn_contract when a contract is
    # supplied; with none, my_team emits null hashes and says why rather than
    # computing a second-best local digest.
    snapshot = my_team_mod.load_latest_snapshot(snapshot_dir, now=generated_at)
    if snapshot is None:
        return {
            "schema_version": my_team_mod.SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "no_current_pick",
            "reason": (f"no ESPN league snapshot found in {snapshot_dir}; the Monitor surface "
                       "reports nothing rather than reusing a prior card"),
            "league": {}, "sources": [],
        }
    projections: dict = {}
    crosswalk: dict = dict(espn_crosswalk or {})
    if len(summaries):
        for row in summaries.to_dict("records"):
            projections[str(row.get("player_id"))] = {
                "mean": float(row.get("mean", 0.0)),
                "p10": float(row.get("p10", row.get("mean", 0.0))),
                "p90": float(row.get("p90", row.get("mean", 0.0))),
            }
    # Projections stay beside the snapshot rather than being spliced into it:
    # the snapshot is a record of what ESPN said, and joining the model into it
    # is what let the reader and the adapter drift into two schemas.
    try:
        return my_team_mod.build(
            snapshot, now=generated_at, contract=contract, waiver_plan=waiver_plan,
            crosswalk=crosswalk, projections=projections)
    # A broken snapshot degrades this one section; it must not stop the publish.
    except Exception as exc:
        print(f"[my-team] contract unavailable this run: {type(exc).__name__}: {exc}")
        return {
            "schema_version": my_team_mod.SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "no_current_pick",
            "reason": f"league snapshot could not be interpreted: {type(exc).__name__}: {exc}",
            "league": {}, "sources": [],
        }


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
    parser.add_argument("--league-snapshot-dir", default="data/espn_league")
    parser.add_argument("--model", default="data/fantasy_model.joblib")
    parser.add_argument("--projection-snapshot", default="data/player_projection_snapshot.json")
    parser.add_argument("--component-samples", default="data/player_projection_samples.parquet")
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
    manifest_path = data_dir / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    historical_audit_path = Path("reports/fantasy_monte_carlo_history.json")
    historical_audit = (
        json.loads(historical_audit_path.read_text())
        if historical_audit_path.exists()
        else {}
    )
    component_validation = {
        "status": "research_only",
        "reason": (
            "The 2023-2025 replay found the raw event center 0.359 MAE worse than the direct "
            "ensemble; component probabilities require market-level validation before use."
        ),
        "evaluated_through": "2025-18",
        "audit_replay_canonical_csv_sha256": historical_audit.get("metadata", {}).get(
            "replay_outputs_canonical_csv_sha256"
        ),
    }
    sample_artifact = write_component_samples(result.components, args.component_samples)
    projection_snapshot = build_projection_snapshot(
        projected,
        result.summaries,
        result.components,
        season=season,
        week=week,
        generated_at=generated,
        information_as_of=str(source_manifest.get("retrieved_at", generated)),
        model_version=os.environ.get("GITHUB_SHA", "local"),
        simulation_metadata=result.metadata,
        sample_artifact=sample_artifact,
        source_manifest=source_manifest,
        component_validation=component_validation,
    )
    write_projection_snapshot(projection_snapshot, args.projection_snapshot)
    player_games = (
        dict(zip(projected["player_id"].astype(str), projected["game_id"].astype(str)))
        if "game_id" in projected.columns
        else {}
    )
    espn_comparison = run_espn_comparison(
        data,
        result.summaries,
        player_games,
        season=season,
        week=week,
        generated_at=generated,
        rules=rules,
    )
    my_team_payload = run_my_team(
        result.summaries, generated_at=generated, snapshot_dir=args.league_snapshot_dir,
    )
    payload = {
        "espn_comparison": espn_comparison,
        "my_team": my_team_payload,
        "generated_at": generated,
        "season": season,
        "week": week,
        "data_quality": frame_quality_report(frame),
        "model_card": artifact.model_card(),
        "simulation": result.metadata,
        "projection_snapshot": {
            "path": str(args.projection_snapshot),
            "players_canonical_csv_sha256": projection_snapshot["players_canonical_csv_sha256"],
            "samples_canonical_csv_sha256": sample_artifact["canonical_csv_sha256"],
            "component_validation": projection_snapshot["component_validation"],
        },
        "players": result.summaries.to_dict("records"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    artifact.save(args.model)
    artifact.write_model_card("reports/fantasy_model_card.json")
    render_fantasy_dashboard(
        result.summaries, args.dashboard, season=season, week=week, generated_at=generated,
        espn_comparison=espn_comparison, my_team=my_team_payload,
    )
    print(f"projected {len(result.summaries)} players for {season} week {week}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
