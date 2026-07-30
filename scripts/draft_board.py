"""Produce the 2026 ceiling-weighted draft board for a 6-team full-PPR league.

Usage: python scripts/draft_board.py [--teams 6] [--ceiling 0.55] [--sims 4000]
Writes data/draft_board_2026.json + .csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nflvalue.fantasy.draft import (
    add_rookie_market_priors,
    apply_offseason_adjustments,
    draft_board,
    pergame_baselines,
    simulate_season,
)
from nflvalue.fantasy.hierarchy import hierarchical_baselines
from nflvalue.fantasy.models import FantasyEnsemble


def load_adp(path: str) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text())
    rows = [
        {
            "name": p["name"],
            "position": p["position"],
            "team": p.get("team"),
            "adp": float(p["adp"]),
            "adp_sd": float(p.get("stdev", 8.0)),
        }
        for p in payload["players"]
        if p.get("position") in {"QB", "RB", "WR", "TE"}
    ]
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default="historical/fantasy/feature_frame.parquet")
    parser.add_argument("--model", default="data/fantasy_model.joblib")
    parser.add_argument("--adp", default="data/adp_ppr_2026.json")
    parser.add_argument("--byes", default="data/byes_2026.json")
    parser.add_argument("--teams", type=int, default=12)
    parser.add_argument("--ceiling", type=float, default=0.55)
    parser.add_argument("--sims", type=int, default=4000)
    parser.add_argument("--source-season", type=int, default=2025)
    parser.add_argument(
        "--pooling", choices=("hierarchical", "flat"), default="hierarchical",
        help="hierarchical = empirical-Bayes partial pooling (retrodiction "
        "Spearman 0.717 vs 0.682 flat; top-24 rate 0.40 vs 0.43)",
    )
    parser.add_argument("--output", default="data/draft_board_2026")
    args = parser.parse_args()

    frame = pd.read_parquet(args.frame)
    model = FantasyEnsemble.load(args.model)
    adp = load_adp(args.adp)
    byes = json.loads(Path(args.byes).read_text())

    if args.pooling == "hierarchical":
        baselines = hierarchical_baselines(frame, model, source_season=args.source_season)
    else:
        baselines = pergame_baselines(frame, model, source_season=args.source_season)
    current_teams = {
        row["name"]: row["team"] for _, row in adp.iterrows() if row.get("team")
    }
    baselines = apply_offseason_adjustments(baselines, current_teams=current_teams)
    baselines = add_rookie_market_priors(baselines, adp)

    outlook = simulate_season(baselines, byes, simulations=args.sims)
    board = draft_board(
        outlook, league_teams=args.teams, ceiling_weight=args.ceiling
    )

    # ADP overlay for value gaps.
    from nflvalue.fantasy.draft import normalize_name
    adp_key = adp.assign(key=adp["name"].map(normalize_name))
    adp_key = adp_key.drop_duplicates("key")
    board = board.assign(key=board["player_name"].map(normalize_name)).merge(
        adp_key[["key", "adp", "adp_sd"]], on="key", how="left"
    ).drop(columns=["key"])
    board["adp_round"] = ((board["adp"] - 1) // args.teams + 1).astype("Int64")
    board["value_gap"] = board["adp"] - board["overall_rank"]

    Path(args.output + ".csv").parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(args.output + ".csv", index=False)
    payload = {
        "league": {"teams": args.teams, "scoring": "ppr", "ceiling_weight": args.ceiling},
        "metadata": outlook.metadata,
        "board": json.loads(board.to_json(orient="records")),
    }
    Path(args.output + ".json").write_text(json.dumps(payload, indent=1))
    outlook.season_points.to_parquet(args.output + "_season_samples.parquet")
    # Save baselines for the trade planner (per-game mu is its market thermometer).
    baselines.to_parquet(args.output + "_baselines.parquet")

    top = board.head(20)[[
        "overall_rank", "tier", "player_name", "position", "team",
        "season_mean", "season_p90", "draft_score", "adp", "basis",
    ]]
    print(top.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
