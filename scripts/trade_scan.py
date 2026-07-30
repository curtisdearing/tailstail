"""Weekly trade scan: who to trade, who to target, and why — from any week.

Preseason (before Week 1) it runs off the draft-board season samples; in
season, rerun scripts/draft_board.py first so baselines reflect the latest
data, then scan the remaining weeks only.

Rosters come from ESPN (league id [+ espn_s2/swid for private leagues]) or a
JSON file: {"teams": [{"name": "...", "players": ["...", ...]}, ...]}.

Examples:
  PYTHONPATH=. python scripts/trade_scan.py --rosters data/rosters.json \
      --my-team "Curtis" --week 5
  PYTHONPATH=. python scripts/trade_scan.py --espn-league 12345 --season 2026 \
      --my-team "Curtis" --week 5 --espn-s2 ... --swid ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nflvalue.fantasy.draft import simulate_season
from nflvalue.fantasy.season import aggregate_rest_of_season  # noqa: F401 (in-season path)
from nflvalue.fantasy.trade_planner import (
    FANTASY_PLAYOFF_WEEKS,
    load_rosters_espn,
    load_rosters_json,
    match_players,
    propose_trades,
)


def _season_sim_from_outlook(board, samples) -> "object":
    """Adapt draft-module season samples to season.SeasonSimulation."""

    from nflvalue.fantasy.season import SeasonSimulation

    meta = board[["player_id", "player_name", "position", "team"]].copy()
    summaries = board[[
        "player_id", "player_name", "position", "team", "season_mean",
    ]].rename(columns={"season_mean": "mean"})
    return SeasonSimulation(
        summaries=summaries, points=samples, player_meta=meta,
        metadata={"source": "draft_outlook"},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", default="data/draft_board_2026.csv")
    parser.add_argument("--baselines", default="data/draft_board_2026_baselines.parquet")
    parser.add_argument("--byes", default="data/byes_2026.json")
    parser.add_argument("--week", type=int, default=1, help="current NFL week (scan covers week..17)")
    parser.add_argument("--rosters", help="rosters JSON path")
    parser.add_argument("--my-team", required=True)
    parser.add_argument("--espn-league", type=int)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--espn-s2")
    parser.add_argument("--swid")
    parser.add_argument("--sims", type=int, default=2000)
    parser.add_argument("--output", default="data/trade_scan.json")
    args = parser.parse_args()

    baselines = pd.read_parquet(args.baselines)
    byes = json.loads(Path(args.byes).read_text())

    if args.espn_league:
        rosters = load_rosters_espn(
            args.espn_league, args.season, args.my_team,
            espn_s2=args.espn_s2, swid=args.swid,
        )
    elif args.rosters:
        rosters = load_rosters_json(args.rosters, args.my_team)
    else:
        raise SystemExit("provide --espn-league or --rosters")

    remaining = list(range(args.week, 18))
    outlook = simulate_season(
        baselines, byes, weeks=remaining, simulations=args.sims, random_seed=6102026,
    )
    season_sim = _season_sim_from_outlook(outlook.board, outlook.season_points)
    playoff_outlook = simulate_season(
        baselines, byes,
        weeks=[w for w in FANTASY_PLAYOFF_WEEKS if w >= args.week],
        simulations=args.sims, random_seed=6102027,
    )
    playoff_sim = _season_sim_from_outlook(playoff_outlook.board, playoff_outlook.season_points)

    merged_board = outlook.board  # carries season_mean for remaining weeks
    proposals = propose_trades(
        season_sim, rosters, merged_board,
        byes=byes, upcoming_weeks=tuple(range(args.week, min(args.week + 3, 18))),
        playoff_season=playoff_sim,
    )

    # Roster hygiene: names we couldn't match (rookies missing from board, DST/K).
    unmatched_report = {}
    for name, players in rosters.teams.items():
        _, unmatched = match_players(players, merged_board)
        if unmatched:
            unmatched_report[name] = unmatched

    payload = {
        "week": args.week,
        "remaining_weeks": remaining,
        "my_team": rosters.my_team,
        "proposals": json.loads(proposals.to_json(orient="records")) if not proposals.empty else [],
        "unmatched_players": unmatched_report,
    }
    Path(args.output).write_text(json.dumps(payload, indent=1))
    if proposals.empty:
        print("No trades pass the two-sided gate this week — hold.")
    else:
        print(proposals.to_string(index=False))
    if unmatched_report:
        print("\nUnmatched names (no projection — DST/K/rookies?):", unmatched_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
