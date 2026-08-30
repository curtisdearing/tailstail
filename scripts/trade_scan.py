#!/usr/bin/env python3
"""Weekly trade scan: ID-based, two-sided, recommendation-only.

Reads a validated ESPN league snapshot (see `scripts/espn_league_snapshot.py`)
and the projection board, and reports packages that improve Curtis's optimal
lineup without making the counterparty's worse. Everything is keyed on ESPN
player ids; names appear only so a human can read the output.

Read-only, in both directions. It talks to no network at all -- the snapshot is
a file -- and it sends, accepts, declines and cancels nothing. A package here is
a recommendation to consider, never a claim about what another manager would do.

Credentials are never arguments: the snapshot step reads ESPN cookies from the
environment, and this step reads no credentials whatsoever.

    PYTHONPATH=. python scripts/trade_scan.py \\
        --snapshot reports/espn/espn-league-1111111111-2026-<stamp>.json \\
        --board data/draft_board_2026.csv --baselines data/draft_board_2026_baselines.parquet

If the two-sided gate admits nothing, the output is an explicit hold state with
the rejection counts -- a finding, not an empty list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue.fantasy import league_trades
from nflvalue.fantasy.draft import simulate_season
from nflvalue.fantasy.season import SeasonSimulation


def _season_sim(outlook) -> SeasonSimulation:
    """Adapt draft-module season samples to season.SeasonSimulation."""
    board = outlook.board
    return SeasonSimulation(
        summaries=board[["player_id", "player_name", "position", "team", "season_mean"]]
        .rename(columns={"season_mean": "mean"}),
        points=outlook.season_points,
        player_meta=board[["player_id", "player_name", "position", "team"]],
        metadata={"source": "draft_outlook"},
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, required=True,
                        help="espn-league/1 snapshot JSON written by espn_league_snapshot.py")
    parser.add_argument("--board", type=Path, default=Path("data/draft_board_2026.csv"))
    parser.add_argument("--baselines", type=Path,
                        default=Path("data/draft_board_2026_baselines.parquet"))
    parser.add_argument("--byes", type=Path, default=Path("data/byes_2026.json"))
    parser.add_argument("--week", type=int,
                        help="current NFL week; defaults to the snapshot's scoring period")
    parser.add_argument("--sims", type=int, default=2000)
    parser.add_argument("--min-gain", type=float, default=0.5,
                        help="minimum mean points my lineup must gain over the simulated span")
    parser.add_argument("--min-prob", type=float, default=0.55,
                        help="maximum share of simulations in which I may end up worse is 1 minus this")
    parser.add_argument("--their-tolerance", type=float, default=0.0,
                        help="how much the counterparty's lineup may lose (0 = not at all)")
    parser.add_argument("--output", type=Path, default=Path("data/trade_scan.json"))
    args = parser.parse_args(argv)

    snapshot = json.loads(args.snapshot.read_text())
    board = pd.read_csv(args.board)
    byes = json.loads(args.byes.read_text())
    baselines = pd.read_parquet(args.baselines)

    week = args.week or int(snapshot["league"]["current_scoring_period"])
    final_week = int(snapshot["league"]["final_scoring_period"]) or 17
    remaining = list(range(week, final_week + 1))
    playoff_weeks = [w for w in league_trades.playoff_scoring_periods(snapshot) if w >= week]

    outlook = simulate_season(baselines, byes, weeks=remaining, simulations=args.sims,
                              random_seed=6102026)
    playoff_outlook = (
        simulate_season(baselines, byes, weeks=playoff_weeks, simulations=args.sims,
                        random_seed=6102027) if playoff_weeks else None)

    try:
        scan = league_trades.scan_trades(
            snapshot, board, _season_sim(outlook),
            playoff_season=_season_sim(playoff_outlook) if playoff_outlook else None,
            byes=byes, upcoming_weeks=tuple(range(week, min(week + 3, final_week + 1))),
            min_my_gain=args.min_gain, min_prob_not_worse=args.min_prob,
            their_tolerance=args.their_tolerance,
            # This CLI *is* the on-demand path: a person ran it, now, against
            # inputs they chose. The weekly card does not call it.
            on_demand=True)
    except league_trades.TradeScanError as exc:
        print(f"[trade-scan] refused: {exc}", file=sys.stderr)
        return 1

    payload = league_trades.scan_to_dict(scan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, default=str) + "\n")

    print(f"[trade-scan] {scan.league['name']} ({scan.league['league_id']}) week {week}; "
          f"playoff weeks {scan.rules['playoff_scoring_periods']}")
    print(f"[trade-scan] identities: {scan.identity['matched']} matched, "
          f"{len(scan.identity['unmatched'])} unmatched, "
          f"{len(scan.identity['ambiguous'])} ambiguous, "
          f"{len(scan.identity['shadow'])} shadow (K/D-ST)")
    for warning in scan.warnings:
        print(f"[trade-scan] WARN {warning}")

    if scan.state == "hold":
        print(f"[trade-scan] {scan.hold_reason}")
    else:
        for package in scan.packages:
            send = ", ".join(f"{p['name']} ({p['position']})" for p in package.mine.sends)
            receive = ", ".join(f"{p['name']} ({p['position']})" for p in package.mine.receives)
            print(f"\n  {package.theirs.team_name}")
            print(f"    send    : {send}")
            print(f"    receive : {receive}")
            print(f"    me      : {package.mine.delta.mean:+.2f} pts "
                  f"[{package.mine.delta.p05:+.2f}, {package.mine.delta.p95:+.2f}], "
                  f"{package.mine.delta.prob_gain:.0%} of sims positive")
            print(f"    them    : {package.theirs.delta.mean:+.2f} pts "
                  f"[{package.theirs.delta.p05:+.2f}, {package.theirs.delta.p95:+.2f}], "
                  f"{package.theirs.delta.prob_gain:.0%} of sims positive")
            print(f"    rosters : mine {package.mine.roster_before}->"
                  f"{package.mine.roster_after}, theirs {package.theirs.roster_before}->"
                  f"{package.theirs.roster_after} (cap {package.mine.legality.roster_cap})")
    print(f"\n[trade-scan] {scan.disclaimer}")
    print(f"[trade-scan] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
