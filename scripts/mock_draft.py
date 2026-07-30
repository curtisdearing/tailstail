"""Evaluate every draft slot via mock drafts against ADP-drafting opponents.

Usage:
    python scripts/mock_draft.py [--board data/draft_board_2026_12team.csv]
        [--samples data/draft_board_2026_12team_season_samples.parquet]
        [--teams 12] [--drafts 40] [--sims 250] [--output reports/mock_draft_2026.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nflvalue.fantasy.mock_draft import evaluate_slot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", default="data/draft_board_2026_12team.csv")
    parser.add_argument(
        "--samples", default="data/draft_board_2026_12team_season_samples.parquet"
    )
    parser.add_argument("--teams", type=int, default=12)
    parser.add_argument("--drafts", type=int, default=40)
    parser.add_argument("--sims", type=int, default=250)
    parser.add_argument("--output", default="reports/mock_draft_2026.json")
    args = parser.parse_args()

    board = pd.read_csv(args.board)
    samples = pd.read_parquet(args.samples)

    results = {}
    for slot in range(1, args.teams + 1):
        result = evaluate_slot(
            board, samples, slot,
            league_teams=args.teams, n_drafts=args.drafts,
            sims_per_draft=args.sims,
        )
        results[slot] = {
            "expected_points": round(result.expected_points, 1),
            "title_probability": round(result.title_probability, 4),
            "round_targets": {
                r: [(name, round(freq, 2)) for name, freq in targets]
                for r, targets in result.round_targets.items()
            },
        }
        print(
            f"slot {slot:>2}: {results[slot]['expected_points']:>7} pts, "
            f"P(title)={results[slot]['title_probability']:.3f}, "
            f"R1={result.round_targets[1][0][0] if result.round_targets[1] else '?'}"
        )

    baseline = 1.0 / args.teams
    report = {
        "league_teams": args.teams,
        "drafts_per_slot": args.drafts,
        "sims_per_draft": args.sims,
        "baseline_title_probability": baseline,
        "note": (
            "Opponents draft best-available by noisy ADP with positional sanity; "
            "we draft by progressive-ceiling round score. Lineups scored on season "
            "totals (documented approximation, symmetric across teams)."
        ),
        "slots": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
