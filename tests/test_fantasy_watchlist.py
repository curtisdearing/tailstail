"""Data-independent tests for league-specific ESPN draft watchlists."""

from __future__ import annotations

import unittest

from nflvalue.fantasy.watchlist import (
    reweight_board_rows,
    snake_pick_numbers,
    watchlist_targets,
)


class WatchlistIntegrationTests(unittest.TestCase):
    def test_reweights_serialized_board_for_eight_team_lineup(self) -> None:
        rows = []
        for position, count, base in (
            ("QB", 12, 300.0),
            ("RB", 35, 260.0),
            ("WR", 35, 250.0),
            ("TE", 12, 190.0),
        ):
            for index in range(count):
                rows.append(
                    {
                        "player_name": f"{position}{index + 1}",
                        "position": position,
                        "season_mean": base - index,
                        "season_p90": base + 40.0 - index,
                        "adp": float(index + 1),
                        "adp_sd": 6.0,
                    }
                )

        board = reweight_board_rows(rows, league_teams=8)
        replacement = {
            position: next(row["replacement_rank"] for row in board if row["position"] == position)
            for position in ("QB", "RB", "WR", "TE")
        }

        self.assertEqual(replacement, {"QB": 10, "RB": 30, "WR": 30, "TE": 12})
        self.assertEqual([row["overall_rank"] for row in board], list(range(1, len(board) + 1)))
        self.assertEqual(board[0]["draft_score"], max(row["draft_score"] for row in board))

    def test_reweight_refreshes_adp_round_and_value_gap(self) -> None:
        rows = [
            {
                "player_name": "Value",
                "position": "RB",
                "season_mean": 100.0,
                "season_p90": 120.0,
                "adp": 17.0,
                "adp_round": 99,
                "value_gap": 99.0,
            },
            {
                "player_name": "Other",
                "position": "RB",
                "season_mean": 90.0,
                "season_p90": 110.0,
                "adp": 9.0,
                "adp_round": 99,
                "value_gap": 99.0,
            },
        ]

        board = reweight_board_rows(rows, league_teams=8)
        by_name = {row["player_name"]: row for row in board}

        self.assertEqual(by_name["Value"]["adp_round"], 3)
        self.assertEqual(by_name["Value"]["value_gap"], 16.0)
        self.assertEqual(by_name["Other"]["adp_round"], 2)
        self.assertEqual(by_name["Other"]["value_gap"], 7.0)

    def test_slot_one_snake_picks_match_live_espn_order(self) -> None:
        self.assertEqual(
            snake_pick_numbers(slot=1, league_teams=8, rounds=16),
            [1, 16, 17, 32, 33, 48, 49, 64, 65, 80, 81, 96, 97, 112, 113, 128],
        )

    def test_watchlist_targets_are_pick_specific_and_unique(self) -> None:
        rows = [
            {
                "player_name": "Early Star",
                "position": "RB",
                "season_mean": 300.0,
                "season_p90": 360.0,
                "vor_mean": 100.0,
                "vor_p90": 130.0,
                "overall_rank": 1,
                "adp": 1.0,
                "adp_sd": 2.0,
            },
            {
                "player_name": "Turn Target",
                "position": "WR",
                "season_mean": 260.0,
                "season_p90": 325.0,
                "vor_mean": 80.0,
                "vor_p90": 115.0,
                "overall_rank": 2,
                "adp": 16.0,
                "adp_sd": 3.0,
            },
            {
                "player_name": "Later Value",
                "position": "TE",
                "season_mean": 220.0,
                "season_p90": 300.0,
                "vor_mean": 70.0,
                "vor_p90": 110.0,
                "overall_rank": 3,
                "adp": 33.0,
                "adp_sd": 3.0,
            },
        ]

        targets = watchlist_targets(
            rows,
            slot=1,
            league_teams=8,
            rounds=5,
            candidates_per_pick=1,
            minimum_availability=0.10,
        )

        self.assertEqual([row["player_name"] for row in targets], ["Early Star", "Turn Target", "Later Value"])
        self.assertEqual([row["target_pick"] for row in targets], [1, 16, 32])
        self.assertEqual(len({row["player_name"] for row in targets}), len(targets))


if __name__ == "__main__":
    unittest.main()
