import pandas as pd

from nflvalue.fantasy.config import LineupRules
from nflvalue.fantasy.season import SeasonSimulation, evaluate_trade, lineup_points


def _season():
    points = pd.DataFrame({
        "qb": [20, 20, 20], "rb1": [15, 15, 15], "rb2": [10, 10, 10],
        "rb3": [3, 30, 3], "wr1": [12, 12, 12], "wr2": [9, 9, 9],
        "te": [7, 7, 7], "new": [18, 18, 18],
    })
    positions = {"qb": "QB", "rb1": "RB", "rb2": "RB", "rb3": "RB",
                 "wr1": "WR", "wr2": "WR", "te": "TE", "new": "RB"}
    meta = pd.DataFrame([
        {"player_id": player, "player_name": player, "position": position, "team": "AAA"}
        for player, position in positions.items()
    ])
    return SeasonSimulation(pd.DataFrame(), points, meta, {"simulations": 3})


def test_lineup_optimization_values_boom_weeks_not_static_ranks():
    roster = ["qb", "rb1", "rb2", "rb3", "wr1", "wr2", "te"]
    values = lineup_points(_season(), roster, LineupRules())
    assert values.tolist() == [76, 103, 76]


def test_trade_is_evaluated_by_lineup_delta_distribution():
    roster = ["qb", "rb1", "rb2", "rb3", "wr1", "wr2", "te"]
    result = evaluate_trade(_season(), roster, ["rb2"], ["new"])
    assert result["mean_delta"] > 0
    assert result["probability_trade_improves_lineup"] == 1
