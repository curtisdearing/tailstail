"""Data-independent tests for the draft board and trade planner."""

from __future__ import annotations

import pandas as pd
import pytest

from nflvalue.fantasy.draft import (
    AGE_CURVES,
    _age_multiplier,
    add_rookie_market_priors,
    apply_offseason_adjustments,
    draft_board,
    normalize_name,
    simulate_season,
    snake_picks,
)
from nflvalue.fantasy.season import SeasonSimulation
from nflvalue.fantasy.trade_planner import (
    LeagueRosters,
    bye_pressure,
    market_temperature,
    match_players,
    propose_trades,
)


def _baselines(n_per_pos: int = 8) -> pd.DataFrame:
    rows = []
    for position, base in (("QB", 18.0), ("RB", 14.0), ("WR", 13.0), ("TE", 9.0)):
        for i in range(n_per_pos):
            mu = base + (n_per_pos - i) * 1.3
            rows.append({
                "player_id": f"{position}{i}",
                "player_name": f"{position} Player{i}",
                "position": position,
                "team": ["DET", "ATL", "BUF", "KC"][i % 4],
                "games_played": 15,
                "mu_pergame_raw": mu,
                "mu_pergame": mu,
                "sigma_own": 5.0,
                "sigma_pergame": 5.0,
                "age": 25.0,
                "draft_number": 50,
                "availability_rate": 0.9,
                "age_multiplier": 1.0,
                "team_changed": False,
                "basis": "model+realized",
            })
    return pd.DataFrame(rows)


BYES = {"DET": [8], "ATL": [5], "BUF": [7], "KC": [10]}


def test_normalize_name_strips_suffixes_and_punctuation():
    assert normalize_name("James Cook Jr.") == "james cook"
    assert normalize_name("Kyle Pitts Sr.") == "kyle pitts"
    assert normalize_name("Amon-Ra St. Brown") == normalize_name("AmonRa St Brown")


def test_age_multiplier_uses_strongest_tier():
    # 30-year-old RB hits the last threshold, not the first.
    assert _age_multiplier("RB", 30.5) == AGE_CURVES["RB"][-1][1]
    assert _age_multiplier("RB", 22.0) == 1.0
    assert _age_multiplier("K", 40.0) == 1.0  # unknown position untouched


def test_simulate_season_shapes_and_bye_effect():
    baselines = _baselines()
    outlook = simulate_season(baselines, BYES, simulations=400, random_seed=1)
    assert len(outlook.board) == len(baselines)
    assert outlook.season_points.shape == (400, len(baselines))
    # A player cannot exceed 16 games (17 weeks minus the bye).
    assert (outlook.board["expected_games"] <= 16.0 + 1e-9).all()
    # Season mean should scale with per-game mu.
    top = outlook.board.sort_values("mu_pergame", ascending=False).iloc[0]
    bottom = outlook.board.sort_values("mu_pergame").iloc[0]
    assert top["season_mean"] > bottom["season_mean"]


def test_draft_board_replacement_and_ceiling_weight():
    outlook = simulate_season(_baselines(), BYES, simulations=400, random_seed=2)
    neutral = draft_board(outlook, league_teams=6, ceiling_weight=0.0)
    ceiling = draft_board(outlook, league_teams=6, ceiling_weight=1.0)
    assert (neutral["draft_score"] - neutral["vor_mean"]).abs().max() < 1e-9
    assert (ceiling["draft_score"] - ceiling["vor_p90"]).abs().max() < 1e-9
    # Onesie positions use shallower replacement than RB/WR.
    qb_rank = neutral.loc[neutral["position"] == "QB", "replacement_rank"].iloc[0]
    rb_rank = neutral.loc[neutral["position"] == "RB", "replacement_rank"].iloc[0]
    assert qb_rank < rb_rank


def test_snake_picks_math():
    picks = snake_picks(1, league_teams=6, rounds=4)
    assert picks == [1, 12, 13, 24]
    picks = snake_picks(6, league_teams=6, rounds=4)
    assert picks == [6, 7, 18, 19]
    with pytest.raises(ValueError):
        snake_picks(7, league_teams=6)


def test_offseason_adjustment_flags_team_change():
    baselines = _baselines()
    moved = apply_offseason_adjustments(
        baselines, current_teams={"rb player0": "SEA"}
    )
    row = moved[moved["player_id"] == "RB0"].iloc[0]
    assert row["team_changed"] and row["team"] == "SEA"
    assert "team_change_prior" in row["basis"]
    # Sigma widened, mu haircut applied.
    original = baselines[baselines["player_id"] == "RB0"].iloc[0]
    assert row["sigma_pergame"] > original["sigma_pergame"]
    assert row["mu_pergame"] < original["mu_pergame"]


def test_rookie_market_prior_added_and_labeled():
    baselines = _baselines()
    adp = pd.DataFrame([
        {"name": "RB Player0", "position": "RB", "team": "DET", "adp": 5.0, "adp_sd": 4.0},
        {"name": "Rookie Star", "position": "WR", "team": "NYG", "adp": 12.0, "adp_sd": 6.0},
    ])
    combined = add_rookie_market_priors(baselines, adp)
    rookie = combined[combined["player_name"] == "Rookie Star"]
    assert len(rookie) == 1
    assert rookie.iloc[0]["basis"] == "rookie_market_prior"
    assert rookie.iloc[0]["sigma_pergame"] > baselines["sigma_pergame"].median()
    # Veteran not duplicated.
    assert (combined["player_name"] == "RB Player0").sum() == 1


def _league(outlook) -> tuple[SeasonSimulation, LeagueRosters, pd.DataFrame]:
    board = outlook.board
    meta = board[["player_id", "player_name", "position", "team"]]
    season = SeasonSimulation(
        summaries=board[["player_id", "player_name", "position", "team", "season_mean"]]
        .rename(columns={"season_mean": "mean"}),
        points=outlook.season_points,
        player_meta=meta,
        metadata={},
    )
    names = board.set_index("player_id")["player_name"]
    mine = ["QB0", "RB4", "RB5", "WR0", "WR1", "TE0", "RB6"]
    theirs = ["QB1", "RB0", "RB1", "WR4", "WR5", "TE1", "RB2"]
    rosters = LeagueRosters(
        teams={
            "Me": [names[p] for p in mine],
            "Them": [names[p] for p in theirs],
        },
        my_team="Me",
    )
    return season, rosters, board


def test_propose_trades_two_sided_gate():
    outlook = simulate_season(_baselines(), BYES, simulations=400, random_seed=3)
    season, rosters, board = _league(outlook)
    proposals = propose_trades(
        season, rosters, board, min_my_gain=0.1, top_candidates=7,
        byes=BYES, upcoming_weeks=(5, 6, 7),
    )
    if not proposals.empty:
        assert (proposals["my_gain_per_sim"] >= 0.1).all()
        assert (proposals["their_gain_per_sim"] >= 0.0).all()
        assert set(proposals["opponent"]) == {"Them"}


def test_match_players_loose_fallback():
    board = pd.DataFrame({
        "player_id": ["1", "2"],
        "player_name": ["Justin Jefferson", "Bijan Robinson"],
    })
    ids, unmatched = match_players(["J. Jefferson", "Bijan Robinson", "Nobody Real"], board)
    assert ids == ["1", "2"]
    assert unmatched == ["Nobody Real"]


def test_bye_pressure_and_market_temperature():
    outlook = simulate_season(_baselines(), BYES, simulations=200, random_seed=4)
    board = outlook.board
    # DET players share the week-8 bye.
    det = board[board["team"] == "DET"]["player_id"].tolist()[:2]
    assert bye_pressure(det, board, BYES, (8,)) == 2
    assert bye_pressure(det, board, BYES, (9,)) == 0
    temperature = market_temperature(board, {det[0]: 99.0})
    assert temperature[det[0]] > 0  # scorching hot vs any mu


def test_fast_lineup_matches_reference_exactly():
    from nflvalue.fantasy.config import LineupRules
    from nflvalue.fantasy.season import lineup_points
    from nflvalue.fantasy.trade_planner import _FastLineup

    outlook = simulate_season(_baselines(), BYES, simulations=150, random_seed=7)
    board = outlook.board
    season = SeasonSimulation(
        summaries=board[["player_id", "player_name", "position", "team", "season_mean"]]
        .rename(columns={"season_mean": "mean"}),
        points=outlook.season_points,
        player_meta=board[["player_id", "player_name", "position", "team"]],
        metadata={},
    )
    rules = LineupRules()
    fast = _FastLineup(season, rules)
    rosters = [
        ["QB0", "RB0", "RB1", "WR0", "WR1", "TE0", "RB2", "WR2"],  # deep
        ["QB0", "RB0", "WR0", "TE0"],                              # short-handed
        ["QB0", "QB1", "RB0", "RB1", "RB2", "TE0", "TE1"],         # no WR at all
    ]
    for roster in rosters:
        reference = float(lineup_points(season, roster, rules).mean())
        assert abs(fast.mean(roster) - reference) < 1e-9


def test_round_ceiling_weight_progression():
    from nflvalue.fantasy.draft import round_ceiling_weight

    assert round_ceiling_weight(1) == pytest.approx(0.30)
    assert round_ceiling_weight(4) == pytest.approx(0.54)
    assert round_ceiling_weight(12) == pytest.approx(0.80)  # capped
    assert round_ceiling_weight(2) > round_ceiling_weight(1)
    with pytest.raises(ValueError):
        round_ceiling_weight(0)


def test_availability_probability_direction():
    """Regression: the sign was flipped once — an ADP-1.7 star showed as
    'available' at pick 24 and unavailable at pick 1."""
    import pandas as pd

    from nflvalue.fantasy.draft import availability_probability

    board = pd.DataFrame({
        "player_name": ["Star", "Mid", "Late"],
        "adp": [1.7, 30.0, 90.0],
        "adp_sd": [4.0, 8.0, 15.0],
    })
    at_pick_1 = availability_probability(board, 1)
    at_pick_24 = availability_probability(board, 24)
    assert at_pick_1.iloc[0] > 0.5          # star nearly always there at 1.01
    assert at_pick_24.iloc[0] < 0.01        # and certainly gone at pick 24
    assert at_pick_24.iloc[2] > 0.99        # ADP-90 player still on the board
    assert at_pick_1.iloc[0] < at_pick_1.iloc[2]  # monotone in ADP distance


def test_pergame_baselines_use_played_flag_not_notna():
    """Regression: the feature frame is a roster-week grid (fantasy_points is
    never NaN; DNP weeks are exact zeros). Counting notna() weeks as games
    deflated per-game means for injury-prone players (19.7 PPG -> 6.9) and
    erased injury history from availability_rate."""
    import pandas as pd

    from nflvalue.fantasy.draft import pergame_baselines

    class _NullModel:
        def predict(self, rows):
            return pd.DataFrame({
                "player_id": rows["player_id"],
                "week": rows["week"],
                "projection_mean": pd.NA,
            })

    def _rows(pid, points, played):
        return pd.DataFrame({
            "season": 2025, "player_id": pid, "player_name": pid,
            "position": "WR", "team": "TST",
            "week": range(1, len(points) + 1),
            "fantasy_points": points, "played": played,
            "birth_date": "1998-01-01", "years_exp": 5, "draft_number": 20,
        })

    frame = pd.concat([
        _rows("ironman", [12.0] * 17, [1] * 17),
        # 7 games at 20.0, then 10 DNP roster weeks logged as 0.0
        _rows("glass", [20.0] * 7 + [0.0] * 10, [1] * 7 + [0] * 10),
    ], ignore_index=True)
    frame["week"] = frame["week"].astype(int)

    out = pergame_baselines(frame, _NullModel(), source_season=2025).set_index("player_id")
    assert out.loc["glass", "games_played"] == 7
    assert out.loc["glass", "mu_pergame_raw"] > 15.0  # not diluted by DNP zeros
    assert out.loc["ironman", "games_played"] == 17
    assert out.loc["glass", "availability_rate"] < out.loc["ironman", "availability_rate"] - 0.15


def test_mock_draft_rosters_are_legal_and_disjoint():
    import numpy as np
    import pandas as pd

    from nflvalue.fantasy.config import LineupRules
    from nflvalue.fantasy.mock_draft import POSITION_CAPS, simulate_draft

    rng = np.random.default_rng(0)
    n = 220
    positions = (["QB"] * 30) + (["RB"] * 70) + (["WR"] * 90) + (["TE"] * 30)
    board = pd.DataFrame({
        "player_id": [f"p{i}" for i in range(n)],
        "player_name": [f"Player {i}" for i in range(n)],
        "position": positions[:n],
        "vor_mean": np.linspace(120, -20, n),
        "vor_p90": np.linspace(160, -10, n),
        "overall_rank": range(1, n + 1),
        "adp": np.arange(1, n + 1, dtype=float),
        "adp_sd": 6.0,
    }).sample(frac=1.0, random_state=1).reset_index(drop=True)

    rosters = simulate_draft(board, my_slot=5, league_teams=12, rounds=14, rng=rng)
    all_picks = [p for picks in rosters.values() for p in picks]
    assert len(all_picks) == len(set(all_picks)) == 12 * 14
    rules = LineupRules()
    for slot, picks in rosters.items():
        counts = board.loc[picks, "position"].value_counts()
        for pos, cap in POSITION_CAPS.items():
            assert counts.get(pos, 0) <= cap, f"slot {slot} exceeds {pos} cap"
        # every team can field a full starting lineup
        for pos, needed in rules.starters.items():
            if pos != "FLEX":
                assert counts.get(pos, 0) >= needed, f"slot {slot} missing {pos}"


def test_best_lineup_matches_bruteforce_on_flex():
    import itertools

    import numpy as np

    from nflvalue.fantasy.config import LineupRules
    from nflvalue.fantasy.mock_draft import _best_lineup

    rules = LineupRules()
    rng = np.random.default_rng(3)
    positions = np.array(["QB", "QB", "RB", "RB", "RB", "WR", "WR", "WR", "TE", "TE"])
    for _ in range(25):
        points = rng.normal(10, 6, size=len(positions))
        got = _best_lineup(points, positions, rules)
        # brute force: choose 1QB,2RB,2WR,1TE + 1 flex from remaining RB/WR/TE
        best = -1e9
        idx = np.arange(len(positions))
        for qb in itertools.combinations(idx[positions == "QB"], 1):
            for rb in itertools.combinations(idx[positions == "RB"], 2):
                for wr in itertools.combinations(idx[positions == "WR"], 2):
                    for te in itertools.combinations(idx[positions == "TE"], 1):
                        base = set(qb) | set(rb) | set(wr) | set(te)
                        flex_pool = [i for i in idx if i not in base
                                     and positions[i] in rules.flex_positions]
                        for fx in flex_pool:
                            best = max(best, points[list(base) + [fx]].sum())
        assert got <= best + 1e-9  # greedy never claims more than optimum
        assert got >= best - 1e-6 or (best - got) / max(abs(best), 1) < 0.02
