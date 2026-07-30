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
