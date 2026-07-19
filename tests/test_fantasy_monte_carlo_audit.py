import numpy as np
import pandas as pd
import pytest

from nflvalue.fantasy.monte_carlo_audit import (
    SIMULATION_INPUT_COLUMNS,
    historical_monte_carlo_replay,
    paired_week_mae_comparison,
    render_monte_carlo_markdown,
)


def _history() -> tuple[pd.DataFrame, pd.DataFrame]:
    roster = [
        ("a_qb", "A QB", "QB", "AAA", 19.0),
        ("a_rb", "A RB", "RB", "AAA", 13.0),
        ("a_wr", "A WR", "WR", "AAA", 14.0),
        ("a_te", "A TE", "TE", "AAA", 8.0),
        ("b_qb", "B QB", "QB", "BBB", 17.0),
        ("b_rb", "B RB", "RB", "BBB", 12.0),
        ("b_wr", "B WR", "WR", "BBB", 13.0),
        ("b_te", "B TE", "TE", "BBB", 7.0),
    ]
    prediction_rows = []
    feature_rows = []
    for week in (1, 2):
        for index, (player_id, player_name, position, team, center) in enumerate(roster):
            actual = center + ((index + week) % 3 - 1) * 2.0
            prediction_rows.append(
                {
                    "season": 2025,
                    "week": week,
                    "player_id": player_id,
                    "player_name": player_name,
                    "position": position,
                    "team": team,
                    "fantasy_points": actual,
                    "projection_mean": center,
                    "projection_lower80": center - 7.0,
                    "projection_upper80": center + 7.0,
                    "total_tds": int(actual > center),
                    "opportunities": 12 + index,
                    "pre_opportunities_ewm4": 11 + index,
                    "team_changed": 0,
                    "qb_changed": 0,
                    "injury_questionable": 0,
                    "practice_dnp": 0,
                }
            )
            is_qb = position == "QB"
            is_receiver = position in {"RB", "WR", "TE"}
            feature_rows.append(
                {
                    "season": 2025,
                    "week": week,
                    "player_id": player_id,
                    "game_id": f"2025_{week}_AAA_BBB",
                    "opponent_team": "BBB" if team == "AAA" else "AAA",
                    "team_spread": -2.0 if team == "AAA" else 2.0,
                    "wind": 8.0,
                    "implied_team_points": 24.0 if team == "AAA" else 21.0,
                    "pre_team_pass_attempts_ewm4": 34.0,
                    "pre_team_rush_attempts_ewm4": 27.0,
                    "pre_target_share_calc_ewm4": 0.18 if is_receiver else 0.005,
                    "pre_carry_share_ewm4": 0.30 if position == "RB" else 0.08 if is_qb else 0.01,
                    "pre_catch_rate_ewm8": 0.66,
                    "pre_yards_per_target_ewm8": 7.8,
                    "pre_yards_per_carry_ewm8": 4.5,
                    "pre_td_per_opportunity_ewm8": 0.04,
                    "pre_interception_rate_ewm8": 0.025,
                    "status_inactive": 0,
                    "injury_out": 0,
                    "injury_doubtful": 0,
                    "practice_limited": 0,
                    "attempts": 32 if is_qb else 0,
                    "completions": 21 if is_qb else 0,
                    "passing_yards": 235 if is_qb else 0,
                    "passing_tds": 1 if is_qb else 0,
                    "passing_interceptions": 1 if is_qb else 0,
                    "carries": 14 if position == "RB" else 4 if is_qb else 1 if position == "WR" else 0,
                    "rushing_yards": 62 if position == "RB" else 20 if is_qb else 3 if position == "WR" else 0,
                    "rushing_tds": 0,
                    "targets": 7 if is_receiver else 0,
                    "receptions": 5 if is_receiver else 0,
                    "receiving_yards": 58 if is_receiver else 0,
                    "receiving_tds": int(position == "WR" and week == 2),
                    "fumbles_lost": 0,
                }
            )
    return pd.DataFrame(feature_rows), pd.DataFrame(prediction_rows)


def test_full_historical_replay_is_complete_reproducible_and_leakage_safe():
    frame, predictions = _history()
    replayed, report = historical_monte_carlo_replay(
        frame,
        predictions,
        simulations=100,
        random_seed=77,
        bootstrap_iterations=200,
    )
    repeated, repeated_report = historical_monte_carlo_replay(
        frame,
        predictions,
        simulations=100,
        random_seed=77,
        bootstrap_iterations=200,
    )

    assert len(replayed) == len(predictions) == 16
    assert report["metadata"]["season_weeks"] == 2
    assert report["metadata"]["total_player_draws"] == 1_600
    assert report["metadata"]["independent_sample_unit"].startswith("season-week")
    assert report["methods"]["direct_ensemble"]["n"] == 16
    assert report["component_accuracy"]["attempts"]["n"] == 4
    assert set(report["metadata"]["simulation_input_columns"]).issubset(SIMULATION_INPUT_COLUMNS)
    assert "fantasy_points" not in report["metadata"]["simulation_input_columns"]
    assert "passing_yards" not in report["metadata"]["simulation_input_columns"]
    assert np.allclose(replayed["mc_calibrated_mean"], repeated["mc_calibrated_mean"])
    assert report["methods"] == repeated_report["methods"]
    assert report["release_gate"]["max_calibrated_center_drift"] < 0.05
    assert "player-draw count is computation" in render_monte_carlo_markdown(report)


def test_duplicate_player_week_is_rejected():
    frame, predictions = _history()
    duplicated = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate player-weeks"):
        historical_monte_carlo_replay(frame, duplicated, simulations=100)


def test_paired_bootstrap_resamples_weeks_and_uses_signed_delta():
    frame = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 1, 2, 2],
            "fantasy_points": [10.0, 20.0, 5.0, 15.0],
            "candidate": [10.0, 20.0, 5.0, 15.0],
            "reference": [13.0, 17.0, 8.0, 12.0],
        }
    )
    result = paired_week_mae_comparison(
        frame, "candidate", "reference", iterations=500, random_seed=9
    )
    assert result["n"] == 4
    assert result["weeks"] == 2
    assert result["mae_delta_candidate_minus_reference"] == -3.0
    assert result["probability_candidate_better"] == 1.0
    assert result["probability_tie"] == 0.0


def test_paired_bootstrap_reports_numerical_identity_as_a_tie():
    frame = pd.DataFrame(
        {
            "season": [2025, 2025],
            "week": [1, 1],
            "fantasy_points": [4.0, 8.0],
            "candidate": [5.0, 7.0],
            "reference": [5.0 + 1e-14, 7.0 - 1e-14],
        }
    )
    result = paired_week_mae_comparison(
        frame, "candidate", "reference", iterations=100, random_seed=3
    )
    assert result["mae_delta_candidate_minus_reference"] == 0.0
    assert result["probability_candidate_better"] == 0.0
    assert result["probability_tie"] == 1.0
