import numpy as np
import pandas as pd

from nflvalue.fantasy.config import SimulationConfig
from nflvalue.fantasy.simulation import simulate_week


def _slate():
    rows = []
    for team, opponent, spread in (("AAA", "BBB", -3), ("BBB", "AAA", 3)):
        for suffix, name, position, mean, target, carry in (
            ("1", f"{team} QB", "QB", 18, 0.01, 0.10),
            ("2", f"{team} WR", "WR", 14, 0.25, 0.01),
            ("3", f"{team} RB", "RB", 13, 0.12, 0.55),
        ):
            rows.append({
                "player_id": f"{team}-{suffix}", "player_name": name, "position": position,
                "team": team, "opponent_team": opponent, "game_id": "GAME",
                "projection_mean": mean, "projection_lower80": mean - 8,
                "projection_upper80": mean + 9, "team_spread": spread,
                "implied_team_points": 24 if spread < 0 else 21,
                "pre_team_pass_attempts_ewm4": 34, "pre_team_rush_attempts_ewm4": 27,
                "pre_target_share_calc_ewm4": target, "pre_carry_share_ewm4": carry,
                "pre_catch_rate_ewm8": 0.67, "pre_yards_per_target_ewm8": 8,
                "pre_yards_per_carry_ewm8": 4.5, "pre_td_per_opportunity_ewm8": 0.04,
                "pre_interception_rate_ewm8": 0.025, "wind": 5,
                "status_inactive": 0, "injury_out": 0, "injury_doubtful": 0,
                "injury_questionable": 0, "practice_dnp": 0, "practice_limited": 0,
            })
    return pd.DataFrame(rows)


def test_simulation_is_reproducible_centered_and_correlated():
    config = SimulationConfig(simulations=1500, random_seed=9)
    first = simulate_week(_slate(), config=config)
    second = simulate_week(_slate(), config=config)
    assert np.allclose(first.points, second.points)
    means = first.summaries.set_index("player_id")["mean"]
    expected = _slate().set_index("player_id")["projection_mean"]
    assert np.allclose(means.sort_index(), expected.sort_index(), atol=0.05)
    widths = first.summaries.set_index("player_id")["p90"] - first.summaries.set_index("player_id")["p10"]
    assert np.allclose(widths, 17.0, atol=0.15)
    assert {"expected_targets", "expected_carries", "component_model_disagreement"}.issubset(
        first.summaries.columns
    )
    assert first.points[["AAA-1", "AAA-2"]].corr().iloc[0, 1] > 0
    modeled_qb = first.components["attempts"]["AAA-1"] > 0
    assert np.all(
        first.components["receiving_tds"].loc[modeled_qb, ["AAA-2", "AAA-3"]].sum(axis=1)
        <= first.components["passing_tds"].loc[modeled_qb, "AAA-1"]
    )


def test_official_inactive_has_zero_mass():
    slate = _slate()
    slate.loc[slate.player_id.eq("AAA-2"), "status_inactive"] = 1
    result = simulate_week(slate, config=SimulationConfig(simulations=500, random_seed=3))
    assert (result.points["AAA-2"] == 0).all()
    assert result.summaries.set_index("player_id").loc["AAA-2", "availability_probability"] == 0


def test_low_negative_center_survives_floor_and_availability_hurdle():
    slate = _slate()
    slate["projection_mean"] = slate["projection_mean"].astype(float)
    row = slate.player_id.eq("AAA-2")
    slate.loc[row, "projection_mean"] = -1.75
    slate.loc[row, "projection_lower80"] = -10.0
    slate.loc[row, "projection_upper80"] = 8.0
    slate.loc[row, "practice_dnp"] = 1
    result = simulate_week(slate, config=SimulationConfig(simulations=1_000, random_seed=17))
    summary = result.summaries.set_index("player_id").loc["AAA-2"]
    assert np.isclose(summary["mean"], -1.75, atol=1e-8)
    assert 0 < summary["availability_probability"] < 1


def test_degenerate_backup_role_uses_explicit_residual_uncertainty():
    slate = _slate()
    backup = slate[slate.player_id.eq("AAA-1")].copy()
    backup["player_id"] = "AAA-backup"
    backup["player_name"] = "AAA Backup QB"
    backup["projection_mean"] = -1.0
    backup["projection_lower80"] = -11.0
    backup["projection_upper80"] = 9.0
    backup["pre_target_share_calc_ewm4"] = 0.0
    backup["pre_carry_share_ewm4"] = 0.0
    slate = pd.concat([slate, backup], ignore_index=True)
    result = simulate_week(slate, config=SimulationConfig(simulations=1_000, random_seed=23))
    summary = result.summaries.set_index("player_id").loc["AAA-backup"]
    assert summary["calibration_residual_fallback"]
    assert np.isclose(summary["p90"] - summary["p10"], 20.0, atol=0.2)
    assert np.isclose(summary["mean"], -1.0, atol=1e-8)
