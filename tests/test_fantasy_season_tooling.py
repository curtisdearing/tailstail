"""Data-independent tests for rest-of-season / lineup / trade / VOR tooling.

These build synthetic ``SeasonSimulation`` and ``SimulationResult`` objects
directly, so no historical parquet frame or model fit is needed.  They pin the
product-core decision logic and its guard rails (missing roster players,
illegal trades, mismatched sample counts).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflvalue.fantasy import season as seas
from nflvalue.fantasy.simulation import SimulationResult

SIMS = 400


def _players():
    # id, name, position, per-sim mean points
    return [
        ("qb1", "Q One", "QB", 22.0),
        ("rb1", "R One", "RB", 18.0),
        ("rb2", "R Two", "RB", 12.0),
        ("rb3", "R Three", "RB", 6.0),
        ("wr1", "W One", "WR", 16.0),
        ("wr2", "W Two", "WR", 11.0),
        ("te1", "T One", "TE", 8.0),
        ("wr3", "W Three", "WR", 4.0),  # bench-quality: forces a real lineup choice
    ]


def _season_simulation(seed: int = 3) -> seas.SeasonSimulation:
    rng = np.random.default_rng(seed)
    players = _players()
    points = pd.DataFrame({
        pid: rng.normal(mu, 4.0, size=SIMS) for pid, _, _, mu in players
    })
    meta = pd.DataFrame(
        [{"player_id": pid, "player_name": nm, "position": pos, "team": "AAA"}
         for pid, nm, pos, _ in players]
    )
    summaries = pd.DataFrame([
        {"player_id": pid, "player_name": nm, "position": pos, "team": "AAA",
         "mean": float(points[pid].mean())}
        for pid, nm, pos, _ in players
    ]).sort_values("mean", ascending=False).reset_index(drop=True)
    return seas.SeasonSimulation(
        summaries=summaries, points=points, player_meta=meta,
        metadata={"simulations": SIMS},
    )


def _weekly_result(seed: int) -> SimulationResult:
    rng = np.random.default_rng(seed)
    players = _players()
    points = pd.DataFrame({
        pid: rng.normal(mu, 4.0, size=SIMS) for pid, _, _, mu in players
    })
    summaries = pd.DataFrame(
        [{"player_id": pid, "player_name": nm, "position": pos, "team": "AAA"}
         for pid, nm, pos, _ in players]
    )
    return SimulationResult(summaries=summaries, points=points, components={},
                            metadata={"simulations": SIMS})


# --------------------------------------------------------------------------- #
# aggregate_rest_of_season
# --------------------------------------------------------------------------- #

def test_aggregate_requires_at_least_one_week():
    with pytest.raises(ValueError, match="at least one"):
        seas.aggregate_rest_of_season({})


def test_aggregate_rejects_mismatched_sample_counts():
    a = _weekly_result(1)
    b = _weekly_result(2)
    b.points = b.points.iloc[: SIMS // 2].reset_index(drop=True)
    b.metadata = {"simulations": SIMS // 2}
    with pytest.raises(ValueError, match="same sample count"):
        seas.aggregate_rest_of_season({1: a, 2: b})


def test_aggregate_sums_weekly_means():
    weekly = {1: _weekly_result(1), 2: _weekly_result(2), 3: _weekly_result(3)}
    combined = seas.aggregate_rest_of_season(weekly)
    qb_mean = combined.summaries.set_index("player_id").loc["qb1", "mean"]
    # three weeks of ~22-point QB expectation
    assert 55.0 < qb_mean < 77.0
    assert combined.metadata["weeks"] == [1, 2, 3]


# --------------------------------------------------------------------------- #
# lineup_points
# --------------------------------------------------------------------------- #

def test_lineup_points_rejects_unknown_roster_player():
    sim = _season_simulation()
    with pytest.raises(ValueError, match="missing from season simulation"):
        seas.lineup_points(sim, ["qb1", "ghost"])


def test_lineup_points_uses_only_starter_slots():
    sim = _season_simulation()
    roster = [pid for pid, *_ in _players()]
    pts = seas.lineup_points(sim, roster)
    # QB(1)+RB(2)+WR(2)+TE(1)+FLEX(1) = 7 starters, but rb3 (weakest) benched:
    # mean lineup should sit below the sum of every player's mean.
    all_mean = float(sim.points[roster].to_numpy().sum(axis=1).mean())
    assert pts.mean() < all_mean
    assert pts.mean() > 0


# --------------------------------------------------------------------------- #
# evaluate_trade
# --------------------------------------------------------------------------- #

def test_trade_rejects_sending_non_roster_player():
    sim = _season_simulation()
    roster = [pid for pid, *_ in _players()]
    with pytest.raises(ValueError, match="not on the current roster"):
        seas.evaluate_trade(sim, roster, send=["ghost"], receive=[])


def test_trade_upgrade_has_positive_expected_delta():
    # Roster with a weak WR2; receive a strong WR by swapping in wr1-level player.
    sim = _season_simulation()
    roster = ["qb1", "rb1", "rb2", "rb3", "wr2", "te1"]  # note: no wr1
    result = seas.evaluate_trade(sim, roster, send=["rb3"], receive=["wr1"])
    assert result["mean_delta"] > 0
    assert 0.0 <= result["probability_trade_improves_lineup"] <= 1.0
    assert result["send"] == ["rb3"] and result["receive"] == ["wr1"]


# --------------------------------------------------------------------------- #
# value_over_replacement
# --------------------------------------------------------------------------- #

def test_value_over_replacement_orders_and_baselines():
    sim = _season_simulation()
    vor = seas.value_over_replacement(sim, league_teams=12)
    assert {"value_over_replacement", "replacement_points", "replacement_rank"} <= set(vor.columns)
    # sorted descending by VOR
    assert vor["value_over_replacement"].is_monotonic_decreasing
    # a starter-quality player beats its positional replacement baseline
    top = vor.iloc[0]
    assert top["value_over_replacement"] >= 0
