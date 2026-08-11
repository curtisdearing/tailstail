"""Tests for the hierarchical empirical-Bayes pooling layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflvalue.fantasy.hierarchy import (
    beta_binomial_posteriors,
    hierarchical_baselines,
    normal_partial_pool,
)


def test_normal_partial_pool_shrinks_low_n_more():
    xbar = pd.Series([20.0, 20.0, 10.0, 11.0, 9.0, 10.5, 9.5, 10.2])
    n = pd.Series([3.0, 15.0, 12.0, 14.0, 10.0, 16.0, 13.0, 11.0])
    within = pd.Series([36.0] * 8)
    post, sd, hyper = normal_partial_pool(xbar, n, within)
    # Both 20-scorers sit above the grand mean; the 3-game one shrinks more.
    assert post.iloc[0] < post.iloc[1]
    assert post.iloc[0] > hyper["grand_mean"]  # still above the pack
    # Posterior lies strictly between observation and grand mean.
    assert hyper["grand_mean"] < post.iloc[1] < 20.0
    # Less data -> more posterior uncertainty.
    assert sd.iloc[0] > sd.iloc[1]


def test_normal_partial_pool_tau_reflects_true_spread():
    rng = np.random.default_rng(0)
    true_means = rng.normal(12.0, 4.0, size=60)          # tau ~ 4
    n = pd.Series(np.full(60, 14.0))
    xbar = pd.Series(true_means + rng.normal(0, np.sqrt(36.0 / 14), size=60))
    within = pd.Series(np.full(60, 36.0))
    _, _, hyper = normal_partial_pool(xbar, n, within)
    assert 8.0 < hyper["tau2"] < 32.0                    # near 16, not the floor


def test_beta_binomial_posterior_orders_availability():
    k = pd.Series([16.0, 7.0, 30.0, 25.0, 14.0, 33.0])
    n = pd.Series([16.0, 33.0, 33.0, 33.0, 16.0, 33.0])
    alpha, beta, hyper = beta_binomial_posteriors(k, n)
    rate = alpha / (alpha + beta)
    assert rate.iloc[0] > rate.iloc[1]                   # iron man > glass cannon
    assert rate.iloc[1] > 7.0 / 33.0                     # pooled upward toward prior
    assert rate.iloc[0] < 1.0                            # never certain
    assert hyper["alpha0"] > 0 and hyper["beta0"] > 0


class _NullModel:
    def predict(self, rows):
        return pd.DataFrame({
            "player_id": rows["player_id"],
            "week": rows["week"],
            "projection_mean": pd.NA,
        })


def _grid(pid, points, played, position="WR"):
    return pd.DataFrame({
        "season": 2025, "player_id": pid, "player_name": pid,
        "position": position, "team": "TST",
        "week": range(1, len(points) + 1),
        "fantasy_points": points, "played": played,
        "birth_date": "1998-01-01", "years_exp": 5, "draft_number": 20,
    })


@pytest.fixture()
def small_frame():
    rng = np.random.default_rng(1)
    frames = []
    for i in range(10):
        pts = list(np.clip(rng.normal(8 + i, 5, size=17), 0, None))
        frames.append(_grid(f"wr{i}", pts, [1] * 17))
    frames.append(_grid("glass", list(np.clip(np.r_[np.full(6, 22.0), np.zeros(11)]
                                              + rng.normal(0, 2, 17), 0, None)),
                        [1] * 6 + [0] * 11))
    frame = pd.concat(frames, ignore_index=True)
    frame["week"] = frame["week"].astype(int)
    return frame


def test_hierarchical_baselines_end_to_end(small_frame):
    out = hierarchical_baselines(
        small_frame, _NullModel(), source_season=2025, model_weight=0.0
    ).set_index("player_id")
    assert out["basis"].str.contains("hier_bayes").all()
    assert {"mu_post_sd", "avail_alpha", "avail_beta"} <= set(out.columns)
    # Glass cannon: high per-game mu preserved (not diluted), low availability.
    assert out.loc["glass", "mu_pergame"] > out.loc["wr5", "mu_pergame"]
    assert out.loc["glass", "availability_rate"] < out["availability_rate"].drop("glass").min()
    # Fewer games -> wider mu posterior.
    assert out.loc["glass", "mu_post_sd"] > out.loc["wr9", "mu_post_sd"]


def test_simulation_consumes_posterior_uncertainty(small_frame):
    from nflvalue.fantasy.draft import simulate_season

    base = hierarchical_baselines(
        small_frame, _NullModel(), source_season=2025, model_weight=0.0
    )
    byes = {"TST": [8]}
    with_unc = simulate_season(base, byes, simulations=1200, random_seed=5)
    flat = base.drop(columns=["mu_post_sd", "avail_alpha", "avail_beta"])
    without = simulate_season(flat, byes, simulations=1200, random_seed=5)
    spread_with = (with_unc.board["season_p90"] - with_unc.board["season_p10"]).mean()
    spread_without = (without.board["season_p90"] - without.board["season_p10"]).mean()
    assert spread_with > spread_without  # parameter uncertainty widens outcomes
