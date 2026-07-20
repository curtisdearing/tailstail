"""Data-independent tests for the Monte-Carlo audit's headline metric machinery
(nflvalue/fantasy/monte_carlo_audit.py).

These are the functions that produce the shipped audit's central evidence -- the
sample CRPS of the calibrated draws and the paired season-week MAE bootstrap that
decided 'raw event center is worse than the direct ensemble'. They are pinned
here on tiny synthetic slates so a regression in that evidence machinery is
caught without replaying 11,481 player-weeks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflvalue.fantasy import monte_carlo_audit as mca

# --------------------------------------------------------------------------- #
# _sample_crps
# --------------------------------------------------------------------------- #

def test_crps_of_a_point_mass_equals_absolute_error():
    # All draws equal c => pairwise term is 0 => CRPS = |c - actual|.
    n = 500
    points = pd.DataFrame({"a": np.full(n, 10.0), "b": np.full(n, 4.0)})
    slate = pd.DataFrame({"player_id": ["a", "b"], "fantasy_points": [7.0, 4.0]})
    out = mca._sample_crps(points, slate).set_index("player_id")["mc_crps"]
    assert out["a"] == pytest.approx(3.0, abs=1e-9)   # |10 - 7|
    assert out["b"] == pytest.approx(0.0, abs=1e-9)   # |4 - 4|


def test_crps_two_point_matches_closed_form():
    # Two-point distribution: half the mass at 0, half at 2a (mean a).
    # CRPS = E|X-y| - 0.5 E|X-X'|.  For a=5, y=5:
    #   E|X-y|      = 0.5*5 + 0.5*5           = 5.0
    #   E|X-X'|     = 0.5*(2a)               = 5.0  -> 0.5 E|X-X'| = 2.5
    #   CRPS                                  = 2.5
    # This isolates the pairwise sharpness term: dropping it yields 5.0.
    a = 5.0
    half = 2000
    draws = np.concatenate([np.zeros(half), np.full(half, 2 * a)])
    points = pd.DataFrame({"x": draws})
    slate = pd.DataFrame({"player_id": ["x"], "fantasy_points": [a]})
    out = mca._sample_crps(points, slate).set_index("player_id")["mc_crps"]
    assert out["x"] == pytest.approx(2.5, abs=1e-6)


def test_crps_rewards_a_sharp_calibrated_distribution():
    rng = np.random.default_rng(0)
    n = 4000
    actual = 15.0
    sharp = rng.normal(actual, 2.0, size=n)     # tight, centered on truth
    diffuse = rng.normal(actual, 9.0, size=n)   # wide, same center
    points = pd.DataFrame({"sharp": sharp, "diffuse": diffuse})
    slate = pd.DataFrame({"player_id": ["sharp", "diffuse"],
                          "fantasy_points": [actual, actual]})
    out = mca._sample_crps(points, slate).set_index("player_id")["mc_crps"]
    assert out["sharp"] < out["diffuse"]


# --------------------------------------------------------------------------- #
# _metrics
# --------------------------------------------------------------------------- #

def test_metrics_reports_coverage_and_width():
    frame = pd.DataFrame({
        "fantasy_points": [10.0, 20.0, 5.0, 30.0],
        "projection_mean": [11.0, 18.0, 6.0, 25.0],
        "lo": [0.0, 0.0, 0.0, 0.0],
        "hi": [12.0, 12.0, 12.0, 12.0],   # covers rows 0 and 2 only
    })
    m = mca._metrics(frame, "projection_mean", lower="lo", upper="hi")
    assert m["n"] == 4
    assert m["coverage80"] == pytest.approx(0.5)
    assert m["mean_interval_width"] == pytest.approx(12.0)
    assert m["mae"] == pytest.approx(np.mean([1, 2, 1, 5]))


# --------------------------------------------------------------------------- #
# paired_week_mae_comparison
# --------------------------------------------------------------------------- #

def _paired_frame(seed: int = 1, *, candidate_better: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for season in (2023, 2024):
        for week in range(1, 15):
            for _ in range(20):
                actual = rng.uniform(2, 30)
                # reference always misses by ~4; candidate misses by ~1 (better)
                ref = actual + rng.choice([-4.0, 4.0])
                cand = actual + rng.choice([-1.0, 1.0]) if candidate_better else ref
                rows.append({"season": season, "week": week,
                             "fantasy_points": actual,
                             "cand": cand, "ref": ref})
    return pd.DataFrame(rows)


def test_paired_comparison_requires_columns():
    frame = _paired_frame(candidate_better=True).drop(columns=["week"])
    with pytest.raises(ValueError, match="missing columns"):
        mca.paired_week_mae_comparison(frame, "cand", "ref")


def test_paired_comparison_favors_the_better_candidate():
    frame = _paired_frame(candidate_better=True)
    result = mca.paired_week_mae_comparison(frame, "cand", "ref", iterations=3000)
    # candidate MAE minus reference MAE is negative and robustly so
    assert result["mae_delta_candidate_minus_reference"] < 0
    assert result["ci95"][1] < 0
    assert result["probability_candidate_better"] > 0.95
    assert result["weeks"] == 28


def test_paired_comparison_reports_exact_tie():
    frame = _paired_frame(candidate_better=False)   # cand == ref
    result = mca.paired_week_mae_comparison(frame, "cand", "ref", iterations=1000)
    assert result["mae_delta_candidate_minus_reference"] == 0.0
    assert result["probability_tie"] == pytest.approx(1.0)
    assert result["probability_candidate_better"] == 0.0


def test_paired_comparison_is_deterministic():
    frame = _paired_frame(candidate_better=True)
    a = mca.paired_week_mae_comparison(frame, "cand", "ref", iterations=2000)
    b = mca.paired_week_mae_comparison(frame, "cand", "ref", iterations=2000)
    assert a["ci95"] == b["ci95"]
