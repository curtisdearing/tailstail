"""Data-independent tests for the fantasy release gate (audit.py) and the
weekly HTML dashboard renderer (dashboard.py).

The release gate is a safety mechanism -- it must FAIL loudly on miscalibrated
intervals or a non-robust improvement claim -- so it is worth pinning under CI
without any historical parquet frame.  All frames here are synthetic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nflvalue.fantasy import audit, dashboard

POSITIONS = ("QB", "RB", "WR", "TE")


def _synthetic_predictions(*, coverage_width_q: float, baseline_offset: float,
                           seed: int = 7) -> pd.DataFrame:
    """A well-behaved synthetic outer-prediction frame.

    ``fantasy_points = projection_mean + noise``; the 80% interval half-width is
    the ``coverage_width_q`` quantile of |noise| so realized coverage is that
    quantile by construction.  Baselines are offset so their error dominates the
    model's, making the paired improvement robustly positive.
    """

    rng = np.random.default_rng(seed)
    n = 1600
    season = rng.choice([2023, 2024], size=n)
    week = rng.integers(1, 18, size=n)
    position = rng.choice(POSITIONS, size=n)
    mean = rng.uniform(2.0, 30.0, size=n)
    noise = rng.normal(0.0, 4.0, size=n)
    actual = mean + noise
    half = np.quantile(np.abs(noise), coverage_width_q)
    return pd.DataFrame({
        "season": season, "week": week, "position": position,
        "projection_mean": mean,
        "fantasy_points": actual,
        "projection_lower80": mean - half,
        "projection_upper80": mean + half,
        "pre_fantasy_points_ewm4": mean + baseline_offset,
        "pre_expected_points_ewm4": mean + baseline_offset,
        "opportunities": rng.uniform(0, 25, size=n),
        "pre_opportunities_ewm4": rng.uniform(0, 25, size=n),
        "total_tds": rng.integers(0, 2, size=n),
    })


# --------------------------------------------------------------------------- #
# paired_week_bootstrap
# --------------------------------------------------------------------------- #

def test_bootstrap_reports_positive_improvement_when_model_beats_baseline():
    preds = _synthetic_predictions(coverage_width_q=0.80, baseline_offset=6.0)
    result = audit.paired_week_bootstrap(preds, "pre_fantasy_points_ewm4", iterations=2000)
    assert result["mae_improvement"] > 0
    assert result["ci95"][0] <= result["ci95"][1]
    assert result["ci95"][0] > 0            # robustly positive
    assert result["probability_nonpositive"] < 0.05
    assert 0.0 <= result["week_win_rate"] <= 1.0


def test_bootstrap_rejects_missing_columns():
    preds = _synthetic_predictions(coverage_width_q=0.80, baseline_offset=6.0)
    with pytest.raises(ValueError, match="missing columns"):
        audit.paired_week_bootstrap(preds.drop(columns=["projection_mean"]),
                                    "pre_fantasy_points_ewm4")


def test_bootstrap_is_deterministic_under_fixed_seed():
    preds = _synthetic_predictions(coverage_width_q=0.80, baseline_offset=6.0)
    a = audit.paired_week_bootstrap(preds, "pre_fantasy_points_ewm4", iterations=1500)
    b = audit.paired_week_bootstrap(preds, "pre_fantasy_points_ewm4", iterations=1500)
    assert a["ci95"] == b["ci95"]


# --------------------------------------------------------------------------- #
# red_team_report / release gate
# --------------------------------------------------------------------------- #

def test_release_gate_passes_on_well_calibrated_frame():
    preds = _synthetic_predictions(coverage_width_q=0.80, baseline_offset=6.0)
    report = audit.red_team_report(preds)
    assert report["release_gate"]["pass"] is True, report["release_gate"]["failures"]
    # regimes and calibration are surfaced for inspection
    assert "role_increase_5_plus" in report["regimes"]
    assert 0.80 <= report["calibration"]["slope"] <= 1.20


def test_release_gate_fails_on_miscalibrated_intervals():
    # Near-zero-width intervals => coverage collapses, gate must fail loudly.
    preds = _synthetic_predictions(coverage_width_q=0.02, baseline_offset=6.0)
    report = audit.red_team_report(preds)
    assert report["release_gate"]["pass"] is False
    assert any("miscalibrated" in f for f in report["release_gate"]["failures"])


def test_release_gate_catches_systematic_bias_even_when_baselines_are_worse():
    # Red-team regression: a projection that under-projects by a constant +5 has
    # calibration slope ~1.0 and can be forced to 80% coverage, so the slope /
    # coverage / relative-MAE checks all pass when the naive baselines are even
    # worse.  The absolute mean-bias gate must still fail it.
    rng = np.random.default_rng(0)
    n = 2000
    mean = rng.uniform(2, 30, n)
    actual = mean + 5.0 + rng.normal(0, 3.0, n)      # systematic +5 bias
    half = float(np.quantile(np.abs(actual - mean), 0.80))
    preds = pd.DataFrame({
        "season": rng.choice([2023, 2024], n), "week": rng.integers(1, 18, n),
        "position": rng.choice(POSITIONS, n),
        "projection_mean": mean, "fantasy_points": actual,
        "projection_lower80": mean - half, "projection_upper80": mean + half,
        "pre_fantasy_points_ewm4": mean + 12.0,       # baselines are even worse
        "pre_expected_points_ewm4": mean + 12.0,
        "opportunities": rng.uniform(0, 25, n),
        "pre_opportunities_ewm4": rng.uniform(0, 25, n),
        "total_tds": rng.integers(0, 2, n),
    })
    report = audit.red_team_report(preds)
    assert report["release_gate"]["pass"] is False
    assert any("systematic mean bias" in f for f in report["release_gate"]["failures"])
    # and it is NOT masked as a mere coverage/slope problem
    assert 0.70 <= report["coverage80"] <= 0.88
    assert 0.80 <= report["calibration"]["slope"] <= 1.20


def test_release_gate_flags_nonrobust_improvement():
    # Baseline equals the model => no real improvement over baseline.
    preds = _synthetic_predictions(coverage_width_q=0.80, baseline_offset=0.0)
    report = audit.red_team_report(preds)
    assert any("improvement over" in f for f in report["release_gate"]["failures"])


# --------------------------------------------------------------------------- #
# dashboard renderer
# --------------------------------------------------------------------------- #

def _summaries() -> pd.DataFrame:
    return pd.DataFrame([
        {"position": "RB", "player_name": "Alpha Back", "team": "AAA",
         "mean": 18.4, "median": 17.9, "event_simulator_mean": 18.0,
         "p10": 8.1, "p90": 29.2, "prob_15_plus": 0.61, "prob_20_plus": 0.42,
         "availability_probability": 0.97, "component_model_disagreement": True},
        {"position": "RB", "player_name": "Beta<script>", "team": "BBB",
         "mean": 22.0, "median": 21.0, "event_simulator_mean": 21.5,
         "p10": 10.0, "p90": 33.0, "prob_15_plus": 0.72, "prob_20_plus": 0.55,
         "availability_probability": 0.90, "component_model_disagreement": False},
        {"position": "QB", "player_name": "Cee Quarterback", "team": "CCC",
         "mean": 24.3, "median": 24.0, "event_simulator_mean": 23.8,
         "p10": 14.0, "p90": 35.0, "prob_15_plus": 0.88, "prob_20_plus": 0.70,
         "availability_probability": 1.0, "component_model_disagreement": False},
    ])


def test_dashboard_writes_file_and_escapes_html(tmp_path: Path):
    out = tmp_path / "nested" / "fantasy.html"
    dashboard.render_fantasy_dashboard(
        _summaries(), out, season=2024, week=5, generated_at="2024-09-01T12:00Z",
    )
    assert out.exists()
    text = out.read_text()
    assert "2024 week 5 fantasy projections" in text
    # user-controlled name with a tag is escaped, never emitted raw
    assert "<script>" not in text
    assert "Beta&lt;script&gt;" in text
    # percentage columns render as integers
    assert "61%" in text and "97%" in text
    # disagreement flag surfaces a review marker
    assert "review" in text


def test_dashboard_orders_by_position_then_descending_mean(tmp_path: Path):
    out = tmp_path / "fantasy.html"
    dashboard.render_fantasy_dashboard(
        _summaries(), out, season=2024, week=5, generated_at="now",
    )
    text = out.read_text()
    # QB sorts before RB; within RB, higher mean first (Beta 22.0 before Alpha 18.4)
    assert text.index("Cee Quarterback") < text.index("Beta")
    assert text.index("Beta") < text.index("Alpha Back")
