"""Adversarial evaluation for projections, intervals, and claimed improvement."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .models import evaluate_predictions

# Max tolerated pooled systematic bias (fantasy points).  The gated calibration
# SLOPE cannot see a constant offset (a +5 under-projection still fits slope 1.0),
# so a globally biased model could pass on relative MAE alone when the naive
# baselines are even worse.  This absolute check closes that false-pass.
MAX_MEAN_BIAS = 1.5


def paired_week_bootstrap(
    predictions: pd.DataFrame,
    baseline: str,
    *,
    iterations: int = 20_000,
    random_seed: int = 6102026,
) -> dict[str, float | list[float] | int]:
    """Paired block bootstrap; every player in a season-week stays together."""

    required = {"season", "week", "fantasy_points", "projection_mean", baseline}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"bootstrap frame missing columns: {missing}")
    valid = predictions[list(required)].dropna()
    valid = valid.assign(
        model_error=(valid["fantasy_points"] - valid["projection_mean"]).abs(),
        baseline_error=(valid["fantasy_points"] - valid[baseline]).abs(),
    )
    blocks = valid.groupby(["season", "week"])[["model_error", "baseline_error"]].apply(
        lambda group: pd.Series({
            "n": len(group),
            "difference_sum": (group["baseline_error"] - group["model_error"]).sum(),
        }),
    ).reset_index()
    rng = np.random.default_rng(random_seed)
    sampled = rng.integers(0, len(blocks), size=(iterations, len(blocks)))
    n = blocks["n"].to_numpy()[sampled].sum(axis=1)
    improvement = blocks["difference_sum"].to_numpy()[sampled].sum(axis=1) / n
    observed = float(blocks["difference_sum"].sum() / blocks["n"].sum())
    return {
        "n": int(len(valid)),
        "weeks": int(len(blocks)),
        "mae_improvement": observed,
        "ci95": [float(value) for value in np.quantile(improvement, [0.025, 0.975])],
        "probability_nonpositive": float(np.mean(improvement <= 0)),
        "week_win_rate": float(np.mean(blocks["difference_sum"] > 0)),
    }


def _regime(group: pd.DataFrame) -> dict[str, float | int]:
    error = group["fantasy_points"] - group["projection_mean"]
    covered = (
        group["fantasy_points"].ge(group["projection_lower80"])
        & group["fantasy_points"].le(group["projection_upper80"])
    )
    return {
        "n": int(len(group)),
        "mae": float(error.abs().mean()),
        "bias_actual_minus_prediction": float(error.mean()),
        "coverage80": float(covered.mean()),
    }


def red_team_report(predictions: pd.DataFrame) -> dict[str, Any]:
    """Report where a good pooled score is concealing brittle behavior."""

    report = evaluate_predictions(predictions)
    report["paired_bootstrap"] = {}
    for baseline in ("pre_fantasy_points_ewm4", "pre_expected_points_ewm4"):
        if baseline in predictions:
            report["paired_bootstrap"][baseline] = paired_week_bootstrap(predictions, baseline)
    actual = predictions["fantasy_points"].to_numpy(dtype=float)
    predicted = predictions["projection_mean"].to_numpy(dtype=float)
    if np.std(predicted) > 0:
        slope, intercept = np.polyfit(predicted, actual, 1)
    else:
        slope, intercept = 0.0, float(np.mean(actual))
    report["calibration"] = {
        "intercept": float(intercept),
        "slope": float(slope),
        "mean_prediction": float(np.mean(predicted)),
        "mean_actual": float(np.mean(actual)),
    }
    regimes: dict[str, dict[str, float | int]] = {}
    if "total_tds" in predictions:
        regimes["scored_touchdown"] = _regime(predictions[predictions["total_tds"] > 0])
        regimes["no_touchdown"] = _regime(predictions[predictions["total_tds"] <= 0])
    if {"opportunities", "pre_opportunities_ewm4"}.issubset(predictions):
        shock = predictions["opportunities"] - predictions["pre_opportunities_ewm4"]
        regimes["role_increase_5_plus"] = _regime(predictions[shock >= 5])
        regimes["role_decrease_5_plus"] = _regime(predictions[shock <= -5])
        regimes["stable_role"] = _regime(predictions[shock.abs() < 3])
        signed_error = predictions["fantasy_points"] - predictions["projection_mean"]
        report["role_shock_correlation"] = float(shock.corr(signed_error))
    for column in ("team_changed", "qb_changed", "injury_questionable", "practice_dnp"):
        if column in predictions:
            regimes[column] = _regime(predictions[predictions[column].fillna(0) > 0])
    report["regimes"] = regimes

    failures = []
    if not 0.70 <= report["coverage80"] <= 0.88:
        failures.append("80% intervals are materially miscalibrated")
    for position, metrics in report["by_position"].items():
        if not 0.70 <= metrics["coverage80"] <= 0.90:
            failures.append(f"{position} 80% intervals are materially miscalibrated")
    for baseline, result in report["paired_bootstrap"].items():
        if result["ci95"][0] <= 0:
            failures.append(f"improvement over {baseline} is not robustly positive")
    if not 0.80 <= slope <= 1.20:
        failures.append("projection calibration slope is outside [0.80, 1.20]")
    mean_bias = report["calibration"]["mean_actual"] - report["calibration"]["mean_prediction"]
    if abs(mean_bias) > MAX_MEAN_BIAS:
        failures.append(
            f"projection has a systematic mean bias of {mean_bias:+.2f} "
            f"(beyond +/-{MAX_MEAN_BIAS} fantasy points)"
        )
    report["release_gate"] = {"pass": not failures, "failures": failures}
    return report
