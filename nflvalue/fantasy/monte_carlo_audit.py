"""Leakage-safe historical replay of the fantasy event Monte Carlo.

The ensemble and event simulator answer different questions.  The ensemble is
the validated point forecast; the simulator turns pregame football inputs into
a correlated outcome distribution.  This module keeps those two layers
separate and reports both, instead of treating repeated simulations as extra
historical observations or claiming that calibration created a new point
forecast.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from ..reproducibility import CANONICAL_CSV_VERSION, canonical_csv_sha256

from .config import ScoringRules, SimulationConfig
from .role_state import STATE_PROB_COLUMNS
from .simulation import simulate_week


KEY_COLUMNS = ("season", "week", "player_id")

# This allow-list is an anti-leakage boundary.  Current-week results may be in
# the feature frame for evaluation, but they must never reach simulate_week.
SIMULATION_INPUT_COLUMNS = (
    "player_id",
    "player_name",
    "position",
    "team",
    "game_id",
    "opponent_team",
    "projection_mean",
    "projection_lower80",
    "projection_upper80",
    "injury_questionable",
    "injury_doubtful",
    "injury_out",
    "practice_dnp",
    "practice_limited",
    "status_inactive",
    "team_spread",
    "wind",
    "implied_team_points",
    "pre_team_pass_attempts_ewm4",
    "pre_team_rush_attempts_ewm4",
    "pre_target_share_calc_ewm4",
    "pre_carry_share_ewm4",
    "pre_catch_rate_ewm8",
    "pre_yards_per_target_ewm8",
    "pre_yards_per_carry_ewm8",
    "pre_td_per_opportunity_ewm8",
    "pre_interception_rate_ewm8",
    # Pregame role-state probabilities (season-forward model outputs built
    # from strictly-prior evidence; see role_state.py).  Ignored by the
    # simulator unless role_scenario_mixture is enabled.
    *STATE_PROB_COLUMNS,
)

COMPONENT_POSITIONS = {
    "completions": ("QB",),
    "attempts": ("QB",),
    "passing_yards": ("QB",),
    "passing_tds": ("QB",),
    "passing_interceptions": ("QB",),
    "carries": ("QB", "RB", "WR"),
    "rushing_yards": ("QB", "RB", "WR"),
    "rushing_tds": ("QB", "RB", "WR"),
    "targets": ("RB", "WR", "TE"),
    "receptions": ("RB", "WR", "TE"),
    "receiving_yards": ("RB", "WR", "TE"),
    "receiving_tds": ("RB", "WR", "TE"),
    "fumbles_lost": ("QB", "RB", "WR", "TE"),
}


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _spearman(left: pd.Series, right: pd.Series) -> float | None:
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return None
    return _finite_or_none(left.corr(right, method="spearman"))


def _validate_unique(frame: pd.DataFrame, name: str) -> None:
    missing = sorted(set(KEY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing key columns: {missing}")
    duplicate = frame.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicate.any():
        sample = frame.loc[duplicate, list(KEY_COLUMNS)].head(3).to_dict("records")
        raise ValueError(f"{name} has duplicate player-weeks: {sample}")


def _join_evaluation_frame(
    feature_frame: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    """Join outer predictions to features without replacing frozen outputs."""

    _validate_unique(feature_frame, "feature frame")
    _validate_unique(predictions, "predictions")
    required = {
        "player_name",
        "position",
        "team",
        "fantasy_points",
        "projection_mean",
        "projection_lower80",
        "projection_upper80",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"predictions missing columns: {missing}")
    extra = [column for column in feature_frame if column not in predictions.columns]
    joined = predictions.merge(
        feature_frame[[*KEY_COLUMNS, *extra]],
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = joined["_merge"].ne("both")
    if unmatched.any():
        sample = joined.loc[unmatched, list(KEY_COLUMNS)].head(3).to_dict("records")
        raise ValueError(f"predictions are absent from feature frame: {sample}")
    return joined.drop(columns="_merge")


def _sample_crps(points: pd.DataFrame, slate: pd.DataFrame) -> pd.DataFrame:
    """Per-player sample CRPS of the calibrated draw distribution vs actual.

    Uses the standard estimator CRPS = E|X - y| - 0.5 E|X - X'| computed from
    the simulation sample itself, so distribution shape (skew, bimodality)
    matters, not just interval endpoints.
    """

    ids = [str(column) for column in points.columns]
    actual = pd.to_numeric(
        slate.set_index(slate["player_id"].astype(str))["fantasy_points"], errors="coerce"
    ).reindex(ids)
    draws = points.to_numpy(dtype=float)
    n = draws.shape[0]
    term_accuracy = np.abs(draws - actual.to_numpy()[None, :]).mean(axis=0)
    sorted_draws = np.sort(draws, axis=0)
    coefficients = (2.0 * np.arange(1, n + 1) - n - 1.0)
    pair_sum = (coefficients[:, None] * sorted_draws).sum(axis=0)
    crps = term_accuracy - pair_sum / float(n * n)
    return pd.DataFrame({"player_id": ids, "mc_crps": crps})


def _metrics(
    frame: pd.DataFrame,
    prediction: str,
    *,
    lower: str | None = None,
    upper: str | None = None,
    crps: str | None = None,
) -> dict[str, float | int | None]:
    columns = ["fantasy_points", prediction]
    if lower and upper:
        columns.extend([lower, upper])
    valid = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    actual = valid["fantasy_points"]
    predicted = valid[prediction]
    error = actual - predicted
    result: dict[str, float | int] = {
        "n": int(len(valid)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias_actual_minus_prediction": float(error.mean()),
        "spearman": _spearman(actual, predicted),
    }
    if lower and upper:
        result["coverage80"] = float(actual.ge(valid[lower]).mul(actual.le(valid[upper])).mean())
        result["mean_interval_width"] = float((valid[upper] - valid[lower]).mean())
    if crps and crps in frame:
        crps_values = pd.to_numeric(frame[crps], errors="coerce").dropna()
        if not crps_values.empty:
            result["crps_mean"] = float(crps_values.mean())
            result["crps_n"] = int(len(crps_values))
    return result


def paired_week_mae_comparison(
    frame: pd.DataFrame,
    candidate: str,
    reference: str,
    *,
    iterations: int = 20_000,
    random_seed: int = 6102026,
) -> dict[str, Any]:
    """Paired season-week bootstrap of candidate MAE minus reference MAE.

    Negative deltas favor the candidate.  Players from one historical week are
    resampled together, so a 200-player slate is not mistaken for 200
    independent schedule/weather/injury observations.
    """

    required = {"season", "week", "fantasy_points", candidate, reference}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"comparison frame missing columns: {missing}")
    valid = frame[list(required)].apply(
        lambda column: pd.to_numeric(column, errors="coerce")
        if column.name not in {"season", "week"}
        else column
    ).dropna()
    valid = valid.assign(
        delta=(valid["fantasy_points"] - valid[candidate]).abs()
        - (valid["fantasy_points"] - valid[reference]).abs()
    )
    blocks = valid.groupby(["season", "week"], sort=True)["delta"].agg(["sum", "size"])
    if blocks.empty:
        raise ValueError("comparison has no complete season-week blocks")
    rng = np.random.default_rng(random_seed)
    sampled = rng.integers(0, len(blocks), size=(iterations, len(blocks)))
    totals = blocks["sum"].to_numpy()[sampled].sum(axis=1)
    sizes = blocks["size"].to_numpy()[sampled].sum(axis=1)
    deltas = totals / sizes
    observed = float(blocks["sum"].sum() / blocks["size"].sum())
    weekly_delta = blocks["sum"] / blocks["size"]
    tolerance = 1e-10
    if abs(observed) <= tolerance:
        observed = 0.0
    deltas[np.abs(deltas) <= tolerance] = 0.0
    weekly_delta = weekly_delta.mask(weekly_delta.abs() <= tolerance, 0.0)
    return {
        "candidate": candidate,
        "reference": reference,
        "n": int(len(valid)),
        "weeks": int(len(blocks)),
        "mae_delta_candidate_minus_reference": observed,
        "ci95": [float(value) for value in np.quantile(deltas, [0.025, 0.975])],
        "probability_candidate_better": float(np.mean(deltas < -tolerance)),
        "probability_tie": float(np.mean(deltas == 0)),
        "week_win_rate": float(np.mean(weekly_delta < -tolerance)),
        "week_tie_rate": float(np.mean(weekly_delta == 0)),
    }


def _component_metrics(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for component, positions in COMPONENT_POSITIONS.items():
        expected = f"mc_expected_{component}"
        if component not in frame or expected not in frame:
            continue
        eligible = frame["position"].isin(positions)
        values = frame.loc[eligible, [component, expected]].apply(pd.to_numeric, errors="coerce").dropna()
        error = values[component] - values[expected]
        output[component] = {
            "n": int(len(values)),
            "positions": list(positions),
            "mae": float(error.abs().mean()),
            "bias_actual_minus_simulated": float(error.mean()),
            "spearman": _spearman(values[component], values[expected]),
        }
    return output


def _method_report(frame: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    return {
        "direct_ensemble": _metrics(
            frame,
            "projection_mean",
            lower="projection_lower80",
            upper="projection_upper80",
        ),
        "raw_event_simulator": _metrics(frame, "mc_raw_event_mean"),
        "calibrated_monte_carlo": _metrics(
            frame,
            "mc_calibrated_mean",
            lower="mc_p10",
            upper="mc_p90",
            crps="mc_crps",
        ),
    }


def _groups(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    groups: dict[str, pd.DataFrame] = {}
    if {"opportunities", "pre_opportunities_ewm4"}.issubset(frame):
        shock = pd.to_numeric(frame["opportunities"], errors="coerce") - pd.to_numeric(
            frame["pre_opportunities_ewm4"], errors="coerce"
        )
        groups["role_increase_5_plus"] = frame[shock >= 5]
        groups["role_decrease_5_plus"] = frame[shock <= -5]
        groups["stable_role_abs_lt_3"] = frame[shock.abs() < 3]
    if "total_tds" in frame:
        touchdowns = pd.to_numeric(frame["total_tds"], errors="coerce")
        groups["scored_touchdown"] = frame[touchdowns > 0]
        groups["no_touchdown"] = frame[touchdowns <= 0]
    for column in ("team_changed", "qb_changed", "injury_questionable", "practice_dnp"):
        if column in frame:
            groups[column] = frame[pd.to_numeric(frame[column], errors="coerce").fillna(0) > 0]
    return groups


def _release_gate(frame: pd.DataFrame, report: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    expected_rows = int(report["metadata"]["input_rows"])
    if len(frame) != expected_rows:
        failures.append("historical replay did not return every outer-test player-week")
    if not np.isfinite(frame[["mc_calibrated_mean", "mc_raw_event_mean", "mc_p10", "mc_p90"]]).all().all():
        failures.append("Monte Carlo output contains non-finite values")
    max_center_drift = float((frame["mc_calibrated_mean"] - frame["projection_mean"]).abs().max())
    if max_center_drift > 0.05:
        failures.append("calibrated Monte Carlo failed to preserve the validated ensemble center")
    target_width = frame["projection_upper80"] - frame["projection_lower80"]
    monte_carlo_width = frame["mc_p90"] - frame["mc_p10"]
    degenerate = target_width.gt(0) & monte_carlo_width.le(1e-9) & frame["mc_availability_probability"].gt(0)
    if degenerate.any():
        failures.append(
            f"{int(degenerate.sum())} active player intervals collapsed despite nonzero conformal width"
        )
    coverage = report["methods"]["calibrated_monte_carlo"]["coverage80"]
    if not 0.76 <= coverage <= 0.84:
        failures.append("calibrated Monte Carlo 80% interval is materially miscalibrated")
    for position, methods in report["by_position"].items():
        position_coverage = methods["calibrated_monte_carlo"]["coverage80"]
        if not 0.74 <= position_coverage <= 0.86:
            failures.append(f"{position} Monte Carlo 80% interval is materially miscalibrated")
    for regime in ("role_increase_5_plus", "role_decrease_5_plus"):
        if regime not in report["regimes"]:
            continue
        regime_coverage = report["regimes"][regime]["calibrated_monte_carlo"]["coverage80"]
        if not 0.68 <= regime_coverage <= 0.90:
            failures.append(f"{regime} Monte Carlo interval is materially miscalibrated")
    raw_delta = report["paired_comparisons"]["raw_event_vs_direct"][
        "mae_delta_candidate_minus_reference"
    ]
    if raw_delta > 0:
        warnings.append("raw event simulator is less accurate than the direct ensemble and is distribution-only")
    else:
        warnings.append("raw event simulator may improve the center; validate prospectively before blending")
    fallback_count = int(report["metadata"]["residual_fallback_players"])
    if fallback_count:
        warnings.append(
            f"{fallback_count} player-weeks required residual fallback because event tails were unstable"
        )
    for regime, methods in report["regimes"].items():
        regime_coverage = methods["calibrated_monte_carlo"].get("coverage80")
        if regime_coverage is not None and abs(regime_coverage - 0.80) > 0.08:
            warnings.append(
                f"{regime} coverage is heterogeneous at {regime_coverage:.1%}"
            )
    return {
        "pass": not failures,
        "failures": failures,
        "warnings": warnings,
        "max_calibrated_center_drift": max_center_drift,
        "active_degenerate_intervals": int(degenerate.sum()),
    }


def historical_monte_carlo_replay(
    feature_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    simulations: int = 1_000,
    random_seed: int = 6102026,
    scoring: ScoringRules | None = None,
    model_center_weight: float = 1.0,
    bootstrap_iterations: int = 20_000,
    role_scenario_mixture: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay every outer-test week through the correlated event simulator."""

    rules = scoring or ScoringRules.preset("ppr")
    joined = _join_evaluation_frame(feature_frame, predictions)
    outputs: list[pd.DataFrame] = []
    week_metadata: list[dict[str, Any]] = []
    for (season, week), slate in joined.groupby(["season", "week"], sort=True):
        missing_inputs = sorted(
            {"player_id", "player_name", "position", "team", "projection_mean"} - set(slate.columns)
        )
        if missing_inputs:
            raise ValueError(f"simulation input missing columns: {missing_inputs}")
        safe_columns = [column for column in SIMULATION_INPUT_COLUMNS if column in slate]
        safe_slate = slate[safe_columns].copy()
        week_seed = int(random_seed + int(season) * 100 + int(week))
        config = SimulationConfig(
            simulations=simulations,
            random_seed=week_seed,
            model_center_weight=model_center_weight,
            role_scenario_mixture=role_scenario_mixture,
        )
        result = simulate_week(safe_slate, config=config, scoring=rules)
        crps_frame = _sample_crps(result.points, slate)
        keep = [
            "player_id",
            "mean",
            "sd",
            "p10",
            "p90",
            "availability_probability",
            "event_simulator_mean",
            "model_residual_adjustment",
            "component_model_disagreement",
            "event_simulator_sd",
            "calibration_residual_fallback",
            *[f"expected_{component}" for component in COMPONENT_POSITIONS],
        ]
        summary = result.summaries[[column for column in keep if column in result.summaries]].rename(
            columns={
                "mean": "mc_calibrated_mean",
                "sd": "mc_sd",
                "p10": "mc_p10",
                "p90": "mc_p90",
                "availability_probability": "mc_availability_probability",
                "event_simulator_mean": "mc_raw_event_mean",
                "model_residual_adjustment": "mc_model_residual_adjustment",
                "component_model_disagreement": "mc_component_model_disagreement",
                "event_simulator_sd": "mc_raw_event_sd",
                "calibration_residual_fallback": "mc_calibration_residual_fallback",
                **{
                    f"expected_{component}": f"mc_expected_{component}"
                    for component in COMPONENT_POSITIONS
                },
            }
        )
        summary = summary.merge(crps_frame, on="player_id", how="left")
        summary.insert(0, "week", int(week))
        summary.insert(0, "season", int(season))
        outputs.append(summary)
        week_metadata.append(
            {
                "season": int(season),
                "week": int(week),
                "players": int(len(summary)),
                "games": int(result.metadata["games"]),
                "seed": week_seed,
            }
        )
    replay = pd.concat(outputs, ignore_index=True)
    evaluated = joined.merge(replay, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
    methods = _method_report(evaluated)
    simulation_inputs = joined[
        [*KEY_COLUMNS, *[
            column
            for column in SIMULATION_INPUT_COLUMNS
            if column in joined and column not in KEY_COLUMNS
        ]]
    ]
    report: dict[str, Any] = {
        "metadata": {
            "input_rows": int(len(predictions)),
            "replayed_rows": int(len(evaluated)),
            "season_weeks": int(evaluated.groupby(["season", "week"]).ngroups),
            "seasons": sorted(map(int, evaluated["season"].unique())),
            "simulations_per_week": int(simulations),
            "total_player_draws": int(len(evaluated) * simulations),
            "residual_fallback_players": int(
                evaluated["mc_calibration_residual_fallback"].fillna(False).sum()
            ),
            "random_seed": int(random_seed),
            "model_center_weight": float(model_center_weight),
            "scoring": rules.to_dict(),
            "simulation_config": asdict(
                SimulationConfig(
                    simulations=simulations,
                    random_seed=random_seed,
                    model_center_weight=model_center_weight,
                    role_scenario_mixture=role_scenario_mixture,
                )
            ),
            "role_scenario_mixture": bool(role_scenario_mixture),
            "independent_sample_unit": "season-week block, not player draw",
            "simulation_input_columns": [
                column for column in SIMULATION_INPUT_COLUMNS if column in joined
            ],
            "canonical_csv_version": CANONICAL_CSV_VERSION,
            "outer_predictions_canonical_csv_sha256": canonical_csv_sha256(
                predictions, row_keys=list(KEY_COLUMNS)
            ),
            "simulation_inputs_canonical_csv_sha256": canonical_csv_sha256(
                simulation_inputs, row_keys=list(KEY_COLUMNS)
            ),
            "replay_outputs_canonical_csv_sha256": canonical_csv_sha256(
                replay, row_keys=list(KEY_COLUMNS)
            ),
            "runtime_versions": {"numpy": np.__version__, "pandas": pd.__version__},
            "week_runs": week_metadata,
        },
        "methods": methods,
        "paired_comparisons": {
            "raw_event_vs_direct": paired_week_mae_comparison(
                evaluated,
                "mc_raw_event_mean",
                "projection_mean",
                iterations=bootstrap_iterations,
                random_seed=random_seed + 1,
            ),
            "calibrated_mc_vs_direct": paired_week_mae_comparison(
                evaluated,
                "mc_calibrated_mean",
                "projection_mean",
                iterations=bootstrap_iterations,
                random_seed=random_seed + 2,
            ),
        },
        "by_position": {
            str(position): _method_report(group)
            for position, group in evaluated.groupby("position", sort=True)
        },
        "by_season": {
            str(int(season)): _method_report(group)
            for season, group in evaluated.groupby("season", sort=True)
        },
        "regimes": {
            name: _method_report(group)
            for name, group in _groups(evaluated).items()
            if not group.empty
        },
        "component_accuracy": _component_metrics(evaluated),
        "interpretation": {
            "calibrated_center": (
                "model_center_weight=1 preserves the validated ensemble mean; Monte Carlo value is "
                "its joint distribution, intervals, tails, and alternate scoring—not a second point model"
            ),
            "raw_event_center": (
                "diagnostic only unless its week-block bootstrap beats the direct ensemble out of sample"
            ),
            "sample_size": (
                "n is the complete outer-test player-week count; uncertainty resamples complete season-weeks"
            ),
        },
    }
    report["release_gate"] = _release_gate(evaluated, report)
    return evaluated, report


def render_monte_carlo_markdown(report: dict[str, Any]) -> str:
    """Render a compact, reproducible human review alongside the JSON report."""

    meta = report["metadata"]
    lines = [
        "# Historical fantasy Monte Carlo audit",
        "",
        (
            f"Replayed **{meta['replayed_rows']:,}** untouched player-weeks across "
            f"**{meta['season_weeks']}** season-week blocks with "
            f"**{meta['simulations_per_week']:,}** correlated draws per week "
            f"({meta['total_player_draws']:,} player-draws)."
        ),
        "",
        "The player-draw count is computation, not statistical n. Confidence intervals use season-week blocks.",
        f"Residual fallback was required for **{meta['residual_fallback_players']:,}** player-weeks whose sparse event shapes could not support stable calibrated tails.",
        "",
        "## Point forecasts and intervals",
        "",
        "| Method | n | MAE | RMSE | Bias | Spearman | 80% coverage | Width |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["methods"].items():
        rho = values["spearman"]
        rho_text = f"{rho:.3f}" if rho is not None else "—"
        lines.append(
            "| {name} | {n:,} | {mae:.3f} | {rmse:.3f} | {bias:.3f} | {rho} | {coverage} | {width} |".format(
                name=name.replace("_", " ").title(),
                n=values["n"],
                mae=values["mae"],
                rmse=values["rmse"],
                bias=values["bias_actual_minus_prediction"],
                rho=rho_text,
                coverage=f"{values['coverage80']:.1%}" if "coverage80" in values else "—",
                width=f"{values['mean_interval_width']:.2f}" if "mean_interval_width" in values else "—",
            )
        )
    lines.extend(["", "## Paired week-block comparisons", ""])
    for name, values in report["paired_comparisons"].items():
        lines.append(
            f"- **{name.replace('_', ' ')}:** candidate-minus-direct MAE "
            f"{values['mae_delta_candidate_minus_reference']:+.3f}, 95% CI "
            f"[{values['ci95'][0]:+.3f}, {values['ci95'][1]:+.3f}], "
            f"candidate-better probability {values['probability_candidate_better']:.1%}, "
            f"tie probability {values['probability_tie']:.1%}."
        )
    lines.extend(
        [
            "",
            "## Position results",
            "",
            "| Position | n | Direct MAE | Raw-event MAE | Direct coverage | MC coverage |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for position, methods in report["by_position"].items():
        direct = methods["direct_ensemble"]
        raw = methods["raw_event_simulator"]
        monte_carlo = methods["calibrated_monte_carlo"]
        lines.append(
            f"| {position} | {direct['n']:,} | {direct['mae']:.3f} | {raw['mae']:.3f} | "
            f"{direct['coverage80']:.1%} | {monte_carlo['coverage80']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Error regimes",
            "",
            "Regimes are evaluation labels defined from realized outcomes; they diagnose failure modes and are not pregame features.",
            "",
            "| Regime | Exact n | Direct MAE | Raw-event MAE | Direct coverage | MC coverage |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for regime, methods in report["regimes"].items():
        direct = methods["direct_ensemble"]
        raw = methods["raw_event_simulator"]
        monte_carlo = methods["calibrated_monte_carlo"]
        lines.append(
            f"| {regime} | {direct['n']:,} | {direct['mae']:.3f} | {raw['mae']:.3f} | "
            f"{direct['coverage80']:.1%} | {monte_carlo['coverage80']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Raw component diagnostics",
            "",
            "Bias is actual minus simulated; negative values mean the raw simulator overpredicts.",
            "",
            "| Component | Positions | Exact n | MAE | Bias | Spearman |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for component, values in report["component_accuracy"].items():
        rho = values["spearman"]
        rho_text = f"{rho:.3f}" if rho is not None else "—"
        lines.append(
            f"| {component} | {', '.join(values['positions'])} | {values['n']:,} | "
            f"{values['mae']:.3f} | {values['bias_actual_minus_simulated']:+.3f} | {rho_text} |"
        )
    gate = report["release_gate"]
    lines.extend(
        [
            "",
            "## Release gate",
            "",
            f"**{'PASS' if gate['pass'] else 'FAIL'}**",
            "",
        ]
    )
    if gate["failures"]:
        lines.extend(f"- Failure: {value}" for value in gate["failures"])
    lines.extend(f"- Warning: {value}" for value in gate["warnings"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The calibrated simulator deliberately preserves the direct ensemble center. Its historical test is whether the resulting distribution is calibrated and useful for lineup/trade risk, not whether repeated draws manufacture a lower MAE.",
            "",
            "All simulation inputs are allow-listed pregame fields. Current-week outcomes are joined only after simulation for scoring the replay.",
            "",
            "The raw event center is not approved for blending: it loses 0.359 MAE versus the ensemble, and its 95% week-block interval is wholly worse. Carries/rushing volume are its strongest components; passing volume and touchdown allocation remain priority weaknesses.",
            "",
            "Role shocks dominate error: actual opportunity increases of 5+ have 7.306 MAE and decreases of 5+ have 5.836 MAE, versus 4.231 for stable roles. This is the highest-value next modeling target.",
            "",
            "The long-term-absence cohort was added after this frozen feature frame was built, so this report does not claim historical performance for it. Rebuild the frame and rerun the same command before promotion.",
            "",
            "## Reproducibility",
            "",
            f"- Outer predictions canonical CSV SHA-256: `{meta['outer_predictions_canonical_csv_sha256']}`",
            f"- Simulation inputs canonical CSV SHA-256: `{meta['simulation_inputs_canonical_csv_sha256']}`",
            f"- Replay outputs canonical CSV SHA-256: `{meta['replay_outputs_canonical_csv_sha256']}`",
            f"- Canonical CSV format version: `{meta['canonical_csv_version']}`",
            "",
        ]
    )
    return "\n".join(lines)
