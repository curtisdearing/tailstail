"""The season-forward audit that stands between the K baseline and a lineup.

The kicker lane shipped with a model card promising coverage, calibration and
sharpness gates, and none of the three had any code behind them. What it had
instead was a determinism claim that was false twice over. So the card's claim
and the lane's evidence are reunited here: this module runs the walk-forward
evaluation, scores it against a league-average-kicker baseline, and returns a
gate that `shadow_kicker` and every consumer can read.

Nothing here promotes anything. `gate()` returns a verdict, and until that
verdict is a pass, `PROMOTION_STATUS["may_enter_lineup_objective"]` stays
False and a kicker's number is displayed, never optimised over. A gate that
cannot be run is a fail, not a pass — an audit nobody executed is exactly the
state the lane was already in.

The evaluation is walk-forward by construction: every projection for week W of
season S is fit on rows strictly before it, through `shadow_kicker.past_only`,
so there is no separate leakage guard to forget to apply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import shadow_kicker as SK

#: A candidate must beat the league-average kicker by this much, with this much
#: confidence, before it is worth calling a model rather than a table lookup.
#: Declared here, before any run, so a disappointing result cannot be answered
#: by moving the line.
GATES: Mapping[str, Any] = {
    "min_weeks": 40,
    "min_kickers": 8,
    "mae_improvement_over_baseline": 0.0,
    "min_probability_of_improvement": 0.90,
    "crps_improvement_over_baseline": 0.0,
    "coverage_50_tolerance": 0.10,
    "coverage_90_tolerance": 0.07,
}


class AuditError(RuntimeError):
    """The audit cannot be run as specified. Never downgraded to a pass."""


@dataclass(frozen=True)
class AuditResult:
    """What the run found, and whether that clears the declared gates."""

    passed: bool
    reasons: tuple[str, ...]
    metrics: Mapping[str, Any]
    gates: Mapping[str, Any] = field(default_factory=lambda: dict(GATES))
    n_rows: int = 0

    def as_dict(self) -> dict:
        return {"passed": self.passed, "reasons": list(self.reasons),
                "metrics": dict(self.metrics), "gates": dict(self.gates),
                "n_rows": self.n_rows}


def league_average_baseline(history: pd.DataFrame, season: int, week: int,
                            contract) -> float | None:
    """The kicker-agnostic reference: league rates, scored by this league.

    Beating this is the whole question. A per-kicker model that cannot is a
    more expensive way to say "kickers score about the same", and shrinkage
    alone will get close to it, which is why MAE is not enough on its own.
    """
    usable = SK.past_only(history, season, week)
    if usable.empty:
        return None
    rates = SK.league_rates(usable)
    total = 0.0
    for bucket in SK.BUCKETS:
        made = rates["attempts"][bucket] * rates["makes"][bucket]
        missed = rates["attempts"][bucket] - made
        total += made * contract.points(bucket)
        total += missed * contract.points("fg_missed_total")
    pat_made = rates["pat_attempts"] * rates["pat_make"]
    total += pat_made * contract.points("pat_made")
    total += (rates["pat_attempts"] - pat_made) * contract.points("pat_missed")
    return float(total)


def _actual_points(row: Mapping[str, Any], contract) -> float:
    total = 0.0
    for bucket in SK.BUCKETS:
        made = sum(float(row.get(column, 0.0) or 0.0) for column in SK.MADE_COLUMNS[bucket])
        missed = sum(float(row.get(column, 0.0) or 0.0) for column in SK.MISSED_COLUMNS[bucket])
        total += made * contract.points(bucket)
        total += missed * contract.points("fg_missed_total")
    pat_made = float(row.get("pat_made", 0.0) or 0.0)
    pat_att = float(row.get("pat_att", 0.0) or 0.0)
    total += pat_made * contract.points("pat_made")
    total += (pat_att - pat_made) * contract.points("pat_missed")
    return float(total)


def walk_forward(history: pd.DataFrame, contract, *, test_seasons: Sequence[int],
                 simulations: int = 2_000, seed: int = 6102026) -> pd.DataFrame:
    """One row per (kicker, season, week) in the test seasons, fit only on the past."""
    if history.empty:
        raise AuditError("no kicker history supplied; an audit with no data is not a pass")
    frame = history.copy()
    for column in ("season", "week"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["season", "week"])
    rows: list[dict[str, Any]] = []
    for season in sorted(int(s) for s in test_seasons):
        weeks = sorted(int(w) for w in frame.loc[frame["season"] == season, "week"].unique())
        for week in weeks:
            baseline = league_average_baseline(frame, season, week, contract)
            if baseline is None:
                continue
            current = frame[(frame["season"] == season) & (frame["week"] == week)]
            for _, actual_row in current.iterrows():
                player_id = str(actual_row["player_id"])
                projection = SK.project(frame, player_id, contract, season=season,
                                        week=week, simulations=simulations, seed=seed,
                                        active=True)
                if projection["status"] != "projected":
                    continue
                rows.append({
                    "season": season, "week": week, "player_id": player_id,
                    "projection_mean": float(projection["distribution"]["mean"]),
                    "projection_p25": float(projection["distribution"]["p25"]),
                    "projection_p75": float(projection["distribution"]["p75"]),
                    "projection_p05": float(projection["distribution"]["p05"]),
                    "projection_p95": float(projection["distribution"]["p95"]),
                    "projection_sd": float(projection["distribution"]["sd"]),
                    "baseline_mean": baseline,
                    "fantasy_points": _actual_points(actual_row, contract),
                })
    return pd.DataFrame(rows)


def _paired_week_bootstrap(predictions: pd.DataFrame, *, iterations: int = 20_000,
                           seed: int = 6102026) -> dict[str, float]:
    """Season-week blocks, so kickers in one week move together."""
    blocks = [group for _, group in predictions.groupby(["season", "week"], sort=True)]
    if not blocks:
        return {"mae_improvement": 0.0, "probability_of_improvement": 0.0}
    per_block = np.array([
        (np.abs(b["baseline_mean"] - b["fantasy_points"]).mean()
         - np.abs(b["projection_mean"] - b["fantasy_points"]).mean())
        for b in blocks
    ], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(per_block), size=(iterations, len(per_block)))
    means = per_block[draws].mean(axis=1)
    return {
        "mae_improvement": float(per_block.mean()),
        "probability_of_improvement": float((means > 0).mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "week_win_rate": float((per_block > 0).mean()),
        "blocks": int(len(per_block)),
    }


def _interval_coverage(predictions: pd.DataFrame) -> dict[str, float]:
    inside_50 = ((predictions["fantasy_points"] >= predictions["projection_p25"]) &
                 (predictions["fantasy_points"] <= predictions["projection_p75"]))
    inside_90 = ((predictions["fantasy_points"] >= predictions["projection_p05"]) &
                 (predictions["fantasy_points"] <= predictions["projection_p95"]))
    return {"coverage_50": float(inside_50.mean()), "coverage_90": float(inside_90.mean())}


def _normal_crps(mean: np.ndarray, sd: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """CRPS under a Normal summary of the simulated distribution.

    Sharpness has to be scored on the whole distribution, not the point: a
    model can win MAE by collapsing to the mean, which is the failure this
    number exists to catch.
    """
    from math import erf, pi, sqrt

    sd = np.maximum(np.asarray(sd, dtype=float), 1e-9)
    z = (np.asarray(actual, dtype=float) - np.asarray(mean, dtype=float)) / sd
    cdf = 0.5 * (1.0 + np.vectorize(lambda v: erf(v / sqrt(2.0)))(z))
    pdf = np.exp(-0.5 * z ** 2) / sqrt(2.0 * pi)
    return sd * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / np.sqrt(pi))


def evaluate(predictions: pd.DataFrame) -> dict[str, Any]:
    """Every declared gate's measurement, computed once."""
    if predictions.empty:
        return {"n_rows": 0}
    actual = predictions["fantasy_points"].to_numpy(dtype=float)
    model_crps = _normal_crps(predictions["projection_mean"].to_numpy(),
                              predictions["projection_sd"].to_numpy(), actual)
    # The baseline is a point, so its sharpest honest summary is the spread of
    # the outcomes it is predicting.
    baseline_sd = np.full(len(predictions), float(np.std(actual, ddof=1)) or 1e-9)
    baseline_crps = _normal_crps(predictions["baseline_mean"].to_numpy(), baseline_sd, actual)
    metrics = {
        "n_rows": int(len(predictions)),
        "n_kickers": int(predictions["player_id"].nunique()),
        "n_weeks": int(predictions.groupby(["season", "week"]).ngroups),
        "mae_model": float(np.abs(predictions["projection_mean"] - actual).mean()),
        "mae_baseline": float(np.abs(predictions["baseline_mean"] - actual).mean()),
        "crps_model": float(model_crps.mean()),
        "crps_baseline": float(baseline_crps.mean()),
        **_interval_coverage(predictions),
        **_paired_week_bootstrap(predictions),
    }
    metrics["crps_improvement"] = metrics["crps_baseline"] - metrics["crps_model"]
    return metrics


def gate(metrics: Mapping[str, Any], gates: Mapping[str, Any] | None = None) -> AuditResult:
    """The verdict. A gate that could not be measured fails."""
    rules = dict(gates or GATES)
    reasons: list[str] = []
    if not metrics or not metrics.get("n_rows"):
        return AuditResult(passed=False,
                           reasons=("the audit produced no rows; an audit that did not run "
                                    "is not an audit that passed",),
                           metrics=dict(metrics), gates=rules, n_rows=0)

    if metrics["n_weeks"] < rules["min_weeks"]:
        reasons.append(f"only {metrics['n_weeks']} scored weeks, below the declared "
                       f"minimum of {rules['min_weeks']}")
    if metrics["n_kickers"] < rules["min_kickers"]:
        reasons.append(f"only {metrics['n_kickers']} kickers, below the declared minimum "
                       f"of {rules['min_kickers']}")
    if metrics["mae_improvement"] <= rules["mae_improvement_over_baseline"]:
        reasons.append(f"MAE improvement over the league-average kicker is "
                       f"{metrics['mae_improvement']:.4f}, not above "
                       f"{rules['mae_improvement_over_baseline']}")
    if metrics["probability_of_improvement"] < rules["min_probability_of_improvement"]:
        reasons.append(f"probability of improvement {metrics['probability_of_improvement']:.3f} "
                       f"is below {rules['min_probability_of_improvement']}")
    if metrics["crps_improvement"] <= rules["crps_improvement_over_baseline"]:
        reasons.append(f"CRPS improvement is {metrics['crps_improvement']:.4f}; beating MAE "
                       "alone can be done by shrinking to the mean, which is why this gate "
                       "exists")
    if abs(metrics["coverage_50"] - 0.50) > rules["coverage_50_tolerance"]:
        reasons.append(f"50% band covers {metrics['coverage_50']:.3f}, outside "
                       f"±{rules['coverage_50_tolerance']} of nominal")
    if abs(metrics["coverage_90"] - 0.90) > rules["coverage_90_tolerance"]:
        reasons.append(f"90% band covers {metrics['coverage_90']:.3f}, outside "
                       f"±{rules['coverage_90_tolerance']} of nominal")

    return AuditResult(passed=not reasons, reasons=tuple(reasons), metrics=dict(metrics),
                       gates=rules, n_rows=int(metrics["n_rows"]))


def run(history: pd.DataFrame, contract, *, test_seasons: Sequence[int],
        simulations: int = 2_000, seed: int = 6102026) -> AuditResult:
    """Walk forward, measure, and return the verdict in one call."""
    predictions = walk_forward(history, contract, test_seasons=test_seasons,
                               simulations=simulations, seed=seed)
    return gate(evaluate(predictions))


def may_enter_lineup_objective(result: AuditResult | None) -> bool:
    """The only question a consumer should ask before optimising over a kicker."""
    return bool(result is not None and result.passed)
