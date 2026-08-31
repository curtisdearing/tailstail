"""End-to-end cutoff poisoning: the future must not reach an earlier cutoff.

A walk-forward harness can *look* leak-free at the top level while a fitted
transformer quietly learned from rows it should never have seen.  Median
imputation, variance thresholding, ``StandardScaler`` centering, PCA
components, the SLSQP stack weights and the conformal residual quantiles are
all fitted objects: each one is a place where a future row can change an
earlier cutoff's answer without any obvious column-level leak.

So these tests do not stop at "the predictions matched".  They rebuild an
identical run against a frame whose future rows have been replaced with
adversarial garbage -- huge magnitudes, sign flips, NaN and infinity, and a
poisoned target -- and then compare the *internals* of every stage as well as
the final numbers.

The frames here are synthetic on purpose: this must run in the offline CI
lane, with no historical cache and no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflvalue.fantasy.config import ModelConfig
from nflvalue.fantasy.features import model_features
from nflvalue.fantasy.models import fit_ensemble, season_forward_backtest

SEASONS = (2020, 2021, 2022, 2023)
CUTOFF = 2022  # first outer test season; 2023 is strictly "the future" for it

CONFIG = ModelConfig(
    positions=("RB",),
    fast=True,
    stack_validation_seasons=2,
    min_train_rows=40,
    min_position_rows=30,
)

BACKTEST_COLUMNS = [
    "season", "week", "player_id", "player_name", "position", "team",
    "fantasy_points", "projection_mean", "projection_lower80",
    "projection_upper80", "pre_fantasy_points_ewm4",
    "pre_expected_points_ewm4", "total_tds", "opportunities",
    "pre_opportunities_ewm4", "team_changed", "qb_changed",
    "injury_questionable", "practice_dnp",
]


def _frame(seed: int = 17) -> pd.DataFrame:
    """A small, well-conditioned walk-forward frame with a real signal."""

    rng = np.random.default_rng(seed)
    features = model_features()
    rows = []
    for season in SEASONS:
        for player in range(40):
            row = {column: float(rng.normal()) for column in features}
            drive = 2.0 * row["pre_opportunities_ewm4"] + 0.8 * row["pre_fantasy_points_ewm4"]
            rows.append({
                **row,
                "season": season,
                "week": player % 17 + 1,
                "player_id": f"00-{player:07d}",
                "player_name": f"Back {player}",
                "position": "RB",
                "team": "AAA",
                "model_eligible": True,
                "status_inactive": 0.0,
                "injury_out": 0.0,
                "injury_questionable": 0.0,
                "practice_dnp": 0.0,
                "team_changed": 0.0,
                "qb_changed": 0.0,
                "total_tds": 0.0,
                "opportunities": 0.0,
                "fantasy_points": max(
                    0.0, 7.0 + player / 8 + drive + float(rng.normal(0, 2))
                ),
            })
    frame = pd.DataFrame(rows)
    for column in BACKTEST_COLUMNS:
        if column not in frame:
            frame[column] = 0.0
    return frame


def _poison(frame: pd.DataFrame, *, strictly_after: int) -> pd.DataFrame:
    """Replace every future row's features and target with adversarial values."""

    poisoned = frame.copy()
    future = poisoned["season"].astype(int) > strictly_after
    assert future.any(), "the poisoning test needs at least one future season"
    rng = np.random.default_rng(99)
    columns = model_features()
    values = rng.normal(loc=5_000.0, scale=25_000.0, size=(int(future.sum()), len(columns)))
    values[:, 0::7] *= -1.0
    values[:, 0::11] = np.nan
    values[:, 0::13] = np.inf
    values[:, 0::17] = -np.inf
    poisoned.loc[future, columns] = values
    poisoned.loc[future, "fantasy_points"] = -9_999.0
    poisoned.loc[future, "total_tds"] = 1e9
    poisoned.loc[future, "opportunities"] = -1e9
    return poisoned


def _fit(frame: pd.DataFrame):
    """Fit exactly what the harness fits for the CUTOFF outer season."""

    return fit_ensemble(frame[frame["season"].astype(int) < CUTOFF], config=CONFIG)


def _stage(estimator, kind: str):
    """Pull one fitted stage out of a pipeline by class name."""

    steps = getattr(estimator, "steps", None)
    if steps is None:  # TransformedTargetRegressor wraps the pipeline
        steps = getattr(estimator.regressor_, "steps", [])
    for _, step in steps:
        if type(step).__name__ == kind:
            return step
    return None


# --------------------------------------------------------------------------
# 1. The whole harness
# --------------------------------------------------------------------------

def test_future_rows_cannot_change_an_earlier_cutoff_prediction():
    clean = _frame()
    poisoned = _poison(clean, strictly_after=CUTOFF)

    baseline, _ = season_forward_backtest(clean, [CUTOFF], config=CONFIG)
    challenged, _ = season_forward_backtest(poisoned, [CUTOFF], config=CONFIG)

    keys = ["season", "week", "player_id"]
    baseline = baseline.sort_values(keys).reset_index(drop=True)
    challenged = challenged.sort_values(keys).reset_index(drop=True)
    assert len(baseline) == len(challenged) > 0
    for column in ("projection_mean", "projection_lower80", "projection_upper80"):
        np.testing.assert_array_equal(
            baseline[column].to_numpy(), challenged[column].to_numpy(),
            err_msg=f"future-season poisoning changed {column} at cutoff {CUTOFF}",
        )


def test_poisoning_the_outer_target_cannot_change_its_own_predictions():
    """Labels are scored, never modelled.  Corrupting the outcome column of
    the test season itself must move the metrics and nothing else."""

    clean = _frame()
    relabelled = clean.copy()
    outer = relabelled["season"].astype(int).eq(CUTOFF)
    relabelled.loc[outer, "fantasy_points"] = -1234.5

    baseline, base_report = season_forward_backtest(clean, [CUTOFF], config=CONFIG)
    challenged, challenge_report = season_forward_backtest(relabelled, [CUTOFF], config=CONFIG)

    keys = ["season", "week", "player_id"]
    baseline = baseline.sort_values(keys).reset_index(drop=True)
    challenged = challenged.sort_values(keys).reset_index(drop=True)
    np.testing.assert_array_equal(
        baseline["projection_mean"].to_numpy(), challenged["projection_mean"].to_numpy()
    )
    # ...and the scorecard must react, otherwise this test proves nothing.
    assert base_report["overall"]["mae"] != challenge_report["overall"]["mae"]


# --------------------------------------------------------------------------
# 2. Every fitted stage inside the ensemble
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_pair():
    clean = _frame()
    poisoned = _poison(clean, strictly_after=CUTOFF)
    return _fit(clean), _fit(poisoned), clean


def test_imputation_medians_are_unchanged(fitted_pair):
    base, challenged, _ = fitted_pair
    for family in ("bayesian", "gradient_boosting", "random_forest", "mlp"):
        a = _stage(base.positions["RB"].models[family], "SimpleImputer")
        b = _stage(challenged.positions["RB"].models[family], "SimpleImputer")
        assert a is not None and b is not None
        np.testing.assert_array_equal(
            a.statistics_, b.statistics_,
            err_msg=f"{family}: future rows moved the imputation medians",
        )


def test_variance_threshold_selection_is_unchanged(fitted_pair):
    base, challenged, _ = fitted_pair
    for family in ("bayesian", "mlp"):
        a = _stage(base.positions["RB"].models[family], "VarianceThreshold")
        b = _stage(challenged.positions["RB"].models[family], "VarianceThreshold")
        assert a is not None and b is not None
        np.testing.assert_array_equal(a.variances_, b.variances_)
        np.testing.assert_array_equal(a.get_support(), b.get_support())


def test_scaler_centering_is_unchanged(fitted_pair):
    base, challenged, _ = fitted_pair
    for family in ("bayesian", "mlp"):
        a = _stage(base.positions["RB"].models[family], "StandardScaler")
        b = _stage(challenged.positions["RB"].models[family], "StandardScaler")
        assert a is not None and b is not None
        np.testing.assert_array_equal(a.mean_, b.mean_)
        np.testing.assert_array_equal(a.scale_, b.scale_)


def test_pca_basis_is_unchanged(fitted_pair):
    base, challenged, _ = fitted_pair
    a = _stage(base.positions["RB"].models["bayesian"], "PCA")
    b = _stage(challenged.positions["RB"].models["bayesian"], "PCA")
    assert a is not None and b is not None
    assert a.n_components_ == b.n_components_
    np.testing.assert_array_equal(a.components_, b.components_)
    np.testing.assert_array_equal(a.explained_variance_, b.explained_variance_)


def test_target_scaler_of_the_network_is_unchanged(fitted_pair):
    base, challenged, _ = fitted_pair
    a = base.positions["RB"].models["mlp"].transformer_
    b = challenged.positions["RB"].models["mlp"].transformer_
    np.testing.assert_array_equal(a.mean_, b.mean_)
    np.testing.assert_array_equal(a.scale_, b.scale_)


def test_stack_weights_are_unchanged(fitted_pair):
    base, challenged, _ = fitted_pair
    assert base.positions["RB"].weights == challenged.positions["RB"].weights


def test_residual_quantiles_are_unchanged(fitted_pair):
    base, challenged, _ = fitted_pair
    assert base.positions["RB"].residual_quantiles == challenged.positions["RB"].residual_quantiles
    assert base.positions["RB"].validation_rows == challenged.positions["RB"].validation_rows
    assert base.positions["RB"].validation_metrics == challenged.positions["RB"].validation_metrics


def test_final_prediction_is_unchanged(fitted_pair):
    base, challenged, clean = fitted_pair
    probe = clean[clean["season"].astype(int).eq(CUTOFF)]
    a = base.predict(probe)
    b = challenged.predict(probe)
    for column in ("projection_mean", "projection_lower80", "projection_upper80",
                   "projection_model_sd"):
        np.testing.assert_array_equal(
            a[column].to_numpy(), b[column].to_numpy(),
            err_msg=f"future rows moved {column}",
        )


def test_the_poison_is_actually_poisonous():
    """Guard against a vacuous suite: if the cutoff filter were removed, the
    same poison must visibly wreck the fit.  Otherwise every assertion above
    passes for the wrong reason."""

    clean = _frame()
    poisoned = _poison(clean, strictly_after=CUTOFF)
    honest = fit_ensemble(clean[clean["season"].astype(int) < CUTOFF], config=CONFIG)
    leaked = fit_ensemble(poisoned, config=CONFIG)  # no cutoff: the future is in
    assert honest.positions["RB"].weights != leaked.positions["RB"].weights or (
        honest.positions["RB"].residual_quantiles
        != leaked.positions["RB"].residual_quantiles
    )
