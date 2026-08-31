"""What the season-forward backtest can and cannot promise about repeatability.

Running the audited backtest twice, same seed, same frame, same machine, does
not produce byte-identical predictions.  The cause is narrow and worth pinning
so nobody rediscovers it as a phantom model change:

``RandomForestRegressor.predict`` with ``n_jobs=-1`` accumulates each tree's
contribution into a shared array in thread-completion order.  Floating-point
addition is not associative, so the last significant digits of every prediction
depend on scheduling.  The *fitted forest* is bit-identical -- ``random_state``
does its job -- and the same forest predicted with ``n_jobs=1`` is bit-identical
too.  Only the parallel sum moves, by about 1e-8 in the resulting MAE.

The practical consequence, and the reason this file exists: a canonical content
hash of the outer predictions is **not** a valid "did the model change?" gate
for this pipeline, and must never be used as one.  Metrics are; hashes are not.

The one-word fix lives in ``nflvalue/fantasy/models.py``, which is the frozen
production forecast centre.  These tests document and pin the behaviour instead
of changing it.
"""

from __future__ import annotations

import hashlib
import warnings

import numpy as np
import pandas as pd

from nflvalue.fantasy.config import ModelConfig
from nflvalue.fantasy.features import model_features
from nflvalue.fantasy.models import season_forward_backtest

SEED = 6102026
#: Reproducibility budget. Measured at ~3.6e-9 on the full 11,482-row outer set;
#: this synthetic frame is smaller and noisier, so the bound is looser.
METRIC_TOLERANCE = 1e-6

CONFIG = ModelConfig(
    positions=("RB",), fast=True, stack_validation_seasons=2,
    min_train_rows=40, min_position_rows=30,
)


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    features = model_features()
    rows = []
    for season in (2020, 2021, 2022):
        for player in range(45):
            row = {column: float(rng.normal()) for column in features}
            rows.append({
                **row,
                "season": season, "week": player % 17 + 1,
                "player_id": f"00-{player:07d}", "player_name": f"Back {player}",
                "position": "RB", "team": "AAA", "model_eligible": True,
                "status_inactive": 0.0, "injury_out": 0.0,
                "injury_questionable": 0.0, "practice_dnp": 0.0,
                "team_changed": 0.0, "qb_changed": 0.0,
                "total_tds": 0.0, "opportunities": 0.0,
                "fantasy_points": max(
                    0.0, 8.0 + 2.0 * row["pre_opportunities_ewm4"] + float(rng.normal(0, 2))
                ),
            })
    return pd.DataFrame(rows)


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def test_backtest_metrics_reproduce_within_the_stated_tolerance():
    frame = _frame()
    _, first = season_forward_backtest(frame, [2022], config=CONFIG)
    _, second = season_forward_backtest(frame, [2022], config=CONFIG)
    assert first["rows"] == second["rows"]
    for metric in ("mae", "rmse", "spearman", "bias_actual_minus_prediction"):
        assert abs(first["overall"][metric] - second["overall"][metric]) < METRIC_TOLERANCE, (
            f"{metric} moved further than the stated reproducibility budget"
        )


def test_the_fitted_forest_is_bit_identical_because_random_state_works():
    """If this ever fails, the instability is no longer confined to the sum."""
    from sklearn.ensemble import RandomForestRegressor

    frame = _frame()
    features = model_features()
    train = frame[frame["season"].astype(int) < 2022]
    digests = []
    for _ in range(2):
        forest = RandomForestRegressor(
            n_estimators=20, max_features=0.65, min_samples_leaf=10,
            n_jobs=-1, random_state=SEED,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forest.fit(train[features].fillna(0.0), train["fantasy_points"])
        digests.append(_digest(
            np.concatenate([tree.tree_.value.ravel() for tree in forest.estimators_])
        ))
    assert digests[0] == digests[1], "the forest itself became nondeterministic"


def test_single_threaded_prediction_of_one_forest_is_bit_identical():
    """The stable direction, pinned: the nondeterminism is the parallel sum
    and nothing else.  Production keeps ``n_jobs=-1`` for speed and accepts a
    ~1e-8 metric wobble; that trade is recorded, not hidden."""
    from sklearn.ensemble import RandomForestRegressor

    frame = _frame()
    features = model_features()
    train = frame[frame["season"].astype(int) < 2022]
    test = frame[frame["season"].astype(int).eq(2022)]
    forest = RandomForestRegressor(
        n_estimators=20, max_features=0.65, min_samples_leaf=10,
        n_jobs=1, random_state=SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forest.fit(train[features].fillna(0.0), train["fantasy_points"])
    repeats = {_digest(forest.predict(test[features].fillna(0.0))) for _ in range(4)}
    assert len(repeats) == 1, "even single-threaded prediction is nondeterministic"
