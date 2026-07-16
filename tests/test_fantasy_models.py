import json

import numpy as np
import pandas as pd
import pytest

from nflvalue.fantasy.config import ModelConfig, ScoringRules
from nflvalue.fantasy.features import model_features
from nflvalue.fantasy.models import fit_ensemble


def _model_frame():
    rng = np.random.default_rng(17)
    rows = []
    features = model_features()
    for season in (2020, 2021, 2022, 2023):
        for player in range(30):
            base = player / 8 + (season - 2020) * 0.15
            row = {column: rng.normal() for column in features}
            row.update({
                "season": season, "week": player % 17 + 1,
                "player_id": f"00-{player:07d}", "player_name": f"Back {player}",
                "position": "RB", "team": "AAA", "model_eligible": True,
                "status_inactive": 0, "injury_out": 0, "injury_questionable": 0,
                "practice_dnp": 0,
                "fantasy_points": max(0, 7 + base + 2 * row["pre_opportunities_ewm4"] + rng.normal(0, 2)),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def test_ensemble_weights_are_constrained_and_predictions_are_finite():
    config = ModelConfig(
        positions=("RB",), fast=True, stack_validation_seasons=2,
        min_train_rows=40, min_position_rows=30,
    )
    frame = _model_frame()
    model = fit_ensemble(frame, config=config)
    weights = model.positions["RB"].weights
    assert np.isclose(sum(weights.values()), 1)
    assert max(weights.values()) <= config.stack_weight_cap + 1e-8
    projected = model.predict(frame.tail(10))
    assert np.isfinite(projected["projection_mean"]).all()
    assert (projected["projection_upper80"] > projected["projection_lower80"]).all()


def test_model_refuses_mismatched_scoring_target():
    frame = _model_frame()
    frame["scoring_rules"] = json.dumps(ScoringRules.preset("ppr").to_dict(), sort_keys=True)
    config = ModelConfig(
        positions=("RB",), fast=True, stack_validation_seasons=1,
        min_train_rows=40, min_position_rows=30,
    )
    with pytest.raises(ValueError, match="scoring"):
        fit_ensemble(frame, config=config, scoring=ScoringRules.preset("half_ppr"))
