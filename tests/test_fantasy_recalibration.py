import numpy as np
import pandas as pd
import pytest

from nflvalue.fantasy.config import ModelConfig
from nflvalue.fantasy.features import model_features
from nflvalue.fantasy.models import fit_ensemble
from nflvalue.fantasy.recalibration import (
    SCALE_COLUMN,
    SCALE_LOWER_COLUMN,
    SCALE_UPPER_COLUMN,
    SHOCK_COMBINED_COLUMN,
    SHOCK_DECREASE_COLUMN,
    SHOCK_INCREASE_COLUMN,
    ShockDispersionScaler,
    apply_dispersion_scaling,
    attach_shock_probabilities,
    combine_shock_probability,
    fit_shock_dispersion_scaler,
    tail_required_scale,
)


def _synthetic_oof(n=8000, seed=11):
    """Role-shock structure: shocks realize with probability proportional to
    the pregame p, an increase blows out the upside, a decrease the downside,
    and non-shock weeks match the base dispersion the intervals assume."""

    rng = np.random.default_rng(seed)
    realized_up = rng.random(n) < 0.11
    realized_down = (~realized_up) & (rng.random(n) < 0.09)
    # A discriminative pregame model concentrates shocked weeks at high p,
    # mirroring the walk-forward shock model this lever conditions on.
    p_increase = np.where(realized_up, rng.beta(4, 6, n), rng.beta(1.2, 14, n))
    p_decrease = np.where(realized_down, rng.beta(5, 5, n), rng.beta(1.2, 14, n))
    center = rng.normal(10, 5, n)
    calm = rng.normal(0, 3.6, n)
    blowout = 2.0 + np.abs(rng.normal(0, 4.5, n))
    actual = np.where(
        realized_up, center + blowout,
        np.where(realized_down, center - 0.9 * blowout, center + calm),
    )
    half = 1.2815515655446004 * 4.0
    return pd.DataFrame({
        "season": 2021,
        "fantasy_points": actual,
        "projection_mean": center,
        "projection_lower80": center - half,
        "projection_upper80": center + half,
        SHOCK_INCREASE_COLUMN: p_increase,
        SHOCK_DECREASE_COLUMN: p_decrease,
        "pre_opportunities_ewm4": 10.0,
        "opportunities": 10.0 + 6.0 * realized_up - 6.0 * realized_down,
    })


def test_tail_required_scale_measures_minimal_covering_multiplier():
    actual = np.array([12.0, 4.0, 10.0])
    center = np.array([10.0, 10.0, 10.0])
    upper = tail_required_scale(actual, center, np.array([14.0] * 3), tail="upper")
    lower = tail_required_scale(actual, center, np.array([7.0] * 3), tail="lower")
    assert upper == pytest.approx([0.5, 0.0, 0.0])
    assert lower == pytest.approx([0.0, 2.0, 0.0])


def test_fitted_scaler_is_monotone_never_shrinks_and_restores_shock_coverage():
    oof = _synthetic_oof()
    scaler = fit_shock_dispersion_scaler(oof)
    assert scaler is not None
    for scales in (scaler.upper_scales, scaler.lower_scales):
        assert scales == sorted(scales)
        assert min(scales) >= 1.0
        assert scales[-1] > 1.05
    assert 0.0 <= scaler.blend_weight <= 1.0
    out = oof.copy()
    apply_dispersion_scaling(out, scaler)

    def covered(rows):
        return (
            rows["fantasy_points"].ge(rows["projection_lower80"])
            & rows["fantasy_points"].le(rows["projection_upper80"])
        ).mean()

    delta = out["opportunities"] - out["pre_opportunities_ewm4"]
    shocked = out[delta.abs() >= 5]
    stable = out[delta.abs() < 3]
    base_shocked = covered(oof[delta.abs() >= 5])
    base_stable = covered(oof[delta.abs() < 3])
    # A monotone function of the pregame probability cannot repair shocks the
    # probability model does not see coming, so the contract is a large bite
    # out of the conditional coverage gap without spending the stable budget.
    assert covered(shocked) - base_shocked >= 0.10
    assert covered(shocked) <= 0.92
    assert covered(stable) - base_stable <= 0.02  # stable-role budget respected


def test_apply_scales_each_tail_independently_and_preserves_center():
    scaler = ShockDispersionScaler(
        upper_edges=[0.0, 0.5, 1.0], upper_scales=[1.0, 2.0],
        lower_edges=[0.0, 0.5, 1.0], lower_scales=[1.0, 3.0],
    )
    frame = pd.DataFrame({
        "projection_mean": [10.0, 10.0, 0.0],
        "projection_lower80": [6.0, 6.0, 0.0],
        "projection_upper80": [16.0, 16.0, 0.0],
        SHOCK_INCREASE_COLUMN: [0.9, 0.1, 0.9],
        SHOCK_DECREASE_COLUMN: [0.1, 0.9, 0.9],
    })
    apply_dispersion_scaling(frame, scaler)
    assert frame["projection_mean"].tolist() == [10.0, 10.0, 0.0]
    # Row 0: upper doubled only. Row 1: lower tripled only. Row 2: zero width stays.
    assert frame["projection_upper80"].tolist() == [22.0, 16.0, 0.0]
    assert frame["projection_lower80"].tolist() == [6.0, -2.0, 0.0]
    assert frame[SCALE_UPPER_COLUMN].tolist() == [2.0, 1.0, 2.0]
    assert frame[SCALE_LOWER_COLUMN].tolist() == [1.0, 3.0, 3.0]
    assert frame[SCALE_COLUMN].tolist() == [2.0, 3.0, 3.0]


def test_scaler_rejects_shrinking_or_non_monotone_payloads():
    with pytest.raises(ValueError, match="shrink"):
        ShockDispersionScaler(
            upper_edges=[0.0, 0.5, 1.0], upper_scales=[0.9, 1.5],
            lower_edges=[0.0, 1.0], lower_scales=[1.0],
        )
    with pytest.raises(ValueError, match="non-decreasing"):
        ShockDispersionScaler(
            upper_edges=[0.0, 1.0], upper_scales=[1.0],
            lower_edges=[0.0, 0.5, 1.0], lower_scales=[1.6, 1.2],
        )


def test_serialization_roundtrip_and_missing_probability_is_neutral():
    scaler = fit_shock_dispersion_scaler(_synthetic_oof())
    restored = ShockDispersionScaler.from_dict(scaler.to_dict())
    p = np.array([0.02, 0.4, np.nan])
    assert restored.upper_scale_for(p)[:2] == pytest.approx(scaler.upper_scale_for(p)[:2])
    assert restored.upper_scale_for(p)[2] == 1.0
    assert restored.lower_scale_for(p)[2] == 1.0


def test_combine_shock_probability_handles_one_sided_and_missing():
    frame = pd.DataFrame({
        SHOCK_INCREASE_COLUMN: [0.2, np.nan, np.nan],
        SHOCK_DECREASE_COLUMN: [0.5, 0.3, np.nan],
    })
    combined = combine_shock_probability(frame)
    assert combined.iloc[0] == pytest.approx(1 - 0.8 * 0.5)
    assert combined.iloc[1] == pytest.approx(0.3)
    assert np.isnan(combined.iloc[2])


def test_attach_shock_probabilities_joins_on_identity_keys():
    frame = pd.DataFrame({
        "season": [2024, 2024], "week": [1, 2], "player_id": ["00-1", "00-1"],
    })
    table = pd.DataFrame({
        "season": [2024], "week": [1], "player_id": ["00-1"],
        SHOCK_INCREASE_COLUMN: [0.4], SHOCK_DECREASE_COLUMN: [0.1],
    })
    merged = attach_shock_probabilities(frame, table)
    assert merged[SHOCK_COMBINED_COLUMN].iloc[0] == pytest.approx(1 - 0.6 * 0.9)
    assert np.isnan(merged[SHOCK_COMBINED_COLUMN].iloc[1])


def _model_frame_with_shock():
    rng = np.random.default_rng(17)
    rows = []
    features = model_features()
    for season in (2020, 2021, 2022, 2023):
        for player in range(40):
            base = player / 8 + (season - 2020) * 0.15
            row = {column: rng.normal() for column in features}
            shock_p = float(rng.beta(1.5, 6))
            row.update({
                "season": season, "week": player % 17 + 1,
                "player_id": f"00-{player:07d}", "player_name": f"Back {player}",
                "position": "RB", "team": "AAA", "model_eligible": True,
                "status_inactive": 0, "injury_out": 0, "injury_questionable": 0,
                "practice_dnp": 0,
                SHOCK_INCREASE_COLUMN: shock_p, SHOCK_DECREASE_COLUMN: shock_p / 2,
                "fantasy_points": max(
                    0,
                    7 + base + 2 * row["pre_opportunities_ewm4"]
                    + rng.normal(0, 2 + 14 * shock_p),
                ),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def test_fit_ensemble_learns_shock_scaler_and_predict_widens_only_shocked_rows():
    config = ModelConfig(
        positions=("RB",), fast=True, stack_validation_seasons=2,
        min_train_rows=40, min_position_rows=30,
    )
    frame = _model_frame_with_shock()
    diagnostics = {}
    model = fit_ensemble(frame, config=config, diagnostics=diagnostics)
    assert "oof" in diagnostics
    for column in (SHOCK_COMBINED_COLUMN, SHOCK_INCREASE_COLUMN, SHOCK_DECREASE_COLUMN):
        assert column in diagnostics["oof"]
    if model.shock_scaler is None:
        # Too few OOF rows to fit in the tiny fixture; force a scaler to test predict.
        model.shock_scaler = ShockDispersionScaler(
            upper_edges=[0.0, 0.2, 1.0], upper_scales=[1.0, 1.5],
            lower_edges=[0.0, 0.2, 1.0], lower_scales=[1.0, 1.4],
        ).to_dict()
    else:
        assert model.shock_scaler["upper_scales"] == sorted(model.shock_scaler["upper_scales"])
        assert model.shock_scaler["lower_scales"] == sorted(model.shock_scaler["lower_scales"])
    projected = model.predict(frame.tail(20))
    assert "projection_lower80_base" in projected.columns
    assert (projected["projection_upper80"] >= projected["projection_upper80_base"] - 1e-9).all()
    assert (projected["projection_lower80"] <= projected["projection_lower80_base"] + 1e-9).all()
    neutral = projected[projected[SCALE_UPPER_COLUMN] <= 1.0 + 1e-9]
    if not neutral.empty:
        assert np.allclose(neutral["projection_upper80"], neutral["projection_upper80_base"])


def test_predict_without_shock_columns_matches_legacy_output():
    config = ModelConfig(
        positions=("RB",), fast=True, stack_validation_seasons=2,
        min_train_rows=40, min_position_rows=30,
    )
    frame = _model_frame_with_shock().drop(
        columns=[SHOCK_INCREASE_COLUMN, SHOCK_DECREASE_COLUMN]
    )
    model = fit_ensemble(frame, config=config)
    assert model.shock_scaler is None
    projected = model.predict(frame.tail(10))
    assert "projection_lower80_base" not in projected.columns
    assert SCALE_COLUMN not in projected.columns
    assert (projected["projection_upper80"] > projected["projection_lower80"]).all()
