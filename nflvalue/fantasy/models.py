"""Position-specific Bayesian/nonlinear ensemble with honest OOF stacking."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import ModelConfig, ScoringRules
from .features import model_features
from .recalibration import (
    SCALE_COLUMN,
    SHOCK_COMBINED_COLUMN,
    SHOCK_DECREASE_COLUMN,
    SHOCK_INCREASE_COLUMN,
    ShockDispersionScaler,
    apply_dispersion_scaling,
    combine_shock_probability,
    fit_shock_dispersion_scaler,
)


@dataclass
class PositionModel:
    position: str
    models: dict[str, Any]
    weights: dict[str, float]
    residual_quantiles: tuple[float, float]
    validation_rows: int
    validation_metrics: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class FantasyEnsemble:
    feature_columns: list[str]
    positions: dict[str, PositionModel]
    config: dict[str, Any]
    scoring: dict[str, float]
    trained_at: str
    training_seasons: list[int]
    target: str = "fantasy_points"
    shock_scaler: dict[str, Any] | None = None

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.feature_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"projection frame missing model features: {missing}")
        out = frame.copy()
        out["projection_mean"] = np.nan
        out["projection_lower80"] = np.nan
        out["projection_upper80"] = np.nan
        out["projection_model_sd"] = np.nan
        for position, artifact in self.positions.items():
            mask = out["position"].eq(position)
            if not mask.any():
                continue
            x = out.loc[mask, self.feature_columns].replace([np.inf, -np.inf], np.nan)
            base = np.column_stack([
                _bounded_predict(artifact.models[name], x)
                for name in artifact.weights
            ])
            weights = np.asarray([artifact.weights[name] for name in artifact.weights])
            center = np.sum(base * weights, axis=1)
            # A direct PPR model can produce tiny negatives around replacement
            # level. Keep QB turnover downside but prevent impossible tails.
            floor = -4.0 if position == "QB" else -2.0
            center = np.clip(center, floor, 60.0)
            disagreement = np.std(base, axis=1)
            lower_q, upper_q = artifact.residual_quantiles
            interval_scale = _interval_heuristic_scale(out.loc[mask])
            lower = center + lower_q * interval_scale
            upper = center + upper_q * interval_scale
            inactive = (
                pd.to_numeric(out.loc[mask].get("status_inactive"), errors="coerce").fillna(0).eq(1)
                | pd.to_numeric(out.loc[mask].get("injury_out"), errors="coerce").fillna(0).eq(1)
            ).to_numpy()
            center[inactive] = 0.0
            lower[inactive] = 0.0
            upper[inactive] = 0.0
            disagreement[inactive] = 0.0
            out.loc[mask, "projection_mean"] = center
            out.loc[mask, "projection_lower80"] = lower
            out.loc[mask, "projection_upper80"] = upper
            out.loc[mask, "projection_model_sd"] = disagreement
        scaler_payload = getattr(self, "shock_scaler", None)
        if scaler_payload:
            probability = combine_shock_probability(out)
            if probability.notna().any():
                out[SHOCK_COMBINED_COLUMN] = probability
                out["projection_lower80_base"] = out["projection_lower80"]
                out["projection_upper80_base"] = out["projection_upper80"]
                apply_dispersion_scaling(out, ShockDispersionScaler.from_dict(scaler_payload))
        return out

    def save(self, path: str | Path) -> None:
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("joblib is required to persist fantasy models") from exc
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "FantasyEnsemble":
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("joblib is required to load fantasy models") from exc
        artifact = joblib.load(path)
        if not isinstance(artifact, cls):
            raise TypeError(f"{path} is not a FantasyEnsemble artifact")
        return artifact

    def model_card(self) -> dict[str, Any]:
        return {
            "trained_at": self.trained_at,
            "training_seasons": self.training_seasons,
            "target": self.target,
            "feature_count": len(self.feature_columns),
            "config": self.config,
            "scoring": self.scoring,
            "shock_scaler": getattr(self, "shock_scaler", None),
            "positions": {
                position: {
                    "weights": artifact.weights,
                    "residual_quantiles": artifact.residual_quantiles,
                    "validation_rows": artifact.validation_rows,
                    "validation_metrics": artifact.validation_metrics,
                }
                for position, artifact in self.positions.items()
            },
        }

    def write_model_card(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_card(), indent=2, sort_keys=True) + "\n")


def _interval_heuristic_scale(rows: pd.DataFrame) -> np.ndarray:
    """Pre-shock heuristic width multiplier shared by predict and OOF fitting."""

    role_short = pd.to_numeric(rows["pre_opportunities_ewm4"], errors="coerce")
    if "pre_opportunities_ewm8" in rows:
        role_long = pd.to_numeric(rows["pre_opportunities_ewm8"], errors="coerce")
    else:
        role_long = role_short.copy()
    role_volatility = ((role_short - role_long).abs() / (role_long.abs() + 2.0)).fillna(0.0).clip(0, 2)
    injury = (
        pd.to_numeric(rows.get("injury_questionable"), errors="coerce").fillna(0.0)
        + pd.to_numeric(rows.get("practice_dnp"), errors="coerce").fillna(0.0)
    ).clip(0, 1)
    return 1.0 + 0.35 * role_volatility.to_numpy() + 0.15 * injury.to_numpy()


def _make_estimator(name: str, config: ModelConfig):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import BayesianRidge
    from sklearn.neural_network import MLPRegressor
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler

    seed = config.random_seed
    if name == "bayesian":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=False, keep_empty_features=True),
            VarianceThreshold(threshold=1e-10),
            StandardScaler(),
            PCA(n_components=0.99, svd_solver="full"),
            BayesianRidge(),
        )
    if name == "gradient_boosting":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            HistGradientBoostingRegressor(
                loss="squared_error", learning_rate=0.045,
                # Larger learners overfit the season-forward player-week
                # sample. Production deliberately uses the validated compact
                # capacity; fast mode is a still-smaller smoke-test profile.
                max_iter=60 if config.fast else 90,
                max_leaf_nodes=15, min_samples_leaf=25,
                l2_regularization=1.5, random_state=seed,
            ),
        )
    if name == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            RandomForestRegressor(
                n_estimators=40 if config.fast else 60,
                max_features=0.65, min_samples_leaf=10,
                max_depth=None, n_jobs=1 if config.fast else -1, random_state=seed,
            ),
        )
    if name == "mlp":
        network = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            VarianceThreshold(threshold=1e-10),
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(16,) if config.fast else (24,),
                activation="relu", solver="adam", alpha=0.02,
                learning_rate_init=0.002, max_iter=75 if config.fast else 100,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=15, random_state=seed,
            ),
        )
        # Scaling the target materially stabilizes gradient descent across QB
        # and low-volume TE distributions.
        return TransformedTargetRegressor(regressor=network, transformer=StandardScaler())
    raise ValueError(f"unknown model family {name!r}")


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    from scipy.stats import spearmanr

    error = actual - predicted
    correlation = spearmanr(actual, predicted, nan_policy="omit").statistic
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias_actual_minus_prediction": float(np.mean(error)),
        "spearman": float(correlation if np.isfinite(correlation) else 0.0),
    }


def _bounded_predict(estimator, features: pd.DataFrame) -> np.ndarray:
    """Convert pathological learner output into an explicit failed-fit signal.

    Bounds are wider than any normal weekly projection.  They stop one
    ill-conditioned fold from poisoning stack optimization while preserving
    the direction of a large miss for validation scoring.
    """

    # NumPy 2 + some Accelerate builds emit spurious matmul overflow warnings
    # for finite PCA arrays; the finite-value assertion below is authoritative.
    with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        predicted = np.asarray(estimator.predict(features), dtype=float)
    if not np.isfinite(predicted).all():
        raise FloatingPointError("model emitted a non-finite fantasy projection")
    return np.clip(predicted, -10.0, 70.0)


def _stack_weights(
    actual: np.ndarray, predictions: np.ndarray, config: ModelConfig
) -> np.ndarray:
    count = predictions.shape[1]
    initial = np.repeat(1.0 / count, count)

    def objective(weights: np.ndarray) -> float:
        predicted = np.sum(predictions * weights, axis=1)
        error = actual - predicted
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(np.square(error)))
        shrink = config.stack_l2 * np.square(weights - initial).sum()
        return float(0.80 * mae + 0.20 * rmse + shrink)

    result = minimize(
        objective, initial, method="SLSQP",
        bounds=[(0.0, config.stack_weight_cap)] * count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success or not np.isfinite(result.x).all():
        return initial
    weights = np.clip(result.x, 0, config.stack_weight_cap)
    return weights / weights.sum()


def fit_ensemble(
    frame: pd.DataFrame,
    *,
    config: ModelConfig | None = None,
    scoring: ScoringRules | None = None,
    target: str = "fantasy_points",
    diagnostics: dict | None = None,
) -> FantasyEnsemble:
    """Fit position models using multi-season out-of-fold stack weights."""

    cfg = config or ModelConfig()
    rules = scoring or ScoringRules()
    if "scoring_rules" in frame:
        observed = set(frame["scoring_rules"].dropna().astype(str).unique())
        expected = json.dumps(rules.to_dict(), sort_keys=True)
        if observed and observed != {expected}:
            raise ValueError(
                "frame scoring does not match requested model scoring; rebuild the feature frame"
            )
    features = model_features()
    missing = sorted(set(features + ["season", "position", target, "model_eligible"]) - set(frame.columns))
    if missing:
        raise ValueError(f"training frame missing columns: {missing}")
    eligible = frame[frame["model_eligible"].fillna(False) & frame[target].notna()].copy()
    eligible[features] = eligible[features].replace([np.inf, -np.inf], np.nan)
    seasons = sorted(eligible["season"].astype(int).unique().tolist())
    if len(eligible) < cfg.min_train_rows or len(seasons) < 2:
        raise ValueError("insufficient eligible history; need multiple seasons and more training rows")
    position_models: dict[str, PositionModel] = {}
    families = list(cfg.model_families)
    oof_records: list[pd.DataFrame] = []

    for position in cfg.positions:
        data = eligible[eligible["position"].eq(position)].copy()
        if len(data) < cfg.min_position_rows:
            continue
        position_seasons = sorted(data["season"].astype(int).unique().tolist())
        candidates = position_seasons[1:]
        validation_seasons = candidates[-cfg.stack_validation_seasons:]
        oof_actual: list[np.ndarray] = []
        oof_by_family: dict[str, list[np.ndarray]] = {name: [] for name in families}
        oof_slices: list[pd.DataFrame] = []

        for validation_season in validation_seasons:
            train = data[data["season"].astype(int) < validation_season]
            validation = data[data["season"].astype(int) == validation_season]
            if len(train) < cfg.min_position_rows or len(validation) < 10:
                continue
            oof_actual.append(validation[target].to_numpy(dtype=float))
            oof_slices.append(validation)
            for family in families:
                estimator = _make_estimator(family, cfg)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    estimator.fit(train[features], train[target])
                oof_by_family[family].append(_bounded_predict(estimator, validation[features]))

        if not oof_actual or any(not values for values in oof_by_family.values()):
            raise ValueError(f"{position} has no valid out-of-fold validation season")
        actual = np.concatenate(oof_actual)
        matrix = np.column_stack([np.concatenate(oof_by_family[name]) for name in families])
        weights = _stack_weights(actual, matrix, cfg)
        stacked = np.sum(matrix * weights, axis=1)
        residual = actual - stacked
        alpha = cfg.conformal_alpha
        residual_quantiles = (
            float(np.quantile(residual, alpha / 2)),
            float(np.quantile(residual, 1 - alpha / 2)),
        )
        oof_slice = pd.concat(oof_slices, ignore_index=True)
        oof_floor = -4.0 if position == "QB" else -2.0
        oof_center = np.clip(stacked, oof_floor, 60.0)
        oof_heuristic = _interval_heuristic_scale(oof_slice)
        oof_record = pd.DataFrame({
            "position": position,
            "fantasy_points": actual,
            "projection_mean": oof_center,
            "projection_lower80": oof_center + residual_quantiles[0] * oof_heuristic,
            "projection_upper80": oof_center + residual_quantiles[1] * oof_heuristic,
            SHOCK_COMBINED_COLUMN: combine_shock_probability(oof_slice).to_numpy(),
        })
        for extra_column in (
            SHOCK_INCREASE_COLUMN, SHOCK_DECREASE_COLUMN,
            "opportunities", "pre_opportunities_ewm4",
        ):
            if extra_column in oof_slice:
                oof_record[extra_column] = pd.to_numeric(
                    oof_slice[extra_column], errors="coerce"
                ).to_numpy()
        if "season" in oof_slice:
            oof_record["season"] = pd.to_numeric(oof_slice["season"], errors="coerce").to_numpy()
        inactive = np.zeros(len(oof_slice), dtype=bool)
        for column in ("status_inactive", "injury_out"):
            if column in oof_slice:
                inactive |= pd.to_numeric(oof_slice[column], errors="coerce").fillna(0).eq(1).to_numpy()
        oof_records.append(oof_record.loc[~inactive])
        validation_metrics = {
            family: _metrics(actual, matrix[:, index])
            for index, family in enumerate(families)
        }
        validation_metrics["stack"] = _metrics(actual, stacked)

        final_models: dict[str, Any] = {}
        for family in families:
            estimator = _make_estimator(family, cfg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(data[features], data[target])
            final_models[family] = estimator
        position_models[position] = PositionModel(
            position=position,
            models=final_models,
            weights={family: float(weight) for family, weight in zip(families, weights)},
            residual_quantiles=residual_quantiles,
            validation_rows=int(len(actual)),
            validation_metrics=validation_metrics,
        )

    if not position_models:
        raise ValueError("no position model could be trained")
    shock_scaler_payload = None
    pooled_oof = pd.concat(oof_records, ignore_index=True) if oof_records else pd.DataFrame()
    if not pooled_oof.empty and pooled_oof[SHOCK_COMBINED_COLUMN].notna().any():
        fitted_scaler = fit_shock_dispersion_scaler(pooled_oof, nominal=1.0 - cfg.conformal_alpha)
        if fitted_scaler is not None:
            shock_scaler_payload = fitted_scaler.to_dict()
    if diagnostics is not None:
        diagnostics["oof"] = pooled_oof
        diagnostics["shock_scaler"] = shock_scaler_payload
    return FantasyEnsemble(
        feature_columns=features,
        positions=position_models,
        config=asdict(cfg),
        scoring=rules.to_dict(),
        trained_at=datetime.now(timezone.utc).isoformat(),
        training_seasons=seasons,
        target=target,
        shock_scaler=shock_scaler_payload,
    )


def season_forward_backtest(
    frame: pd.DataFrame,
    test_seasons: list[int],
    *,
    config: ModelConfig | None = None,
    scoring: ScoringRules | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Refit before every test season and return untouched outer predictions."""

    cfg = config or ModelConfig(fast=True, stack_validation_seasons=2)
    predictions: list[pd.DataFrame] = []
    for season in sorted(set(map(int, test_seasons))):
        train = frame[frame["season"].astype(int) < season]
        test = frame[
            frame["season"].astype(int).eq(season)
            & frame["model_eligible"].fillna(False)
        ]
        if train["season"].nunique() < 2 or test.empty:
            continue
        artifact = fit_ensemble(train, config=cfg, scoring=scoring)
        projected = artifact.predict(test)
        columns = [
            "season", "week", "player_id", "player_name", "position", "team",
            "fantasy_points", "projection_mean", "projection_lower80",
            "projection_upper80", "pre_fantasy_points_ewm4",
            "pre_expected_points_ewm4", "total_tds", "opportunities",
            "pre_opportunities_ewm4", "team_changed", "qb_changed",
            "injury_questionable", "practice_dnp",
        ]
        columns += [
            column
            for column in (
                SHOCK_COMBINED_COLUMN, SHOCK_INCREASE_COLUMN, SHOCK_DECREASE_COLUMN,
                SCALE_COLUMN, "projection_lower80_base", "projection_upper80_base",
            )
            if column in projected.columns
        ]
        predictions.append(projected[columns])
    if not predictions:
        raise ValueError("no season-forward predictions were produced")
    output = pd.concat(predictions, ignore_index=True)
    report = evaluate_predictions(output)
    return output, report


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    actual = predictions["fantasy_points"].to_numpy(dtype=float)
    projected = predictions["projection_mean"].to_numpy(dtype=float)
    report: dict[str, Any] = {
        "rows": int(len(predictions)),
        "overall": _metrics(actual, projected),
        "coverage80": float(np.mean(
            (actual >= predictions["projection_lower80"].to_numpy(dtype=float))
            & (actual <= predictions["projection_upper80"].to_numpy(dtype=float))
        )),
        "by_position": {},
        "by_season": {},
    }
    for position, group in predictions.groupby("position"):
        position_metrics = _metrics(
            group["fantasy_points"].to_numpy(dtype=float),
            group["projection_mean"].to_numpy(dtype=float),
        )
        position_metrics["coverage80"] = float(np.mean(
            group["fantasy_points"].ge(group["projection_lower80"])
            & group["fantasy_points"].le(group["projection_upper80"])
        ))
        report["by_position"][position] = position_metrics
    for season, group in predictions.groupby("season"):
        season_metrics = _metrics(
            group["fantasy_points"].to_numpy(dtype=float),
            group["projection_mean"].to_numpy(dtype=float),
        )
        season_metrics["coverage80"] = float(np.mean(
            group["fantasy_points"].ge(group["projection_lower80"])
            & group["fantasy_points"].le(group["projection_upper80"])
        ))
        report["by_season"][str(int(season))] = season_metrics
    for baseline in ("pre_fantasy_points_ewm4", "pre_expected_points_ewm4"):
        if baseline in predictions:
            values = pd.to_numeric(predictions[baseline], errors="coerce")
            valid = values.notna()
            report[f"baseline_{baseline}"] = _metrics(
                predictions.loc[valid, "fantasy_points"].to_numpy(dtype=float),
                values[valid].to_numpy(dtype=float),
            )
    return report
