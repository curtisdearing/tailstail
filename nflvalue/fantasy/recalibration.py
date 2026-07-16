"""P(role-shock)-conditioned interval and dispersion recalibration (M-F1v2).

Global 80% interval coverage is calibrated, but it is heterogeneous by role
cohort: weeks where a player's realized opportunities jump well above or
below their trailing expectation under-cover, while stable-role weeks are
slightly over-covered.  Blanket widening would fix the former by breaking
the latter.

This module conditions each interval tail on a *pregame* role-shock
probability (an external walk-forward model keyed by season/week/player_id):
the upper half-width is multiplied by a monotone non-decreasing step function
of P(role increase) and the lower half-width by one of P(role decrease).
Role-increase weeks miss above the upper bound and role-decrease weeks miss
below the lower bound, so per-tail conditioning targets each defect without
inflating the other side.  Both step functions are fitted on out-of-fold
validation rows -- seasons strictly before any evaluation fold -- by setting
each probability bin's multiplier to the empirical quantile that restores the
nominal per-tail exceedance rate, then enforcing monotonicity (PAVA) and a
floor of 1.0 so stable-role intervals are never shrunk.

Because the Monte Carlo simulator derives each player's target dispersion
from the projected interval width (``target_sd = (upper80 - lower80) /
(2 * z80)`` in ``simulation.py``), scaling the interval is exactly a scaling
of simulated output dispersion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

SHOCK_INCREASE_COLUMN = "p_role_increase"
SHOCK_DECREASE_COLUMN = "p_role_decrease"
SHOCK_COMBINED_COLUMN = "p_role_shock"
SCALE_COLUMN = "interval_shock_scale"
SCALE_UPPER_COLUMN = "interval_shock_scale_upper"
SCALE_LOWER_COLUMN = "interval_shock_scale_lower"

_INCREASE_CANDIDATES = ("p_role_increase", "p_increase_gbdt", "p_increase_logit", "p_increase")
_DECREASE_CANDIDATES = ("p_role_decrease", "p_decrease_gbdt", "p_decrease_logit", "p_decrease")

DEFAULT_BIN_EDGES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 1.0)


def load_shock_table(path) -> pd.DataFrame:
    """Load an external shock-probability table into the canonical schema.

    Accepts any parquet with season/week/player_id keys and increase or
    decrease probability columns under common names.  Probabilities must be
    strictly pregame and walk-forward out-of-sample for every season they
    cover; this loader cannot verify that, so only attach tables whose
    provenance documents it.
    """

    table = pd.read_parquet(path)
    missing = {"season", "week", "player_id"} - set(table.columns)
    if missing:
        raise ValueError(f"shock table missing key columns: {sorted(missing)}")
    out = pd.DataFrame({
        "season": pd.to_numeric(table["season"], errors="coerce").astype("Int64"),
        "week": pd.to_numeric(table["week"], errors="coerce").astype("Int64"),
        "player_id": table["player_id"].astype(str),
    })
    for target, candidates in (
        (SHOCK_INCREASE_COLUMN, _INCREASE_CANDIDATES),
        (SHOCK_DECREASE_COLUMN, _DECREASE_CANDIDATES),
    ):
        for name in candidates:
            if name in table.columns:
                out[target] = pd.to_numeric(table[name], errors="coerce").clip(0.0, 1.0)
                break
    if SHOCK_INCREASE_COLUMN not in out and SHOCK_DECREASE_COLUMN not in out:
        raise ValueError(
            f"shock table has no probability column; expected one of "
            f"{_INCREASE_CANDIDATES + _DECREASE_CANDIDATES}"
        )
    out = out.dropna(subset=["season", "week"])
    out = out.drop_duplicates(subset=["season", "week", "player_id"], keep="first")
    return out


def attach_shock_probabilities(frame: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Left-join canonical shock probabilities onto a feature frame.

    Rows without a match keep NaN probabilities and are never rescaled.
    """

    for column in (SHOCK_INCREASE_COLUMN, SHOCK_DECREASE_COLUMN, SHOCK_COMBINED_COLUMN):
        if column in frame.columns:
            frame = frame.drop(columns=[column])
    keyed = frame.assign(
        _join_season=pd.to_numeric(frame["season"], errors="coerce").astype("Int64"),
        _join_week=pd.to_numeric(frame["week"], errors="coerce").astype("Int64"),
        _join_player=frame["player_id"].astype(str),
    )
    right = table.rename(columns={
        "season": "_join_season", "week": "_join_week", "player_id": "_join_player",
    })
    merged = keyed.merge(
        right, on=["_join_season", "_join_week", "_join_player"], how="left", validate="many_to_one",
    ).drop(columns=["_join_season", "_join_week", "_join_player"])
    merged[SHOCK_COMBINED_COLUMN] = combine_shock_probability(merged)
    return merged


def combine_shock_probability(frame: pd.DataFrame) -> pd.Series:
    """P(any role shock) = 1 - (1 - P(increase)) * (1 - P(decrease)).

    A missing side counts as zero; rows missing both sides stay NaN so the
    scaler leaves them untouched.
    """

    increase = pd.to_numeric(frame.get(SHOCK_INCREASE_COLUMN, np.nan), errors="coerce")
    decrease = pd.to_numeric(frame.get(SHOCK_DECREASE_COLUMN, np.nan), errors="coerce")
    if np.isscalar(increase):
        increase = pd.Series(increase, index=frame.index, dtype=float)
    if np.isscalar(decrease):
        decrease = pd.Series(decrease, index=frame.index, dtype=float)
    both_missing = increase.isna() & decrease.isna()
    combined = 1.0 - (1.0 - increase.fillna(0.0)) * (1.0 - decrease.fillna(0.0))
    combined[both_missing] = np.nan
    return combined.clip(0.0, 1.0)


def tail_required_scale(
    actual: np.ndarray, center: np.ndarray, bound: np.ndarray, *, tail: str
) -> np.ndarray:
    """Minimal half-width multiplier at which one tail covers the actual.

    Rows on the other side of the center return 0: any non-negative scale
    keeps them covered by this tail.
    """

    actual = np.asarray(actual, dtype=float)
    center = np.asarray(center, dtype=float)
    half = np.maximum(np.abs(np.asarray(bound, dtype=float) - center), 1e-9)
    if tail == "upper":
        return np.where(actual > center, (actual - center) / half, 0.0)
    if tail == "lower":
        return np.where(actual < center, (center - actual) / half, 0.0)
    raise ValueError(f"unknown tail {tail!r}")


def _isotonic_non_decreasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators for a non-decreasing sequence."""

    merged: list[list[float]] = []
    counts: list[int] = []
    for value, weight in zip(values, weights):
        merged.append([float(value), float(weight)])
        counts.append(1)
        while len(merged) > 1 and merged[-2][0] > merged[-1][0]:
            v2, w2 = merged.pop()
            c2 = counts.pop()
            v1, w1 = merged[-1]
            total = w1 + w2
            merged[-1] = [(v1 * w1 + v2 * w2) / max(total, 1e-9), total]
            counts[-1] += c2
    out: list[float] = []
    for (value, _), count in zip(merged, counts):
        out.extend([value] * count)
    return np.asarray(out)


def _fit_tail_bins(
    probability: np.ndarray,
    tail_scale: np.ndarray,
    *,
    quantile: float,
    bin_edges: tuple[float, ...],
    min_bin_rows: int,
    max_scale: float,
    shocked: np.ndarray | None = None,
    min_shock_rows: int = 40,
    blend_weight: float = 0.0,
    neutral_below: float = 0.0,
) -> tuple[list[float], list[np.ndarray], np.ndarray] | None:
    """Per-bin quantile calibration with sparse-tail merging.

    Each bin multiplier blends two targets: the ``quantile`` of the required
    scale over every row in the bin (bin-marginal calibration) and the same
    quantile over only the rows whose role shock materialized
    (shock-conditional calibration).  ``blend_weight`` moves between them.
    Bins entirely at or below ``neutral_below`` stay at 1.0 so low-probability
    -- overwhelmingly stable -- rows are never touched.  Returns the raw bin
    structure (edges, row masks over valid rows, blended scales before
    monotonicity/floor); the caller applies PAVA and clipping.
    """

    valid = np.isfinite(probability)
    if valid.sum() < min_bin_rows:
        return None
    p = probability[valid]
    r = tail_scale[valid]
    shocked_valid = shocked[valid] if shocked is not None else None
    edges = np.asarray(bin_edges, dtype=float)
    assignment = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    kept_edges = [float(edges[0])]
    kept_masks: list[np.ndarray] = []
    pending = np.zeros(len(p), dtype=bool)
    for bin_index in range(len(edges) - 1):
        pending = pending | (assignment == bin_index)
        if pending.sum() >= min_bin_rows:
            kept_masks.append(pending)
            kept_edges.append(float(edges[bin_index + 1]))
            pending = np.zeros(len(p), dtype=bool)
    if pending.any():
        if kept_masks:
            kept_masks[-1] = kept_masks[-1] | pending
            kept_edges[-1] = float(edges[-1])
        else:
            kept_masks = [np.ones(len(p), dtype=bool)]
            kept_edges = [float(edges[0]), float(edges[-1])]
    elif kept_masks and kept_edges[-1] != float(edges[-1]):
        kept_edges[-1] = float(edges[-1])
    if not kept_masks:
        return None
    scales = []
    for index, mask in enumerate(kept_masks):
        if kept_edges[index + 1] <= neutral_below:
            scales.append(1.0)
            continue
        marginal = float(np.quantile(r[mask], quantile))
        conditional = marginal
        if shocked_valid is not None and blend_weight > 0:
            shocked_in_bin = mask & shocked_valid
            if shocked_in_bin.sum() >= min_shock_rows:
                conditional = float(np.quantile(r[shocked_in_bin], quantile))
        scales.append((1.0 - blend_weight) * marginal + blend_weight * max(conditional, marginal))
    return kept_edges, kept_masks, np.asarray(scales, dtype=float)


def _finalize_tail(scales: np.ndarray, masks: list[np.ndarray], max_scale: float) -> list[float]:
    counts = np.asarray([float(mask.sum()) for mask in masks])
    ordered = _isotonic_non_decreasing(scales, counts)
    return [float(v) for v in np.clip(ordered, 1.0, max_scale)]


def _validate_tail(edges: list[float], scales: list[float], name: str) -> None:
    if len(edges) != len(scales) + 1:
        raise ValueError(f"{name} bin_edges must have exactly one more entry than scales")
    if any(b < a for a, b in zip(scales, scales[1:])):
        raise ValueError(f"{name} scales must be non-decreasing")
    if any(scale < 1.0 for scale in scales):
        raise ValueError(f"{name} scales must never shrink an interval")


def _tail_scale_for(edges: list[float], scales: list[float], probability) -> np.ndarray:
    p = pd.to_numeric(pd.Series(probability), errors="coerce").to_numpy(dtype=float)
    out = np.ones(len(p), dtype=float)
    valid = np.isfinite(p)
    if valid.any() and scales:
        index = np.searchsorted(np.asarray(edges), p[valid], side="right") - 1
        index = np.clip(index, 0, len(scales) - 1)
        out[valid] = np.asarray(scales)[index]
    return out


@dataclass
class ShockDispersionScaler:
    """Per-tail monotone step functions p -> half-width multiplier.

    The upper tail is conditioned on P(role increase), the lower tail on
    P(role decrease).  Missing probabilities always map to 1.0 (no change).
    JSON-serializable for model cards and artifact storage.
    """

    upper_edges: list[float]
    upper_scales: list[float]
    lower_edges: list[float]
    lower_scales: list[float]
    nominal: float = 0.80
    fitted_rows: int = 0
    fitted_seasons: list[int] = field(default_factory=list)
    blend_weight: float = 0.0
    aggressiveness: float = 1.0

    def __post_init__(self) -> None:
        _validate_tail(self.upper_edges, self.upper_scales, "upper")
        _validate_tail(self.lower_edges, self.lower_scales, "lower")

    def upper_scale_for(self, probability) -> np.ndarray:
        return _tail_scale_for(self.upper_edges, self.upper_scales, probability)

    def lower_scale_for(self, probability) -> np.ndarray:
        return _tail_scale_for(self.lower_edges, self.lower_scales, probability)

    def to_dict(self) -> dict[str, Any]:
        return {
            "upper_edges": [float(v) for v in self.upper_edges],
            "upper_scales": [float(v) for v in self.upper_scales],
            "lower_edges": [float(v) for v in self.lower_edges],
            "lower_scales": [float(v) for v in self.lower_scales],
            "nominal": float(self.nominal),
            "fitted_rows": int(self.fitted_rows),
            "fitted_seasons": [int(s) for s in self.fitted_seasons],
            "blend_weight": float(self.blend_weight),
            "aggressiveness": float(self.aggressiveness),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ShockDispersionScaler":
        return cls(
            upper_edges=list(payload["upper_edges"]),
            upper_scales=list(payload["upper_scales"]),
            lower_edges=list(payload["lower_edges"]),
            lower_scales=list(payload["lower_scales"]),
            nominal=float(payload.get("nominal", 0.80)),
            fitted_rows=int(payload.get("fitted_rows", 0)),
            fitted_seasons=list(payload.get("fitted_seasons", [])),
            blend_weight=float(payload.get("blend_weight", 0.0)),
            aggressiveness=float(payload.get("aggressiveness", 1.0)),
        )


ROLE_SHOCK_OPPORTUNITY_DELTA = 5.0
ROLE_STABLE_OPPORTUNITY_DELTA = 3.0


def fit_shock_dispersion_scaler(
    oof: pd.DataFrame,
    *,
    nominal: float = 0.80,
    bin_edges: tuple[float, ...] = DEFAULT_BIN_EDGES,
    min_bin_rows: int = 60,
    min_fit_rows: int = 500,
    max_scale: float = 2.5,
    min_shock_rows: int = 25,
    neutral_below: float = 0.10,
    blend_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    stable_budget_pp: float = 1.4,
) -> ShockDispersionScaler | None:
    """Fit both per-tail scalers on out-of-fold predictions.

    ``oof`` needs columns ``fantasy_points`` (actual), ``projection_mean``,
    ``projection_lower80``, ``projection_upper80`` plus ``p_role_increase``
    and/or ``p_role_decrease``; ``opportunities`` and
    ``pre_opportunities_ewm4`` additionally enable shock-conditional
    calibration.  Rows must come from seasons strictly before any fold the
    scaler is applied to; the caller owns that guarantee (``fit_ensemble``
    uses stack validation seasons, which precede every test season by
    construction).

    Each probability bin's multiplier blends two out-of-fold targets: the
    ``1 - alpha/2`` required-scale quantile over every bin row (calibrating
    the bin's marginal exceedance rate) and the same quantile over only the
    bin rows whose role shock materialized (calibrating coverage
    *conditional on the shock happening*).  Marginal-only calibration is
    provably insufficient for shock-conditional coverage when most
    high-probability weeks still resolve stable, so the blend weight is
    chosen on the out-of-fold rows themselves: the largest realized-shock
    coverage whose realized-stable coverage drift stays within
    ``stable_budget_pp`` percentage points.
    """

    required_columns = {"fantasy_points", "projection_mean", "projection_lower80", "projection_upper80"}
    missing = sorted(required_columns - set(oof.columns))
    if missing:
        raise ValueError(f"scaler fit frame missing columns: {missing}")
    numeric = {column: pd.to_numeric(oof[column], errors="coerce") for column in required_columns}
    for column in (
        SHOCK_INCREASE_COLUMN, SHOCK_DECREASE_COLUMN,
        "opportunities", "pre_opportunities_ewm4",
    ):
        numeric[column] = (
            pd.to_numeric(oof[column], errors="coerce")
            if column in oof.columns else pd.Series(np.nan, index=oof.index)
        )
    frame = pd.DataFrame(numeric)
    frame = frame.dropna(subset=list(required_columns))
    frame = frame[frame["projection_upper80"] > frame["projection_lower80"]]
    has_probability = frame[[SHOCK_INCREASE_COLUMN, SHOCK_DECREASE_COLUMN]].notna().any(axis=1)
    frame = frame[has_probability]
    if len(frame) < min_fit_rows:
        return None
    quantile = (1.0 + nominal) / 2.0  # per-tail non-exceedance rate
    actual = frame["fantasy_points"].to_numpy()
    center = frame["projection_mean"].to_numpy()
    lower_bound = frame["projection_lower80"].to_numpy()
    upper_bound = frame["projection_upper80"].to_numpy()
    upper_required = tail_required_scale(actual, center, upper_bound, tail="upper")
    lower_required = tail_required_scale(actual, center, lower_bound, tail="lower")
    p_up = frame[SHOCK_INCREASE_COLUMN].to_numpy(dtype=float)
    p_down = frame[SHOCK_DECREASE_COLUMN].to_numpy(dtype=float)

    delta = frame["opportunities"].to_numpy() - frame["pre_opportunities_ewm4"].to_numpy()
    with np.errstate(invalid="ignore"):
        realized_up = np.where(np.isfinite(delta), delta >= ROLE_SHOCK_OPPORTUNITY_DELTA, False)
        realized_down = np.where(np.isfinite(delta), delta <= -ROLE_SHOCK_OPPORTUNITY_DELTA, False)
        realized_stable = np.where(
            np.isfinite(delta), np.abs(delta) < ROLE_STABLE_OPPORTUNITY_DELTA, False
        )
    outcomes_known = bool(np.isfinite(delta).any())
    weights = list(blend_grid) if outcomes_known else [0.0]

    def tails_for(weight: float):
        upper_fit = _fit_tail_bins(
            p_up, upper_required, quantile=quantile, bin_edges=bin_edges,
            min_bin_rows=min_bin_rows, max_scale=max_scale,
            shocked=realized_up, min_shock_rows=min_shock_rows,
            blend_weight=weight, neutral_below=neutral_below,
        )
        lower_fit = _fit_tail_bins(
            p_down, lower_required, quantile=quantile, bin_edges=bin_edges,
            min_bin_rows=min_bin_rows, max_scale=max_scale,
            shocked=realized_down, min_shock_rows=min_shock_rows,
            blend_weight=weight, neutral_below=neutral_below,
        )
        if upper_fit is None and lower_fit is None:
            return None
        neutral = ([0.0, 1.0], [1.0])
        if upper_fit is not None:
            upper_edges = upper_fit[0]
            upper_scales = _finalize_tail(upper_fit[2], upper_fit[1], max_scale)
        else:
            upper_edges, upper_scales = neutral
        if lower_fit is not None:
            lower_edges = lower_fit[0]
            lower_scales = _finalize_tail(lower_fit[2], lower_fit[1], max_scale)
        else:
            lower_edges, lower_scales = neutral
        return upper_edges, upper_scales, lower_edges, lower_scales

    def oof_coverage(candidate, mask: np.ndarray) -> float:
        if not mask.any():
            return float("nan")
        upper_edges, upper_scales, lower_edges, lower_scales = candidate
        up_scale = _tail_scale_for(upper_edges, upper_scales, p_up[mask])
        down_scale = _tail_scale_for(lower_edges, lower_scales, p_down[mask])
        scaled_lower = center[mask] - down_scale * (center[mask] - lower_bound[mask])
        scaled_upper = center[mask] + up_scale * (upper_bound[mask] - center[mask])
        return float(np.mean((actual[mask] >= scaled_lower) & (actual[mask] <= scaled_upper)))

    realized_shock = realized_up | realized_down
    baseline_stable = (
        float(np.mean((actual[realized_stable] >= lower_bound[realized_stable])
                      & (actual[realized_stable] <= upper_bound[realized_stable])))
        if realized_stable.any() else float("nan")
    )

    def dampen(candidate, aggressiveness: float):
        upper_edges, upper_scales, lower_edges, lower_scales = candidate
        return (
            upper_edges, [1.0 + aggressiveness * (v - 1.0) for v in upper_scales],
            lower_edges, [1.0 + aggressiveness * (v - 1.0) for v in lower_scales],
        )

    def side_improvement(candidate, mask: np.ndarray) -> float:
        if not mask.any():
            return 0.0
        base = float(np.mean((actual[mask] >= lower_bound[mask]) & (actual[mask] <= upper_bound[mask])))
        new = oof_coverage(candidate, mask)
        return abs(nominal - base) - abs(nominal - new)

    chosen = None
    chosen_weight = 0.0
    chosen_aggressiveness = 1.0
    best_objective = (-np.inf, -np.inf)
    for weight in weights:
        candidate = tails_for(weight)
        if candidate is None:
            continue
        # Damping toward neutral is the release valve: the largest multiplier
        # profile whose out-of-fold stable-role drift stays inside the budget.
        # aggressiveness 0 is always inside it, so every weight yields one.
        for aggressiveness in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
            damped = dampen(candidate, aggressiveness)
            if outcomes_known and realized_stable.any():
                drift = 100.0 * (oof_coverage(damped, realized_stable) - baseline_stable)
                if drift > stable_budget_pp:
                    continue
            break
        # The accuracy gate binds on whichever realized-shock side improves
        # less, so optimize the minimum side improvement, then the sum.
        up_gain = side_improvement(damped, realized_up)
        down_gain = side_improvement(damped, realized_down)
        objective = (min(up_gain, down_gain), up_gain + down_gain)
        if chosen is None or objective > best_objective:
            chosen = damped
            chosen_weight = weight
            chosen_aggressiveness = aggressiveness
            best_objective = objective
    if chosen is None:
        return None
    upper_edges, upper_scales, lower_edges, lower_scales = chosen
    seasons = sorted(
        pd.to_numeric(oof.get("season"), errors="coerce").dropna().astype(int).unique().tolist()
    ) if "season" in oof.columns else []
    scaler = ShockDispersionScaler(
        upper_edges=upper_edges, upper_scales=upper_scales,
        lower_edges=lower_edges, lower_scales=lower_scales,
        nominal=float(nominal), fitted_rows=int(len(frame)), fitted_seasons=seasons,
    )
    scaler.blend_weight = float(chosen_weight)
    scaler.aggressiveness = float(chosen_aggressiveness)
    return scaler


def apply_dispersion_scaling(
    frame: pd.DataFrame,
    scaler: ShockDispersionScaler,
) -> pd.DataFrame:
    """Scale interval half-widths in place around an untouched center.

    The projection mean is never moved, so point-accuracy metrics are
    unchanged by construction.  Zero-width rows (forced-inactive players)
    are unaffected for any scale, and missing probabilities scale by 1.
    """

    upper_scales = scaler.upper_scale_for(frame.get(SHOCK_INCREASE_COLUMN, np.nan))
    lower_scales = scaler.lower_scale_for(frame.get(SHOCK_DECREASE_COLUMN, np.nan))
    center = pd.to_numeric(frame["projection_mean"], errors="coerce").to_numpy(dtype=float)
    lower = pd.to_numeric(frame["projection_lower80"], errors="coerce").to_numpy(dtype=float)
    upper = pd.to_numeric(frame["projection_upper80"], errors="coerce").to_numpy(dtype=float)
    frame[SCALE_UPPER_COLUMN] = upper_scales
    frame[SCALE_LOWER_COLUMN] = lower_scales
    frame[SCALE_COLUMN] = np.maximum(upper_scales, lower_scales)
    frame["projection_lower80"] = center - lower_scales * (center - lower)
    frame["projection_upper80"] = center + upper_scales * (upper - center)
    return frame
