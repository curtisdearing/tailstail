"""Correlated game-event Monte Carlo translated to arbitrary fantasy scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import ScoringRules, SimulationConfig
from .scoring import score_components


@dataclass
class SimulationResult:
    summaries: pd.DataFrame
    points: pd.DataFrame
    components: dict[str, pd.DataFrame]
    metadata: dict[str, Any]


def _first_numeric(frame: pd.DataFrame, column: str, fallback: float) -> float:
    if column not in frame:
        return fallback
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else fallback


def _numeric(frame: pd.DataFrame, column: str, fallback: float) -> np.ndarray:
    if column not in frame:
        return np.repeat(fallback, len(frame)).astype(float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(fallback).to_numpy(dtype=float)


def _availability(frame: pd.DataFrame) -> np.ndarray:
    probability = np.repeat(0.995, len(frame))
    probability -= 0.18 * _numeric(frame, "injury_questionable", 0.0)
    probability -= 0.28 * _numeric(frame, "practice_dnp", 0.0)
    probability -= 0.08 * _numeric(frame, "practice_limited", 0.0)
    doubtful = _numeric(frame, "injury_doubtful", 0.0).astype(bool)
    out = (
        _numeric(frame, "status_inactive", 0.0).astype(bool)
        | _numeric(frame, "injury_out", 0.0).astype(bool)
    )
    probability[doubtful] = np.minimum(probability[doubtful], 0.30)
    probability[out] = 0.0
    return np.clip(probability, 0.0, 0.995)


def _share_prior(frame: pd.DataFrame, column: str, kind: str) -> np.ndarray:
    position = frame["position"].astype(str).to_numpy()
    if kind == "target":
        fallback_map = {"RB": 0.09, "WR": 0.15, "TE": 0.10, "QB": 0.005}
    else:
        fallback_map = {"RB": 0.22, "WR": 0.012, "TE": 0.002, "QB": 0.09}
    fallback = np.asarray([fallback_map.get(value, 0.01) for value in position])
    values = _numeric(frame, column, np.nan)
    values = np.where(np.isfinite(values) & (values >= 0), values, fallback)
    return np.clip(values, 0.0005, 0.80)


def _draw_shares(
    rng: np.random.Generator,
    base: np.ndarray,
    available: np.ndarray,
    concentration: float,
) -> np.ndarray:
    """Draw player shares plus an unmodeled-roster bucket, then gate availability."""

    total = float(np.clip(base.sum(), 0.05, 0.94))
    normalized = base * (total / base.sum())
    alpha = np.r_[np.maximum(normalized * concentration, 0.15), max((1 - total) * concentration, 0.8)]
    draws = rng.gamma(shape=alpha, scale=1.0, size=(len(available), len(alpha)))
    draws[:, :-1] *= available
    denominator = draws.sum(axis=1, keepdims=True)
    denominator[denominator == 0] = 1.0
    return draws / denominator


def _row_multinomial(
    rng: np.random.Generator, totals: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    output = np.zeros_like(probabilities, dtype=int)
    for index, total in enumerate(totals.astype(int)):
        if total > 0:
            output[index] = rng.multinomial(total, probabilities[index])
    return output


def _assign_touchdowns(
    rng: np.random.Generator,
    totals: np.ndarray,
    event_counts: np.ndarray,
    prior: np.ndarray,
) -> np.ndarray:
    """Allocate team touchdowns to modeled players and an implicit other bucket."""

    simulations, players = event_counts.shape
    output = np.zeros((simulations, players), dtype=int)
    for index, total in enumerate(totals.astype(int)):
        if total <= 0:
            continue
        participated = event_counts[index] > 0
        weights = (
            (event_counts[index].astype(float) + 0.35)
            * np.maximum(prior, 0.02)
            * participated
        )
        other = max(float(np.mean(weights)) if players else 1.0, 0.5)
        probabilities = np.r_[weights, other]
        probabilities /= probabilities.sum()
        output[index] = rng.multinomial(total, probabilities)[:-1]
    return output


def _center_hurdle_distribution(
    raw: np.ndarray,
    available: np.ndarray,
    target_mean: float,
    target_sd: float,
    position: str,
    target_interval_width: float | None = None,
    residual_noise: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Calibrate the event simulator while preserving a true inactive zero mass."""

    p = float(available.mean())
    if p <= 0:
        return np.zeros_like(raw), False
    active_values = raw[available]
    active_mean = float(active_values.mean()) if len(active_values) else 0.0
    active_var = float(active_values.var()) if len(active_values) else 0.0
    conditional_target = target_mean / p
    target_var = max(float(target_sd) ** 2, 1.0)
    hurdle_var = p * (1 - p) * conditional_target**2
    remaining = max(target_var - hurdle_var, 0.25)
    scale = np.sqrt(remaining / max(p * active_var, 0.25))
    scale = float(np.clip(scale, 0.45, 2.75))
    default_floor = -6.0 if position == "QB" else -3.0
    residual_floor = (
        conditional_target - 1.5 * target_interval_width
        if target_interval_width is not None
        else default_floor
    )
    floor = min(default_floor, conditional_target - 0.25, residual_floor)

    def locate(centered: np.ndarray) -> np.ndarray:
        """Shift a clipped active distribution to its exact conditional mean."""

        low = floor - float(centered.max()) - 1.0
        high = conditional_target - float(centered.min()) + 1.0
        for _ in range(45):
            midpoint = (low + high) / 2.0
            candidate_mean = float(np.maximum(centered + midpoint, floor).mean())
            if candidate_mean < conditional_target:
                low = midpoint
            else:
                high = midpoint
        return np.maximum(centered + (low + high) / 2.0, floor)

    def calibrate_shape(active_shape: np.ndarray) -> np.ndarray:
        centered = scale * active_shape
        output = np.zeros_like(raw, dtype=float)
        output[available] = locate(centered)
        # SD matching assumes a symmetric distribution and was narrowing
        # p10-p90 for skewed event draws. Match the already out-of-sample
        # conformal width directly while retaining the supplied shape.
        if target_interval_width is not None and target_interval_width > 0:
            for _ in range(3):
                current_width = float(np.quantile(output, 0.90) - np.quantile(output, 0.10))
                if current_width <= 1e-9:
                    break
                ratio = float(np.clip(target_interval_width / current_width, 0.25, 4.0))
                if abs(ratio - 1.0) < 0.005:
                    break
                active = output[available]
                output[available] = locate((active - active.mean()) * ratio)
        return output

    calibrated = calibrate_shape(active_values - active_mean)
    used_residual_fallback = False
    if target_interval_width is not None and target_interval_width > 0:
        calibrated_width = float(np.quantile(calibrated, 0.90) - np.quantile(calibrated, 0.10))
        width_ratio = calibrated_width / target_interval_width
        unstable_scale = float(calibrated.std()) > 2.0 * max(target_sd, 1.0)
        if (not 0.90 <= width_ratio <= 1.10 or unstable_scale) and residual_noise is not None:
            fallback_shape = np.asarray(residual_noise, dtype=float)[available]
            fallback_shape -= fallback_shape.mean()
            fallback_shape /= max(float(fallback_shape.std()), 1e-9)
            calibrated = calibrate_shape(fallback_shape)
            used_residual_fallback = True
    return calibrated, used_residual_fallback


def simulate_week(
    slate: pd.DataFrame,
    *,
    config: SimulationConfig | None = None,
    scoring: ScoringRules | None = None,
) -> SimulationResult:
    """Simulate a slate with shared game pace, team volume, targets, carries and TDs.

    ``slate`` should be the output of ``FantasyEnsemble.predict``.  All roster
    rows may be supplied; only rows with a finite model projection are emitted.
    """

    cfg = config or SimulationConfig()
    rules = scoring or ScoringRules()
    required = {"player_id", "player_name", "position", "team", "projection_mean"}
    missing = sorted(required - set(slate.columns))
    if missing:
        raise ValueError(f"simulation slate missing columns: {missing}")
    if slate["player_id"].duplicated().any():
        raise ValueError("simulation slate must contain one row per player")
    players = slate[slate["projection_mean"].notna()].copy().reset_index(drop=True)
    if players.empty:
        raise ValueError("simulation slate contains no projected players")
    n = cfg.simulations
    p = len(players)
    rng = np.random.default_rng(cfg.random_seed)
    component_names = (
        "completions", "attempts", "passing_yards", "passing_tds",
        "passing_interceptions", "carries", "rushing_yards", "rushing_tds",
        "targets", "receptions", "receiving_yards", "receiving_tds",
        "fumbles_lost",
    )
    components = {name: np.zeros((n, p), dtype=float) for name in component_names}
    availability_prob = _availability(players)
    available_all = rng.random((n, p)) < availability_prob

    if "game_id" in players:
        game_values = players["game_id"].astype("object")
    else:
        game_values = pd.Series(np.nan, index=players.index, dtype="object")
    opponent = players.get("opponent_team", pd.Series("unknown", index=players.index)).fillna("unknown")
    fallback_game = ["_".join(sorted((str(team), str(opp)))) for team, opp in zip(players["team"], opponent)]
    players["_simulation_game"] = game_values.where(game_values.notna(), fallback_game)
    game_key = "_simulation_game"
    for game_value, game in players.groupby(game_key, dropna=False, sort=True):
        # One pace draw per game creates the required cross-team correlation.
        sigma = np.sqrt(np.log1p(cfg.team_volume_cv**2))
        pace = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=n)
        for team, team_frame in game.groupby("team", sort=True):
            idx = team_frame.index.to_numpy(dtype=int)
            available = available_all[:, idx].copy()
            positions = team_frame["position"].astype(str).to_numpy()

            # If an injury draw removes the nominal starter, the next available
            # quarterback inherits the team attempts. This is the role-shock
            # branch a one-number projection cannot represent.
            qb_local = np.flatnonzero(positions == "QB")
            qb_choice = np.full(n, -1, dtype=int)
            if len(qb_local):
                order = qb_local[np.argsort(-_numeric(team_frame.iloc[qb_local], "projection_mean", 0.0))]
                for candidate in order:
                    choose = (qb_choice < 0) & available[:, candidate]
                    qb_choice[choose] = candidate

            spread = _first_numeric(team_frame, "team_spread", 0.0)
            wind = _first_numeric(team_frame, "wind", 5.0)
            pass_mean = _first_numeric(team_frame, "pre_team_pass_attempts_ewm4", 34.0)
            rush_mean = _first_numeric(team_frame, "pre_team_rush_attempts_ewm4", 27.0)
            pass_lambda = np.clip(pass_mean * pace * (1 + spread / 75.0), 15, 55)
            rush_lambda = np.clip(rush_mean * pace * (1 - spread / 100.0), 12, 45)
            team_pass_attempts = rng.poisson(pass_lambda)
            team_carries = rng.poisson(rush_lambda)

            target_base = _share_prior(team_frame, "pre_target_share_calc_ewm4", "target")
            target_prob = _draw_shares(rng, target_base, available, cfg.target_share_concentration)
            target_counts_all = _row_multinomial(rng, team_pass_attempts, target_prob)
            targets = target_counts_all[:, :-1]
            other_targets = target_counts_all[:, -1]
            components["targets"][:, idx] = targets

            catch_default = np.asarray([0.72 if pos == "RB" else 0.64 for pos in positions])
            catch_rate = _numeric(team_frame, "pre_catch_rate_ewm8", np.nan)
            catch_rate = np.where(np.isfinite(catch_rate), catch_rate, catch_default)
            catch_rate = np.clip(catch_rate, 0.35, 0.90)
            catch_draw = rng.beta(catch_rate * 30 + 1, (1 - catch_rate) * 30 + 1, size=(n, len(idx)))
            receptions = rng.binomial(targets.astype(int), catch_draw)
            other_receptions = rng.binomial(other_targets.astype(int), 0.64)
            components["receptions"][:, idx] = receptions

            ypt = _numeric(team_frame, "pre_yards_per_target_ewm8", np.nan)
            ypt_default = np.asarray([5.8 if pos == "RB" else 7.7 if pos == "TE" else 8.2 for pos in positions])
            ypt = np.where(np.isfinite(ypt), ypt, ypt_default)
            yards_per_reception = np.clip(ypt / np.maximum(catch_rate, 0.25), 5.0, 22.0)
            shape_per_reception = max(1.0 / max((cfg.efficiency_cv * 3.0) ** 2, 0.05), 1.0)
            shape = np.maximum(receptions * shape_per_reception, 0.01)
            receiving_yards = rng.gamma(shape=shape, scale=yards_per_reception / shape_per_reception)
            receiving_yards[receptions == 0] = 0.0
            other_yards = rng.gamma(
                np.maximum(other_receptions * shape_per_reception, 0.01),
                11.5 / shape_per_reception,
            )
            other_yards[other_receptions == 0] = 0.0
            components["receiving_yards"][:, idx] = receiving_yards

            carry_base = _share_prior(team_frame, "pre_carry_share_ewm4", "carry")
            carry_eligible = np.isin(positions, ["QB", "RB", "WR"])
            carry_base = np.where(carry_eligible, carry_base, 0.0005)
            carry_prob = _draw_shares(rng, carry_base, available, cfg.carry_share_concentration)
            carry_counts_all = _row_multinomial(rng, team_carries, carry_prob)
            carries = carry_counts_all[:, :-1]
            components["carries"][:, idx] = carries
            ypc = _numeric(team_frame, "pre_yards_per_carry_ewm8", np.nan)
            ypc_default = np.asarray([5.2 if pos == "QB" else 4.3 for pos in positions])
            ypc = np.where(np.isfinite(ypc), ypc, ypc_default)
            rush_mean_matrix = carries * np.clip(ypc, 2.0, 8.5)
            rush_sd = np.sqrt(np.maximum(carries, 1)) * 3.1 * (cfg.efficiency_cv / 0.20)
            rushing_yards = rng.normal(rush_mean_matrix, rush_sd)
            rushing_yards[carries == 0] = 0.0
            rushing_yards = np.maximum(rushing_yards, -8.0)
            components["rushing_yards"][:, idx] = rushing_yards

            implied = _first_numeric(team_frame, "implied_team_points", 22.5)
            expected_tds = np.clip(implied * cfg.implied_points_offense_share / 7.0, 0.8, 4.5)
            offensive_tds = np.clip(rng.poisson(expected_tds, size=n), 0, 8)
            pass_fraction = np.clip(0.63 + spread / 180.0 - max(wind - 15, 0) / 120.0, 0.42, 0.78)
            pass_tds = rng.binomial(offensive_tds.astype(int), pass_fraction)
            rush_tds = offensive_tds - pass_tds
            td_prior = np.clip(_numeric(team_frame, "pre_td_per_opportunity_ewm8", 0.035), 0.005, 0.20)
            receiving_tds = _assign_touchdowns(rng, pass_tds, targets, td_prior)
            rushing_tds = _assign_touchdowns(rng, rush_tds, carries, td_prior)
            components["receiving_tds"][:, idx] = receiving_tds
            components["rushing_tds"][:, idx] = rushing_tds

            team_passing_yards = receiving_yards.sum(axis=1) + other_yards
            team_completions = receptions.sum(axis=1) + other_receptions
            interception_rate = np.clip(_numeric(team_frame, "pre_interception_rate_ewm8", 0.024), 0.005, 0.08)
            for simulation, local_qb in enumerate(qb_choice):
                if local_qb < 0:
                    continue
                global_qb = idx[local_qb]
                attempts = int(team_pass_attempts[simulation])
                components["attempts"][simulation, global_qb] = attempts
                components["completions"][simulation, global_qb] = team_completions[simulation]
                components["passing_yards"][simulation, global_qb] = team_passing_yards[simulation]
                components["passing_tds"][simulation, global_qb] = pass_tds[simulation]
                components["passing_interceptions"][simulation, global_qb] = rng.binomial(
                    attempts, interception_rate[local_qb]
                )

            touches = carries + receptions
            components["fumbles_lost"][:, idx] = rng.binomial(
                touches.astype(int), np.clip(0.008 + (positions == "QB") * 0.004, 0, 0.03)
            )

    raw_points = score_components(components, rules)
    calibrated = np.zeros_like(raw_points)
    residual_fallback = np.zeros(p, dtype=bool)
    for index, row in players.iterrows():
        model_mean = float(row["projection_mean"])
        target_mean = (
            cfg.model_center_weight * model_mean
            + (1.0 - cfg.model_center_weight) * float(raw_points[:, index].mean())
        )
        lower = float(row.get("projection_lower80", np.nan))
        upper = float(row.get("projection_upper80", np.nan))
        if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
            target_sd = (upper - lower) / (2 * 1.2815515655446004)
            target_interval_width = upper - lower
        else:
            target_sd = {"QB": 7.5, "RB": 6.5, "WR": 6.5, "TE": 5.5}.get(row["position"], 6.5)
            target_interval_width = None
        calibrated[:, index], residual_fallback[index] = _center_hurdle_distribution(
            raw_points[:, index],
            available_all[:, index],
            target_mean,
            target_sd,
            str(row["position"]),
            target_interval_width,
            rng.standard_normal(n),
        )

    ids = players["player_id"].astype(str).tolist()
    point_frame = pd.DataFrame(calibrated, columns=ids)
    component_frames = {
        name: pd.DataFrame(values, columns=ids) for name, values in components.items()
    }
    summary_rows = []
    for index, row in players.iterrows():
        values = calibrated[:, index]
        raw = raw_points[:, index]
        adjustment = float(values.mean() - raw.mean())
        component_means = {
            f"expected_{name}": float(component_values[:, index].mean())
            for name, component_values in components.items()
        }
        summary_rows.append({
            "player_id": str(row["player_id"]),
            "player_name": row["player_name"],
            "position": row["position"],
            "team": row["team"],
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "sd": float(values.std()),
            "p10": float(np.quantile(values, 0.10)),
            "p25": float(np.quantile(values, 0.25)),
            "p75": float(np.quantile(values, 0.75)),
            "p90": float(np.quantile(values, 0.90)),
            "prob_10_plus": float(np.mean(values >= 10)),
            "prob_15_plus": float(np.mean(values >= 15)),
            "prob_20_plus": float(np.mean(values >= 20)),
            "prob_25_plus": float(np.mean(values >= 25)),
            "availability_probability": float(available_all[:, index].mean()),
            "event_simulator_mean": float(raw.mean()),
            "event_simulator_sd": float(raw.std()),
            "model_residual_adjustment": adjustment,
            "calibration_residual_fallback": bool(residual_fallback[index]),
            "component_model_disagreement": bool(
                abs(adjustment) > max(5.0, 0.40 * abs(float(values.mean())))
            ),
            **component_means,
        })
    summaries = pd.DataFrame(summary_rows).sort_values("mean", ascending=False).reset_index(drop=True)
    return SimulationResult(
        summaries=summaries,
        points=point_frame,
        components=component_frames,
        metadata={
            "simulations": n,
            "random_seed": cfg.random_seed,
            "players": p,
            "games": int(players[game_key].nunique(dropna=False)),
            "scoring": rules.to_dict(),
            "calibration": "hurdle-preserving center and interval scale",
        },
    )
