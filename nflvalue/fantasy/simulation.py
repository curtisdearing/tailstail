"""Correlated game-event Monte Carlo translated to arbitrary fantasy scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import ScoringRules, SimulationConfig
from .role_state import ACTIVE_STATES, STATE_PROB_COLUMNS, state_multiplier_table
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
    multiplier: np.ndarray | None = None,
) -> np.ndarray:
    """Draw player shares plus an unmodeled-roster bucket, then gate availability.

    ``multiplier`` optionally rescales each (draw, player) share before the
    joint normalization; because every draw is renormalized across the team
    (including the unmodeled bucket), team volume is conserved exactly.
    """

    total = float(np.clip(base.sum(), 0.05, 0.94))
    normalized = base * (total / base.sum())
    alpha = np.r_[np.maximum(normalized * concentration, 0.15), max((1 - total) * concentration, 0.8)]
    draws = rng.gamma(shape=alpha, scale=1.0, size=(len(available), len(alpha)))
    gate = available if multiplier is None else available * multiplier
    draws[:, :-1] *= gate
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


def _sample_role_states(
    rng: np.random.Generator,
    active_probs: np.ndarray,
    simulations: int,
) -> np.ndarray:
    """Sample one active role state per (draw, player) from pregame probabilities."""

    cumulative = np.cumsum(active_probs, axis=1)
    uniforms = rng.random((simulations, active_probs.shape[0]))
    state_index = (uniforms[:, :, None] >= cumulative[None, :, :]).sum(axis=2)
    return np.clip(state_index, 0, active_probs.shape[1] - 1)


def _role_mixture_inputs(
    players: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Active-state probabilities plus per-player state multiplier rows.

    Missing or degenerate probability rows fall back to certain-``stable``,
    which reproduces baseline behaviour for that player.  ``up_mass`` and
    ``down_mass`` are the pregame probabilities of role-increase and
    role-decrease states; the directional interval-width law keys on them.
    (Multiplier-variance width laws were tested and rejected: ratio-scale
    variance is dominated by fringe players whose conformal widths already
    price their noise, so it fails to separate genuine role-shock candidates.
    Interval misses in the role cohorts are almost purely one-sided, so the
    tail stretch is directional.  See reports/role_shock_engine.md.)
    """

    table = state_multiplier_table()
    ones = np.ones(len(ACTIVE_STATES))
    zeros = np.zeros(len(ACTIVE_STATES))
    probs = np.column_stack([
        _numeric(players, f"p_state_{state}", np.nan) for state in ACTIVE_STATES
    ])
    probs = np.where(np.isfinite(probs) & (probs >= 0.0), probs, 0.0)
    row_sum = probs.sum(axis=1, keepdims=True)
    has_probs = row_sum[:, 0] > 1e-9
    stable_row = np.zeros(len(ACTIVE_STATES))
    stable_row[ACTIVE_STATES.index("stable")] = 1.0
    probs = np.where(has_probs[:, None], probs / np.where(row_sum <= 1e-9, 1.0, row_sum), stable_row)
    positions = players["position"].astype(str).to_numpy()
    target_rows = np.stack([table.get(pos, {"target": ones})["target"] for pos in positions])
    carry_rows = np.stack([table.get(pos, {"carry": ones})["carry"] for pos in positions])
    opportunity_rows = np.stack([
        table.get(pos, {"opportunity": ones})["opportunity"] for pos in positions
    ])
    within_var_rows = np.stack([
        table.get(pos, {"opportunity_var": zeros})["opportunity_var"] for pos in positions
    ])
    del opportunity_rows, within_var_rows  # width law keys on shock masses directly
    up_mass = (
        probs[:, ACTIVE_STATES.index("moderate_increase")]
        + probs[:, ACTIVE_STATES.index("major_increase")]
    )
    down_mass = probs[:, ACTIVE_STATES.index("limited_decrease")]
    return probs, target_rows, carry_rows, up_mass, down_mass, has_probs


def _directional_tail_stretch(
    values: np.ndarray,
    available: np.ndarray,
    stretch_up: float,
    stretch_down: float,
    position: str,
) -> np.ndarray:
    """Stretch the active tails asymmetrically, preserving the exact mean.

    Draws above the conditional mean scale by ``stretch_up``; draws below it
    by ``stretch_down``.  The inactive zero mass is untouched and an additive
    recentering restores the conditional (hence unconditional) mean exactly,
    so the validated ensemble center never moves.
    """

    if not available.any() or (stretch_up == 1.0 and stretch_down == 1.0):
        return values
    active = values[available]
    center = float(active.mean())
    stretched = center + (active - center) * np.where(
        active > center, stretch_up, stretch_down
    )
    floor = -6.0 if position == "QB" else -3.0
    stretched = np.maximum(stretched, min(floor, float(active.min())))
    stretched -= float(stretched.mean()) - center
    output = values.copy()
    output[available] = stretched
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

    # Scenario-mixture role engine: per-draw role states sampled from pregame
    # probabilities.  Flag OFF (default) adds no random draws and leaves the
    # availability heuristic untouched, keeping the baseline bit-identical.
    role_mixture = bool(cfg.role_scenario_mixture) and all(
        column in players.columns for column in STATE_PROB_COLUMNS
    )
    stretch_up = np.ones(p)
    stretch_down = np.ones(p)
    if role_mixture:
        (active_state_probs, target_mult_rows, carry_mult_rows,
         up_mass, down_mass, has_probs) = _role_mixture_inputs(players)
        # Modeled inactive risk: the report-flag heuristic misses healthy
        # scratches and committee benchings the classifier sees (depth chart,
        # snap trend, missed-week streak).  Taking the minimum only ever adds
        # zero mass, so report-listed absences keep their hard gate.
        inactive_raw = _numeric(players, "p_state_inactive", np.nan)
        # The 0.5 floor keeps the pinned-center hurdle solvable: the direct
        # ensemble still projects points for surprise scratches, and forcing
        # its mean through a >50% zero mass would demand absurd conditional
        # tails (and collapsed intervals) rather than admit the center is
        # what is wrong on those rows.
        model_available = np.clip(1.0 - inactive_raw, 0.50, 0.995)
        usable = has_probs & np.isfinite(model_available)
        availability_prob = np.where(
            usable, np.minimum(availability_prob, model_available), availability_prob
        )
        # Interval misses in role-shock cohorts are almost purely one-sided
        # (realized role-up outcomes overshoot the upper bound, role-down the
        # lower), and per-position mass scales differ by 5x, so each tail
        # stretches with the player's shock mass RELATIVE to same-position
        # slate peers: (mass / position reference) ** gamma.
        gamma = float(cfg.role_width_inflation)
        budget = float(cfg.role_width_budget)
        positions_all = players["position"].astype(str).to_numpy()
        up_reference = np.full(p, np.nan)
        down_reference = np.full(p, np.nan)
        for position_value in np.unique(positions_all):
            pos_mask = positions_all == position_value
            ref_mask = pos_mask & has_probs
            if ref_mask.any():
                up_reference[pos_mask] = up_mass[ref_mask].mean()
                down_reference[pos_mask] = down_mass[ref_mask].mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            up_ratio = up_mass / np.maximum(up_reference, 0.02)
            down_ratio = down_mass / np.maximum(down_reference, 0.02)
        stretch_up = np.where(
            has_probs & np.isfinite(up_ratio),
            np.clip(up_ratio ** gamma, 0.8, 2.2) * budget, 1.0,
        )
        stretch_down = np.where(
            has_probs & np.isfinite(down_ratio),
            np.clip(down_ratio ** gamma, 0.8, 2.2) * budget, 1.0,
        )
    available_all = rng.random((n, p)) < availability_prob
    team_totals: dict[str, dict[str, np.ndarray]] = {}

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

            target_multiplier = None
            carry_multiplier = None
            if role_mixture:
                state_index = _sample_role_states(rng, active_state_probs[idx], n)
                local = np.arange(len(idx))[None, :]
                target_multiplier = target_mult_rows[idx][local, state_index]
                carry_multiplier = carry_mult_rows[idx][local, state_index]

            spread = _first_numeric(team_frame, "team_spread", 0.0)
            wind = _first_numeric(team_frame, "wind", 5.0)
            pass_mean = _first_numeric(team_frame, "pre_team_pass_attempts_ewm4", 34.0)
            rush_mean = _first_numeric(team_frame, "pre_team_rush_attempts_ewm4", 27.0)
            pass_lambda = np.clip(pass_mean * pace * (1 + spread / 75.0), 15, 55)
            rush_lambda = np.clip(rush_mean * pace * (1 - spread / 100.0), 12, 45)
            team_pass_attempts = rng.poisson(pass_lambda)
            team_carries = rng.poisson(rush_lambda)

            target_base = _share_prior(team_frame, "pre_target_share_calc_ewm4", "target")
            target_prob = _draw_shares(
                rng, target_base, available, cfg.target_share_concentration,
                multiplier=target_multiplier,
            )
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
            carry_prob = _draw_shares(
                rng, carry_base, available, cfg.carry_share_concentration,
                multiplier=carry_multiplier,
            )
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
            if cfg.capture_team_totals:
                team_totals[str(team)] = {
                    "team_pass_attempts": team_pass_attempts.copy(),
                    "team_carries": team_carries.copy(),
                    "player_targets": targets.sum(axis=1),
                    "other_targets": other_targets.copy(),
                    "player_carries": carries.sum(axis=1),
                    "other_carries": carry_counts_all[:, -1].copy(),
                }

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
        if role_mixture:
            calibrated[:, index] = _directional_tail_stretch(
                calibrated[:, index],
                available_all[:, index],
                float(stretch_up[index]),
                float(stretch_down[index]),
                str(row["position"]),
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
            "role_scenario_mixture": bool(role_mixture),
            **({"team_totals": team_totals} if cfg.capture_team_totals else {}),
        },
    )
