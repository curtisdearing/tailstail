"""Stratified accuracy scoring and decision-level regret for champion/challenger runs.

A pooled MAE hides the only failures that cost a fantasy manager a week: the
player whose role just changed, the one who is questionable, the rookie with
no history, and the one who did not play at all.  Every stratum here is
declared in ``analysis/shadow_challenge_2026.json`` before any challenger is
scored, and every cell carries its exact ``n`` so a flattering slice cannot be
quoted without its sample size.

Two deliberate honesty constraints:

* ``crps`` is reported only where a predictive *sample* exists.  The direct
  ensemble emits a centre and a conformal interval, not draws, so its
  probabilistic quality is scored with the pinball loss at the two quantiles
  it actually claims -- never with a normal-approximation CRPS dressed up as
  the real thing.
* Lineup regret is paired.  Every arm is handed the identical randomly drawn
  rosters, so a regret difference is a decision difference and not a
  difference in which teams happened to be sampled.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .lineup import from_positions, lineup_value, optimize

#: Starting-lineup shape used for decision-level regret.  Standard redraft;
#: K and D/ST are excluded because the frozen centre does not model them.
REGRET_SLOTS: Mapping[str, int] = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
REGRET_ROSTER_SIZE = 14
REGRET_TEAMS_PER_WEEK = 12
REGRET_SEED = 6102026

NOMINAL_COVERAGE = 0.80
LOWER_QUANTILE = 0.10
UPPER_QUANTILE = 0.90

#: Role change measured on prior-only opportunity trend, so a stratum label
#: never depends on the outcome being scored.
ROLE_UP_THRESHOLD = 2.0
ROLE_DOWN_THRESHOLD = -2.0
#: "Cold start" is fewer than this many prior played games, counted from
#: strictly-prior evidence -- rookies, and returns from a long absence.
COLD_START_MAX_PRIOR_GAMES = 3.0


def _spearman(actual: np.ndarray, predicted: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if len(actual) < 3:
        return float("nan")
    value = spearmanr(actual, predicted, nan_policy="omit").statistic
    return float(value) if np.isfinite(value) else float("nan")


def _pinball(actual: np.ndarray, quantile_prediction: np.ndarray, tau: float) -> float:
    delta = actual - quantile_prediction
    return float(np.mean(np.maximum(tau * delta, (tau - 1.0) * delta)))


def score(frame: pd.DataFrame) -> dict[str, float | int]:
    """Every point and interval metric for one set of rows."""

    actual = pd.to_numeric(frame["fantasy_points"], errors="coerce").to_numpy(dtype=float)
    predicted = pd.to_numeric(frame["projection_mean"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[keep], predicted[keep]
    if actual.size == 0:
        return {"n": 0}
    lower = pd.to_numeric(frame["projection_lower80"], errors="coerce").to_numpy(dtype=float)[keep]
    upper = pd.to_numeric(frame["projection_upper80"], errors="coerce").to_numpy(dtype=float)[keep]
    error = actual - predicted
    covered = (actual >= lower) & (actual <= upper)
    return {
        "n": int(actual.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias_actual_minus_prediction": float(np.mean(error)),
        "spearman": _spearman(actual, predicted),
        "coverage80": float(np.mean(covered)),
        "coverage_error_pp": float((np.mean(covered) - NOMINAL_COVERAGE) * 100.0),
        "mean_interval_width": float(np.mean(upper - lower)),
        "median_interval_width": float(np.median(upper - lower)),
        "pinball_q10": _pinball(actual, lower, LOWER_QUANTILE),
        "pinball_q90": _pinball(actual, upper, UPPER_QUANTILE),
    }


def crps_from_samples(actual: np.ndarray, samples: np.ndarray) -> float:
    """Empirical CRPS from real draws.  Never call this without draws."""

    actual = np.asarray(actual, dtype=float)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] != actual.shape[0]:
        raise ValueError("samples must be (rows, draws) aligned with actual")
    draws = samples.shape[1]
    if draws < 2:
        raise ValueError("CRPS needs at least two draws per row")
    ordered = np.sort(samples, axis=1)
    term_one = np.mean(np.abs(ordered - actual[:, None]), axis=1)
    # E|X - X'| for the empirical distribution, in O(n log n) via the sorted
    # order statistics identity rather than an O(n^2) pairwise matrix.
    index = np.arange(1, draws + 1, dtype=float)
    weights = (2.0 * index - draws - 1.0)
    term_two = 2.0 * np.sum(ordered * weights, axis=1) / (draws * draws)
    return float(np.mean(term_one - 0.5 * term_two))


def add_strata(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach every preregistered stratum label using pregame evidence only.

    Each label is derived from ``pre_*`` columns, so no stratum can be defined
    by the outcome it is used to score.  A column the caller did not join in
    yields an explicit ``*_unknown`` level rather than a silent default.
    """

    out = frame.copy()
    short = pd.to_numeric(out.get("pre_opportunities_ewm4"), errors="coerce")
    long_run = pd.to_numeric(out.get("pre_opportunities_ewm8"), errors="coerce")
    delta = short - long_run
    out["role_delta"] = delta
    role = np.where(
        delta >= ROLE_UP_THRESHOLD, "role_up",
        np.where(delta <= ROLE_DOWN_THRESHOLD, "role_down", "stable_role"),
    )
    out["stratum_role"] = np.where(delta.isna().to_numpy(), "role_unknown", role)

    questionable = pd.to_numeric(out.get("injury_questionable"), errors="coerce").fillna(0.0)
    dnp = pd.to_numeric(out.get("practice_dnp"), errors="coerce").fillna(0.0)
    out["stratum_availability"] = np.where(
        dnp.gt(0), "practice_dnp",
        np.where(questionable.gt(0), "injury_questionable", "status_available"),
    )

    played_games = pd.to_numeric(out.get("pre_played_games"), errors="coerce")
    experience = np.where(
        played_games.fillna(0.0) < COLD_START_MAX_PRIOR_GAMES, "cold_start", "established"
    )
    out["stratum_experience"] = np.where(
        played_games.isna().to_numpy(), "experience_unknown", experience
    )

    opportunities = pd.to_numeric(out.get("opportunities"), errors="coerce").fillna(0.0)
    points = pd.to_numeric(out.get("fantasy_points"), errors="coerce").fillna(0.0)
    out["stratum_participation"] = np.where(
        (opportunities > 0) | (points != 0.0), "played", "dnp"
    )
    return out


STRATUM_COLUMNS = (
    ("position", "position"),
    ("role", "stratum_role"),
    ("availability", "stratum_availability"),
    ("experience", "stratum_experience"),
    ("participation", "stratum_participation"),
)


def stratified_scorecard(frame: pd.DataFrame) -> dict[str, object]:
    """Overall, per-season and per-stratum metrics, each with its exact n."""

    work = add_strata(frame)
    card: dict[str, object] = {"overall": score(work)}
    card["by_season"] = {
        str(int(season)): score(group)
        for season, group in work.groupby(work["season"].astype(int))
    }
    strata: dict[str, dict[str, object]] = {}
    for name, column in STRATUM_COLUMNS:
        strata[name] = {
            str(level): score(group) for level, group in work.groupby(work[column].astype(str))
        }
    card["by_stratum"] = strata
    return card


# --------------------------------------------------------------------------
# Decision-level regret
# --------------------------------------------------------------------------

def _weekly_rosters(
    ids: Sequence[object], rng: np.random.Generator, *, teams: int, size: int
) -> list[list[object]]:
    pool = list(ids)
    if len(pool) < size:
        return []
    order = rng.permutation(len(pool))
    drawn = [pool[i] for i in order]
    usable = min(teams, len(drawn) // size)
    return [drawn[team * size : (team + 1) * size] for team in range(usable)]


def lineup_regret(
    frames: Mapping[str, pd.DataFrame],
    *,
    slots: Mapping[str, int] | None = None,
    teams: int = REGRET_TEAMS_PER_WEEK,
    roster_size: int = REGRET_ROSTER_SIZE,
    seed: int = REGRET_SEED,
) -> dict[str, object]:
    """Points left on the bench, paired across arms on identical rosters.

    Every arm sees the same randomly drawn teams for a given season-week, so a
    regret gap reflects the projections and nothing else.  The oracle is the
    retrospectively optimal legal lineup for the same roster.
    """

    slot_counts = dict(slots or REGRET_SLOTS)
    names = list(frames)
    if not names:
        raise ValueError("at least one arm is required")
    keys = ["season", "week", "player_id"]
    aligned: dict[str, pd.DataFrame] = {}
    for name, frame in frames.items():
        work = frame.dropna(subset=["projection_mean", "fantasy_points"]).copy()
        work["player_id"] = work["player_id"].astype(str)
        aligned[name] = work.set_index(keys, drop=False)

    common = None
    for work in aligned.values():
        index = set(work.index)
        common = index if common is None else (common & index)
    common = sorted(common or set())
    if not common:
        raise ValueError("arms share no player-weeks")

    shared = pd.DataFrame(common, columns=keys)
    totals: dict[str, list[float]] = {name: [] for name in names}
    oracle_totals: list[float] = []
    team_blocks: list[tuple[int, int]] = []
    weeks_scored = 0
    teams_scored = 0

    for (season, week), block in shared.groupby(["season", "week"], sort=True):
        rng = np.random.default_rng((seed, int(season), int(week)))
        index = list(zip(block["season"], block["week"], block["player_id"]))
        first = aligned[names[0]].loc[index]
        positions = dict(zip(first["player_id"].astype(str), first["position"].astype(str)))
        actual = dict(zip(
            first["player_id"].astype(str),
            pd.to_numeric(first["fantasy_points"], errors="coerce").astype(float),
        ))
        projections = {
            name: dict(zip(
                aligned[name].loc[index]["player_id"].astype(str),
                pd.to_numeric(aligned[name].loc[index]["projection_mean"], errors="coerce").astype(float),
            ))
            for name in names
        }
        rosters = _weekly_rosters(
            sorted(positions), rng, teams=teams, size=roster_size
        )
        if not rosters:
            continue
        weeks_scored += 1
        for roster in rosters:
            players = from_positions(
                [(pid, positions[pid]) for pid in roster], slot_counts
            )
            oracle = lineup_value({pid: actual[pid] for pid in roster}, players, slot_counts)
            oracle_totals.append(float(oracle))
            team_blocks.append((int(season), int(week)))
            teams_scored += 1
            for name in names:
                chosen = optimize(
                    {pid: projections[name][pid] for pid in roster}, players, slot_counts
                )
                started = [
                    pid for seat in chosen.assignment.values() for pid in seat
                ]
                totals[name].append(float(sum(actual[pid] for pid in started)))

    oracle = np.asarray(oracle_totals, dtype=float)
    report: dict[str, object] = {
        "slots": slot_counts,
        "teams_per_week": teams,
        "roster_size": roster_size,
        "seed": seed,
        "weeks_scored": weeks_scored,
        "teams_scored": teams_scored,
        "shared_player_weeks": int(len(shared)),
        "oracle_mean_points": float(np.mean(oracle)) if oracle.size else float("nan"),
        "arms": {},
    }
    for name in names:
        achieved = np.asarray(totals[name], dtype=float)
        regret = oracle - achieved
        report["arms"][name] = {
            "n_teams": int(achieved.size),
            "mean_started_points": float(np.mean(achieved)) if achieved.size else float("nan"),
            "mean_regret": float(np.mean(regret)) if regret.size else float("nan"),
            "median_regret": float(np.median(regret)) if regret.size else float("nan"),
            "p90_regret": float(np.quantile(regret, 0.90)) if regret.size else float("nan"),
        }

    # A regret gap of a few hundredths of a point is meaningless without its
    # spread, so quantify it on the same season-week blocks used everywhere
    # else.  Teams inside one week share players and a game script.
    blocks = np.asarray(team_blocks, dtype=np.int64)
    if blocks.size:
        keys, block_index = np.unique(blocks, axis=0, return_inverse=True)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(keys), size=(2000, len(keys)))
        membership = [np.flatnonzero(block_index == i) for i in range(len(keys))]
        report["paired_vs_champion"] = {}
        champion_name = "champion_ensemble" if "champion_ensemble" in names else names[0]
        champion_regret = oracle - np.asarray(totals[champion_name], dtype=float)
        for name in names:
            if name == champion_name:
                continue
            difference = champion_regret - (oracle - np.asarray(totals[name], dtype=float))
            sums = np.asarray([difference[rows].sum() for rows in membership])
            counts = np.asarray([len(rows) for rows in membership], dtype=float)
            resampled = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
            report["paired_vs_champion"][name] = {
                "champion_name": champion_name,
                "blocks": int(len(keys)),
                "iterations": 2000,
                "regret_reduction_vs_champion": float(difference.mean()),
                "ci95": [float(v) for v in np.quantile(resampled, [0.025, 0.975])],
                "probability_of_improvement": float(np.mean(resampled > 0)),
            }
    return report


def top_n_overlap(
    champion: pd.DataFrame, challenger: pd.DataFrame, *, n: int = 10
) -> dict[str, float | int]:
    """Weekly agreement on who the top ``n`` projected players are."""

    keys = ["season", "week"]
    a_tops = {
        key: set(group.nlargest(n, "projection_mean")["player_id"].astype(str))
        for key, group in champion.dropna(subset=["projection_mean"]).groupby(keys)
    }
    b_tops = {
        key: set(group.nlargest(n, "projection_mean")["player_id"].astype(str))
        for key, group in challenger.dropna(subset=["projection_mean"]).groupby(keys)
    }
    overlaps = [
        len(a_tops[key] & b_tops[key]) / float(min(len(a_tops[key]), len(b_tops[key])))
        for key in sorted(set(a_tops) & set(b_tops))
        if a_tops[key] and b_tops[key]
    ]
    return {
        "top_n": n,
        "weeks": len(overlaps),
        "mean_overlap": float(np.mean(overlaps)) if overlaps else float("nan"),
        "min_overlap": float(np.min(overlaps)) if overlaps else float("nan"),
    }


def paired_block_bootstrap(
    champion: pd.DataFrame,
    challenger: pd.DataFrame,
    *,
    iterations: int = 20_000,
    seed: int = REGRET_SEED,
) -> dict[str, object]:
    """Champion-minus-challenger MAE, resampled by season-week block.

    A positive ``mae_improvement`` means the challenger is better.  Blocks are
    whole season-weeks because player errors inside one week are correlated by
    weather, game script and the same handful of offences.
    """

    keys = ["season", "week", "player_id"]
    left = champion[keys + ["fantasy_points", "projection_mean"]].dropna()
    right = challenger[keys + ["fantasy_points", "projection_mean"]].dropna()
    merged = left.merge(right, on=keys, suffixes=("_champion", "_challenger"))
    if merged.empty:
        raise ValueError("champion and challenger share no scored player-weeks")
    if not np.allclose(
        merged["fantasy_points_champion"], merged["fantasy_points_challenger"]
    ):
        raise ValueError("paired rows disagree about the observed outcome")
    merged = merged.assign(
        champion_error=(merged["fantasy_points_champion"] - merged["projection_mean_champion"]).abs(),
        challenger_error=(merged["fantasy_points_champion"] - merged["projection_mean_challenger"]).abs(),
    )
    grouped = merged.groupby(["season", "week"])
    blocks = pd.DataFrame({
        "n": grouped["champion_error"].size(),
        "difference_sum": (
            grouped["champion_error"].sum() - grouped["challenger_error"].sum()
        ),
    })
    counts = blocks["n"].to_numpy(dtype=float)
    differences = blocks["difference_sum"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(blocks), size=(iterations, len(blocks)))
    resampled = differences[draw].sum(axis=1) / counts[draw].sum(axis=1)
    observed = float(differences.sum() / counts.sum())
    return {
        "n": int(len(merged)),
        "weeks": int(len(blocks)),
        "iterations": iterations,
        "seed": seed,
        "mae_improvement": observed,
        "ci95": [float(v) for v in np.quantile(resampled, [0.025, 0.975])],
        "probability_of_improvement": float(np.mean(resampled > 0)),
        "week_win_rate": float(np.mean(differences > 0)),
    }
