"""Hierarchical empirical-Bayes pooling for draft baselines.

Replaces two hand-tuned heuristics in ``draft.pergame_baselines`` with
closed-form posteriors whose strength is *estimated from the data*:

1. **Per-game scoring** — a normal-normal hierarchy per position.  Each
   player's observed per-game mean ``xbar_i`` (over ``n_i`` games, within-
   player variance pooled at the position level) is partially pooled toward
   the position mean with a between-player variance ``tau^2`` estimated by
   method of moments.  The old ``n/(n+6)`` shrinkage assumed the same
   pseudo-count everywhere; here a position where players genuinely differ
   a lot (WR) pools less than one where they don't.

2. **Availability** — a Beta-Binomial hierarchy per position.  Games-played
   counts over eligible weeks get a Beta posterior, so a 17-of-17 iron man
   and a 7-of-33 glass cannon carry *distributions*, not point rates.  The
   old version shrank every player toward 0.88 with pseudo-n 8.

Both posteriors carry uncertainty columns (``mu_post_sd``, ``avail_alpha``,
``avail_beta``) that ``draft.simulate_season`` consumes when present:
parameter uncertainty is sampled per simulation instead of being ignored.

Everything is closed-form empirical Bayes — no MCMC dependency — and every
row it touches gets ``+hier_bayes`` appended to its ``basis`` flag.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .models import FantasyEnsemble

__all__ = [
    "beta_binomial_posteriors",
    "hierarchical_baselines",
    "normal_partial_pool",
]

_MIN_TAU2 = 0.25  # floor on between-player variance (points^2 per game)
_MIN_WITHIN = 4.0  # floor on pooled within-player weekly variance


def normal_partial_pool(
    xbar: pd.Series,
    n: pd.Series,
    within_var: pd.Series,
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    """Normal-normal partial pooling for one position group.

    Returns (posterior mean, posterior sd, hyperparameters).  ``within_var``
    is each player's weekly scoring variance; it is pooled (df-weighted)
    into a single position-level ``s2`` before use, then the between-player
    ``tau^2`` comes from the method of moments:

        Var(xbar_i) ≈ tau^2 + mean(s2 / n_i)
    """

    n = n.astype(float).clip(lower=1.0)
    valid = within_var.dropna()
    if len(valid):
        weights = (n.loc[valid.index] - 1.0).clip(lower=1.0)
        s2 = float(np.average(valid, weights=weights))
    else:
        s2 = 25.0
    s2 = max(s2, _MIN_WITHIN)

    sampling_var = s2 / n
    tau2 = max(float(xbar.var(ddof=1)) - float(sampling_var.mean()), _MIN_TAU2)
    grand_mean = float(np.average(xbar, weights=n))

    precision = n / s2 + 1.0 / tau2
    post_mean = (xbar * n / s2 + grand_mean / tau2) / precision
    post_sd = np.sqrt(1.0 / precision)
    hyper = {"grand_mean": grand_mean, "tau2": tau2, "within_var": s2}
    return post_mean, post_sd, hyper


def beta_binomial_posteriors(
    successes: pd.Series,
    trials: pd.Series,
) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    """Beta-Binomial posteriors for one position group.

    Hyperparameters (alpha0, beta0) come from moment-matching the observed
    rates' overdispersion relative to pure binomial sampling.  Returns
    (alpha_i, beta_i) posterior parameter Series plus the hyperparameters.
    """

    trials = trials.astype(float).clip(lower=1.0)
    rates = successes / trials
    m = float(np.average(rates, weights=trials))
    m = min(max(m, 0.05), 0.98)
    observed_var = float(rates.var(ddof=1)) if len(rates) > 1 else 0.0
    binomial_var = float((m * (1 - m) / trials).mean())
    excess = max(observed_var - binomial_var, 1e-4)
    # Beta variance m(1-m)/(alpha0+beta0+1) = excess  =>  concentration:
    concentration = max(m * (1 - m) / excess - 1.0, 2.0)
    alpha0, beta0 = m * concentration, (1 - m) * concentration

    alpha_post = alpha0 + successes
    beta_post = beta0 + (trials - successes)
    hyper = {"alpha0": float(alpha0), "beta0": float(beta0), "mean": m}
    return alpha_post, beta_post, hyper


def hierarchical_baselines(
    frame: pd.DataFrame,
    model: FantasyEnsemble,
    *,
    source_season: int,
    model_weight: float = 0.6,
    min_games: int = 3,
    availability_clip: tuple[float, float] = (0.30, 0.98),
) -> pd.DataFrame:
    """``draft.pergame_baselines`` with hierarchical pooling swapped in.

    Runs the standard baseline builder (model blending, sigma fit) and then
    replaces its ad hoc shrinkage and availability heuristics with the
    posteriors above.  Adds ``mu_post_sd``, ``avail_alpha``, ``avail_beta``
    and tags ``basis`` with ``+hier_bayes``.
    """

    from .draft import pergame_baselines  # local import to avoid a cycle

    baselines = pergame_baselines(
        frame, model, source_season=source_season,
        model_weight=model_weight, min_games=min_games,
    )

    # --- per-game mu: pool the *raw* (pre-shrinkage) means -----------------
    hypers: dict[str, dict[str, float]] = {}
    for position, group in baselines.groupby("position"):
        post_mean, post_sd, hyper = normal_partial_pool(
            group["mu_pergame_raw"],
            group["games_played"],
            group["sigma_own"] ** 2,
        )
        baselines.loc[group.index, "mu_pergame"] = post_mean
        baselines.loc[group.index, "mu_post_sd"] = post_sd
        hypers[f"mu:{position}"] = hyper

    # --- availability: Beta-Binomial over the last two seasons -------------
    recent = frame[frame["season"].isin([source_season - 1, source_season])]
    played_counts = (
        recent[recent["played"].astype(bool)].groupby("player_id").size()
    )
    weeks_per_season = recent.groupby("season")["week"].nunique() - 1
    season_sets = recent.groupby("player_id")["season"].unique()
    eligible = season_sets.map(
        lambda seasons: float(sum(weeks_per_season.get(s, 16) for s in seasons))
    )

    ids = baselines["player_id"]
    k = ids.map(played_counts).fillna(0.0)
    n = ids.map(eligible).fillna(weeks_per_season.max() if len(weeks_per_season) else 16.0)
    k = np.minimum(k, n)
    for position, group in baselines.groupby("position"):
        alpha, beta, hyper = beta_binomial_posteriors(k.loc[group.index], n.loc[group.index])
        baselines.loc[group.index, "avail_alpha"] = alpha
        baselines.loc[group.index, "avail_beta"] = beta
        baselines.loc[group.index, "availability_rate"] = (
            (alpha / (alpha + beta)).clip(*availability_clip)
        )
        hypers[f"avail:{position}"] = hyper

    baselines["basis"] = baselines["basis"] + "+hier_bayes"
    baselines.attrs["hierarchy_hyperparameters"] = hypers
    return baselines


def hyperparameter_table(baselines: pd.DataFrame) -> pd.DataFrame:
    """Readable table of the fitted hyperpriors (for reports/audits)."""

    hypers: Mapping[str, Mapping[str, float]] = baselines.attrs.get(
        "hierarchy_hyperparameters", {}
    )
    rows = [{"group": key, **values} for key, values in hypers.items()]
    return pd.DataFrame(rows)
