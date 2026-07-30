"""Draft-day season outlook: per-player 2026 season distributions and a
ceiling-weighted value-over-replacement draft board.

Methodology (honest about what is model and what is prior):

1. **Per-game baseline** — the trained ensemble's weekly ``projection_mean``
   over the player's most recent season, blended with realized per-game
   fantasy points and shrunk toward the position mean by games played.
   This inherits the shipped model's accuracy evidence (frozen protocol,
   MAE 5.091 full PPR) for the *per-game center*.
2. **Season aggregation** — Monte Carlo over the target season's real
   schedule (bye weeks respected): weekly points drawn from a Student-t
   around the per-game baseline, availability drawn per week from a
   shrunken games-played rate plus a position-level season-ending hazard.
3. **Priors, labeled as priors** — age curves, team-change widening, and
   the rookie ADP market prior are documented heuristics, NOT model
   output.  Every row carries ``basis`` flags so the board never hides
   which numbers are evidence and which are priors.
4. **Replacement + ceiling weighting** — VOR against the league's actual
   replacement rank; the draft score deliberately overweights the P90
   season outcome (championships in shallow leagues come from right-tail
   seasons, and replacement level is high enough that floor is cheap).

The retrodiction grade for this module lives in
``analysis/draft_retrodiction.py`` — rankings built from season N-1 data
are scored against realized season N totals before anyone trusts a board.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import LineupRules
from .models import FantasyEnsemble

# ---------------------------------------------------------------------------
# Priors (documented heuristics — kept small, explicit, and overridable)
# ---------------------------------------------------------------------------

#: Multiplicative per-game age adjustments by position.  Values are mild on
#: purpose; the model's trailing-usage features already price most decline.
AGE_CURVES: Mapping[str, Sequence[tuple[float, float]]] = {
    # (age threshold, multiplier applied when age >= threshold)
    "RB": ((26.0, 0.97), (28.0, 0.93), (30.0, 0.86)),
    "WR": ((29.0, 0.97), (31.0, 0.93), (33.0, 0.88)),
    "TE": ((30.0, 0.97), (32.0, 0.93)),
    "QB": ((36.0, 0.96), (39.0, 0.90)),
}

#: Season-ending injury hazard per week, by position (probability that an
#: active player's season effectively ends at any given week).  Calibrated
#: coarsely so simulated games-played distributions match the historical
#: per-position spread; a documented heuristic, not a fitted model.
SEASON_END_HAZARD: Mapping[str, float] = {
    "RB": 0.009, "WR": 0.007, "TE": 0.006, "QB": 0.005,
}

TEAM_CHANGE_SIGMA_INFLATION = 1.15
TEAM_CHANGE_MU_HAIRCUT: Mapping[str, float] = {"WR": 0.97, "TE": 0.96, "RB": 0.99, "QB": 0.98}
ROOKIE_SIGMA_INFLATION = 1.40

_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(value: str) -> str:
    """Lowercase, alpha-only, suffix-stripped player-name key for joins."""

    cleaned = "".join(ch for ch in str(value).lower() if ch.isalpha() or ch == " ")
    parts = [part for part in cleaned.split() if part not in _NAME_SUFFIXES]
    return " ".join(parts)


@dataclass
class SeasonOutlook:
    """Season simulation output: one row per player plus the sample matrix."""

    board: pd.DataFrame
    season_points: pd.DataFrame  # simulations x players
    metadata: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-game baselines from the trained model
# ---------------------------------------------------------------------------

def _age_on(birth_date: pd.Series, season: int) -> pd.Series:
    born = pd.to_datetime(birth_date, errors="coerce")
    ref = pd.Timestamp(year=season, month=9, day=1)
    return (ref - born).dt.days / 365.25


def _age_multiplier(position: str, age: float) -> float:
    if not np.isfinite(age):
        return 1.0
    factor = 1.0
    for threshold, multiplier in AGE_CURVES.get(position, ()):  # cumulative? no: strongest tier
        if age >= threshold:
            factor = multiplier
    return factor


def pergame_baselines(
    frame: pd.DataFrame,
    model: FantasyEnsemble,
    *,
    source_season: int,
    model_weight: float = 0.6,
    shrink_games: float = 6.0,
    min_games: int = 3,
) -> pd.DataFrame:
    """Per-player per-game (mu, sigma) built from the source season.

    ``mu`` blends the ensemble's weekly predictions with realized points and
    shrinks toward the position mean by games played; ``sigma`` blends the
    player's own weekly spread with a position-level sigma-vs-mu fit.
    """

    rows = frame[(frame["season"] == source_season)].copy()
    if rows.empty:
        raise ValueError(f"no rows for season {source_season} in frame")
    predictions = model.predict(rows)
    merged = rows[[
        "player_id", "player_name", "position", "team", "week",
        "fantasy_points", "played", "birth_date", "years_exp", "draft_number",
    ]].merge(
        predictions[["player_id", "week", "projection_mean"]],
        on=["player_id", "week"], how="left",
    )
    # The feature frame is a full roster-week grid: fantasy_points is NEVER
    # NaN (DNP weeks are exact zeros), so notna() would count every roster
    # week as a game. The frame's `played` flag is the real predicate.
    played = merged[merged["played"].astype(bool)]

    position_mu = played.groupby("position")["fantasy_points"].mean()

    records = []
    for player_id, group in played.groupby("player_id"):
        games = len(group)
        if games < min_games:
            continue
        position = group["position"].iloc[0]
        model_mu = float(group["projection_mean"].mean()) if group["projection_mean"].notna().any() else np.nan
        actual_mu = float(group["fantasy_points"].mean())
        raw_mu = (
            model_weight * model_mu + (1 - model_weight) * actual_mu
            if np.isfinite(model_mu) else actual_mu
        )
        prior_mu = float(position_mu.get(position, raw_mu))
        weight = games / (games + shrink_games)
        mu = weight * raw_mu + (1 - weight) * prior_mu
        sigma_own = float(group["fantasy_points"].std(ddof=1)) if games > 1 else np.nan
        records.append({
            "player_id": str(player_id),
            "player_name": group["player_name"].iloc[0],
            "position": position,
            "team": group["team"].iloc[-1],
            "games_played": games,
            "mu_pergame_raw": raw_mu,
            "mu_pergame": mu,
            "sigma_own": sigma_own,
            "age": float(_age_on(group["birth_date"], source_season + 1).iloc[0]),
            "draft_number": group["draft_number"].iloc[0],
            "basis": "model+realized",
        })
    baselines = pd.DataFrame(records)

    # Position-level sigma-vs-mu fit (weekly volatility grows with volume).
    fit = {}
    for position, group in baselines.dropna(subset=["sigma_own"]).groupby("position"):
        if len(group) >= 8:
            slope, intercept = np.polyfit(group["mu_pergame"], group["sigma_own"], 1)
        else:
            slope, intercept = 0.45, 2.0
        fit[position] = (float(slope), float(intercept))
    def _sigma(row) -> float:
        slope, intercept = fit.get(row["position"], (0.45, 2.0))
        structural = max(slope * row["mu_pergame"] + intercept, 1.5)
        if np.isfinite(row["sigma_own"]) and row["games_played"] >= 6:
            w = row["games_played"] / (row["games_played"] + 8.0)
            return float(w * row["sigma_own"] + (1 - w) * structural)
        return float(structural)
    baselines["sigma_pergame"] = baselines.apply(_sigma, axis=1)

    # Availability rate over the last two seasons (games / eligible weeks).
    recent = frame[frame["season"].isin([source_season - 1, source_season])]
    counts = recent[recent["played"].astype(bool)].groupby("player_id").size()
    # Eligible weeks = team game-weeks per season: calendar weeks in the
    # frame minus the bye (17 since the 2021 18-week schedule; 16 before).
    weeks_per_season = recent.groupby("season")["week"].nunique() - 1
    season_sets = recent.groupby("player_id")["season"].unique()
    eligible = season_sets.map(
        lambda seasons: float(sum(weeks_per_season.get(s, 16) for s in seasons))
    )
    rate = (counts / eligible).clip(0.05, 1.0)
    baselines["availability_rate"] = baselines["player_id"].map(rate).fillna(0.8)
    # Shrink toward a position prior of 0.88 with pseudo-n of 8 games.
    n_obs = baselines["player_id"].map(counts).fillna(0.0)
    w = n_obs / (n_obs + 8.0)
    baselines["availability_rate"] = (
        w * baselines["availability_rate"] + (1 - w) * 0.88
    ).clip(0.30, 0.98)
    return baselines


# ---------------------------------------------------------------------------
# Offseason adjustments (priors, labeled)
# ---------------------------------------------------------------------------

def apply_offseason_adjustments(
    baselines: pd.DataFrame,
    *,
    current_teams: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Age curves + team-change adjustments.  ``current_teams`` maps
    player_name (normalized lower) -> team for the upcoming season; entries
    that differ from the source-season team trigger the team-change prior."""

    adjusted = baselines.copy()
    factors = adjusted.apply(lambda r: _age_multiplier(r["position"], r["age"]), axis=1)
    adjusted["age_multiplier"] = factors
    adjusted["mu_pergame"] *= factors

    adjusted["team_changed"] = False
    if current_teams:
        normalized_teams = {normalize_name(k): v for k, v in current_teams.items()}

        def _lookup(name: str) -> str | None:
            return normalized_teams.get(normalize_name(name))
        new_team = adjusted["player_name"].map(_lookup)
        moved = new_team.notna() & (new_team != adjusted["team"])
        adjusted.loc[moved, "team_changed"] = True
        adjusted.loc[moved, "team"] = new_team[moved]
        for position, haircut in TEAM_CHANGE_MU_HAIRCUT.items():
            mask = moved & (adjusted["position"] == position)
            adjusted.loc[mask, "mu_pergame"] *= haircut
        adjusted.loc[moved, "sigma_pergame"] *= TEAM_CHANGE_SIGMA_INFLATION
        adjusted.loc[moved, "basis"] = adjusted.loc[moved, "basis"] + "+team_change_prior"
    mask_age = adjusted["age_multiplier"] < 1.0
    adjusted.loc[mask_age, "basis"] = adjusted.loc[mask_age, "basis"] + "+age_prior"
    return adjusted


def add_rookie_market_priors(
    baselines: pd.DataFrame,
    adp: pd.DataFrame,
) -> pd.DataFrame:
    """Add rows for drafted rookies absent from history, priced off the
    veteran ADP -> projected-points curve.  ``adp`` needs columns
    name/position/team/adp.  Rookie rows are labeled ``rookie_market_prior``
    and get inflated sigma — the market knows things the model cannot."""

    known = set(baselines["player_name"].map(normalize_name))
    veterans = adp[adp["name"].map(normalize_name).isin(known)].copy()
    merged = veterans.assign(key=veterans["name"].map(normalize_name)).merge(
        baselines[["player_name", "mu_pergame"]].assign(
            key=baselines["player_name"].map(normalize_name)
        ),
        on="key", how="inner",
    )
    if len(merged) >= 20:
        coefficients = np.polyfit(np.log(merged["adp"].clip(1)), merged["mu_pergame"], 1)
    else:  # conservative fallback curve
        coefficients = (-3.2, 18.0)

    rows = []
    for _, row in adp.iterrows():
        name = normalize_name(row["name"])
        if name in known or row.get("position") not in {"QB", "RB", "WR", "TE"}:
            continue
        mu = float(np.polyval(coefficients, np.log(max(float(row["adp"]), 1.0))))
        mu = float(np.clip(mu, 4.0, 22.0))
        position_sigma = baselines.loc[
            baselines["position"] == row["position"], "sigma_pergame"
        ].median()
        rows.append({
            "player_id": f"adp:{name}",
            "player_name": row["name"],
            "position": row["position"],
            "team": row.get("team", "UNK"),
            "games_played": 0,
            "mu_pergame_raw": mu,
            "mu_pergame": mu,
            "sigma_own": np.nan,
            "sigma_pergame": float(position_sigma * ROOKIE_SIGMA_INFLATION),
            "age": np.nan,
            "draft_number": np.nan,
            "availability_rate": 0.85,
            "age_multiplier": 1.0,
            "team_changed": False,
            "basis": "rookie_market_prior",
        })
    if not rows:
        return baselines
    return pd.concat([baselines, pd.DataFrame(rows)], ignore_index=True)


# ---------------------------------------------------------------------------
# Season Monte Carlo over the real schedule
# ---------------------------------------------------------------------------

def simulate_season(
    baselines: pd.DataFrame,
    byes: Mapping[str, Sequence[int]],
    *,
    weeks: Sequence[int] = tuple(range(1, 18)),
    simulations: int = 4000,
    random_seed: int = 6102026,
    t_dof: float = 6.0,
) -> SeasonOutlook:
    """Simulate season totals: weekly Student-t draws around the per-game
    baseline, per-week availability, and a season-ending hazard."""

    rng = np.random.default_rng(random_seed)
    players = baselines.reset_index(drop=True)
    p = len(players)
    week_list = list(weeks)
    n_weeks = len(week_list)

    bye_mask = np.ones((p, n_weeks), dtype=bool)  # True = scheduled to play
    for i, row in players.iterrows():
        for j, week in enumerate(week_list):
            if week in set(byes.get(str(row["team"]), ())):
                bye_mask[i, j] = False

    mu = players["mu_pergame"].to_numpy()[None, :, None]
    sigma = players["sigma_pergame"].to_numpy()[None, :, None]
    weekly = mu + sigma * rng.standard_t(t_dof, size=(simulations, p, n_weeks)) / np.sqrt(
        t_dof / (t_dof - 2.0)
    )
    np.clip(weekly, -4.0, None, out=weekly)

    active_rate = players["availability_rate"].to_numpy()
    active = rng.random((simulations, p, n_weeks)) < active_rate[None, :, None]

    hazard = players["position"].map(SEASON_END_HAZARD).fillna(0.006).to_numpy()
    ended = rng.random((simulations, p, n_weeks)) < hazard[None, :, None]
    season_over = np.cumsum(ended, axis=2) > 0

    playable = bye_mask[None, :, :] & active & ~season_over
    totals = np.where(playable, weekly, 0.0).sum(axis=2)
    games = playable.sum(axis=2)

    board = players.copy()
    board["season_mean"] = totals.mean(axis=0)
    board["season_median"] = np.median(totals, axis=0)
    board["season_p10"] = np.quantile(totals, 0.10, axis=0)
    board["season_p90"] = np.quantile(totals, 0.90, axis=0)
    board["expected_games"] = games.mean(axis=0)
    points = pd.DataFrame(totals, columns=players["player_id"].astype(str))
    return SeasonOutlook(
        board=board,
        season_points=points,
        metadata={
            "weeks": week_list, "simulations": simulations,
            "random_seed": random_seed, "t_dof": t_dof,
        },
    )


# ---------------------------------------------------------------------------
# Replacement level, ceiling-weighted score, tiers, snake targets
# ---------------------------------------------------------------------------

#: Positional bench depth actually rostered, by position.  Onesie positions
#: (QB/TE) are streamed in shallow leagues — almost nobody carries a backup —
#: so their replacement level sits just past the starter count, while RB/WR
#: benches run deep.
BENCH_MULTIPLIERS: Mapping[str, float] = {"QB": 1.2, "TE": 1.1, "RB": 1.6, "WR": 1.6}


def draft_board(
    outlook: SeasonOutlook,
    *,
    league_teams: int = 6,
    rules: LineupRules | None = None,
    bench_multipliers: Mapping[str, float] | None = None,
    ceiling_weight: float = 0.55,
    tier_gap_fraction: float = 0.35,
) -> pd.DataFrame:
    """Ceiling-weighted VOR board.

    ``ceiling_weight`` is the share of the draft score carried by P90 VOR
    (vs mean VOR).  In shallow leagues replacement level is high, floors
    are free on waivers, and the score should chase right tails.
    """

    lineup = rules or LineupRules()
    multipliers = dict(BENCH_MULTIPLIERS)
    if bench_multipliers:
        multipliers.update(bench_multipliers)
    board = outlook.board.copy()
    flex_n = int(lineup.starters.get("FLEX", 0))
    output = []
    for position, group in board.groupby("position"):
        direct = int(lineup.starters.get(position, 0))
        flex_share = flex_n / max(len(lineup.flex_positions), 1) if position in lineup.flex_positions else 0.0
        bench_multiplier = float(multipliers.get(position, 1.5))
        replacement_rank = max(int(np.ceil(league_teams * (direct + flex_share) * bench_multiplier)), 1)
        ordered = group.sort_values("season_mean", ascending=False).reset_index(drop=True)
        idx = min(replacement_rank - 1, len(ordered) - 1)
        replacement_mean = float(ordered.loc[idx, "season_mean"])
        replacement_p90 = float(ordered.loc[idx, "season_p90"])
        ordered["replacement_rank"] = replacement_rank
        ordered["vor_mean"] = ordered["season_mean"] - replacement_mean
        ordered["vor_p90"] = ordered["season_p90"] - replacement_p90
        output.append(ordered)
    result = pd.concat(output, ignore_index=True)
    result["draft_score"] = (
        (1.0 - ceiling_weight) * result["vor_mean"] + ceiling_weight * result["vor_p90"]
    )
    result = result.sort_values("draft_score", ascending=False).reset_index(drop=True)
    result["overall_rank"] = np.arange(1, len(result) + 1)

    # Tier breaks: a gap between consecutive scores that is large relative to
    # the local gap scale (median of the trailing window) starts a new tier.
    # A relative rule survives the steep-then-flat shape of draft value curves,
    # where any fixed threshold either splinters the top or merges the middle.
    scores = result["draft_score"].to_numpy()
    tiers = np.ones(len(result), dtype=int)
    if len(result) > 1:
        gaps = -np.diff(scores)
        tier = 1
        window = 8
        for i, gap in enumerate(gaps):
            if i >= 80:
                tiers[i + 1] = tier
                continue
            local = gaps[max(0, i - window): i + window + 1]
            local_scale = max(float(np.median(local[local > 0])) if (local > 0).any() else 0.0, 1e-9)
            if gap > max(2.0 * local_scale, tier_gap_fraction):
                tier += 1
            tiers[i + 1] = tier
    result["tier"] = tiers
    result["position_rank"] = result.groupby("position")["draft_score"].rank(
        ascending=False, method="first"
    ).astype(int)
    return result


def snake_picks(slot: int, *, league_teams: int = 6, rounds: int = 15) -> list[int]:
    """Overall pick numbers for a snake slot (1-indexed)."""

    if not 1 <= slot <= league_teams:
        raise ValueError("slot must be within league size")
    picks = []
    for round_number in range(1, rounds + 1):
        if round_number % 2 == 1:
            pick = (round_number - 1) * league_teams + slot
        else:
            pick = round_number * league_teams - slot + 1
        picks.append(pick)
    return picks


def availability_probability(
    board: pd.DataFrame, pick_number: int, *, adp_sd_floor: float = 6.0
) -> pd.Series:
    """P(player still available at ``pick_number``) from ADP (normal CDF).
    Players with no ADP are treated as always available."""

    from scipy.stats import norm  # scipy ships with scikit-learn's stack

    adp = board.get("adp")
    if adp is None:
        return pd.Series(1.0, index=board.index)
    sd = board.get("adp_sd", pd.Series(np.nan, index=board.index)).fillna(adp_sd_floor).clip(lower=adp_sd_floor)
    z = (adp - pick_number) / sd
    # P(available at pick k) = P(market drafts him at a pick >= k) = CDF(z):
    # a player with ADP far *after* your pick (z >> 0) is almost surely there;
    # one with ADP far before it (z << 0) is almost surely gone.
    prob = pd.Series(norm.cdf(z), index=board.index)
    return prob.where(adp.notna(), 1.0)


def round_ceiling_weight(
    round_number: int,
    *,
    start: float = 0.30,
    step: float = 0.08,
    cap: float = 0.80,
) -> float:
    """Progressive ceiling tilt by draft round.

    Early rounds buy reliable production (busts there sink a season);
    later rounds buy lottery tickets (their busts cost a waiver claim).
    Round 1 -> ~0.30 P90 weight, rising ~0.08 per round, capped at 0.80.
    """

    if round_number < 1:
        raise ValueError("round_number is 1-indexed")
    return float(min(start + step * (round_number - 1), cap))
