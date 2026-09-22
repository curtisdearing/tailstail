"""Shadow / research-only K and D/ST projections under this league's contract.

Why this module exists
----------------------
The league starts one K and one D/ST every week and the promoted model
projects neither, so two of nine starting seats are decided blind.  This
module gives those two seats a *number* with a *distribution*.  It does not
give them a promotion: the 2026 protocol freeze
(`docs/PROTOCOL_FREEZE_2026.md`) governs what may enter a lineup objective,
and nothing here has passed it.  `PROMOTION_STATUS` says so in code, and
`my_team._shadow` keeps rendering NO CURRENT PICK until an audit says
otherwise.

Isolation
---------
Nothing in this module is imported by the frozen QB/RB/WR/TE path.  It reads
`espn_contract` and `special_scoring` (pure scoring), and nflverse history.
It never touches `ModelConfig.positions`, `features.py`, `simulation.py`, or
`PlayerProjectionSnapshot`.

The feature clock
-----------------
Every projection for (season S, week W) is fit on rows *strictly before*
(S, W) -- see `past_only`, which is the only place the boundary is
expressed.  The repository has already been burned once by a feature that
read same-week presence (`features.py`, `expected_points_missing`), so the
boundary here is a single function with a leakage test that poisons future
rows and asserts the earlier forecast is bit-identical.

Why the D/ST mean is not a regression on past fantasy points
------------------------------------------------------------
Under this league's overrides a single defensive or return touchdown is worth
12-30 points against a sack's 1.0, so a defense's *mean* is dominated by an
event that happens a few times a season.  Regressing on historical fantasy
points would fit mostly to which defenses happened to catch one.  So the
touchdown hazard is modelled explicitly, per category, as a rate shrunk
toward a measured league base rate (`LEAGUE_BASE_RATES`, recomputed from
history at fit time and reported on the result), and the points come out of a
Monte Carlo over events rather than out of a fitted mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .espn_contract import (
    FIELD_GOAL_BUCKETS,
    POINTS_ALLOWED_BANDS,
    YARDS_ALLOWED_BANDS,
    LeagueContract,
)
from .special_scoring import score_dst, score_kicker

MODEL_VERSION = "special_teams-0.1.0-shadow"

#: Mirrors `shadow_kicker.PROMOTION_STATUS`.  Read by any consumer before it
#: is tempted to rank these rows into an objective.
PROMOTION_STATUS: Mapping[str, Any] = {
    "status": "shadow",
    "may_enter_lineup_objective": False,
    "reason": (
        "No season-forward promotion audit has passed for K or D/ST under the "
        "2026 protocol freeze. These numbers are research output."
    ),
}

#: Pseudo-counts for the empirical-Bayes shrinkage of a per-team or
#: per-kicker rate toward the league rate.  Units are games (or attempts, for
#: make rates).  Larger where the event is rarer, because a defense with one
#: pick-six in nine games has not demonstrated a pick-six rate.
PRIOR_GAMES: Mapping[str, float] = {
    "sacks": 8.0,
    "interceptions": 10.0,
    "fumble_recoveries": 10.0,
    "safeties": 40.0,
    "blocked_kicks": 30.0,
    "return_td": 40.0,
    "points": 6.0,
    "yards": 6.0,
    "fg_att": 8.0,
    "pat_att": 8.0,
}

#: Pseudo-attempts for a kicker's per-bucket make rate.  A kicker with three
#: 50-59 attempts must not get a bespoke long-range rate; the n-gate in the
#: K model card is expressed as this shrinkage rather than as a hard cutoff,
#: because a hard cutoff throws away the little information there is.
FG_PRIOR_ATTEMPTS: Mapping[str, float] = {
    "fg_made_0_39": 12.0,
    "fg_made_40_49": 10.0,
    "fg_made_50_59": 8.0,
    "fg_made_60_plus": 4.0,
}

DEFAULT_SIMULATIONS = 20000


class SpecialTeamsError(RuntimeError):
    """The projection cannot be produced honestly on the inputs given."""


# --------------------------------------------------------------------------- #
# The feature clock
# --------------------------------------------------------------------------- #
def past_only(frame: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Rows strictly before (``season``, ``week``).

    The ONLY expression of the information boundary in this module.  Every
    rate, every shrinkage target and every league base rate is computed from
    what this returns, so there is one place to audit and one place to test.
    Same-week rows are excluded even for the team being projected: a week's
    own result is not pregame information, however tempting the join.
    """
    if frame.empty:
        return frame
    season = int(season)
    week = int(week)
    mask = (frame["season"] < season) | ((frame["season"] == season) & (frame["week"] < week))
    return frame.loc[mask].copy()


# --------------------------------------------------------------------------- #
# History construction
# --------------------------------------------------------------------------- #
def _require(frame: pd.DataFrame, columns: Iterable[str], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise SpecialTeamsError(f"{what} is missing column(s) {missing}")


def build_dst_history(
    team_stats: pd.DataFrame,
    schedules: pd.DataFrame,
    opponent_tds_against: pd.DataFrame,
    contract: LeagueContract,
) -> pd.DataFrame:
    """One row per team-week, with the league's D/ST points as the target.

    ``opponent_tds_against`` counts touchdowns the OPPONENT scored while this
    team's offence or special teams had possession.  ESPN excludes exactly
    six points per such touchdown from a defense's points allowed -- and only
    the six: the extra point that follows is still charged.  That is not a
    guess; it is the rule that reproduces every 2026 week 1-2 applied total
    (see `tests/test_special_teams.py`).
    """
    _require(team_stats, ["season", "week", "team", "def_sacks"], "team_stats")
    _require(schedules, ["season", "week", "home_team", "away_team",
                         "home_score", "away_score"], "schedules")

    home = schedules.rename(columns={"home_team": "team", "away_team": "opponent",
                                     "home_score": "points_for", "away_score": "points_against"})
    away = schedules.rename(columns={"away_team": "team", "home_team": "opponent",
                                     "away_score": "points_for", "home_score": "points_against"})
    cols = ["season", "week", "team", "opponent", "points_for", "points_against"]
    games = pd.concat([home[cols], away[cols]], ignore_index=True)
    games = games.dropna(subset=["points_for", "points_against"])
    games["team_win"] = (games["points_for"] > games["points_against"]).astype(int)

    games = games.merge(opponent_tds_against, on=["season", "week", "team"], how="left")
    games["td_against_us"] = games["td_against_us"].fillna(0.0)
    games["points_allowed"] = games["points_against"] - 6.0 * games["td_against_us"]
    games["points_allowed"] = games["points_allowed"].clip(lower=0.0)

    off = team_stats[["season", "week", "team", "passing_yards", "rushing_yards",
                      "sack_yards_lost"]].rename(columns={"team": "opponent"})
    games = games.merge(off, on=["season", "week", "opponent"], how="left")
    own_off = team_stats[["season", "week", "team", "passing_yards", "rushing_yards",
                          "sack_yards_lost", "sacks_suffered"]].rename(columns={
        "passing_yards": "own_pass", "rushing_yards": "own_rush",
        "sack_yards_lost": "own_sack_yards"})
    games = games.merge(own_off, on=["season", "week", "team"], how="left")
    games["offense_yards"] = (games["own_pass"].fillna(0.0) + games["own_rush"].fillna(0.0)
                              + games["own_sack_yards"].fillna(0.0))
    games["sacks_allowed"] = games["sacks_suffered"].fillna(0.0)
    # nflverse reports sack_yards_lost as a negative number, so the official
    # "total net yards" is a sum, not a difference.  Adding it as a difference
    # inflates every defense's yards allowed by twice the sack yardage, which
    # is how this was wrong the first time.
    games["yards_allowed"] = (games["passing_yards"].fillna(0.0)
                              + games["rushing_yards"].fillna(0.0)
                              + games["sack_yards_lost"].fillna(0.0))

    defense = team_stats[[
        "season", "week", "team", "def_sacks", "def_interceptions", "def_safeties",
        "def_punt_blocks", "def_pat_blocks", "def_fg_blocks", "fumble_recovery_opp",
    ]].copy()
    games = games.merge(defense, on=["season", "week", "team"], how="left")
    games["sacks"] = games["def_sacks"].fillna(0.0)
    games["interceptions"] = games["def_interceptions"].fillna(0.0)
    games["fumble_recoveries"] = games["fumble_recovery_opp"].fillna(0.0)
    games["safeties"] = games["def_safeties"].fillna(0.0)
    games["blocked_kicks"] = (games["def_punt_blocks"].fillna(0.0)
                              + games["def_pat_blocks"].fillna(0.0)
                              + games["def_fg_blocks"].fillna(0.0))

    for column in ("interception_return_td", "fumble_return_td", "kickoff_return_td",
                   "punt_return_td", "blocked_kick_return_td", "two_point_return",
                   "one_point_safety"):
        if column not in games.columns:
            games[column] = 0.0

    games["fantasy_points"] = score_dst_frame(games, contract)
    games["defensive_tds"] = (games["interception_return_td"] + games["fumble_return_td"]
                              + games["blocked_kick_return_td"])
    games["return_tds"] = games["kickoff_return_td"] + games["punt_return_td"]
    return games.sort_values(["season", "week", "team"]).reset_index(drop=True)


#: History column -> the contract key `score_dst` prices it under.  These are
#: deliberately different vocabularies: the history frame is named for what
#: nflverse calls things, the contract is named for what ESPN calls things,
#: and the translation is written down once here.  Passing the history row
#: straight to `score_dst` silently scores every per-event category as zero,
#: because a missing key is a real "the league does not pay for this" answer
#: and cannot also mean "you spelled it wrong".
DST_HISTORY_TO_CONTRACT: Mapping[str, str] = {
    "sacks": "defensive_sack",
    "interceptions": "defensive_interception",
    "fumble_recoveries": "defensive_fumble_recovery",
    "safeties": "defensive_safety",
    "blocked_kicks": "blocked_kick",
    "interception_return_td": "interception_return_td",
    "fumble_return_td": "fumble_return_td",
    "kickoff_return_td": "kickoff_return_td",
    "punt_return_td": "punt_return_td",
    "blocked_kick_return_td": "blocked_kick_return_td",
    "two_point_return": "two_point_return",
    "one_point_safety": "one_point_safety",
}


def dst_line(row: Mapping[str, Any]) -> dict[str, Any]:
    """A history row rewritten into the stat line `score_dst` expects."""
    line: dict[str, Any] = {
        contract_key: row.get(history_key, 0.0) or 0.0
        for history_key, contract_key in DST_HISTORY_TO_CONTRACT.items()
    }
    line["points_allowed"] = row["points_allowed"]
    line["yards_allowed"] = row["yards_allowed"]
    line["team_win"] = row.get("team_win", 0)
    return line


def score_dst_frame(frame: pd.DataFrame, contract: LeagueContract) -> list[float]:
    """League D/ST points for every row of a history frame."""
    return [score_dst(dst_line(row), contract) for row in frame.to_dict("records")]


def build_kicker_history(
    player_stats: pd.DataFrame,
    schedules: pd.DataFrame,
    contract: LeagueContract,
) -> pd.DataFrame:
    """One row per kicker-week, with the league's K points as the target."""
    _require(player_stats, ["season", "week", "team", "player_id", "fg_made_0_19"],
             "player_stats")
    frame = player_stats.copy()
    frame["fg_made_0_39"] = (frame["fg_made_0_19"].fillna(0.0)
                             + frame["fg_made_20_29"].fillna(0.0)
                             + frame["fg_made_30_39"].fillna(0.0))
    frame["fg_made_40_49"] = frame["fg_made_40_49"].fillna(0.0)
    frame["fg_made_50_59"] = frame["fg_made_50_59"].fillna(0.0)
    frame["fg_made_60_plus"] = frame["fg_made_60_"].fillna(0.0)
    # A blocked field goal is a miss this league charges at -3; nflverse keeps
    # blocks out of `fg_missed`, so folding them in is the difference between
    # matching ESPN and being three points light per block.
    frame["field_goals_missed"] = (frame["fg_missed"].fillna(0.0)
                                   + frame["fg_blocked"].fillna(0.0))
    frame["pat_made"] = frame["pat_made"].fillna(0.0)
    frame["pat_missed"] = (frame["pat_missed"].fillna(0.0)
                           + frame["pat_blocked"].fillna(0.0))

    for _low, _high, key in FIELD_GOAL_BUCKETS:
        att = f"{key}_att"
        if key == "fg_made_0_39":
            missed = (frame["fg_missed_0_19"].fillna(0.0) + frame["fg_missed_20_29"].fillna(0.0)
                      + frame["fg_missed_30_39"].fillna(0.0))
        elif key == "fg_made_40_49":
            missed = frame["fg_missed_40_49"].fillna(0.0)
        elif key == "fg_made_50_59":
            missed = frame["fg_missed_50_59"].fillna(0.0)
        else:
            missed = frame["fg_missed_60_"].fillna(0.0)
        frame[att] = frame[key] + missed

    home = schedules.rename(columns={"home_team": "team", "away_team": "opponent",
                                     "home_score": "points_for", "away_score": "points_against"})
    away = schedules.rename(columns={"away_team": "team", "home_team": "opponent",
                                     "away_score": "points_for", "home_score": "points_against"})
    cols = ["season", "week", "team", "opponent", "points_for", "points_against"]
    games = pd.concat([home[cols], away[cols]], ignore_index=True).dropna(
        subset=["points_for", "points_against"])
    games["team_win"] = (games["points_for"] > games["points_against"]).astype(int)
    frame = frame.merge(games, on=["season", "week", "team"], how="left")
    frame = frame.dropna(subset=["team_win"])
    frame["team_win"] = frame["team_win"].astype(int)

    frame["fantasy_points"] = [
        score_kicker(row, contract) for row in frame.to_dict("records")
    ]
    keep = ["season", "week", "player_id", "player_display_name", "team", "opponent",
            "team_win", "points_for", "fantasy_points", "field_goals_missed",
            "pat_made", "pat_missed"]
    keep += [key for _, _, key in FIELD_GOAL_BUCKETS]
    keep += [f"{key}_att" for _, _, key in FIELD_GOAL_BUCKETS]
    keep = [c for c in keep if c in frame.columns]
    return frame[keep].sort_values(["season", "week", "team"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Shrinkage helpers
# --------------------------------------------------------------------------- #
def _shrunk_rate(events: float, exposure: float, league_rate: float, prior: float) -> float:
    """Empirical-Bayes rate: ``(events + prior*league) / (exposure + prior)``.

    With no exposure this returns the league rate exactly, which is the honest
    answer for a team or kicker with no history -- not zero, and not a bespoke
    rate fitted to nothing.
    """
    exposure = max(float(exposure), 0.0)
    return (float(events) + float(prior) * float(league_rate)) / (exposure + float(prior))


def _factor(events: float, exposure: float, league_rate: float, prior: float,
            lo: float = 0.5, hi: float = 2.0) -> float:
    """A multiplicative strength factor, shrunk and then clipped.

    Clipping is not cosmetic: an unclipped ratio lets one team-week with two
    sacks allowed against a league mean of 2.3 become a 0.0 factor and zero
    out the whole term.
    """
    if league_rate <= 0:
        return 1.0
    rate = _shrunk_rate(events, exposure, league_rate, prior)
    return float(np.clip(rate / league_rate, lo, hi))


def _band_probabilities(samples: np.ndarray,
                        bands: Sequence[tuple[int, float, str | None]],
                        contract: LeagueContract) -> np.ndarray:
    """Points contributed by a tier ladder, per simulation draw."""
    out = np.zeros_like(samples, dtype=float)
    for low, high, key in bands:
        if key is None:
            continue
        inside = (samples >= low) & (samples <= high)
        out = out + inside * contract.dst_points(key)
    return out


# --------------------------------------------------------------------------- #
# Measured league base rates
# --------------------------------------------------------------------------- #
def league_base_rates(history: pd.DataFrame) -> dict[str, float]:
    """Per-team-game league rates, measured on whatever ``history`` is given.

    Called only on `past_only` output, so these are prior rates.  They are
    returned on every projection so a reader can see the base rate the rare
    events were shrunk toward rather than having to trust it.
    """
    n = max(len(history), 1)
    rates = {
        "games": float(len(history)),
        "sacks_per_game": float(history["sacks"].mean()) if len(history) else 2.4,
        "interceptions_per_game": float(history["interceptions"].mean()) if len(history) else 0.75,
        "fumble_recoveries_per_game": float(history["fumble_recoveries"].mean()) if len(history) else 0.6,
        "safeties_per_game": float(history["safeties"].sum()) / n,
        "blocked_kicks_per_game": float(history["blocked_kicks"].sum()) / n,
        "kickoff_return_td_per_game": float(history["kickoff_return_td"].sum()) / n,
        "punt_return_td_per_game": float(history["punt_return_td"].sum()) / n,
        "blocked_kick_return_td_per_game": float(history["blocked_kick_return_td"].sum()) / n,
        "points_allowed_mean": float(history["points_allowed"].mean()) if len(history) else 22.0,
        "points_allowed_sd": float(history["points_allowed"].std(ddof=1)) if len(history) > 1 else 10.0,
        "yards_allowed_mean": float(history["yards_allowed"].mean()) if len(history) else 330.0,
        "yards_allowed_sd": float(history["yards_allowed"].std(ddof=1)) if len(history) > 1 else 80.0,
        "sacks_allowed_per_game": float(history["sacks_allowed"].mean()) if len(history) else 2.4,
        "offense_yards_mean": float(history["offense_yards"].mean()) if len(history) else 330.0,
    }
    ints = float(history["interceptions"].sum()) if len(history) else 0.0
    frs = float(history["fumble_recoveries"].sum()) if len(history) else 0.0
    rates["int_return_td_per_interception"] = (
        float(history["interception_return_td"].sum()) / ints if ints > 0 else 0.055)
    rates["fumble_return_td_per_recovery"] = (
        float(history["fumble_return_td"].sum()) / frs if frs > 0 else 0.070)
    return rates


def league_fg_rates(history: pd.DataFrame) -> dict[str, float]:
    """League make rate and attempts-per-game for each scoring bucket."""
    out: dict[str, float] = {}
    n = max(len(history), 1)
    for _, _, key in FIELD_GOAL_BUCKETS:
        att = float(history[f"{key}_att"].sum()) if len(history) else 0.0
        made = float(history[key].sum()) if len(history) else 0.0
        out[f"{key}_make"] = made / att if att > 0 else 0.5
        out[f"{key}_att_per_game"] = att / n
    out["pat_att_per_game"] = (
        float((history["pat_made"] + history["pat_missed"]).sum()) / n if len(history) else 2.4)
    out["pat_make"] = (
        float(history["pat_made"].sum()) / max(float((history["pat_made"] + history["pat_missed"]).sum()), 1.0)
        if len(history) else 0.95)
    return out


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Projection:
    """One seat's distribution, plus the components it was built from."""

    subject: str
    position: str
    team: str
    opponent: str
    season: int
    week: int
    mean: float
    sd: float
    p10: float
    p50: float
    p90: float
    p_zero_or_less: float
    components: Mapping[str, float] = field(default_factory=dict)
    status: str = "shadow"
    model_version: str = MODEL_VERSION
    n_prior_games: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject, "position": self.position, "team": self.team,
            "opponent": self.opponent, "season": int(self.season), "week": int(self.week),
            "mean": round(float(self.mean), 4), "sd": round(float(self.sd), 4),
            "p10": round(float(self.p10), 4), "p50": round(float(self.p50), 4),
            "p90": round(float(self.p90), 4),
            "p_zero_or_less": round(float(self.p_zero_or_less), 4),
            "components": {k: round(float(v), 5) for k, v in sorted(self.components.items())},
            "status": self.status, "model_version": self.model_version,
            "promoted": False, "n_prior_games": int(self.n_prior_games),
        }


def _stable_seed(*parts: Any) -> int:
    """A seed that does not move with ``PYTHONHASHSEED``.

    Python's builtin ``hash()`` is salted per process, which is exactly how
    `shadow_kicker` shipped a determinism claim that was false; this uses a
    stable digest instead.
    """
    import hashlib
    blob = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") % (2**32 - 1)


def _score_rate(history: pd.DataFrame, team: str, column: str, prior: float,
                league_rate: float) -> float:
    rows = history.loc[history["team"] == team]
    return _shrunk_rate(rows[column].sum(), len(rows), league_rate, prior)


def project_dst(history: pd.DataFrame, team: str, opponent: str, season: int, week: int,
                contract: LeagueContract, *, simulations: int = DEFAULT_SIMULATIONS,
                ) -> Projection:
    """Project one team defense for (``season``, ``week``).

    ``history`` may contain any rows; only `past_only` rows are read.
    """
    prior = past_only(history, season, week)
    if prior.empty:
        raise SpecialTeamsError(
            f"no prior rows before {season} week {week}: a defense with no history gets no "
            "distribution rather than the league average wearing its name")
    base = league_base_rates(prior)
    own = prior.loc[prior["team"] == team]
    opp = prior.loc[prior["team"] == opponent]

    rng = np.random.default_rng(_stable_seed(MODEL_VERSION, season, week, team, opponent))
    n = int(simulations)

    # -- event rates: league base x our defence x their offence ------------- #
    lam_sack = (base["sacks_per_game"]
                * _factor(own["sacks"].sum(), len(own), base["sacks_per_game"], PRIOR_GAMES["sacks"])
                * _factor(opp["sacks_allowed"].sum(), len(opp),
                          base["sacks_allowed_per_game"], PRIOR_GAMES["sacks"]))
    lam_int = (base["interceptions_per_game"]
               * _factor(own["interceptions"].sum(), len(own), base["interceptions_per_game"],
                         PRIOR_GAMES["interceptions"]))
    lam_fr = (base["fumble_recoveries_per_game"]
              * _factor(own["fumble_recoveries"].sum(), len(own), base["fumble_recoveries_per_game"],
                        PRIOR_GAMES["fumble_recoveries"]))
    lam_saf = _score_rate(prior, team, "safeties", PRIOR_GAMES["safeties"], base["safeties_per_game"])
    lam_blk = _score_rate(prior, team, "blocked_kicks", PRIOR_GAMES["blocked_kicks"],
                          base["blocked_kicks_per_game"])
    lam_kr = _score_rate(prior, team, "kickoff_return_td", PRIOR_GAMES["return_td"],
                         base["kickoff_return_td_per_game"])
    lam_pr = _score_rate(prior, team, "punt_return_td", PRIOR_GAMES["return_td"],
                         base["punt_return_td_per_game"])
    lam_bkr = base["blocked_kick_return_td_per_game"]
    p_int_td = _shrunk_rate(own["interception_return_td"].sum(), own["interceptions"].sum(),
                            base["int_return_td_per_interception"], 25.0)
    p_fum_td = _shrunk_rate(own["fumble_return_td"].sum(), own["fumble_recoveries"].sum(),
                            base["fumble_return_td_per_recovery"], 25.0)

    sacks = rng.poisson(lam_sack, n)
    ints = rng.poisson(lam_int, n)
    frs = rng.poisson(lam_fr, n)
    safeties = rng.poisson(max(lam_saf, 0.0), n)
    blocks = rng.poisson(max(lam_blk, 0.0), n)
    int_tds = rng.binomial(ints, min(max(p_int_td, 0.0), 1.0))
    fum_tds = rng.binomial(frs, min(max(p_fum_td, 0.0), 1.0))
    kr_tds = rng.poisson(max(lam_kr, 0.0), n)
    pr_tds = rng.poisson(max(lam_pr, 0.0), n)
    bkr_tds = rng.poisson(max(lam_bkr, 0.0), n)

    # -- points and yards allowed, and the win, from a shared score draw ---- #
    mu_pa = (base["points_allowed_mean"]
             * _factor(own["points_allowed"].sum(), len(own), base["points_allowed_mean"],
                       PRIOR_GAMES["points"])
             * _factor(opp["points_for"].sum(), len(opp), base["points_allowed_mean"],
                       PRIOR_GAMES["points"]))
    mu_pf = (base["points_allowed_mean"]
             * _factor(own["points_for"].sum(), len(own), base["points_allowed_mean"],
                       PRIOR_GAMES["points"])
             * _factor(opp["points_allowed"].sum(), len(opp), base["points_allowed_mean"],
                       PRIOR_GAMES["points"]))
    # Negative binomial: scores are over-dispersed relative to Poisson, and a
    # Poisson here would understate the shutout tail that this league pays 5
    # points for.
    pa = _negbin(rng, mu_pa, base["points_allowed_sd"], n)
    pf = _negbin(rng, mu_pf, base["points_allowed_sd"], n)
    wins = (pf > pa).astype(float)

    mu_ya = (base["yards_allowed_mean"]
             * _factor(own["yards_allowed"].sum(), len(own), base["yards_allowed_mean"],
                       PRIOR_GAMES["yards"])
             * _factor(opp["offense_yards"].sum(), len(opp),
                       base["offense_yards_mean"], PRIOR_GAMES["yards"]))
    ya = np.maximum(rng.normal(mu_ya, base["yards_allowed_sd"], n), 0.0)

    points = (
        sacks * contract.dst_points("defensive_sack")
        + ints * contract.dst_points("defensive_interception")
        + frs * contract.dst_points("defensive_fumble_recovery")
        + safeties * contract.dst_points("defensive_safety")
        + blocks * contract.dst_points("blocked_kick")
        + int_tds * contract.dst_points("interception_return_td")
        + fum_tds * contract.dst_points("fumble_return_td")
        + kr_tds * contract.dst_points("kickoff_return_td")
        + pr_tds * contract.dst_points("punt_return_td")
        + bkr_tds * contract.dst_points("blocked_kick_return_td")
        + wins * contract.dst_points("team_win")
        + _band_probabilities(pa, POINTS_ALLOWED_BANDS, contract)
        + _band_probabilities(ya, YARDS_ALLOWED_BANDS, contract)
    )

    td_points = (int_tds * contract.dst_points("interception_return_td")
                 + fum_tds * contract.dst_points("fumble_return_td")
                 + kr_tds * contract.dst_points("kickoff_return_td")
                 + pr_tds * contract.dst_points("punt_return_td")
                 + bkr_tds * contract.dst_points("blocked_kick_return_td"))
    components = {
        "lambda_sacks": lam_sack, "lambda_interceptions": lam_int,
        "lambda_fumble_recoveries": lam_fr, "lambda_safeties": lam_saf,
        "lambda_blocked_kicks": lam_blk, "lambda_kick_return_td": lam_kr,
        "lambda_punt_return_td": lam_pr, "lambda_blocked_kick_return_td": lam_bkr,
        "p_td_given_interception": p_int_td, "p_td_given_fumble_recovery": p_fum_td,
        "expected_points_allowed": mu_pa, "expected_points_for": mu_pf,
        "expected_yards_allowed": mu_ya, "p_win": float(wins.mean()),
        "p_any_touchdown": float((int_tds + fum_tds + kr_tds + pr_tds + bkr_tds > 0).mean()),
        "mean_touchdown_points": float(td_points.mean()),
        "share_of_mean_from_touchdowns": float(td_points.mean() / points.mean())
        if points.mean() != 0 else float("nan"),
        "league_base_sacks_per_game": base["sacks_per_game"],
        "league_base_int_td_per_interception": base["int_return_td_per_interception"],
        "league_base_fumble_td_per_recovery": base["fumble_return_td_per_recovery"],
        "league_base_kick_return_td_per_game": base["kickoff_return_td_per_game"],
        "league_base_punt_return_td_per_game": base["punt_return_td_per_game"],
        "league_base_safety_per_game": base["safeties_per_game"],
    }
    return Projection(
        subject=f"{team} D/ST", position="D/ST", team=team, opponent=opponent,
        season=int(season), week=int(week), mean=float(points.mean()), sd=float(points.std(ddof=1)),
        p10=float(np.percentile(points, 10)), p50=float(np.percentile(points, 50)),
        p90=float(np.percentile(points, 90)),
        p_zero_or_less=float((points <= 0).mean()),
        components=components, n_prior_games=int(len(own)),
    )


def _negbin(rng: np.random.Generator, mean: float, sd: float, n: int) -> np.ndarray:
    """Negative-binomial draws with the given mean and (at least Poisson) sd."""
    mean = max(float(mean), 0.1)
    var = max(float(sd) ** 2, mean * 1.05)
    p = mean / var
    r = mean * p / (1.0 - p)
    return rng.negative_binomial(max(r, 0.01), min(max(p, 1e-6), 1 - 1e-6), n).astype(float)


def project_kicker(history: pd.DataFrame, dst_history: pd.DataFrame, player_id: str,
                   team: str, opponent: str, season: int, week: int,
                   contract: LeagueContract, *, simulations: int = DEFAULT_SIMULATIONS,
                   name: str | None = None) -> Projection:
    """Project one kicker for (``season``, ``week``).

    Volume comes from the TEAM (how often this offence reaches scoring range),
    accuracy comes from the KICKER.  Splitting them is what lets a new kicker
    on a good offence be projected at all: he inherits the team's attempt
    rate and the league's make rate, visibly, instead of inheriting the
    incumbent's numbers.
    """
    prior = past_only(history, season, week)
    if prior.empty:
        raise SpecialTeamsError(
            f"no prior kicker rows before {season} week {week}")
    league = league_fg_rates(prior)
    team_rows = prior.loc[prior["team"] == team]
    own = prior.loc[prior["player_id"] == player_id]
    opp_rows = prior.loc[prior["opponent"] == opponent]

    rng = np.random.default_rng(_stable_seed(MODEL_VERSION, season, week, "K", player_id, team))
    n = int(simulations)
    points = np.zeros(n, dtype=float)
    components: dict[str, float] = {}

    for _low, _high, key in FIELD_GOAL_BUCKETS:
        att_col = f"{key}_att"
        lam_team = _shrunk_rate(team_rows[att_col].sum(), len(team_rows),
                                league[f"{key}_att_per_game"], PRIOR_GAMES["fg_att"])
        opp_factor = _factor(opp_rows[att_col].sum(), len(opp_rows),
                             league[f"{key}_att_per_game"], PRIOR_GAMES["fg_att"])
        lam = max(lam_team * opp_factor, 0.0)
        make = _shrunk_rate(own[key].sum(), own[att_col].sum(),
                            league[f"{key}_make"], FG_PRIOR_ATTEMPTS[key])
        make = float(np.clip(make, 0.0, 1.0))
        attempts = rng.poisson(lam, n)
        made = rng.binomial(attempts, make)
        missed = attempts - made
        points = points + made * contract.points(key) + missed * contract.points("fg_missed_total")
        components[f"lambda_att_{key}"] = lam
        components[f"p_make_{key}"] = make

    lam_pat = _shrunk_rate((team_rows["pat_made"] + team_rows["pat_missed"]).sum(),
                           len(team_rows), league["pat_att_per_game"], PRIOR_GAMES["pat_att"])
    p_pat = float(np.clip(_shrunk_rate(own["pat_made"].sum(),
                                       (own["pat_made"] + own["pat_missed"]).sum(),
                                       league["pat_make"], 20.0), 0.0, 1.0))
    pat_att = rng.poisson(max(lam_pat, 0.0), n)
    pat_made = rng.binomial(pat_att, p_pat)
    points = points + pat_made * contract.points("pat_made") \
        + (pat_att - pat_made) * contract.points("pat_missed")

    p_win = _win_probability(dst_history, team, opponent, season, week)
    wins = (rng.random(n) < p_win).astype(float)
    points = points + wins * contract.points("team_win")

    components.update({
        "lambda_pat_attempts": lam_pat, "p_make_pat": p_pat, "p_win": p_win,
        "league_make_0_39": league["fg_made_0_39_make"],
        "league_make_40_49": league["fg_made_40_49_make"],
        "league_make_50_59": league["fg_made_50_59_make"],
        "league_make_60_plus": league["fg_made_60_plus_make"],
    })
    return Projection(
        subject=name or str(player_id), position="K", team=team, opponent=opponent,
        season=int(season), week=int(week), mean=float(points.mean()),
        sd=float(points.std(ddof=1)), p10=float(np.percentile(points, 10)),
        p50=float(np.percentile(points, 50)), p90=float(np.percentile(points, 90)),
        p_zero_or_less=float((points <= 0).mean()), components=components,
        n_prior_games=int(len(own)),
    )


def _win_probability(dst_history: pd.DataFrame, team: str, opponent: str,
                     season: int, week: int) -> float:
    """P(team beats opponent), from prior scoring rates only.

    No market line is read.  The repo has an odds source, but wiring a paid
    feed into a shadow module would make the shadow lane depend on a
    credential the audit path does not have; the gap is declared rather than
    filled with a number that looks like a market and is not.
    """
    prior = past_only(dst_history, season, week)
    if prior.empty:
        return 0.5
    base = float(prior["points_for"].mean())
    own = prior.loc[prior["team"] == team]
    opp = prior.loc[prior["team"] == opponent]
    mu_for = base * _factor(own["points_for"].sum(), len(own), base, PRIOR_GAMES["points"]) \
        * _factor(opp["points_allowed"].sum(), len(opp), base, PRIOR_GAMES["points"])
    mu_against = base * _factor(opp["points_for"].sum(), len(opp), base, PRIOR_GAMES["points"]) \
        * _factor(own["points_allowed"].sum(), len(own), base, PRIOR_GAMES["points"])
    sd = float(prior["points_for"].std(ddof=1)) or 10.0
    from math import erf, sqrt
    z = (mu_for - mu_against) / (sd * sqrt(2.0))
    return float(min(max(0.5 * (1.0 + erf(z)), 0.02), 0.98))
