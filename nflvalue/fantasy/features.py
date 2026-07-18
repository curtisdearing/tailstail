"""Roster-first, prior-only fantasy feature construction.

Every model feature is either a pregame fact or begins with ``pre_``/``z_``
and is computed after a player/team shift.  Current-week results remain in the
frame solely as labels and simulation-audit fields; they are never returned by
``model_features``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .config import ScoringRules
from .data import HistoricalData
from .scoring import add_fantasy_points

POSITIONS = ("QB", "RB", "WR", "TE")
KEYS = ["season", "week", "player_id"]

STAT_DEFAULTS = (
    "completions", "attempts", "passing_yards", "passing_tds",
    "passing_interceptions", "passing_2pt_conversions", "carries",
    "rushing_yards", "rushing_tds", "rushing_2pt_conversions", "receptions",
    "targets", "receiving_yards", "receiving_tds", "receiving_2pt_conversions",
    "target_share", "air_yards_share", "wopr", "passing_epa", "rushing_epa",
    "receiving_epa", "passing_cpoe",
)

EXPECTED_COMPONENTS = {
    "pass_fantasy_points_exp": "expected_pass_points",
    "rush_fantasy_points_exp": "expected_rush_points",
    "rec_fantasy_points_exp": "expected_receiving_points",
    "total_touchdown_exp": "expected_touchdowns",
    "total_yards_gained_exp": "expected_yards",
    "receptions_exp": "expected_receptions",
    "pass_completions_exp": "expected_completions",
    "pass_interception_exp": "expected_interceptions",
    "total_first_down_exp": "expected_first_downs",
}

ROLLING_BASES = (
    "fantasy_points", "expected_points", *EXPECTED_COMPONENTS.values(),
    "attempts", "completions", "carries",
    "targets", "receptions", "touches", "opportunities", "passing_yards",
    "rushing_yards", "receiving_yards", "total_tds", "fumbles_lost",
    "offense_pct", "target_share_calc", "carry_share", "air_yards_share",
    "wopr", "yards_per_attempt", "yards_per_carry", "yards_per_target",
    "catch_rate", "td_per_opportunity", "passing_epa", "rushing_epa",
    "receiving_epa", "passing_cpoe", "interception_rate", "fumble_rate",
)

ROLE_TREND_BASES = (
    "attempts", "carries", "targets", "receptions", "opportunities",
    "offense_pct", "target_share_calc", "carry_share", "expected_points",
)

PREGAME_NUMERIC = (
    "is_home", "team_spread", "total_line", "implied_team_points", "rest",
    "opponent_rest", "rest_advantage", "division_game", "primetime", "dome",
    "outdoors", "turf", "temperature", "wind", "age", "years_exp",
    "draft_number", "team_changed", "qb_changed", "status_inactive",
    "injury_out", "injury_doubtful", "injury_questionable", "practice_dnp",
    "practice_limited", "vacated_target_share", "vacated_carry_share",
    "wr1_out", "rb1_out", "te1_out", "unavailable_skill_count",
    "schedule_missing", "line_missing", "weather_missing", "age_missing",
    "snaps_missing", "expected_points_missing",
    "coach_history_missing", "roster_snapshot_carried",
)

CONDITION_FEATURES = (
    "stadium", "referee", "surface", "primetime_key", "current_qb_id",
)

NORMALIZE_BASES = (
    "pre_fantasy_points_ewm4", "pre_expected_points_ewm4",
    "pre_opportunities_ewm4", "pre_offense_pct_ewm4",
    "pre_target_share_calc_ewm4", "pre_carry_share_ewm4",
    "pre_total_tds_ewm8", "pre_points_over_expected_ewm4",
)

FEATURE_GROUPS = {
    "history": [
        "pre_fantasy_points_ewm2", "pre_fantasy_points_ewm4",
        "pre_fantasy_points_ewm8", "pre_fantasy_points_sd8",
        "pre_expected_points_ewm2", "pre_expected_points_ewm4",
        "pre_expected_points_ewm8", "pre_points_over_expected_ewm4",
        "pre_expected_pass_points_ewm4", "pre_expected_rush_points_ewm4",
        "pre_expected_receiving_points_ewm4",
        "pre_played_games", "pre_roster_weeks",
    ],
    "opportunity": [
        "pre_attempts_ewm4", "pre_carries_ewm4", "pre_targets_ewm4",
        "pre_receptions_ewm4", "pre_touches_ewm4", "pre_opportunities_ewm4",
        "pre_offense_pct_ewm4", "pre_target_share_calc_ewm4",
        "pre_carry_share_ewm4", "pre_air_yards_share_ewm4", "pre_wopr_ewm4",
        "pre_expected_touchdowns_ewm4", "pre_expected_touchdowns_ewm8",
        "pre_expected_yards_ewm4", "pre_expected_receptions_ewm4",
        "pre_expected_completions_ewm4", "pre_expected_interceptions_ewm8",
        "pre_expected_first_downs_ewm4",
        *[f"pre_{column}_trend_2v8" for column in ROLE_TREND_BASES],
    ],
    "efficiency": [
        "pre_yards_per_attempt_ewm8", "pre_yards_per_carry_ewm8",
        "pre_yards_per_target_ewm8", "pre_catch_rate_ewm8",
        "pre_td_per_opportunity_ewm8", "pre_passing_epa_ewm8",
        "pre_rushing_epa_ewm8", "pre_receiving_epa_ewm8",
        "pre_passing_cpoe_ewm8",
    ],
    "team_role": [
        "pre_team_pass_attempts_ewm4", "pre_team_rush_attempts_ewm4",
        "pre_team_points_ewm4", "pre_team_expected_points_ewm4",
        "pre_coach_pass_rate_ewm8", "pre_coach_pace_ewm8",
        "pre_coach_rb_target_share_ewm8", "pre_coach_wr_target_share_ewm8",
        "pre_coach_te_target_share_ewm8", "pre_coach_qb_rush_share_ewm8",
        "pre_reconciled_attempt_share", "pre_reconciled_target_share",
        "pre_reconciled_carry_share", "pre_topdown_attempts",
        "pre_topdown_targets", "pre_topdown_carries",
        "pre_topdown_opportunities", "pre_position_role_rank",
        "pre_usage_share_overflow", "coach_history_missing",
        "vacated_target_share", "vacated_carry_share", "wr1_out", "rb1_out",
        "te1_out", "unavailable_skill_count", "team_changed", "qb_changed",
    ],
    "opponent": ["pre_opponent_position_points_ewm8"],
    "market_game": [
        "is_home", "team_spread", "total_line", "implied_team_points", "rest",
        "opponent_rest", "rest_advantage", "division_game", "primetime",
        "schedule_missing", "line_missing",
    ],
    "environment": ["dome", "outdoors", "turf", "temperature", "wind", "weather_missing"],
    "injury": [
        "status_inactive", "injury_out", "injury_doubtful",
        "injury_questionable", "practice_dnp", "practice_limited",
    ],
    "player_context": [
        "age", "years_exp", "draft_number", "pre_stadium_uplift",
        "pre_referee_uplift", "pre_surface_uplift", "pre_primetime_key_uplift",
        "pre_current_qb_id_uplift", "pre_current_qb_id_games",
        "age_missing", "snaps_missing", "expected_points_missing",
        "roster_snapshot_carried",
    ],
    "normalized": [f"z_{column[4:]}" for column in NORMALIZE_BASES],
}


def model_features() -> list[str]:
    return list(dict.fromkeys(column for columns in FEATURE_GROUPS.values() for column in columns))


def _numeric(frame: pd.DataFrame, columns: Iterable[str], fill: float = 0.0) -> None:
    for column in columns:
        if column not in frame:
            frame[column] = fill
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(fill)


def _coalesce(frame: pd.DataFrame, target: str, candidates: list[str], default=None) -> None:
    values = None
    for column in candidates:
        if column not in frame:
            continue
        values = frame[column] if values is None else values.combine_first(frame[column])
    if values is None:
        frame[target] = default
    else:
        frame[target] = values if default is None else values.fillna(default)


def _schedule_long(schedules: pd.DataFrame) -> pd.DataFrame:
    games = schedules.copy()
    if "game_type" in games:
        games = games[games["game_type"].fillna("REG").eq("REG")]
    common = [
        "season", "week", "game_id", "gameday", "weekday", "gametime",
        "total_line", "div_game", "roof", "surface", "temp", "wind",
        "referee", "stadium", "stadium_id",
    ]
    for column in common:
        if column not in games:
            games[column] = np.nan

    rows = []
    for side, opponent in (("home", "away"), ("away", "home")):
        selected = games[common].copy()
        selected["team"] = games[f"{side}_team"]
        selected["opponent_team"] = games[f"{opponent}_team"]
        selected["is_home"] = float(side == "home")
        selected["rest"] = pd.to_numeric(games.get(f"{side}_rest"), errors="coerce")
        selected["opponent_rest"] = pd.to_numeric(games.get(f"{opponent}_rest"), errors="coerce")
        home_spread = pd.to_numeric(games.get("spread_line"), errors="coerce")
        selected["team_spread"] = home_spread if side == "home" else -home_spread
        selected["current_qb_id"] = games.get(f"{side}_qb_id")
        selected["opponent_qb_id"] = games.get(f"{opponent}_qb_id")
        selected["coach"] = games.get(f"{side}_coach")
        rows.append(selected)
    out = pd.concat(rows, ignore_index=True)
    out["implied_team_points"] = (
        pd.to_numeric(out["total_line"], errors="coerce") / 2.0
        - pd.to_numeric(out["team_spread"], errors="coerce") / 2.0
    )
    out["rest_advantage"] = out["rest"] - out["opponent_rest"]
    time = out["gametime"].fillna("").astype(str).str.extract(r"^(\d{1,2})")[0]
    hour = pd.to_numeric(time, errors="coerce")
    weekday = out["weekday"].fillna("").astype(str).str.lower()
    out["primetime"] = ((hour >= 20) | weekday.isin(["monday", "thursday"])).astype(float)
    roof = out["roof"].fillna("unknown").astype(str).str.lower()
    surface = out["surface"].fillna("unknown").astype(str).str.lower()
    out["dome"] = roof.isin(["dome", "closed"]).astype(float)
    out["outdoors"] = roof.eq("outdoors").astype(float)
    out["turf"] = surface.str.contains("turf|artificial|astro", regex=True).astype(float)
    out["division_game"] = pd.to_numeric(out["div_game"], errors="coerce").fillna(0.0)
    out["temperature"] = pd.to_numeric(out["temp"], errors="coerce")
    out["wind"] = pd.to_numeric(out["wind"], errors="coerce")
    out["primetime_key"] = np.where(out["primetime"].eq(1), "primetime", "daytime")
    return out.drop_duplicates(["season", "week", "team"], keep="last")


def _prepare_rosters(rosters: pd.DataFrame) -> pd.DataFrame:
    r = rosters.copy()
    if "game_type" in r:
        r = r[r["game_type"].fillna("REG").eq("REG")]
    r = r[r["position"].isin(POSITIONS)].copy()
    r["player_id"] = r["gsis_id"].fillna("").astype(str)
    r = r[r["player_id"].str.startswith("00-")].copy()
    active = r.get("status", "").fillna("").astype(str).str.upper().isin(["ACT", "ACTIVE"])
    r["_active_priority"] = active.astype(int)
    r = r.sort_values(KEYS + ["_active_priority"]).drop_duplicates(KEYS, keep="last")
    keep = KEYS + [
        column for column in (
            "team", "position", "status", "full_name", "birth_date", "height",
            "weight", "years_exp", "entry_year", "rookie_year", "draft_club",
            "draft_number", "pfr_id", "depth_chart_position",
            "status_description_abbr",
            "snapshot_carried_forward",
        ) if column in r
    ]
    return r[keep]


def _prepare_stats(stats: pd.DataFrame, rules: ScoringRules) -> pd.DataFrame:
    s = stats.copy()
    season_type = "season_type" if "season_type" in s else "game_type"
    if season_type in s:
        s = s[s[season_type].fillna("REG").eq("REG")]
    s = s[s["position"].isin(POSITIONS)].copy()
    s = s.sort_values(KEYS).drop_duplicates(KEYS, keep="last")
    _numeric(s, STAT_DEFAULTS)
    if "fumbles_lost_total" in s:
        s["fumbles_lost"] = pd.to_numeric(s["fumbles_lost_total"], errors="coerce").fillna(0.0)
    else:
        fumble_columns = [
            column for column in ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost")
            if column in s
        ]
        s["fumbles_lost"] = s[fumble_columns].sum(axis=1) if fumble_columns else 0.0
    s = add_fantasy_points(s, rules, output="fantasy_points")
    keep = KEYS + [
        column for column in (
            "player_name", "player_display_name", "position", "team", "opponent_team",
            "game_id", *STAT_DEFAULTS, "fumbles_lost", "fantasy_points",
        ) if column in s
    ]
    return s[keep]


def _prior_ewm(frame: pd.DataFrame, group: list[str], column: str, span: int) -> pd.Series:
    return frame.groupby(group, sort=False)[column].transform(
        lambda values: values.shift(1).ewm(
            span=span, min_periods=1, adjust=False, ignore_na=True
        ).mean()
    )


def _prior_sd(frame: pd.DataFrame, group: list[str], column: str, window: int = 8) -> pd.Series:
    return frame.groupby(group, sort=False)[column].transform(
        lambda values: values.shift(1).rolling(window, min_periods=2).std()
    )


def _merge_optional(frame: pd.DataFrame, data: HistoricalData) -> pd.DataFrame:
    out = frame
    if data.expected_points is not None and not data.expected_points.empty:
        x = data.expected_points.copy()
        x["season"] = pd.to_numeric(x["season"], errors="coerce").astype("Int64")
        x["week"] = pd.to_numeric(x["week"], errors="coerce").astype("Int64")
        expected_col = "total_fantasy_points_exp"
        if expected_col in x:
            rename = {expected_col: "expected_points"}
            rename.update({source: target for source, target in EXPECTED_COMPONENTS.items() if source in x})
            x = x.rename(columns=rename)
            expected_columns = ["expected_points", *EXPECTED_COMPONENTS.values()]
            expected_columns = [column for column in expected_columns if column in x]
            x = x[["season", "week", "player_id", *expected_columns]].drop_duplicates(KEYS)
            out = out.merge(x, on=KEYS, how="left")
    if "expected_points" not in out:
        out["expected_points"] = np.nan
    for column in EXPECTED_COMPONENTS.values():
        if column not in out:
            out[column] = np.nan

    if data.snaps is not None and not data.snaps.empty and "pfr_id" in out:
        snaps = data.snaps.copy()
        if "game_type" in snaps:
            snaps = snaps[snaps["game_type"].fillna("REG").eq("REG")]
        snaps = snaps.rename(columns={"pfr_player_id": "pfr_id"})
        if {"season", "week", "pfr_id", "offense_pct"}.issubset(snaps):
            snaps = snaps[["season", "week", "pfr_id", "offense_pct"]].drop_duplicates(
                ["season", "week", "pfr_id"], keep="last"
            )
            out = out.merge(snaps, on=["season", "week", "pfr_id"], how="left")
    if "offense_pct" not in out:
        out["offense_pct"] = np.nan

    if data.injuries is not None and not data.injuries.empty:
        injuries = data.injuries.copy().rename(columns={"gsis_id": "player_id"})
        cols = KEYS + [
            column for column in (
                "report_status", "practice_status", "report_primary_injury",
                "practice_primary_injury",
            ) if column in injuries
        ]
        injuries = injuries[cols].drop_duplicates(KEYS, keep="last")
        out = out.merge(injuries, on=KEYS, how="left")
    for column in ("report_status", "practice_status", "report_primary_injury", "practice_primary_injury"):
        if column not in out:
            out[column] = ""
    return out


def _add_team_priors(frame: pd.DataFrame) -> pd.DataFrame:
    source = frame.copy()
    if "schedule_missing" in source:
        outcome_columns = ["attempts", "carries", "fantasy_points", "expected_points"]
        source.loc[source["schedule_missing"].eq(1), outcome_columns] = np.nan
    totals = source.groupby(["season", "week", "team"], as_index=False).agg(
        team_pass_attempts=("attempts", "sum"),
        team_rush_attempts=("carries", "sum"),
        team_points=("fantasy_points", "sum"),
        team_expected_points=("expected_points", "sum"),
        _scheduled=("schedule_missing", lambda values: values.eq(0).any()),
    ).sort_values(["team", "season", "week"])
    total_columns = (
        "team_pass_attempts", "team_rush_attempts", "team_points", "team_expected_points"
    )
    totals.loc[~totals["_scheduled"], list(total_columns)] = np.nan
    for column in total_columns:
        totals[f"pre_{column}_ewm4"] = _prior_ewm(totals, ["team"], column, 4)
    keep = ["season", "week", "team"] + [column for column in totals if column.startswith("pre_")]
    return frame.merge(totals[keep], on=["season", "week", "team"], how="left")


def _add_coach_priors(frame: pd.DataFrame) -> pd.DataFrame:
    """Add lagged play-volume and positional-allocation tendencies by coach.

    Professional projection systems explicitly reconcile a team's expected
    play mix with coach history.  These priors use only completed earlier
    games and stay missing for a coach with no public NFL history.
    """

    source = frame.copy()
    if "schedule_missing" in source:
        source.loc[
            source["schedule_missing"].eq(1), ["attempts", "carries", "targets"]
        ] = np.nan
    team_week = source.groupby(["season", "week", "team", "coach"], dropna=False).agg(
        coach_pass_attempts=("attempts", "sum"),
        coach_rush_attempts=("carries", "sum"),
        coach_targets=("targets", "sum"),
        _scheduled=("schedule_missing", lambda values: values.eq(0).any()),
    ).reset_index()
    position = source.groupby(
        ["season", "week", "team", "coach", "position"], dropna=False
    ).agg(targets=("targets", "sum"), carries=("carries", "sum")).reset_index()
    targets = position.pivot_table(
        index=["season", "week", "team", "coach"], columns="position",
        values="targets", aggfunc="sum", fill_value=0,
    ).add_prefix("targets_").reset_index()
    carries = position.pivot_table(
        index=["season", "week", "team", "coach"], columns="position",
        values="carries", aggfunc="sum", fill_value=0,
    ).add_prefix("carries_").reset_index()
    tendencies = team_week.merge(
        targets, on=["season", "week", "team", "coach"], how="left"
    ).merge(carries, on=["season", "week", "team", "coach"], how="left")
    plays = tendencies["coach_pass_attempts"] + tendencies["coach_rush_attempts"]
    tendencies["coach_pass_rate"] = tendencies["coach_pass_attempts"] / plays.replace(0, np.nan)
    tendencies["coach_pace"] = plays
    for position_name in ("RB", "WR", "TE"):
        values = tendencies.get(f"targets_{position_name}", 0.0)
        tendencies[f"coach_{position_name.lower()}_target_share"] = (
            values / tendencies["coach_targets"].replace(0, np.nan)
        )
    tendencies["coach_qb_rush_share"] = (
        tendencies.get("carries_QB", 0.0) / tendencies["coach_rush_attempts"].replace(0, np.nan)
    )
    tendency_columns = (
        "coach_pass_rate", "coach_pace", "coach_rb_target_share",
        "coach_wr_target_share", "coach_te_target_share", "coach_qb_rush_share",
    )
    tendencies.loc[~tendencies["_scheduled"], list(tendency_columns)] = np.nan
    # Reduce to one observation per coach-week before shifting. This prevents a
    # duplicate team row (or rare same-name collision) from making another row
    # in the same week look like prior history.
    coach_week = tendencies.groupby(
        ["season", "week", "coach"], dropna=False, as_index=False
    )[list(tendency_columns)].mean()
    coach_week = coach_week.sort_values(["coach", "season", "week"])
    for column in tendency_columns:
        coach_week[f"pre_{column}_ewm8"] = _prior_ewm(coach_week, ["coach"], column, 8)
    keep = [
        "season", "week", "coach",
        *[f"pre_{column}_ewm8" for column in tendency_columns],
    ]
    return frame.merge(coach_week[keep], on=["season", "week", "coach"], how="left")


def _add_opponent_priors(frame: pd.DataFrame) -> pd.DataFrame:
    allowed = frame.groupby(
        ["season", "week", "opponent_team", "position"], as_index=False
    )["fantasy_points"].sum().rename(
        columns={"opponent_team": "defense", "fantasy_points": "position_points_allowed"}
    )
    allowed = allowed.sort_values(["defense", "position", "season", "week"])
    allowed["pre_opponent_position_points_ewm8"] = _prior_ewm(
        allowed, ["defense", "position"], "position_points_allowed", 8
    )
    keep = ["season", "week", "defense", "position", "pre_opponent_position_points_ewm8"]
    return frame.merge(
        allowed[keep], left_on=["season", "week", "opponent_team", "position"],
        right_on=["season", "week", "defense", "position"], how="left",
    ).drop(columns=["defense"], errors="ignore")


def _condition_prior(frame: pd.DataFrame, condition: str, shrink_games: float = 8.0) -> None:
    key = frame[condition].fillna("unknown").astype(str)
    work = frame.assign(_condition_key=key, _played_points=frame["fantasy_points"].where(frame["played"]))
    group = work.groupby(["player_id", "_condition_key"], sort=False)["_played_points"]
    prior_mean = group.transform(lambda values: values.shift(1).expanding(min_periods=1).mean())
    prior_n = group.transform(lambda values: values.shift(1).expanding(min_periods=1).count()).fillna(0.0)
    baseline = frame["pre_fantasy_points_ewm8"].fillna(frame["position_prior_points"])
    shrunk = (prior_n * prior_mean.fillna(baseline) + shrink_games * baseline) / (prior_n + shrink_games)
    frame[f"pre_{condition}_uplift"] = (shrunk - baseline).fillna(0.0)
    frame[f"pre_{condition}_games"] = prior_n


def _add_teammate_state(frame: pd.DataFrame) -> None:
    frame["_role_rank"] = frame.groupby(["season", "week", "team", "position"])[
        "pre_opportunities_ewm4"
    ].rank(method="first", ascending=False)
    frame["pre_position_role_rank"] = frame["_role_rank"].fillna(99.0).clip(1, 99)
    unavailable = frame["status_inactive"].eq(1) | frame["injury_out"].eq(1)
    skill = frame["position"].isin(["RB", "WR", "TE"])
    frame["_vac_target"] = frame["pre_target_share_calc_ewm4"].fillna(0.0).where(unavailable & skill, 0.0)
    frame["_vac_carry"] = frame["pre_carry_share_ewm4"].fillna(0.0).where(unavailable & skill, 0.0)
    grouped = frame.groupby(["season", "week", "team"])
    frame["vacated_target_share"] = grouped["_vac_target"].transform("sum").clip(0, 1)
    frame["vacated_carry_share"] = grouped["_vac_carry"].transform("sum").clip(0, 1)
    frame["unavailable_skill_count"] = (unavailable & skill).astype(int).groupby(
        [frame["season"], frame["week"], frame["team"]]
    ).transform("sum")
    for position in ("WR", "RB", "TE"):
        flag = (unavailable & frame["position"].eq(position) & frame["_role_rank"].eq(1)).astype(int)
        frame[f"{position.lower()}1_out"] = flag.groupby(
            [frame["season"], frame["week"], frame["team"]]
        ).transform("max")

    active = ~unavailable
    share_inputs = {
        "attempt": frame["pre_attempts_ewm4"].fillna(0.0).where(active & frame["position"].eq("QB"), 0.0),
        "target": frame["pre_target_share_calc_ewm4"].fillna(0.0).where(
            active & frame["position"].isin(["RB", "WR", "TE"]), 0.0
        ),
        "carry": frame["pre_carry_share_ewm4"].fillna(0.0).where(active, 0.0),
    }
    team_keys = [frame["season"], frame["week"], frame["team"]]
    overflow = pd.Series(0.0, index=frame.index)
    for name, values in share_inputs.items():
        total = values.groupby(team_keys).transform("sum")
        denominator = total.clip(lower=1.0)
        frame[f"pre_reconciled_{name}_share"] = values / denominator
        overflow = np.maximum(overflow, (total - 1.0).clip(lower=0.0))
    frame["pre_usage_share_overflow"] = overflow
    frame["pre_topdown_attempts"] = (
        frame["pre_team_pass_attempts_ewm4"] * frame["pre_reconciled_attempt_share"]
    )
    frame["pre_topdown_targets"] = (
        frame["pre_team_pass_attempts_ewm4"] * frame["pre_reconciled_target_share"]
    )
    frame["pre_topdown_carries"] = (
        frame["pre_team_rush_attempts_ewm4"] * frame["pre_reconciled_carry_share"]
    )
    frame["pre_topdown_opportunities"] = frame[
        ["pre_topdown_attempts", "pre_topdown_targets", "pre_topdown_carries"]
    ].sum(axis=1, min_count=1)
    frame.drop(columns=["_role_rank", "_vac_target", "_vac_carry"], inplace=True)


def _add_normalized(frame: pd.DataFrame) -> None:
    group = frame.groupby(["season", "week", "position"])
    for column in NORMALIZE_BASES:
        values = frame[column]
        mean = group[column].transform("mean")
        sd = group[column].transform("std").replace(0, np.nan)
        frame[f"z_{column[4:]}"] = ((values - mean) / sd).fillna(0.0).clip(-5, 5)


def build_feature_frame(
    data: HistoricalData, scoring: ScoringRules | None = None
) -> pd.DataFrame:
    """Build one roster-first row per player-week with only pregame features."""

    data.validate()
    rules = scoring or ScoringRules()
    rosters = _prepare_rosters(data.rosters)
    stats = _prepare_stats(data.stats, rules)
    frame = rosters.merge(stats, on=KEYS, how="outer", suffixes=("_roster", "_stat"), indicator=True)
    _coalesce(frame, "team", ["team_roster", "team_stat"])
    _coalesce(frame, "position", ["position_roster", "position_stat"])
    _coalesce(frame, "player_name", ["full_name", "player_display_name", "player_name"])
    _coalesce(frame, "status", ["status"], "unknown")
    frame = frame[frame["position"].isin(POSITIONS) & frame["player_id"].notna()].copy()
    frame["roster_source"] = frame["_merge"].astype(str)
    frame["roster_snapshot_carried"] = frame.get("snapshot_carried_forward", False)
    frame.drop(columns=["_merge"], inplace=True)
    _numeric(frame, (*STAT_DEFAULTS, "fumbles_lost", "fantasy_points", "roster_snapshot_carried"))
    frame = _merge_optional(frame, data)
    frame["expected_points_missing"] = frame["expected_points"].isna().astype(float)
    frame["snaps_missing"] = frame["offense_pct"].isna().astype(float)
    _numeric(frame, ("expected_points", *EXPECTED_COMPONENTS.values(), "offense_pct"))

    schedule = _schedule_long(data.schedules)
    schedule["_schedule_present"] = 1.0
    frame = frame.merge(schedule, on=["season", "week", "team"], how="left", suffixes=("", "_schedule"))
    frame["schedule_missing"] = frame["_schedule_present"].isna().astype(float)
    frame["line_missing"] = frame["total_line"].isna().astype(float)
    frame["weather_missing"] = (frame["temp"].isna() | frame["wind"].isna()).astype(float)
    _coalesce(frame, "opponent_team", ["opponent_team", "opponent_team_schedule"])
    _coalesce(frame, "game_id", ["game_id", "game_id_schedule"])
    frame = frame.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    status = frame["status"].fillna("unknown").astype(str).str.upper()
    inactive_codes = {"INA", "RES", "IR", "PUP", "DEV", "EXE", "SUS", "NFI", "RET"}
    frame["status_inactive"] = status.isin(inactive_codes).astype(float)
    report = frame["report_status"].fillna("").astype(str).str.lower()
    practice = frame["practice_status"].fillna("").astype(str).str.lower()
    frame["injury_out"] = report.str.fullmatch("out").fillna(False).astype(float)
    frame["injury_doubtful"] = report.str.contains("doubt", regex=False).astype(float)
    frame["injury_questionable"] = report.str.contains("question", regex=False).astype(float)
    frame["practice_dnp"] = practice.str.contains("did not|dnp", regex=True).astype(float)
    frame["practice_limited"] = practice.str.contains("limited", regex=False).astype(float)

    team_week = frame.groupby(["season", "week", "team"])
    team_pass = team_week["attempts"].transform("sum").replace(0, np.nan)
    team_rush = team_week["carries"].transform("sum").replace(0, np.nan)
    frame["target_share_calc"] = (frame["targets"] / team_pass).fillna(frame["target_share"]).fillna(0.0)
    frame["carry_share"] = (frame["carries"] / team_rush).fillna(0.0)
    frame["touches"] = frame["carries"] + frame["receptions"]
    frame["opportunities"] = frame["attempts"] + frame["carries"] + frame["targets"]
    frame["total_tds"] = frame["passing_tds"] + frame["rushing_tds"] + frame["receiving_tds"]
    frame["yards_per_attempt"] = frame["passing_yards"] / frame["attempts"].replace(0, np.nan)
    frame["yards_per_carry"] = frame["rushing_yards"] / frame["carries"].replace(0, np.nan)
    frame["yards_per_target"] = frame["receiving_yards"] / frame["targets"].replace(0, np.nan)
    frame["catch_rate"] = frame["receptions"] / frame["targets"].replace(0, np.nan)
    frame["td_per_opportunity"] = frame["total_tds"] / frame["opportunities"].replace(0, np.nan)
    frame["interception_rate"] = frame["passing_interceptions"] / frame["attempts"].replace(0, np.nan)
    frame["fumble_rate"] = frame["fumbles_lost"] / (frame["touches"] + frame["attempts"]).replace(0, np.nan)
    frame["points_over_expected"] = frame["fantasy_points"] - frame["expected_points"]
    frame["played"] = frame["opportunities"].gt(0) | frame["fantasy_points"].ne(0)

    group = ["player_id"]
    history_frame = frame.copy()
    history_columns = list(dict.fromkeys([*ROLLING_BASES, "points_over_expected"]))
    history_frame.loc[history_frame["schedule_missing"].eq(1), history_columns] = np.nan
    rolling_features: dict[str, pd.Series] = {}
    for column in ROLLING_BASES:
        for span in (2, 4, 8):
            if span == 2 and column not in {"fantasy_points", "expected_points", *ROLE_TREND_BASES}:
                continue
            rolling_features[f"pre_{column}_ewm{span}"] = _prior_ewm(
                history_frame, group, column, span
            )
    for column in ROLE_TREND_BASES:
        rolling_features[f"pre_{column}_trend_2v8"] = (
            rolling_features[f"pre_{column}_ewm2"] - rolling_features[f"pre_{column}_ewm8"]
        )
    rolling_features["pre_fantasy_points_sd8"] = _prior_sd(
        history_frame, group, "fantasy_points"
    )
    rolling_features["pre_points_over_expected_ewm4"] = _prior_ewm(
        history_frame, group, "points_over_expected", 4
    )
    rolling_features["pre_played_games"] = frame.groupby("player_id")["played"].transform(
        lambda values: values.shift(1).expanding(min_periods=1).sum()
    ).fillna(0.0)
    rolling_features["pre_roster_weeks"] = frame.groupby("player_id").cumcount().astype(float)
    frame = pd.concat([frame, pd.DataFrame(rolling_features, index=frame.index)], axis=1)

    position_prior = frame.groupby(["position", "season", "week"])["fantasy_points"].transform("mean")
    position_week = frame[["position", "season", "week", "fantasy_points"]].copy()
    position_week = position_week.groupby(["position", "season", "week"], as_index=False)["fantasy_points"].mean()
    position_week = position_week.sort_values(["position", "season", "week"])
    position_week["position_prior_points"] = _prior_ewm(position_week, ["position"], "fantasy_points", 8)
    frame = frame.merge(
        position_week[["position", "season", "week", "position_prior_points"]],
        on=["position", "season", "week"], how="left",
    )
    del position_prior

    frame = _add_team_priors(frame)
    frame = _add_coach_priors(frame)
    frame = _add_opponent_priors(frame)
    previous_team = frame.groupby("player_id")["team"].shift(1)
    frame["team_changed"] = (previous_team.notna() & previous_team.ne(frame["team"])).astype(float)
    team_qb = schedule.sort_values(["team", "season", "week"])
    team_qb["prior_qb_id"] = team_qb.groupby("team")["current_qb_id"].shift(1)
    qb_changed = team_qb[["season", "week", "team", "prior_qb_id", "current_qb_id"]].copy()
    qb_changed["qb_changed"] = (
        qb_changed["prior_qb_id"].notna()
        & qb_changed["current_qb_id"].notna()
        & qb_changed["prior_qb_id"].ne(qb_changed["current_qb_id"])
    ).astype(float)
    frame = frame.drop(columns=["qb_changed"], errors="ignore").merge(
        qb_changed[["season", "week", "team", "qb_changed"]],
        on=["season", "week", "team"], how="left",
    )

    gameday = pd.to_datetime(frame["gameday"], errors="coerce")
    birth_values = frame["birth_date"] if "birth_date" in frame else pd.Series(pd.NaT, index=frame.index)
    birth = pd.to_datetime(birth_values, errors="coerce")
    frame["age"] = (gameday - birth).dt.days / 365.2425
    frame["age_missing"] = frame["age"].isna().astype(float)
    _numeric(frame, ("years_exp", "draft_number"))
    frame["draft_number"] = frame["draft_number"].replace(0, np.nan).fillna(300).clip(1, 300)
    _numeric(frame, PREGAME_NUMERIC)

    coach_columns = [column for column in frame if column.startswith("pre_coach_")]
    frame["coach_history_missing"] = frame[coach_columns].isna().all(axis=1).astype(float)

    _add_teammate_state(frame)
    for condition in CONDITION_FEATURES:
        if condition not in frame:
            frame[condition] = "unknown"
        _condition_prior(frame, condition)
    _add_normalized(frame)

    # Prior-only relevance gate.  It avoids flattering the model with hundreds
    # of obvious bench zeroes while still admitting high-draft cold starts.
    qb = frame["position"].eq("QB") & frame["pre_attempts_ewm4"].ge(10)
    rb = frame["position"].eq("RB") & frame["pre_opportunities_ewm4"].ge(4)
    receiver = frame["position"].isin(["WR", "TE"]) & frame["pre_targets_ewm4"].ge(2)
    xfp = frame["pre_expected_points_ewm4"].ge(4)
    cold_start = frame["pre_roster_weeks"].lt(3) & frame["draft_number"].le(100)
    frame["model_eligible"] = (
        (qb | rb | receiver | xfp | cold_start)
        & frame["schedule_missing"].eq(0)
        & frame["status_inactive"].eq(0)
        & frame["injury_out"].eq(0)
    )
    frame["active_participant"] = frame["played"] & frame["status_inactive"].eq(0)
    frame["scoring_rules"] = json.dumps(rules.to_dict(), sort_keys=True)
    for column in model_features():
        if column not in frame:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    assert_feature_contract(frame)
    return frame


def assert_feature_contract(frame: pd.DataFrame) -> None:
    missing = sorted(set(KEYS + ["position", "fantasy_points", "model_eligible"]) - set(frame.columns))
    if missing:
        raise ValueError(f"fantasy frame missing required columns: {missing}")
    if frame.duplicated(KEYS).any():
        duplicates = int(frame.duplicated(KEYS, keep=False).sum())
        raise ValueError(f"fantasy frame has {duplicates} duplicate player-week rows")
    forbidden = {
        "fantasy_points", "expected_points", "attempts", "carries", "targets",
        "passing_yards", "rushing_yards", "receiving_yards", "total_tds",
    }
    leaked = sorted(forbidden & set(model_features()))
    if leaked:
        raise ValueError(f"current-week outcome columns leaked into model features: {leaked}")


def frame_quality_report(frame: pd.DataFrame) -> dict[str, object]:
    features = model_features()
    return {
        "rows": int(len(frame)),
        "seasons": sorted(frame["season"].dropna().astype(int).unique().tolist()),
        "eligible_rows": int(frame["model_eligible"].sum()),
        "active_participant_rows": int(frame["active_participant"].sum()),
        "positions": frame["position"].value_counts().sort_index().astype(int).to_dict(),
        "roster_sources": frame["roster_source"].value_counts().astype(int).to_dict(),
        "feature_count": len(features),
        "feature_missing_rate": {
            column: float(frame[column].isna().mean()) for column in features
        },
    }
