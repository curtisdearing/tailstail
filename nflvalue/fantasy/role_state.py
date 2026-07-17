"""Pregame role-state features, labels, and the scenario-mixture role model.

Role shocks (realized opportunity changes of 5+) are the dominant fantasy
error regime.  This module makes role state a first-class pregame quantity:

1. ``build_role_state_frame`` extends the roster-first feature frame with
   strictly-pregame role evidence: lagged snap share and route participation,
   as-of official depth-chart rank, practice/injury progression, roster-status
   transitions, and exact-seat teammate vacancy states (QB1/RB1/RB2/WR1/WR2/
   WR3/TE1) split into short-notice vs long-term and early vs established
   replacement windows.
2. ``RoleStateModel`` is a season-forward multinomial classifier over
   {inactive, limited_decrease, stable, moderate_increase, major_increase}.
   Labels are defined from REALIZED opportunity change (audit labels); the
   predictors are pregame-only.
3. ``DEFAULT_STATE_MULTIPLIERS`` parameterize the scenario-mixture Monte
   Carlo in ``simulation.py`` (state-conditional share reallocation).

As-of clock: every ``pre_``-prefixed column uses only completed prior weeks.
Current-week columns are limited to the same pregame facts the base frame
already treats as pregame-known (official injury report, roster-of-record
status, official depth chart snapshots strictly before gameday).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

KEYS = ["season", "week", "player_id"]
POSITIONS = ("QB", "RB", "WR", "TE")

ROLE_STATES = (
    "inactive", "limited_decrease", "stable", "moderate_increase", "major_increase",
)
STATE_PROB_COLUMNS = tuple(f"p_state_{state}" for state in ROLE_STATES)

# Realized-opportunity-change label thresholds (audit-aligned: the frozen
# error regimes use +/-5; major_increase isolates the extreme tail).
DECREASE_THRESHOLD = -5.0
INCREASE_THRESHOLD = 5.0
MAJOR_INCREASE_THRESHOLD = 12.0

SEATS = ("qb1", "rb1", "rb2", "wr1", "wr2", "wr3", "te1")
SEAT_DEFINITIONS = {
    "qb1": ("QB", 1), "rb1": ("RB", 1), "rb2": ("RB", 2),
    "wr1": ("WR", 1), "wr2": ("WR", 2), "wr3": ("WR", 3), "te1": ("TE", 1),
}

FEATURE_FRAME_V2_PATH = "reports/fantasy_feature_frame_v2.parquet"
ROUTES_CACHE_DIR = "/tmp/exp_roleshock/routes"

OL_POSITIONS = ("T", "G", "C", "OL", "OT", "OG", "LS")
DB_POSITIONS = ("CB", "S", "SS", "FS", "DB")
FRONT7_POSITIONS = ("DE", "DT", "NT", "LB", "ILB", "OLB", "MLB", "EDGE")

# Roster statuses that mean a multi-week (reserve-list) absence.
RESERVE_STATUSES = {"RES", "PUP", "NFI", "SUS", "EXE", "RSR", "IR"}

ROLE_FEATURES = (
    # lagged participation levels and trends
    "pre_snap_pct_last", "pre_snap_pct_lag1", "pre_snap_pct_ewm3",
    "pre_snap_pct_ewm8", "pre_snap_pct_delta", "pre_snap_rank_in_pos",
    "pre_route_share_last", "pre_route_share_ewm3", "pre_route_share_delta",
    "route_share_missing",
    # official depth chart, as-of
    "depth_rank_asof", "depth_rank_missing", "depth_rank_change",
    "depth_is_starter",
    # practice/injury progression (current report is a pregame fact)
    "practice_ord", "practice_ord_prev", "practice_ramp",
    "injury_designation_ord",
    # roster transitions
    "ir_return_flag", "new_to_roster", "weeks_on_team",
    # absence / return
    "pre_missed_streak", "weeks_since_played", "returning_incumbent",
    # own-seat context
    "my_seat_rank", "same_pos_starter_out", "replacement_window",
    # teammate seat vacancies
    *[f"seat_{seat}_out" for seat in SEATS],
    *[f"seat_{seat}_out_shortnotice" for seat in SEATS],
    *[f"seat_{seat}_out_longterm" for seat in SEATS],
    *[f"seat_{seat}_doubtful" for seat in SEATS],
    *[f"seat_{seat}_ir" for seat in SEATS],
    *[f"seat_{seat}_vacancy_week" for seat in SEATS],
    *[f"seat_{seat}_returns" for seat in SEATS],
    # injury clusters
    "team_ol_out_ct", "opp_db_out_ct", "opp_front7_out_ct", "opp_qb1_out",
)

# Base-frame pregame columns reused by the classifier.
BASE_MODEL_FEATURES = (
    "pre_offense_pct_ewm4", "pre_opportunities_ewm4", "pre_targets_ewm4",
    "pre_carries_ewm4", "pre_attempts_ewm4", "pre_receptions_ewm4",
    "pre_target_share_calc_ewm4", "pre_carry_share_ewm4",
    "pre_opportunities_trend_2v8", "pre_targets_trend_2v8",
    "pre_carries_trend_2v8", "pre_offense_pct_trend_2v8",
    "pre_played_games", "pre_roster_weeks", "pre_expected_points_ewm4",
    "injury_questionable", "injury_doubtful", "practice_dnp",
    "practice_limited", "team_changed", "qb_changed", "week",
    "draft_number", "years_exp", "vacated_target_share",
    "vacated_carry_share", "unavailable_skill_count",
)


def _numeric(series: pd.Series, fill: float | None = None) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values if fill is None else values.fillna(fill)


def _grouped_shifted_ewm(frame: pd.DataFrame, column: str, halflife: float) -> pd.Series:
    return frame.groupby("player_id", sort=False)[column].transform(
        lambda values: values.shift(1).ewm(
            halflife=halflife, min_periods=1, ignore_na=True
        ).mean()
    )


def load_routes_table(routes_dir: str | Path = ROUTES_CACHE_DIR) -> pd.DataFrame | None:
    """Load the per-season charted pass-snap participation cache, if present."""

    routes_dir = Path(routes_dir)
    files = sorted(routes_dir.glob("routes_*.parquet"))
    if not files:
        return None
    routes = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    routes = routes.drop_duplicates(["season", "week", "player_id"], keep="last")
    return routes[["season", "week", "player_id", "route_share"]]


def _depth_chart_asof(historical_dir: Path, game_days: pd.DataFrame) -> pd.DataFrame:
    """One official-depth-chart rank per (season, week, team, player), as-of.

    2019-2024 files are the weekly club-submitted charts (published early in
    game week, a standard pregame artifact).  2025+ files are timestamped
    snapshots; only snapshots strictly before the team's gameday are used.
    """

    tables: list[pd.DataFrame] = []
    for path in sorted(historical_dir.glob("depth_charts_*.parquet")):
        season = int(path.stem.split("_")[-1])
        chart = pd.read_parquet(path)
        if "depth_team" in chart.columns:  # weekly format (<=2024)
            chart = chart[chart.get("formation").fillna("").eq("Offense")]
            chart = chart[chart["position"].isin(POSITIONS)].copy()
            chart["season"] = _numeric(chart["season"]).astype("Int64")
            chart["week"] = _numeric(chart["week"]).astype("Int64")
            chart["depth_rank"] = _numeric(chart["depth_team"])
            chart = chart.rename(columns={"club_code": "team", "gsis_id": "player_id"})
            chart = chart.dropna(subset=["player_id", "week", "depth_rank"])
            chart = chart.sort_values("depth_rank").drop_duplicates(
                ["season", "week", "team", "player_id"], keep="first"
            )
            tables.append(chart[["season", "week", "team", "player_id", "depth_rank"]])
            continue
        # timestamped format (2025+): offensive skill rows, ordered slots
        chart = chart[chart["pos_abb"].isin(POSITIONS)].copy()
        if chart.empty:
            continue
        chart["dt"] = pd.to_datetime(chart["dt"], errors="coerce", utc=True)
        chart = chart.dropna(subset=["dt", "gsis_id"])
        chart["season"] = season
        days = game_days[game_days["season"].eq(season)].copy()
        if days.empty:
            continue
        days["gameday_ts"] = pd.to_datetime(days["gameday"], errors="coerce", utc=True)
        days = days.dropna(subset=["gameday_ts"])
        snapshots = chart[["team", "dt"]].drop_duplicates().sort_values("dt")
        days = days.sort_values("gameday_ts")
        picked = pd.merge_asof(
            days, snapshots.rename(columns={"dt": "chart_dt"}),
            left_on="gameday_ts", right_on="chart_dt", by="team",
            direction="backward", allow_exact_matches=False,
        ).dropna(subset=["chart_dt"])
        chart = chart.merge(
            picked[["season", "week", "team", "chart_dt"]],
            left_on=["season", "team", "dt"], right_on=["season", "team", "chart_dt"],
            how="inner",
        )
        # Cross-slot rank: starters of each slot first (WR has 3 slots).
        chart = chart.sort_values(["pos_rank", "pos_slot"])
        chart["depth_rank"] = chart.groupby(
            ["season", "week", "team", "pos_abb"]
        ).cumcount() + 1.0
        chart = chart.rename(columns={"gsis_id": "player_id"})
        chart = chart.drop_duplicates(["season", "week", "team", "player_id"], keep="first")
        tables.append(chart[["season", "week", "team", "player_id", "depth_rank"]])
    if not tables:
        return pd.DataFrame(columns=["season", "week", "team", "player_id", "depth_rank"])
    out = pd.concat(tables, ignore_index=True)
    out["season"] = _numeric(out["season"]).astype(int)
    out["week"] = _numeric(out["week"]).astype(int)
    return out.drop_duplicates(["season", "week", "team", "player_id"], keep="first")


def _injury_cluster_counts(historical_dir: Path) -> pd.DataFrame:
    """Per team-week counts of listed-Out regulars on OL, DB, and front seven.

    Starter weighting uses strictly-lagged snap shares (EWM of prior weeks),
    so a current-week snap count never influences its own week's flag.
    """

    injuries = pd.read_parquet(historical_dir / "fantasy" / "injuries.parquet")
    snaps = pd.read_parquet(historical_dir / "fantasy" / "snap_counts.parquet")
    rosters = pd.read_parquet(
        historical_dir / "fantasy" / "weekly_rosters.parquet",
        columns=["season", "week", "gsis_id", "pfr_id"],
    )
    crosswalk = rosters.dropna(subset=["gsis_id", "pfr_id"]).drop_duplicates(
        ["season", "gsis_id"], keep="last"
    )[["season", "gsis_id", "pfr_id"]]
    if "game_type" in snaps:
        snaps = snaps[snaps["game_type"].fillna("REG").eq("REG")]
    snaps = snaps.rename(columns={"pfr_player_id": "pfr_id"})
    snaps["season"] = _numeric(snaps["season"]).astype(int)
    snaps["week"] = _numeric(snaps["week"]).astype(int)
    snaps = snaps.merge(crosswalk, on=["season", "pfr_id"], how="inner")
    snaps = snaps.sort_values(["gsis_id", "season", "week"])
    for side in ("offense_pct", "defense_pct"):
        snaps[side] = _numeric(snaps[side])
        snaps[f"pre_{side}_ewm"] = snaps.groupby("gsis_id", sort=False)[side].transform(
            lambda values: values.shift(1).ewm(halflife=3, min_periods=1, ignore_na=True).mean()
        )
    regular = snaps[["season", "week", "gsis_id", "pre_offense_pct_ewm", "pre_defense_pct_ewm"]]

    injuries = injuries.copy()
    injuries["season"] = _numeric(injuries["season"]).astype("Int64")
    injuries["week"] = _numeric(injuries["week"]).astype("Int64")
    injuries = injuries.dropna(subset=["season", "week", "gsis_id"])
    injuries["out"] = (
        injuries["report_status"].fillna("").astype(str).str.lower().eq("out")
    )
    injuries = injuries[injuries["out"]]
    injuries = injuries.merge(
        regular, left_on=["season", "week", "gsis_id"],
        right_on=["season", "week", "gsis_id"], how="left",
    )
    position = injuries["position"].fillna("").astype(str).str.upper()
    off_regular = _numeric(injuries["pre_offense_pct_ewm"], 0.0).ge(0.40)
    def_regular = _numeric(injuries["pre_defense_pct_ewm"], 0.0).ge(0.40)
    injuries["is_ol_out"] = position.isin(OL_POSITIONS) & off_regular
    injuries["is_db_out"] = position.isin(DB_POSITIONS) & def_regular
    injuries["is_front7_out"] = position.isin(FRONT7_POSITIONS) & def_regular
    counts = injuries.groupby(["season", "week", "team"], as_index=False).agg(
        team_ol_out_ct=("is_ol_out", "sum"),
        team_db_out_ct=("is_db_out", "sum"),
        team_front7_out_ct=("is_front7_out", "sum"),
    )
    for column in ("team_ol_out_ct", "team_db_out_ct", "team_front7_out_ct"):
        counts[column] = counts[column].astype(float)
    counts["season"] = counts["season"].astype(int)
    counts["week"] = counts["week"].astype(int)
    return counts


def build_role_state_frame(
    base_frame: pd.DataFrame,
    historical_dir: str | Path = "historical",
    routes_dir: str | Path = ROUTES_CACHE_DIR,
) -> pd.DataFrame:
    """Extend the roster-first feature frame with pregame role-state columns.

    The input frame is the output of ``features.build_feature_frame``; its
    rows and existing columns are preserved unchanged.
    """

    historical_dir = Path(historical_dir)
    frame = base_frame.copy()
    frame = frame.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    player_group = frame.groupby("player_id", sort=False)

    # --- lagged snap participation (offense_pct is a post-game fact; only
    # shifted values feed features) -------------------------------------
    snap = _numeric(frame["offense_pct"])
    frame["_snap_obs"] = snap.where(frame["played"].astype(bool))
    shifted_snap = player_group["_snap_obs"].shift(1)
    frame["pre_snap_pct_lag1"] = shifted_snap
    frame["pre_snap_pct_last"] = shifted_snap.groupby(frame["player_id"]).ffill()
    frame["pre_snap_pct_ewm3"] = _grouped_shifted_ewm(frame, "_snap_obs", halflife=3.0)
    frame["pre_snap_pct_ewm8"] = _grouped_shifted_ewm(frame, "_snap_obs", halflife=8.0)
    frame["pre_snap_pct_delta"] = frame["pre_snap_pct_last"] - frame["pre_snap_pct_ewm8"]
    frame["pre_snap_rank_in_pos"] = frame.groupby(
        ["season", "week", "team", "position"]
    )["pre_snap_pct_ewm3"].rank(method="first", ascending=False)

    # --- charted route participation (strictly lagged) ------------------
    routes = load_routes_table(routes_dir)
    if routes is not None:
        routes = routes.copy()
        routes["season"] = _numeric(routes["season"]).astype(int)
        routes["week"] = _numeric(routes["week"]).astype(int)
        frame = frame.merge(routes, on=KEYS, how="left")
    else:
        frame["route_share"] = np.nan
    frame["route_share_missing"] = frame["route_share"].isna().astype(float)
    frame["_route_obs"] = _numeric(frame["route_share"]).where(frame["played"].astype(bool))
    player_group = frame.groupby("player_id", sort=False)
    shifted_route = player_group["_route_obs"].shift(1)
    frame["pre_route_share_lag1"] = shifted_route
    frame["pre_route_share_last"] = shifted_route.groupby(frame["player_id"]).ffill()
    frame["pre_route_share_ewm3"] = _grouped_shifted_ewm(frame, "_route_obs", halflife=3.0)
    frame["pre_route_share_ewm8"] = _grouped_shifted_ewm(frame, "_route_obs", halflife=8.0)
    frame["pre_route_share_delta"] = frame["pre_route_share_last"] - frame["pre_route_share_ewm8"]

    # --- official depth chart, as-of ------------------------------------
    game_days = frame[["season", "week", "team", "gameday"]].drop_duplicates()
    depth = _depth_chart_asof(historical_dir, game_days)
    if not depth.empty:
        frame = frame.merge(
            depth.rename(columns={"depth_rank": "depth_rank_asof"}),
            on=["season", "week", "team", "player_id"], how="left",
        )
    else:
        frame["depth_rank_asof"] = np.nan
    frame["depth_rank_missing"] = frame["depth_rank_asof"].isna().astype(float)
    frame["depth_is_starter"] = frame["depth_rank_asof"].eq(1).astype(float)
    frame = frame.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    player_group = frame.groupby("player_id", sort=False)
    prior_depth = player_group["depth_rank_asof"].shift(1)
    frame["depth_rank_change"] = (prior_depth - frame["depth_rank_asof"]).fillna(0.0)
    frame["depth_rank_asof"] = frame["depth_rank_asof"].fillna(99.0)

    # --- practice/injury progression ------------------------------------
    practice = frame["practice_status"].fillna("").astype(str).str.lower()
    practice_ord = pd.Series(3.0, index=frame.index)  # 3 = no practice report
    practice_ord[practice.str.contains("full", regex=False)] = 2.0
    practice_ord[practice.str.contains("limited", regex=False)] = 1.0
    practice_ord[practice.str.contains("did not|dnp", regex=True)] = 0.0
    frame["practice_ord"] = practice_ord
    frame["practice_ord_prev"] = player_group["practice_ord"].shift(1).fillna(3.0)
    on_report = frame["practice_ord"].lt(3) & frame["practice_ord_prev"].lt(3)
    frame["practice_ramp"] = (
        (frame["practice_ord"] - frame["practice_ord_prev"]).where(on_report, 0.0)
    )
    report = frame["report_status"].fillna("").astype(str).str.lower()
    designation = pd.Series(0.0, index=frame.index)
    designation[report.str.contains("question", regex=False)] = 1.0
    designation[report.str.contains("doubt", regex=False)] = 2.0
    designation[report.str.fullmatch("out").fillna(False)] = 3.0
    designation[frame["status_inactive"].astype(float).eq(1.0)] = 3.0
    frame["injury_designation_ord"] = designation

    # --- roster-status transitions ---------------------------------------
    status = frame["status"].fillna("unknown").astype(str).str.upper()
    frame["_on_reserve"] = status.isin(RESERVE_STATUSES).astype(float)
    prev_reserve = player_group["_on_reserve"].shift(1).fillna(0.0)
    active_now = frame["status_inactive"].astype(float).eq(0.0) & frame["injury_out"].astype(float).eq(0.0)
    frame["ir_return_flag"] = (prev_reserve.eq(1.0) & active_now).astype(float)
    frame["new_to_roster"] = (
        frame["team_changed"].astype(float).eq(1.0) | player_group.cumcount().eq(0)
    ).astype(float)
    stint = frame.groupby("player_id", sort=False)["new_to_roster"].cumsum()
    frame["weeks_on_team"] = frame.groupby(
        [frame["player_id"], stint], sort=False
    ).cumcount().astype(float)

    # --- absence and return ------------------------------------------------
    # Consecutive prior SCHEDULED weeks without playing.  Bye weeks neither
    # increment nor reset the streak (they are not absences).
    played = frame["played"].astype(bool)
    scheduled = _numeric(frame["schedule_missing"], 1.0).eq(0.0)
    run_inclusive = pd.Series(np.nan, index=frame.index)
    sub = frame.loc[scheduled, ["player_id"]].copy()
    sub_missed = (~played[scheduled]).astype(int)
    sub_reset = sub_missed.eq(0).groupby(sub["player_id"]).cumsum()
    run_inclusive.loc[scheduled] = sub_missed.groupby(
        [sub["player_id"], sub_reset]
    ).cumsum().astype(float)
    run_inclusive = run_inclusive.groupby(frame["player_id"]).ffill()
    frame["pre_missed_streak"] = (
        run_inclusive.groupby(frame["player_id"]).shift(1).fillna(0.0).astype(float)
    )
    prior_played_any = player_group["played"].transform(
        lambda values: values.shift(1).expanding(min_periods=1).sum()
    ).fillna(0.0)
    frame["weeks_since_played"] = np.where(
        prior_played_any.gt(0), frame["pre_missed_streak"] + 1.0, 99.0
    ).clip(0, 99)
    frame["returning_incumbent"] = (
        frame["pre_missed_streak"].ge(2) & prior_played_any.gt(0) & active_now
    ).astype(float)

    # --- exact-seat vacancy states ----------------------------------------
    # Seat holder = prior-usage incumbent.  ignore_na EWMs survive missed
    # weeks, so an injured RB1 keeps holding the RB1 seat while absent.
    seat_score = frame["pre_snap_pct_ewm3"].fillna(0.0) * 4.0
    seat_score = seat_score + frame["pre_opportunities_ewm4"].fillna(0.0) / 10.0
    seat_score = seat_score + frame["pre_target_share_calc_ewm4"].fillna(0.0)
    seat_score = seat_score + frame["pre_carry_share_ewm4"].fillna(0.0)
    seat_score = seat_score.where(prior_played_any.gt(0), -1.0)
    frame["_seat_score"] = seat_score
    frame["my_seat_rank"] = frame.groupby(["season", "week", "team", "position"])[
        "_seat_score"
    ].rank(method="first", ascending=False).clip(1, 99)

    out_now = (
        frame["status_inactive"].astype(float).eq(1.0)
        | frame["injury_out"].astype(float).eq(1.0)
    )
    doubtful_now = frame["injury_doubtful"].astype(float).eq(1.0)
    on_reserve = frame["_on_reserve"].astype(float).eq(1.0)
    vacancy_week = np.where(out_now, frame["pre_missed_streak"] + 1.0, 0.0)

    occupant = pd.DataFrame({
        "season": frame["season"], "week": frame["week"], "team": frame["team"],
        "position": frame["position"], "rank": frame["my_seat_rank"],
        "out": out_now.astype(float), "doubtful": doubtful_now.astype(float),
        "ir": on_reserve.astype(float), "vacancy_week": vacancy_week,
        "returns": frame["returning_incumbent"].astype(float),
        "had_history": prior_played_any.gt(0).astype(float),
    })
    for seat, (position, rank) in SEAT_DEFINITIONS.items():
        holder = occupant[
            occupant["position"].eq(position)
            & occupant["rank"].eq(float(rank))
            & occupant["had_history"].eq(1.0)
        ]
        holder = holder.drop_duplicates(["season", "week", "team"], keep="first")
        seat_state = holder[["season", "week", "team"]].copy()
        seat_state[f"seat_{seat}_out"] = holder["out"].to_numpy()
        seat_state[f"seat_{seat}_doubtful"] = holder["doubtful"].to_numpy()
        seat_state[f"seat_{seat}_ir"] = (holder["out"].to_numpy() * holder["ir"].to_numpy())
        seat_state[f"seat_{seat}_vacancy_week"] = holder["vacancy_week"].to_numpy()
        vw = holder["vacancy_week"].to_numpy()
        seat_state[f"seat_{seat}_out_shortnotice"] = ((vw == 1.0)).astype(float)
        seat_state[f"seat_{seat}_out_longterm"] = ((vw >= 3.0)).astype(float)
        seat_state[f"seat_{seat}_returns"] = holder["returns"].to_numpy()
        frame = frame.merge(seat_state, on=["season", "week", "team"], how="left")
        for suffix in ("out", "doubtful", "ir", "vacancy_week", "out_shortnotice", "out_longterm", "returns"):
            column = f"seat_{seat}_{suffix}"
            frame[column] = frame[column].fillna(0.0)

    # own-position starter vacancy from the beneficiary's point of view
    starter_seat = {"QB": "qb1", "RB": "rb1", "WR": "wr1", "TE": "te1"}
    same_out = pd.Series(0.0, index=frame.index)
    same_vw = pd.Series(0.0, index=frame.index)
    for position, seat in starter_seat.items():
        mask = frame["position"].eq(position)
        same_out[mask] = frame.loc[mask, f"seat_{seat}_out"]
        same_vw[mask] = frame.loc[mask, f"seat_{seat}_vacancy_week"]
    # a starter who is himself out is not his own beneficiary
    is_holder = frame["my_seat_rank"].eq(1.0)
    frame["same_pos_starter_out"] = same_out.where(~is_holder, 0.0)
    replacement = np.zeros(len(frame))
    open_now = frame["same_pos_starter_out"].eq(1.0)
    replacement[open_now & same_vw.le(2.0)] = 1.0   # early replacement window
    replacement[open_now & same_vw.ge(3.0)] = 2.0   # established replacement
    frame["replacement_window"] = replacement

    # --- injury clusters (own OL, opponent DB / front seven) --------------
    clusters = _injury_cluster_counts(historical_dir)
    frame = frame.merge(clusters, on=["season", "week", "team"], how="left")
    frame = frame.merge(
        clusters.rename(columns={
            "team": "opponent_team", "team_ol_out_ct": "_opp_ol_out_ct",
            "team_db_out_ct": "opp_db_out_ct", "team_front7_out_ct": "opp_front7_out_ct",
        }),
        on=["season", "week", "opponent_team"], how="left",
    )
    opp_qb = frame[["season", "week", "team", "seat_qb1_out"]].drop_duplicates(
        ["season", "week", "team"]
    ).rename(columns={"team": "opponent_team", "seat_qb1_out": "opp_qb1_out"})
    frame = frame.merge(opp_qb, on=["season", "week", "opponent_team"], how="left")
    for column in ("team_ol_out_ct", "team_db_out_ct", "team_front7_out_ct",
                   "_opp_ol_out_ct", "opp_db_out_ct", "opp_front7_out_ct", "opp_qb1_out"):
        frame[column] = _numeric(frame.get(column, np.nan), 0.0)

    frame.drop(
        columns=["_snap_obs", "_route_obs", "_on_reserve", "_seat_score", "_opp_ol_out_ct"],
        inplace=True, errors="ignore",
    )
    for column in ROLE_FEATURES:
        if column not in frame:
            frame[column] = 0.0
    frame["role_state_label"] = realized_role_state_labels(frame)
    assert_role_state_contract(frame)
    return frame.reset_index(drop=True)


def realized_role_state_labels(frame: pd.DataFrame) -> pd.Series:
    """Audit-aligned realized role states.  Evaluation/training labels ONLY."""

    delta = _numeric(frame["opportunities"]) - _numeric(frame["pre_opportunities_ewm4"]).fillna(0.0)
    labels = pd.Series("stable", index=frame.index)
    labels[delta.le(DECREASE_THRESHOLD)] = "limited_decrease"
    labels[delta.ge(INCREASE_THRESHOLD)] = "moderate_increase"
    labels[delta.ge(MAJOR_INCREASE_THRESHOLD)] = "major_increase"
    labels[~frame["played"].astype(bool)] = "inactive"
    return labels


def assert_role_state_contract(frame: pd.DataFrame) -> None:
    missing = sorted(set(ROLE_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"role-state frame missing columns: {missing}")
    if frame.duplicated(KEYS).any():
        raise ValueError("role-state frame has duplicate player-week rows")
    for seat in SEATS:
        short = frame[f"seat_{seat}_out_shortnotice"]
        long_term = frame[f"seat_{seat}_out_longterm"]
        if (short.astype(float) * long_term.astype(float)).any():
            raise ValueError(f"seat {seat}: short-notice and long-term cohorts overlap")


def load_feature_frame_v2(path: str | Path = FEATURE_FRAME_V2_PATH) -> pd.DataFrame:
    """Load the persisted role-state feature frame."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; build it with scripts/build_feature_frame_v2 or "
            "role_state.build_role_state_frame"
        )
    frame = pd.read_parquet(path)
    assert_role_state_contract(frame)
    return frame


# ---------------------------------------------------------------------------
# Role-state classifier
# ---------------------------------------------------------------------------

def role_model_features() -> list[str]:
    """Pregame-only predictor list for the role-state classifier."""

    return list(dict.fromkeys([*ROLE_FEATURES, *BASE_MODEL_FEATURES]))


def _position_dummies(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for position in POSITIONS:
        out[f"pos_{position}"] = frame["position"].eq(position).astype(float)
    return out


class RoleStateModel:
    """HistGB multinomial P(role state) with per-class isotonic calibration.

    Trained strictly season-forward: ``fit`` uses seasons ``<= max_season``;
    the final training season is held out as the inner calibration fold, then
    per-class isotonic maps are fit on it and probabilities renormalized.
    """

    def __init__(self, random_seed: int = 6102026, max_iter: int = 120) -> None:
        self.random_seed = random_seed
        self.max_iter = max_iter
        self.classifier = None
        self.calibrators: dict[str, object] = {}
        self.features: list[str] = []
        self.classes_: list[str] = []
        self.metadata: dict[str, object] = {}

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        columns = [
            pd.to_numeric(frame.get(name, np.nan), errors="coerce")
            for name in self.features
        ]
        base = pd.concat(columns, axis=1)
        base.columns = self.features
        return pd.concat([base, _position_dummies(frame)], axis=1).to_numpy(dtype=float)

    def fit(self, frame: pd.DataFrame, max_season: int) -> "RoleStateModel":
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.isotonic import IsotonicRegression

        self.features = role_model_features()
        train = frame[
            frame["season"].astype(int).le(max_season)
            & frame["model_eligible"].fillna(False)
        ].copy()
        if train.empty:
            raise ValueError("no eligible training rows for role-state model")
        labels = train["role_state_label"].astype(str)
        seasons = train["season"].astype(int)
        inner_season = int(seasons.max())
        core = seasons.lt(inner_season)
        calib = seasons.eq(inner_season)
        if core.sum() < 500 or calib.sum() < 200:
            core = pd.Series(True, index=train.index)
            calib = pd.Series(False, index=train.index)
        self.classifier = HistGradientBoostingClassifier(
            loss="log_loss", max_iter=self.max_iter, learning_rate=0.08,
            max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
            random_state=self.random_seed, early_stopping=False,
        )
        self.classifier.fit(self._matrix(train[core]), labels[core])
        self.classes_ = [str(value) for value in self.classifier.classes_]
        self.calibrators = {}
        if calib.any():
            raw = self.classifier.predict_proba(self._matrix(train[calib]))
            for index, state in enumerate(self.classes_):
                target = labels[calib].eq(state).astype(float).to_numpy()
                iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1.0)
                iso.fit(raw[:, index], target)
                self.calibrators[state] = iso
        self.metadata = {
            "max_train_season": int(max_season),
            "inner_calibration_season": inner_season if calib.any() else None,
            "train_rows": int(core.sum()),
            "calibration_rows": int(calib.sum()),
            "class_base_rates": labels.value_counts(normalize=True).to_dict(),
        }
        return self

    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.classifier is None:
            raise RuntimeError("fit the role-state model first")
        raw = self.classifier.predict_proba(self._matrix(frame))
        if self.calibrators:
            calibrated = np.column_stack([
                self.calibrators[state].predict(raw[:, index])
                for index, state in enumerate(self.classes_)
            ])
        else:
            calibrated = raw
        calibrated = np.clip(calibrated, 1e-6, 1.0)
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        columns = {}
        for state in ROLE_STATES:
            if state in self.classes_:
                columns[f"p_state_{state}"] = calibrated[:, self.classes_.index(state)]
            else:
                columns[f"p_state_{state}"] = np.zeros(len(frame))
        out = pd.DataFrame(columns, index=frame.index)
        return out


def season_forward_role_probabilities(
    frame: pd.DataFrame,
    test_seasons: Iterable[int],
    *,
    random_seed: int = 6102026,
) -> tuple[pd.DataFrame, dict[int, RoleStateModel]]:
    """Train <= (season-1) and predict each test season; never look forward."""

    outputs: list[pd.DataFrame] = []
    models: dict[int, RoleStateModel] = {}
    for season in sorted(int(s) for s in test_seasons):
        model = RoleStateModel(random_seed=random_seed).fit(frame, max_season=season - 1)
        test = frame[
            frame["season"].astype(int).eq(season) & frame["model_eligible"].fillna(False)
        ]
        if test.empty:
            continue
        probs = model.predict_proba(test)
        probs = pd.concat([test[KEYS].reset_index(drop=True),
                           probs.reset_index(drop=True)], axis=1)
        outputs.append(probs)
        models[season] = model
    if not outputs:
        raise ValueError("no test rows produced role probabilities")
    return pd.concat(outputs, ignore_index=True), models


def attach_role_probabilities(frame: pd.DataFrame, probabilities: pd.DataFrame) -> pd.DataFrame:
    """Merge p_state_* columns onto a frame by player-week key."""

    merged = frame.merge(
        probabilities[[*KEYS, *STATE_PROB_COLUMNS]], on=KEYS, how="left",
    )
    return merged


# ---------------------------------------------------------------------------
# Scenario-mixture parameters
# ---------------------------------------------------------------------------

def estimate_state_multipliers(
    frame: pd.DataFrame, max_season: int
) -> dict[str, dict[str, float]]:
    """Median realized share/opportunity ratios per position x active state.

    Estimated on training seasons only.  These parameterize the
    state-conditional reallocation inside the scenario-mixture Monte Carlo.
    """

    train = frame[
        frame["season"].astype(int).le(max_season)
        & frame["model_eligible"].fillna(False)
        & frame["played"].astype(bool)
    ].copy()
    labels = train["role_state_label"].astype(str)
    out: dict[str, dict[str, float]] = {}
    opportunity_ratio = _numeric(train["opportunities"]) / _numeric(
        train["pre_opportunities_ewm4"]
    ).clip(lower=1.0)
    target_ratio = _numeric(train["target_share_calc"]) / _numeric(
        train["pre_target_share_calc_ewm4"]
    ).clip(lower=0.02)
    carry_ratio = _numeric(train["carry_share"]) / _numeric(
        train["pre_carry_share_ewm4"]
    ).clip(lower=0.02)
    for position in POSITIONS:
        rows = train["position"].eq(position)
        for state in ("limited_decrease", "stable", "moderate_increase", "major_increase"):
            mask = rows & labels.eq(state)
            key = f"{position}:{state}"
            if mask.sum() < 25:
                out[key] = {"opportunity": 1.0, "target": 1.0, "carry": 1.0, "n": int(mask.sum())}
                continue
            out[key] = {
                "opportunity": float(np.clip(opportunity_ratio[mask].median(), 0.2, 4.0)),
                "target": float(np.clip(target_ratio[mask].median(), 0.2, 4.0)),
                "carry": float(np.clip(carry_ratio[mask].median(), 0.2, 4.0)),
                "n": int(mask.sum()),
            }
    return out


def relative_state_multipliers(
    raw: Mapping[str, Mapping[str, float]]
) -> dict[str, dict[str, float]]:
    """Normalize raw state medians so ``stable`` is exactly 1.0 per channel.

    Raw medians divide by clipped denominators, which distorts channels a
    position rarely uses (e.g. WR carries).  Expressing every active state
    relative to the same position's ``stable`` median cancels that clip.
    Sparse ``major_increase`` cells (n < 25) extrapolate from the position's
    ``moderate_increase`` value using the cross-position median
    major/moderate ratio.
    """

    states = ("limited_decrease", "stable", "moderate_increase", "major_increase")
    channels = ("opportunity", "target", "carry")
    ratios: dict[str, list[float]] = {channel: [] for channel in channels}
    for position in POSITIONS:
        major = raw[f"{position}:major_increase"]
        moderate = raw[f"{position}:moderate_increase"]
        if major["n"] >= 25 and moderate["n"] >= 25:
            for channel in channels:
                ratios[channel].append(major[channel] / max(moderate[channel], 1e-6))
    major_ratio = {
        channel: float(np.median(values)) if values else 1.4
        for channel, values in ratios.items()
    }
    output: dict[str, dict[str, float]] = {}
    for position in POSITIONS:
        stable = raw[f"{position}:stable"]
        for state in states:
            cell = raw[f"{position}:{state}"]
            entry: dict[str, float] = {}
            for channel in channels:
                if state == "stable":
                    entry[channel] = 1.0
                elif cell["n"] >= 25:
                    entry[channel] = round(float(np.clip(
                        cell[channel] / max(stable[channel], 1e-6), 0.2, 4.0
                    )), 3)
                elif state == "major_increase":
                    moderate_value = output[f"{position}:moderate_increase"][channel]
                    entry[channel] = round(float(np.clip(
                        moderate_value * major_ratio[channel], 0.2, 4.0
                    )), 3)
                else:
                    entry[channel] = 1.0
            entry["n"] = float(cell["n"])
            output[f"{position}:{state}"] = entry
    return output


# Frozen output of relative_state_multipliers(estimate_state_multipliers(
# feature_frame_v2, max_season=2022)), plus "opportunity_var" = robust
# ((q3-q1)/1.349)^2 of the stable-normalized realized opportunity ratio within
# each state (train rows; sparse cells inherit the cross-position median of
# the state; /tmp/exp_roleshock/make_within_var.py) -- 2019-2022 only, so
# the simulator carries no runtime data dependency and no test-season
# information.  Regenerate via /tmp/exp_roleshock/make_multipliers.py (or the
# two calls above) whenever the frame definition changes.
DEFAULT_STATE_MULTIPLIERS: Mapping[str, Mapping[str, float]] = {
    "QB:limited_decrease": {"opportunity": 0.734, "target": 1.0, "carry": 0.722, "n": 472, "opportunity_var": 0.02},
    "QB:stable": {"opportunity": 1.0, "target": 1.0, "carry": 1.0, "n": 743, "opportunity_var": 0.011},
    "QB:moderate_increase": {"opportunity": 1.228, "target": 1.0, "carry": 1.241, "n": 409, "opportunity_var": 0.01},
    "QB:major_increase": {"opportunity": 1.705, "target": 1.0, "carry": 1.67, "n": 413, "opportunity_var": 0.408},
    "RB:limited_decrease": {"opportunity": 0.494, "target": 0.508, "carry": 0.568, "n": 614, "opportunity_var": 0.058},
    "RB:stable": {"opportunity": 1.0, "target": 1.0, "carry": 1.0, "n": 2331, "opportunity_var": 0.09},
    "RB:moderate_increase": {"opportunity": 1.75, "target": 1.586, "carry": 1.525, "n": 693, "opportunity_var": 0.184},
    "RB:major_increase": {"opportunity": 2.692, "target": 2.297, "carry": 2.207, "n": 225, "opportunity_var": 1.0},
    "WR:limited_decrease": {"opportunity": 0.329, "target": 0.409, "carry": 1.0, "n": 229, "opportunity_var": 0.027},
    "WR:stable": {"opportunity": 1.0, "target": 1.0, "carry": 1.0, "n": 5220, "opportunity_var": 0.244},
    "WR:moderate_increase": {"opportunity": 2.396, "target": 2.004, "carry": 1.0, "n": 500, "opportunity_var": 0.559},
    "WR:major_increase": {"opportunity": 3.506, "target": 2.453, "carry": 1.396, "n": 16, "opportunity_var": 0.704},
    "TE:limited_decrease": {"opportunity": 0.308, "target": 0.322, "carry": 1.0, "n": 47, "opportunity_var": 0.023},
    "TE:stable": {"opportunity": 1.0, "target": 1.0, "carry": 1.0, "n": 2182, "opportunity_var": 0.343},
    "TE:moderate_increase": {"opportunity": 2.464, "target": 2.034, "carry": 1.0, "n": 111, "opportunity_var": 0.627},
    "TE:major_increase": {"opportunity": 3.606, "target": 2.49, "carry": 1.396, "n": 2, "opportunity_var": 0.704},
}

ACTIVE_STATES = ("limited_decrease", "stable", "moderate_increase", "major_increase")


def state_multiplier_table(
    multipliers: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-position (target, carry, opportunity) arrays ordered by ACTIVE_STATES."""

    table = multipliers or DEFAULT_STATE_MULTIPLIERS
    output: dict[str, dict[str, np.ndarray]] = {}
    for position in POSITIONS:
        output[position] = {
            channel: np.asarray([
                float(table[f"{position}:{state}"].get(channel, 0.0))
                for state in ACTIVE_STATES
            ])
            for channel in ("target", "carry", "opportunity", "opportunity_var")
        }
    return output
