import pandas as pd

from nflvalue.fantasy.data import HistoricalData, materialize_projection_week
from nflvalue.fantasy.features import _prior_ewm, build_feature_frame, model_features


def _bundle(receiving_yards_week_two=60):
    stats = pd.DataFrame([
        {"season": 2022, "week": week, "season_type": "REG", "game_id": f"2022_0{week}_AAA_BBB",
         "player_id": player, "player_name": name, "position": position, "team": team,
         "opponent_team": "BBB" if team == "AAA" else "AAA", "attempts": attempts,
         "completions": attempts * 0.65, "passing_yards": attempts * 7,
         "passing_tds": 1, "passing_interceptions": 0, "carries": carries,
         "rushing_yards": carries * 4, "rushing_tds": 0, "targets": targets,
         "receptions": targets * 0.7, "receiving_yards": yards, "receiving_tds": 0}
        for week in (1, 2, 3)
        for player, name, position, team, attempts, carries, targets, yards in (
            ("00-0000001", "Alpha QB", "QB", "AAA", 30, 3, 0, 0),
            ("00-0000002", "Alpha WR", "WR", "AAA", 0, 0, 8,
             receiving_yards_week_two if week == 2 else 50),
            ("00-0000003", "Beta QB", "QB", "BBB", 28, 2, 0, 0),
            ("00-0000004", "Beta WR", "WR", "BBB", 0, 0, 7, 55),
        )
    ])
    rosters = pd.DataFrame([
        {"season": 2022, "week": week, "game_type": "REG", "gsis_id": player,
         "full_name": name, "position": position, "team": team, "status": "ACT",
         "birth_date": "1995-01-01", "years_exp": 3, "draft_number": 50,
         "pfr_id": player}
        for week in (1, 2, 3)
        for player, name, position, team in (
            ("00-0000001", "Alpha QB", "QB", "AAA"),
            ("00-0000002", "Alpha WR", "WR", "AAA"),
            ("00-0000003", "Beta QB", "QB", "BBB"),
            ("00-0000004", "Beta WR", "WR", "BBB"),
        )
    ])
    schedules = pd.DataFrame([
        {"season": 2022, "week": week, "game_type": "REG", "game_id": f"2022_0{week}_AAA_BBB",
         "gameday": f"2022-09-{week + 7:02d}", "weekday": "Sunday", "gametime": "13:00",
         "away_team": "AAA", "home_team": "BBB", "away_rest": 7, "home_rest": 7,
         "spread_line": -3.0, "total_line": 45.0, "div_game": 0, "roof": "outdoors",
         "surface": "grass", "temp": 70, "wind": 5, "referee": "Ref A", "stadium": "Park",
         "away_qb_id": "00-0000001", "home_qb_id": "00-0000003",
         "away_coach": "Coach A", "home_coach": "Coach B"}
        for week in (1, 2, 3)
    ])
    return HistoricalData(stats=stats, rosters=rosters, schedules=schedules)


def test_same_week_outcome_cannot_change_same_week_features():
    original = build_feature_frame(_bundle(60))
    changed = build_feature_frame(_bundle(600))
    key = (original.player_id.eq("00-0000002"))
    week_two_a = original[key & original.week.eq(2)].iloc[0]
    week_two_b = changed[key & changed.week.eq(2)].iloc[0]
    assert week_two_a[model_features()].equals(week_two_b[model_features()])
    week_three_a = original[key & original.week.eq(3)].iloc[0]
    week_three_b = changed[key & changed.week.eq(3)].iloc[0]
    assert week_three_a["pre_fantasy_points_ewm4"] != week_three_b["pre_fantasy_points_ewm4"]


def test_expected_opportunity_components_are_shifted_before_use():
    original = _bundle()
    expected = pd.DataFrame([
        {"season": 2022, "week": week, "player_id": player,
         "total_fantasy_points_exp": 10.0, "total_touchdown_exp": touchdown,
         "total_yards_gained_exp": 80.0, "receptions_exp": 4.0,
         "total_first_down_exp": 5.0, "pass_fantasy_points_exp": 0.0,
         "rush_fantasy_points_exp": 0.0, "rec_fantasy_points_exp": 8.0}
        for week in (1, 2, 3)
        for player in ("00-0000001", "00-0000002", "00-0000003", "00-0000004")
        for touchdown in [0.25]
    ])
    original.expected_points = expected
    changed = _bundle()
    changed.expected_points = expected.copy()
    changed.expected_points.loc[
        changed.expected_points.week.eq(2)
        & changed.expected_points.player_id.eq("00-0000002"),
        "total_touchdown_exp",
    ] = 4.0

    frame_a = build_feature_frame(original)
    frame_b = build_feature_frame(changed)
    player = frame_a.player_id.eq("00-0000002")
    assert frame_a[player & frame_a.week.eq(2)][model_features()].iloc[0].equals(
        frame_b[player & frame_b.week.eq(2)][model_features()].iloc[0]
    )
    assert (
        frame_a[player & frame_a.week.eq(3)].iloc[0]["pre_expected_touchdowns_ewm4"]
        != frame_b[player & frame_b.week.eq(3)].iloc[0]["pre_expected_touchdowns_ewm4"]
    )


def test_coach_tendencies_are_prior_only_and_usage_is_reconciled():
    original = _bundle()
    changed = _bundle()
    changed.stats.loc[
        changed.stats.week.eq(2) & changed.stats.position.eq("QB"), "attempts"
    ] = 70
    frame_a = build_feature_frame(original)
    frame_b = build_feature_frame(changed)
    key = frame_a.player_id.eq("00-0000002")
    assert frame_a[key & frame_a.week.eq(2)][model_features()].iloc[0].equals(
        frame_b[key & frame_b.week.eq(2)][model_features()].iloc[0]
    )
    assert (
        frame_a[key & frame_a.week.eq(3)].iloc[0]["pre_coach_pass_rate_ewm8"]
        != frame_b[key & frame_b.week.eq(3)].iloc[0]["pre_coach_pass_rate_ewm8"]
    )

    active = frame_a[frame_a.week.ge(2) & frame_a.status_inactive.eq(0)]
    reconciled = active.groupby(["season", "week", "team"])[
        ["pre_reconciled_attempt_share", "pre_reconciled_target_share",
         "pre_reconciled_carry_share"]
    ].sum()
    assert (reconciled <= 1.0 + 1e-12).all().all()


def test_schedule_spread_is_translated_from_each_teams_view():
    frame = build_feature_frame(_bundle())
    home = frame[(frame.week == 1) & (frame.team == "BBB")].iloc[0]
    away = frame[(frame.week == 1) & (frame.team == "AAA")].iloc[0]
    assert home.team_spread == -3
    assert away.team_spread == 3
    assert home.implied_team_points == 24
    assert away.implied_team_points == 21


def test_feature_frame_is_roster_first_and_unique():
    frame = build_feature_frame(_bundle())
    assert len(frame) == 12
    assert not frame.duplicated(["season", "week", "player_id"]).any()
    assert {"pre_stadium_uplift", "pre_current_qb_id_uplift", "vacated_target_share"}.issubset(frame)


def test_future_week_carries_identity_but_not_results():
    data = _bundle()
    future_schedule = data.schedules.iloc[[0]].copy()
    future_schedule["week"] = 4
    future_schedule["game_id"] = "2022_04_AAA_BBB"
    data.schedules = pd.concat([data.schedules, future_schedule], ignore_index=True)
    extended = materialize_projection_week(data, 2022, 4)
    target = extended.rosters[extended.rosters.week.eq(4)]
    assert len(target) == 4
    assert target["snapshot_carried_forward"].all()
    assert not (extended.stats.week == 4).any()


def test_no_game_rows_are_ineligible_and_do_not_decay_ewm_history():
    data = _bundle()
    extra = data.rosters[data.rosters.week.eq(3)].copy()
    extra["week"] = 4
    data.rosters = pd.concat([data.rosters, extra], ignore_index=True)
    frame = build_feature_frame(data)
    no_game = frame[frame.week.eq(4)]
    assert no_game.schedule_missing.eq(1).all()
    assert not no_game.model_eligible.any()

    history = pd.DataFrame({
        "player_id": ["p"] * 4,
        "value": [10.0, float("nan"), 20.0, 0.0],
    })
    # With span=3, alpha=.5: the missing no-game row is ignored, so the prior
    # entering row four is .5*20 + .5*10 = 15 rather than a time-decayed value.
    assert _prior_ewm(history, ["player_id"], "value", 3).iloc[3] == 15.0
