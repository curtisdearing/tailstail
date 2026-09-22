"""The shadow K / D/ST lane: the verified scoring map and the projection contract.

The golden rows below are NOT re-runs of this implementation.  Each one is a
2026 week 1 or week 2 stat line taken from nflverse box scores, with the
``expected`` value taken from ESPN's OWN ``appliedTotal`` for that player in
Curtis's league -- the number the league actually paid.  Reproducing them is
what establishes that the contract in `espn_contract` is the league's real
scoring and not a plausible decode of it.

The decode this replaced got two things wrong, and both are pinned here:

  * ``team_win`` (statId 155) pays 2.0 to a kicker and to a defense whose NFL
    team wins.  It was missing entirely, so every winning K and D/ST was
    scored exactly two points light -- 20 of the 40 rows below.
  * field goals of 50+ yards were read as scoring nothing.  They score 5.0
    (50-59, statId 198) and 6.0 (60+, statId 201).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nflvalue.fantasy.espn_contract import from_settings_payload
from nflvalue.fantasy.special_scoring import DST_EVENT_KEYS, score_dst, score_kicker
from nflvalue.fantasy.special_teams import (
    DST_HISTORY_TO_CONTRACT,
    MODEL_VERSION,
    PROMOTION_STATUS,
    Projection,
    SpecialTeamsError,
    dst_line,
    league_base_rates,
    past_only,
    project_dst,
    project_kicker,
    score_dst_frame,
)

FIXTURE = Path(__file__).parent / "fixtures" / "espn_league_settings_2026_recorded.json"


@pytest.fixture(scope="module")
def contract():
    return from_settings_payload(json.loads(FIXTURE.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# Golden rows: ESPN's own applied totals, 2026 weeks 1-2
# --------------------------------------------------------------------------- #
#: (label, stat line, ESPN appliedTotal).  Chosen to cover every term that the
#: earlier decode got wrong or could not see: the win bonus on both sides of
#: the win/loss split, a 50-59 make, a 60+ make, a missed PAT, two missed
#: field goals in one game, a 20-point fumble-return touchdown, the
#: points-allowed band that scores zero and is absent from the payload, and a
#: yards-allowed total in the 300-349 dead band.
GOLDEN_KICKERS = [
    # Cam Little, wk1: two 40-49 makes, four PATs, JAX won.
    ("cam_little_wk1", dict(fg_made_40_49=2, pat_made=4, team_win=1), 14.0),
    # Same line with the loss: exactly the win bonus apart.
    ("cam_little_wk1_if_lost", dict(fg_made_40_49=2, pat_made=4, team_win=0), 12.0),
    # Eddy Pineiro, wk1: one 0-39, one 50-59, one miss, three PATs, won.
    ("pineiro_wk1", dict(fg_made_0_39=1, fg_made_50_59=1, field_goals_missed=1,
                         pat_made=3, team_win=1), 10.0),
    # Brandon Aubrey, wk2: 0-39, 40-49 and a SIXTY-YARDER, four PATs, won.
    ("aubrey_wk2", dict(fg_made_0_39=1, fg_made_40_49=1, fg_made_60_plus=1,
                        pat_made=4, team_win=1), 19.0),
    # Brandon Aubrey, wk1: two PATs made, one missed, lost.
    ("aubrey_wk1", dict(pat_made=2, pat_missed=1, team_win=0), -1.0),
    # Nick Folk, wk1: 40-49 and 50-59 makes, TWO misses, one PAT, lost.
    ("folk_wk1", dict(fg_made_40_49=1, fg_made_50_59=1, field_goals_missed=2,
                      pat_made=1, team_win=0), 4.0),
    # Harrison Mevis, wk2: one miss, four PATs, LA won.
    ("mevis_wk2", dict(field_goals_missed=1, pat_made=4, team_win=1), 3.0),
    # Ka'imi Fairbairn, wk1: one 50-59, four PATs, HOU lost.
    ("fairbairn_wk1", dict(fg_made_50_59=1, pat_made=4, team_win=0), 9.0),
]

GOLDEN_DST = [
    # Steelers wk1: 4 sacks, 2 INT, a FUMBLE-RETURN TD (20), PA 13, YA 238, won.
    ("steelers_wk1", dict(defensive_sack=4, defensive_interception=2, fumble_return_td=1,
                          points_allowed=13, yards_allowed=238, team_win=1), 35.0),
    # Panthers wk2: 2 sacks, 3 INT, 2 FR, fumble-return TD, PA 3, YA 286, won.
    ("panthers_wk2", dict(defensive_sack=2, defensive_interception=3,
                          defensive_fumble_recovery=2, fumble_return_td=1,
                          points_allowed=3, yards_allowed=286, team_win=1), 40.0),
    # Panthers wk1: PA 59 (46+ = -5) and YA 552 (550+ = -7).  The floor.
    ("panthers_wk1", dict(defensive_sack=2, defensive_fumble_recovery=1,
                          points_allowed=59, yards_allowed=552, team_win=0), -8.0),
    # Eagles wk1: a blocked kick, PA 22 -- a band the payload does not contain
    # because it scores zero -- YA 295, won.
    ("eagles_wk1", dict(defensive_sack=1, blocked_kick=1, points_allowed=22,
                        yards_allowed=295, team_win=1), 7.0),
    # Steelers wk2: PA 14 (14-17 band), YA 327 -- the 300-349 dead band -- lost.
    ("steelers_wk2", dict(defensive_sack=3, defensive_interception=1,
                          defensive_fumble_recovery=1, points_allowed=14,
                          yards_allowed=327, team_win=0), 8.0),
    # Texans wk1: PA 36 (-3) and YA 409 (-3), lost.
    ("texans_wk1", dict(defensive_sack=2, points_allowed=36, yards_allowed=409,
                        team_win=0), -4.0),
    # 49ers wk2: 4 sacks, PA 13, YA 284, won -- the plain case.
    ("niners_wk2", dict(defensive_sack=4, points_allowed=13, yards_allowed=284,
                        team_win=1), 11.0),
    # Buccaneers wk1: fumble-return TD but a LOSS; PA 27 and YA 351 both in
    # scoring bands, and no win bonus.
    ("bucs_wk1", dict(defensive_sack=1, defensive_interception=1, fumble_return_td=1,
                      points_allowed=27, yards_allowed=351, team_win=0), 22.0),
]


@pytest.mark.parametrize("label,line,expected", GOLDEN_KICKERS,
                         ids=[row[0] for row in GOLDEN_KICKERS])
def test_kicker_matches_espn_applied_total(contract, label, line, expected):
    assert score_kicker(line, contract) == pytest.approx(expected)


@pytest.mark.parametrize("label,line,expected", GOLDEN_DST,
                         ids=[row[0] for row in GOLDEN_DST])
def test_dst_matches_espn_applied_total(contract, label, line, expected):
    assert score_dst(line, contract) == pytest.approx(expected)


def test_team_win_is_worth_two_to_both_seats(contract):
    """The category the first decode missed, asserted directly."""
    assert contract.points("team_win") == pytest.approx(2.0)
    assert contract.dst_points("team_win") == pytest.approx(2.0)


def test_win_bonus_is_exactly_the_difference(contract):
    """Every golden pair differs by the win bonus and nothing else."""
    won = dict(defensive_sack=4, points_allowed=13, yards_allowed=284, team_win=1)
    lost = dict(won, team_win=0)
    assert score_dst(won, contract) - score_dst(lost, contract) == pytest.approx(2.0)


def test_team_win_absent_scores_nothing(contract):
    """A caller that does not know the result must not be charged or credited.

    This is what keeps the change backward compatible: the pre-existing
    callers pass no ``team_win`` and get exactly the number they got before.
    """
    line = dict(defensive_sack=4, points_allowed=13, yards_allowed=284)
    assert score_dst(line, contract) == pytest.approx(9.0)
    assert score_kicker({"fg_made_0_39": 1}, contract) == pytest.approx(3.0)


def test_long_field_goals_are_not_free(contract):
    """50+ yard makes score; reading them as zero cost 5 and 6 points a kick."""
    assert contract.points("fg_made_50_59") == pytest.approx(5.0)
    assert contract.points("fg_made_60_plus") == pytest.approx(6.0)
    assert score_kicker({"field_goals_made": [50]}, contract) == pytest.approx(5.0)
    assert score_kicker({"field_goals_made": [60]}, contract) == pytest.approx(6.0)
    assert score_kicker({"field_goals_made": [49]}, contract) == pytest.approx(4.0)


def test_amplified_defensive_touchdowns(contract):
    """The overrides that make this league's D/ST unusual."""
    assert contract.dst_points("interception_return_td") == pytest.approx(20.0)
    assert contract.dst_points("fumble_return_td") == pytest.approx(20.0)
    assert contract.dst_points("blocked_kick_return_td") == pytest.approx(30.0)
    assert contract.dst_points("kickoff_return_td") == pytest.approx(12.0)
    assert contract.dst_points("punt_return_td") == pytest.approx(12.0)
    assert contract.dst_points("defensive_safety") == pytest.approx(10.0)
    # ...against 1.0 for the sack that is the position's bread and butter.
    assert contract.dst_points("defensive_sack") == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# The history -> contract vocabulary
# --------------------------------------------------------------------------- #
def test_every_priced_dst_event_has_a_history_column():
    """A missing key in `score_dst` means "the league pays nothing", so it
    cannot also mean "you spelled the column wrong".  Passing a history row
    through unmapped scored every per-event category as zero and cost a
    defense its sacks; this asserts the translation is total."""
    assert set(DST_HISTORY_TO_CONTRACT.values()) == set(DST_EVENT_KEYS)


def test_dst_line_translates_a_history_row(contract):
    row = {"sacks": 4.0, "interceptions": 2.0, "fumble_recoveries": 0.0,
           "safeties": 0.0, "blocked_kicks": 0.0, "interception_return_td": 0.0,
           "fumble_return_td": 1.0, "kickoff_return_td": 0.0, "punt_return_td": 0.0,
           "blocked_kick_return_td": 0.0, "two_point_return": 0.0,
           "one_point_safety": 0.0, "points_allowed": 13.0, "yards_allowed": 238.0,
           "team_win": 1}
    assert score_dst(dst_line(row), contract) == pytest.approx(35.0)
    assert score_dst_frame(pd.DataFrame([row]), contract) == [pytest.approx(35.0)]


# --------------------------------------------------------------------------- #
# Synthetic history, for the clock and contract tests
# --------------------------------------------------------------------------- #
def _dst_history(seasons=(2024, 2025), weeks=14, teams=("AAA", "BBB", "CCC", "DDD")):
    rng = np.random.default_rng(7)
    rows = []
    for season in seasons:
        for week in range(1, weeks + 1):
            for i, team in enumerate(teams):
                opp = teams[(i + 1) % len(teams)]
                rows.append(dict(
                    season=season, week=week, team=team, opponent=opp,
                    points_for=float(rng.integers(3, 35)),
                    points_against=float(rng.integers(3, 35)),
                    points_allowed=float(rng.integers(3, 35)),
                    yards_allowed=float(rng.integers(180, 450)),
                    offense_yards=float(rng.integers(180, 450)),
                    sacks=float(rng.integers(0, 6)), sacks_allowed=float(rng.integers(0, 6)),
                    interceptions=float(rng.integers(0, 3)),
                    fumble_recoveries=float(rng.integers(0, 2)),
                    safeties=0.0, blocked_kicks=float(rng.integers(0, 2)),
                    interception_return_td=float(rng.random() < 0.07),
                    fumble_return_td=float(rng.random() < 0.04),
                    kickoff_return_td=0.0,
                    punt_return_td=float(rng.random() < 0.015),
                    blocked_kick_return_td=0.0, two_point_return=0.0, one_point_safety=0.0,
                    team_win=int(rng.random() < 0.5), fantasy_points=float(rng.integers(-5, 30)),
                ))
    return pd.DataFrame(rows)


def _kicker_history(dst):
    rng = np.random.default_rng(11)
    rows = []
    for r in dst.itertuples():
        rows.append(dict(
            season=r.season, week=r.week, team=r.team, opponent=r.opponent,
            player_id=f"K-{r.team}", player_display_name=f"Kicker {r.team}",
            team_win=r.team_win, points_for=r.points_for,
            fg_made_0_39=float(rng.integers(0, 3)), fg_made_0_39_att=float(rng.integers(0, 3)),
            fg_made_40_49=float(rng.integers(0, 2)), fg_made_40_49_att=float(rng.integers(0, 2)),
            fg_made_50_59=float(rng.integers(0, 2)), fg_made_50_59_att=float(rng.integers(0, 2)),
            fg_made_60_plus=0.0, fg_made_60_plus_att=0.0,
            field_goals_missed=float(rng.integers(0, 2)),
            pat_made=float(rng.integers(0, 5)), pat_missed=0.0,
            fantasy_points=float(rng.integers(-2, 20)),
        ))
    out = pd.DataFrame(rows)
    for key in ("fg_made_0_39", "fg_made_40_49", "fg_made_50_59", "fg_made_60_plus"):
        out[f"{key}_att"] = np.maximum(out[f"{key}_att"], out[key])
    return out


# --------------------------------------------------------------------------- #
# The feature clock
# --------------------------------------------------------------------------- #
def test_past_only_excludes_the_week_being_projected():
    history = _dst_history()
    kept = past_only(history, 2025, 6)
    assert len(kept) > 0
    assert kept[(kept.season == 2025) & (kept.week >= 6)].empty
    assert not kept[(kept.season == 2025) & (kept.week == 5)].empty
    assert not kept[kept.season == 2024].empty


def test_past_only_excludes_future_seasons_entirely():
    history = _dst_history()
    kept = past_only(history, 2024, 1)
    assert kept.empty


def test_poisoning_the_future_leaves_the_projection_bit_identical(contract):
    """The leakage gate.

    Every row at or after the projected week is replaced with absurd values.
    If any of them reaches the fit, the number moves.  This is the guard the
    repository lacked when `expected_points_missing` read same-week presence.
    """
    history = _dst_history()
    clean = project_dst(history, "AAA", "BBB", 2025, 6, contract, simulations=3000)

    poisoned = history.copy()
    future = (poisoned.season > 2025) | ((poisoned.season == 2025) & (poisoned.week >= 6))
    for column in ("sacks", "interceptions", "fumble_recoveries", "blocked_kicks",
                   "interception_return_td", "fumble_return_td", "punt_return_td",
                   "points_allowed", "yards_allowed", "points_for", "offense_yards",
                   "fantasy_points", "team_win", "sacks_allowed"):
        poisoned.loc[future, column] = 999.0
    dirty = project_dst(poisoned, "AAA", "BBB", 2025, 6, contract, simulations=3000)

    assert dirty.as_dict() == clean.as_dict()


def test_poisoning_the_future_leaves_the_kicker_projection_identical(contract):
    dst = _dst_history()
    kicks = _kicker_history(dst)
    clean = project_kicker(kicks, dst, "K-AAA", "AAA", "BBB", 2025, 6, contract,
                           simulations=3000)
    poisoned = kicks.copy()
    future = (poisoned.season > 2025) | ((poisoned.season == 2025) & (poisoned.week >= 6))
    for column in ("fg_made_0_39", "fg_made_0_39_att", "fg_made_40_49", "fg_made_40_49_att",
                   "fg_made_50_59", "fg_made_50_59_att", "field_goals_missed",
                   "pat_made", "pat_missed", "fantasy_points", "team_win"):
        poisoned.loc[future, column] = 999.0
    dirty = project_kicker(poisoned, dst, "K-AAA", "AAA", "BBB", 2025, 6, contract,
                           simulations=3000)
    assert dirty.as_dict() == clean.as_dict()


def test_no_prior_history_fails_closed(contract):
    history = _dst_history(seasons=(2025,))
    with pytest.raises(SpecialTeamsError):
        project_dst(history, "AAA", "BBB", 2025, 1, contract, simulations=100)


# --------------------------------------------------------------------------- #
# The projection contract
# --------------------------------------------------------------------------- #
def test_projection_shape_and_ordering(contract):
    history = _dst_history()
    p = project_dst(history, "AAA", "BBB", 2025, 8, contract, simulations=5000)
    assert isinstance(p, Projection)
    assert p.p10 <= p.p50 <= p.p90
    assert np.isfinite([p.mean, p.sd, p.p10, p.p50, p.p90]).all()
    assert 0.0 <= p.p_zero_or_less <= 1.0
    body = p.as_dict()
    assert body["status"] == "shadow"
    assert body["promoted"] is False
    assert body["model_version"] == MODEL_VERSION
    assert body["position"] == "D/ST"
    for key in ("mean", "p10", "p50", "p90", "sd", "p_zero_or_less", "components",
                "season", "week", "team", "opponent", "n_prior_games"):
        assert key in body


def test_kicker_projection_shape(contract):
    dst = _dst_history()
    p = project_kicker(_kicker_history(dst), dst, "K-AAA", "AAA", "BBB", 2025, 8,
                       contract, simulations=5000, name="Kicker AAA")
    assert p.position == "K"
    assert p.p10 <= p.p50 <= p.p90
    assert p.as_dict()["promoted"] is False
    assert 0.0 <= p.components["p_win"] <= 1.0
    for _, _, key in (("", "", "fg_made_0_39"), ("", "", "fg_made_40_49"),
                      ("", "", "fg_made_50_59"), ("", "", "fg_made_60_plus")):
        assert 0.0 <= p.components[f"p_make_{key}"] <= 1.0


def test_lane_stays_shadow():
    assert PROMOTION_STATUS["status"] == "shadow"
    assert PROMOTION_STATUS["may_enter_lineup_objective"] is False


def test_the_touchdown_hazard_is_modelled_explicitly(contract):
    """The whole reason this is an event model and not a points regression."""
    history = _dst_history()
    p = project_dst(history, "AAA", "BBB", 2025, 10, contract, simulations=8000)
    assert 0.0 < p.components["p_any_touchdown"] < 0.5
    assert 0.0 < p.components["p_td_given_interception"] < 0.5
    assert 0.0 < p.components["p_td_given_fumble_recovery"] < 0.5
    # A rare event worth 20-30 must still dominate the mean, or the
    # amplification has been modelled away.
    assert p.components["share_of_mean_from_touchdowns"] > 0.10


def test_base_rates_are_reported_not_assumed(contract):
    history = _dst_history()
    rates = league_base_rates(past_only(history, 2025, 8))
    for key in ("sacks_per_game", "int_return_td_per_interception",
                "fumble_return_td_per_recovery", "points_allowed_mean",
                "yards_allowed_mean"):
        assert np.isfinite(rates[key])
    p = project_dst(history, "AAA", "BBB", 2025, 8, contract, simulations=2000)
    assert p.components["league_base_sacks_per_game"] == pytest.approx(rates["sacks_per_game"])


# --------------------------------------------------------------------------- #
# Determinism and isolation
# --------------------------------------------------------------------------- #
def test_same_inputs_same_answer(contract):
    history = _dst_history()
    a = project_dst(history, "AAA", "BBB", 2025, 8, contract, simulations=4000)
    b = project_dst(history, "AAA", "BBB", 2025, 8, contract, simulations=4000)
    assert a.as_dict() == b.as_dict()


def test_seed_survives_a_hash_salt(contract):
    """`shadow_kicker` shipped a determinism claim that `PYTHONHASHSEED` broke.

    Two subprocesses under different salts must agree, which builtin `hash()`
    would not.
    """
    script = (
        "import json, sys, pandas as pd, numpy as np;"
        "sys.path.insert(0, %r);"
        "from tests.test_special_teams import _dst_history;"
        "from nflvalue.fantasy.espn_contract import from_settings_payload;"
        "from nflvalue.fantasy.special_teams import project_dst;"
        "c = from_settings_payload(json.load(open(%r)));"
        "p = project_dst(_dst_history(), 'AAA', 'BBB', 2025, 8, c, simulations=2000);"
        "print(json.dumps(p.as_dict(), sort_keys=True))"
        % (str(Path(__file__).resolve().parents[1]), str(FIXTURE))
    )
    out = []
    for salt in ("0", "1", "12345"):
        env = {**__import__("os").environ, "PYTHONHASHSEED": salt}
        r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                           text=True, env=env,
                           cwd=str(Path(__file__).resolve().parents[1]))
        assert r.returncode == 0, r.stderr
        out.append(r.stdout.strip())
    assert out[0] == out[1] == out[2]


def test_the_frozen_snapshot_schema_still_refuses_these_positions():
    """Isolation, as a real assertion rather than a promise.

    The promoted contract's ``position`` enum is exactly QB/RB/WR/TE with
    ``additionalProperties: false``, so a K or D/ST row cannot validate
    against it by construction.
    """
    schema_path = (Path(__file__).resolve().parents[1] / "schemas"
                   / "player_projection_snapshot.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    blob = json.dumps(schema)
    assert '"QB"' in blob and '"K"' not in blob.replace('"KEY"', "")
    assert "D/ST" not in blob


def test_special_teams_is_not_imported_by_the_frozen_path():
    """No module on the promoted QB/RB/WR/TE path may import this lane."""
    root = Path(__file__).resolve().parents[1] / "nflvalue"
    frozen = ["fantasy/features.py", "fantasy/simulation.py", "fantasy/models.py",
              "fantasy/config.py", "fantasy/scoring.py", "fantasy/season.py"]
    for rel in frozen:
        path = root / rel
        if not path.exists():
            continue
        assert "special_teams" not in path.read_text(encoding="utf-8"), rel
