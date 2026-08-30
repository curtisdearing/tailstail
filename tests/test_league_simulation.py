"""The league is the unit, not the player.

A player-season simulation answers "how many points will he score". A LEAGUE
simulation answers "does my team make the playoffs", and those are different
questions: the second one is decided by an eight-team head-to-head schedule,
a legal weekly lineup drawn from a real roster, a standings sort with an exact
tiebreaker order, a seeding rule, and a bracket whose rounds are two matchup
periods long. None of that is a function of one player's mean.

Every fixture here is hand-checkable on purpose. A simulation that cannot be
checked by hand on four teams and three weeks cannot be trusted on eight teams
and seventeen, and the failure mode this file exists to prevent is a league
model that looks plausible and is quietly wrong about who advances.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue.fantasy import league_sim as LS
from tests.test_espn_league_adapter import _fixture, normalize

# --------------------------------------------------------------------------- #
# Hand-checkable schedules
# --------------------------------------------------------------------------- #
FOUR_TEAMS = (1, 2, 3, 4)

#: A complete single round robin. Each team plays each other exactly once, so
#: every team plays 3 games across 3 periods and every period is a perfect
#: pairing. Every assertion below can be checked by eye against this table.
RR4 = {
    1: ((1, 2), (3, 4)),
    2: ((1, 3), (2, 4)),
    3: ((1, 4), (2, 3)),
}

EIGHT_TEAMS = tuple(range(1, 9))


def circle_schedule(team_ids, periods):
    """Standard circle method — the shape ESPN produces for an even league."""
    teams = list(team_ids)
    n = len(teams)
    schedule = {}
    for period in range(1, periods + 1):
        games = []
        for i in range(n // 2):
            home, away = teams[i], teams[n - 1 - i]
            games.append((home, away) if period % 2 else (away, home))
        schedule[period] = tuple(games)
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return schedule


RS14 = circle_schedule(EIGHT_TEAMS, 14)


# --------------------------------------------------------------------------- #
# 1 · Schedule validation — fail closed, and say which team
# --------------------------------------------------------------------------- #
def test_a_complete_round_robin_validates():
    report = LS.validate_schedule(FOUR_TEAMS, RR4, periods=(1, 2, 3))
    assert report.ok
    assert report.games_per_team == {1: 3, 2: 3, 3: 3, 4: 3}


def test_every_team_appears_in_every_period():
    """The check that catches a dropped game: eight teams, fourteen periods,
    four games each, and nobody missing anywhere."""
    report = LS.validate_schedule(EIGHT_TEAMS, RS14, periods=tuple(range(1, 15)))
    assert report.ok
    assert set(report.games_per_team) == set(EIGHT_TEAMS)
    assert set(report.games_per_team.values()) == {14}


def test_every_matchup_is_reciprocal():
    """If 1 hosts 2 in period 1 then 2's period-1 opponent must be 1. A
    schedule where the two halves disagree is unsimulatable, not a nuance."""
    assert LS.validate_schedule(EIGHT_TEAMS, RS14, periods=tuple(range(1, 15))).ok
    broken = dict(RR4)
    broken[1] = ((1, 2), (3, 1))          # team 1 twice, team 4 nowhere
    with pytest.raises(LS.ScheduleError) as exc:
        LS.validate_schedule(FOUR_TEAMS, broken, periods=(1, 2, 3))
    assert "period 1" in str(exc.value)
    assert "1" in str(exc.value) and "4" in str(exc.value)


def test_a_team_cannot_play_itself():
    broken = dict(RR4)
    broken[2] = ((1, 1), (2, 4))
    with pytest.raises(LS.ScheduleError, match="itself"):
        LS.validate_schedule(FOUR_TEAMS, broken, periods=(1, 2, 3))


def test_an_unknown_team_id_is_refused():
    broken = dict(RR4)
    broken[3] = ((1, 4), (2, 99))
    with pytest.raises(LS.ScheduleError, match="99"):
        LS.validate_schedule(FOUR_TEAMS, broken, periods=(1, 2, 3))


def test_a_missing_period_is_refused():
    partial = {1: RR4[1], 3: RR4[3]}
    with pytest.raises(LS.ScheduleError, match="period 2"):
        LS.validate_schedule(FOUR_TEAMS, partial, periods=(1, 2, 3))


def test_a_bye_is_refused_unless_the_league_declares_odd_sizes():
    """An eight-team league has no byes. A period with three games means a
    game was lost, and guessing which one is exactly the wrong repair."""
    broken = dict(RR4)
    broken[1] = ((1, 2),)
    with pytest.raises(LS.ScheduleError):
        LS.validate_schedule(FOUR_TEAMS, broken, periods=(1, 2, 3))


# --------------------------------------------------------------------------- #
# 2 · Legal weekly lineups
# --------------------------------------------------------------------------- #
SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "D/ST": 1}


def _p(pid, name, slots, pts):
    return LS.RosterSpot(player_id=pid, name=name, eligible_slots=tuple(slots)), pts


def _roster(rows):
    roster = tuple(spot for spot, _ in rows)
    points = {spot.player_id: pts for spot, pts in rows}
    return roster, points


def test_the_flex_takes_the_best_leftover_not_the_first_one():
    """Greedy position-filling is the classic bug: it fills RB/WR first and
    hands the FLEX whatever is left, which can leave points on the bench."""
    roster, points = _roster([
        _p(1, "QB", ["QB"], 20.0),
        _p(2, "RB1", ["RB", "FLEX"], 18.0),
        _p(3, "RB2", ["RB", "FLEX"], 9.0),
        _p(4, "RB3", ["RB", "FLEX"], 8.0),
        _p(5, "WR1", ["WR", "FLEX"], 15.0),
        _p(6, "WR2", ["WR", "FLEX"], 12.0),
        _p(7, "WR3", ["WR", "FLEX"], 11.0),
        _p(8, "TE1", ["TE", "FLEX"], 7.0),
        _p(9, "K1", ["K"], 6.0),
        _p(10, "DST1", ["D/ST"], 5.0),
    ])
    best = LS.optimal_lineup(points, roster, SLOTS)
    # QB20 + RB 18,9 + WR 15,12 + TE 7 + FLEX = WR3 11 + K6 + DST5 = 103
    assert best.total == pytest.approx(103.0)
    assert best.assignment["FLEX"] == (7,)


def test_a_slot_with_no_eligible_player_scores_zero_and_is_named():
    roster, points = _roster([
        _p(1, "QB", ["QB"], 20.0),
        _p(2, "RB1", ["RB", "FLEX"], 10.0),
    ])
    best = LS.optimal_lineup(points, roster, SLOTS)
    assert best.total == pytest.approx(30.0)
    # the RB fills the RB slot, not the FLEX: base slots are preferred when the
    # points are identical, so the leftover FLEX is reported empty too
    assert best.empty_slots == ("D/ST", "FLEX", "K", "RB", "TE", "WR", "WR")
    assert best.assignment["FLEX"] == ()
    assert best.assignment["RB"] == (2,)


def test_a_player_fills_exactly_one_slot():
    roster, points = _roster([
        _p(1, "QB", ["QB"], 20.0),
        _p(2, "RB1", ["RB", "FLEX"], 18.0),
    ])
    best = LS.optimal_lineup(points, roster, SLOTS)
    used = [pid for slot in best.assignment.values() for pid in slot]
    assert len(used) == len(set(used))
    assert best.total == pytest.approx(38.0)


def test_lineup_optimisation_is_deterministic_under_ties():
    roster, points = _roster([
        _p(1, "QB", ["QB"], 10.0),
        _p(2, "RBa", ["RB", "FLEX"], 5.0),
        _p(3, "RBb", ["RB", "FLEX"], 5.0),
        _p(4, "RBc", ["RB", "FLEX"], 5.0),
    ])
    first = LS.optimal_lineup(points, roster, SLOTS)
    for _ in range(5):
        again = LS.optimal_lineup(points, roster, SLOTS)
        assert again.assignment == first.assignment
        assert again.total == first.total


# --------------------------------------------------------------------------- #
# 3 · Matchup scoring and ties
# --------------------------------------------------------------------------- #
def test_equal_scores_are_a_tie_when_the_league_does_not_break_them():
    result = LS.score_matchup(101.5, 101.5, tie_rule="NONE")
    assert result == "TIE"


def test_a_tie_rule_of_home_awards_the_home_team():
    assert LS.score_matchup(101.5, 101.5, tie_rule="HOME") == "HOME"


def test_an_unknown_tie_rule_fails_closed():
    with pytest.raises(LS.LeagueSimError, match="tie rule"):
        LS.score_matchup(1.0, 1.0, tie_rule="COIN_FLIP_MAYBE")


def test_the_home_bonus_is_applied_before_comparison():
    assert LS.score_matchup(100.0, 101.0, tie_rule="NONE", home_bonus=2.0) == "HOME"


# --------------------------------------------------------------------------- #
# 4 · Standings and the exact tiebreaker order
# --------------------------------------------------------------------------- #
def _rec(team_id, w, losses, t, pf, pa):
    return LS.TeamRecord(team_id=team_id, wins=w, losses=losses, ties=t,
                         points_for=pf, points_against=pa)


def test_record_sorts_before_any_tiebreaker():
    table = LS.standings(
        [_rec(1, 2, 1, 0, 300.0, 290.0), _rec(2, 3, 0, 0, 200.0, 190.0)],
        tiebreakers=("head_to_head", "points_for"),
        head_to_head={},
    )
    assert [row.team_id for row in table] == [2, 1]


def test_head_to_head_breaks_a_tie_before_points_for():
    """Team 1 scored more all year; team 2 beat it. ESPN's order says team 2."""
    table = LS.standings(
        [_rec(1, 2, 1, 0, 400.0, 300.0), _rec(2, 2, 1, 0, 310.0, 300.0)],
        tiebreakers=("head_to_head", "points_for"),
        head_to_head={(2, 1): (1, 0, 0)},
    )
    assert [row.team_id for row in table] == [2, 1]


def test_points_for_breaks_a_tie_when_head_to_head_is_split():
    table = LS.standings(
        [_rec(1, 2, 1, 0, 400.0, 300.0), _rec(2, 2, 1, 0, 310.0, 300.0)],
        tiebreakers=("head_to_head", "points_for"),
        head_to_head={(2, 1): (1, 1, 0)},
    )
    assert [row.team_id for row in table] == [1, 2]


def test_the_declared_order_is_obeyed_even_when_it_is_unusual():
    """The order comes from the league payload, not from our preferences."""
    table = LS.standings(
        [_rec(1, 2, 1, 0, 400.0, 300.0), _rec(2, 2, 1, 0, 310.0, 250.0)],
        tiebreakers=("points_against", "points_for"),
        head_to_head={},
    )
    assert [row.team_id for row in table] == [2, 1]


def test_ties_count_as_half_a_win():
    table = LS.standings(
        [_rec(1, 2, 0, 1, 300.0, 200.0), _rec(2, 2, 1, 0, 900.0, 200.0)],
        tiebreakers=("points_for",),
        head_to_head={},
    )
    assert [row.team_id for row in table] == [1, 2]


def test_an_unknown_tiebreaker_fails_closed():
    with pytest.raises(LS.LeagueSimError, match="tiebreaker"):
        LS.standings([_rec(1, 1, 0, 0, 1.0, 1.0)],
                     tiebreakers=("vibes",), head_to_head={})


def test_a_total_tie_falls_back_to_a_declared_deterministic_rule():
    """Two teams identical on every declared tiebreaker. The result must be
    stable across runs and must SAY that it was arbitrary."""
    table = LS.standings(
        [_rec(7, 2, 1, 0, 300.0, 300.0), _rec(3, 2, 1, 0, 300.0, 300.0)],
        tiebreakers=("points_for",), head_to_head={},
    )
    assert [row.team_id for row in table] == [3, 7]
    assert table[0].unbroken_tie_with == (7,)


# --------------------------------------------------------------------------- #
# 5 · Playoff settings — fail closed on contradictions
# --------------------------------------------------------------------------- #
def test_two_week_rounds_are_built_from_the_declared_length():
    rounds = LS.playoff_rounds(playoff_team_count=4, first_period=15,
                               matchup_period_length=2)
    assert [r.name for r in rounds] == ["semifinal", "final"]
    assert rounds[0].matchup_periods == (15, 16)
    assert rounds[1].matchup_periods == (17, 18)


def test_one_week_rounds_still_work():
    rounds = LS.playoff_rounds(playoff_team_count=4, first_period=15,
                               matchup_period_length=1)
    assert [r.matchup_periods for r in rounds] == [(15,), (16,)]


def test_six_playoff_teams_give_the_top_two_a_bye():
    rounds = LS.playoff_rounds(playoff_team_count=6, first_period=15,
                               matchup_period_length=2)
    assert [r.name for r in rounds] == ["quarterfinal", "semifinal", "final"]
    assert rounds[0].byes == (1, 2)


def test_a_playoff_field_larger_than_the_league_fails_closed():
    with pytest.raises(LS.PlayoffSettingsError, match="8"):
        LS.playoff_rounds(playoff_team_count=10, first_period=15,
                          matchup_period_length=2, team_count=8)


def test_a_playoff_field_of_one_fails_closed():
    with pytest.raises(LS.PlayoffSettingsError):
        LS.playoff_rounds(playoff_team_count=1, first_period=15,
                          matchup_period_length=2)


def test_a_zero_length_matchup_period_fails_closed():
    with pytest.raises(LS.PlayoffSettingsError, match="length"):
        LS.playoff_rounds(playoff_team_count=4, first_period=15,
                          matchup_period_length=0)


def test_declared_periods_that_contradict_the_length_fail_closed():
    """The adapter derives playoff periods in SCORING-period units while the
    bracket runs in MATCHUP periods. Silently picking one is how a two-week
    final becomes a one-week final; the contradiction has to surface."""
    with pytest.raises(LS.PlayoffSettingsError, match="contradict"):
        LS.playoff_rounds(playoff_team_count=4, first_period=15,
                          matchup_period_length=2,
                          declared_periods=(15, 16, 17))


def test_declared_periods_consistent_with_the_length_are_accepted():
    rounds = LS.playoff_rounds(playoff_team_count=4, first_period=15,
                               matchup_period_length=2,
                               declared_periods=(15, 16, 17, 18))
    assert [r.matchup_periods for r in rounds] == [(15, 16), (17, 18)]


# --------------------------------------------------------------------------- #
# 6 · The bracket: two-week aggregates, reseeding, byes
# --------------------------------------------------------------------------- #
def test_a_two_week_round_is_decided_on_the_aggregate_not_week_one():
    """Team 4 wins the first week by 10 and loses the round by 5. A bracket
    that decides on week one alone sends the wrong team to the final."""
    rounds = LS.playoff_rounds(playoff_team_count=4, first_period=15,
                               matchup_period_length=2)
    points = {
        (1, 15): 90.0, (1, 16): 110.0,   # seed 1 aggregate 200
        (4, 15): 100.0, (4, 16): 95.0,   # seed 4 aggregate 195
        (2, 15): 100.0, (2, 16): 100.0,
        (3, 15): 80.0, (3, 16): 80.0,
        (1, 17): 105.0, (1, 18): 105.0,   # final, also two weeks
        (2, 17): 100.0, (2, 18): 100.0,
    }
    outcome = LS.run_bracket(seeds=(1, 2, 3, 4), rounds=rounds,
                             team_period_points=points, reseed=False,
                             tie_rule="NONE")
    assert outcome.rounds[0].winners == (1, 2)
    assert outcome.champion == 1


def test_the_higher_seed_meets_the_lower_seed_in_round_one():
    rounds = LS.playoff_rounds(playoff_team_count=4, first_period=15,
                               matchup_period_length=1)
    outcome = LS.run_bracket(
        seeds=(11, 22, 33, 44), rounds=rounds,
        team_period_points={(11, 15): 1, (44, 15): 0, (22, 15): 1, (33, 15): 0,
                            (11, 16): 1, (22, 16): 0},
        reseed=False, tie_rule="NONE")
    assert outcome.rounds[0].pairings == ((11, 44), (22, 33))
    assert outcome.champion == 11


def test_without_reseeding_the_bracket_is_fixed():
    """Seeds 3 and 4 both win. A fixed bracket pairs them by bracket slot."""
    rounds = LS.playoff_rounds(playoff_team_count=4, first_period=15,
                               matchup_period_length=1)
    points = {(1, 15): 0.0, (4, 15): 1.0, (2, 15): 0.0, (3, 15): 1.0,
              (3, 16): 5.0, (4, 16): 4.0}
    outcome = LS.run_bracket(seeds=(1, 2, 3, 4), rounds=rounds,
                             team_period_points=points, reseed=False,
                             tie_rule="NONE")
    assert outcome.rounds[0].winners == (4, 3)
    # the bracket seats the better surviving seed at home, so 3 hosts 4
    assert outcome.rounds[1].pairings == ((3, 4),)
    assert outcome.champion == 3


def test_with_reseeding_the_best_surviving_seed_gets_the_worst():
    """Six teams, top two on byes. Seeds 5 and 6 both win round one. With
    reseeding, seed 1 must draw 6 and seed 2 must draw 5 — not the fixed
    bracket's pairing."""
    rounds = LS.playoff_rounds(playoff_team_count=6, first_period=15,
                               matchup_period_length=1)
    points = {(3, 15): 0.0, (6, 15): 1.0, (4, 15): 0.0, (5, 15): 1.0}
    fixed = LS.run_bracket(seeds=(1, 2, 3, 4, 5, 6), rounds=rounds,
                           team_period_points={**points, (1, 16): 1, (6, 16): 0,
                                               (2, 16): 1, (5, 16): 0,
                                               (1, 17): 1, (2, 17): 0},
                           reseed=False, tie_rule="NONE")
    reseeded = LS.run_bracket(seeds=(1, 2, 3, 4, 5, 6), rounds=rounds,
                              team_period_points={**points, (1, 16): 1, (6, 16): 0,
                                                  (2, 16): 1, (5, 16): 0,
                                                  (1, 17): 1, (2, 17): 0},
                              reseed=True, tie_rule="NONE")
    assert fixed.rounds[1].pairings == ((1, 5), (2, 6))
    assert reseeded.rounds[1].pairings == ((1, 6), (2, 5))


def test_a_bye_team_does_not_play_round_one():
    rounds = LS.playoff_rounds(playoff_team_count=6, first_period=15,
                               matchup_period_length=1)
    outcome = LS.run_bracket(
        seeds=(1, 2, 3, 4, 5, 6), rounds=rounds,
        team_period_points={(3, 15): 1, (6, 15): 0, (4, 15): 1, (5, 15): 0,
                            (1, 16): 1, (4, 16): 0, (2, 16): 1, (3, 16): 0,
                            (1, 17): 1, (2, 17): 0},
        reseed=False, tie_rule="NONE")
    played = {t for pair in outcome.rounds[0].pairings for t in pair}
    assert 1 not in played and 2 not in played
    assert outcome.rounds[0].byes == (1, 2)


def test_a_playoff_tie_uses_the_playoff_tie_rule_not_the_regular_one():
    rounds = LS.playoff_rounds(playoff_team_count=4, first_period=15,
                               matchup_period_length=1)
    points = {(1, 15): 100.0, (4, 15): 100.0, (2, 15): 100.0, (3, 15): 99.0,
              (1, 16): 10.0, (2, 16): 9.0}
    outcome = LS.run_bracket(seeds=(1, 2, 3, 4), rounds=rounds,
                             team_period_points=points, reseed=False,
                             tie_rule="HIGHER_SEED")
    assert outcome.rounds[0].winners == (1, 2)
    assert outcome.champion == 1


def test_missing_playoff_points_fail_closed_rather_than_scoring_zero():
    rounds = LS.playoff_rounds(playoff_team_count=4, first_period=15,
                               matchup_period_length=2)
    with pytest.raises(LS.LeagueSimError, match="no points"):
        LS.run_bracket(seeds=(1, 2, 3, 4), rounds=rounds,
                       team_period_points={(1, 15): 10.0}, reseed=False,
                       tie_rule="NONE")


# --------------------------------------------------------------------------- #
# 7 · End to end, and the honesty properties
# --------------------------------------------------------------------------- #
def _league(**overrides):
    spec = dict(
        team_ids=EIGHT_TEAMS,
        schedule=RS14,
        regular_season_periods=tuple(range(1, 15)),
        starting_slots=SLOTS,
        playoff_team_count=4,
        matchup_period_length=2,
        reseed=False,
        seeding_rule="TOTAL_POINTS_SCORED",
        tiebreakers=("head_to_head", "points_for"),
        matchup_tie_rule="NONE",
        playoff_tie_rule="NONE",
        home_bonus=0.0,
        source_hashes={"settings": "abc123"},
    )
    spec.update(overrides)
    return LS.LeagueFormat(**spec)


def _rosters():
    rosters = {}
    for team in EIGHT_TEAMS:
        rosters[team] = tuple(
            LS.RosterSpot(player_id=team * 100 + i, name=f"T{team}P{i}",
                          eligible_slots=slots)
            for i, slots in enumerate((
                ("QB",), ("RB", "FLEX"), ("RB", "FLEX"), ("WR", "FLEX"),
                ("WR", "FLEX"), ("WR", "FLEX"), ("TE", "FLEX"), ("K",), ("D/ST",),
            ))
        )
    return rosters


def _means():
    # team 1 is the best team, team 8 the worst, monotonically
    means = {}
    for team in EIGHT_TEAMS:
        for i in range(9):
            means[team * 100 + i] = 12.0 + (8 - team) * 0.8
    return means


def test_a_full_league_runs_and_conserves_games():
    result = LS.simulate_league(
        _league(), rosters=_rosters(), player_means=_means(),
        player_sds=dict.fromkeys(_means(), 5.0),
        simulations=40, seed=6102026,
        period_basis={p: ("weekly_projection" if p == 1
                          else "rest_of_season_assumption")
                      for p in range(1, 19)},
    )
    for team in EIGHT_TEAMS:
        row = result.teams[team]
        assert row.wins + row.losses + row.ties == pytest.approx(14.0, abs=1e-9)
    assert sum(r.championship_probability for r in result.teams.values()) == pytest.approx(1.0)
    assert sum(r.made_playoffs for r in result.teams.values()) == pytest.approx(4.0)


def test_the_stronger_team_wins_more_often_than_the_weaker_one():
    result = LS.simulate_league(
        _league(), rosters=_rosters(), player_means=_means(),
        player_sds=dict.fromkeys(_means(), 5.0),
        simulations=60, seed=6102026,
        period_basis=dict.fromkeys(range(1, 19), "rest_of_season_assumption"),
    )
    assert result.teams[1].wins > result.teams[8].wins
    assert result.teams[1].championship_probability > result.teams[8].championship_probability


def test_the_same_seed_reproduces_the_run_exactly():
    kwargs = dict(rosters=_rosters(), player_means=_means(),
                  player_sds=dict.fromkeys(_means(), 5.0),
                  simulations=25, seed=99,
                  period_basis=dict.fromkeys(range(1, 19), "rest_of_season_assumption"))
    a = LS.simulate_league(_league(), **kwargs)
    b = LS.simulate_league(_league(), **kwargs)
    assert a.teams[3].championship_probability == b.teams[3].championship_probability
    assert a.teams[3].wins == b.teams[3].wins
    assert a.config_hash == b.config_hash


def test_a_different_seed_gives_a_different_run():
    kwargs = dict(rosters=_rosters(), player_means=_means(),
                  player_sds=dict.fromkeys(_means(), 5.0),
                  simulations=25,
                  period_basis=dict.fromkeys(range(1, 19), "rest_of_season_assumption"))
    a = LS.simulate_league(_league(), seed=1, **kwargs)
    b = LS.simulate_league(_league(), seed=2, **kwargs)
    assert a.seed != b.seed
    assert a.config_hash != b.config_hash


def test_source_hashes_and_the_projection_basis_survive_into_the_result():
    result = LS.simulate_league(
        _league(), rosters=_rosters(), player_means=_means(),
        player_sds=dict.fromkeys(_means(), 5.0),
        simulations=10, seed=7,
        period_basis={p: ("weekly_projection" if p <= 2
                          else "rest_of_season_assumption")
                      for p in range(1, 19)},
    )
    assert result.source_hashes == {"settings": "abc123"}
    assert result.periods_by_basis["weekly_projection"] == 2
    assert result.periods_by_basis["rest_of_season_assumption"] == 16


def test_a_period_with_no_declared_basis_fails_closed():
    """A week the model has no feature frame for is a modelling assumption.
    Leaving it unlabelled is how a projection quietly becomes a forecast."""
    with pytest.raises(LS.LeagueSimError, match="basis"):
        LS.simulate_league(
            _league(), rosters=_rosters(), player_means=_means(),
            player_sds=dict.fromkeys(_means(), 5.0),
            simulations=5, seed=7,
            period_basis={1: "weekly_projection"},
        )


def test_the_result_refuses_to_call_itself_calibrated():
    result = LS.simulate_league(
        _league(), rosters=_rosters(), player_means=_means(),
        player_sds=dict.fromkeys(_means(), 5.0),
        simulations=10, seed=7,
        period_basis=dict.fromkeys(range(1, 19), "rest_of_season_assumption"),
    )
    text = result.disclaimer.lower()
    assert "model" in text
    assert "not" in text and "calibrated" in text
    assert result.probability_kind == "model_relative_frequency"


def test_correlated_teammates_move_together_more_than_opponents():
    """A shared team week is the whole reason a league sim differs from eight
    independent player sims: teammates boom together, which fattens both tails
    of a matchup margin."""
    independent = LS.team_week_correlation_check(rho=0.0, seed=5)
    correlated = LS.team_week_correlation_check(rho=0.6, seed=5)
    assert correlated > independent


def test_an_invalid_schedule_stops_the_whole_simulation():
    broken = dict(RS14)
    broken[7] = ((1, 2), (3, 4), (5, 6), (7, 7))
    with pytest.raises(LS.ScheduleError):
        LS.simulate_league(
            _league(schedule=broken), rosters=_rosters(), player_means=_means(),
            player_sds=dict.fromkeys(_means(), 5.0),
            simulations=5, seed=7,
            period_basis=dict.fromkeys(range(1, 19), "rest_of_season_assumption"),
        )


# --------------------------------------------------------------------------- #
# 8 · Built from the real read-only ESPN contract, not a parallel structure
# --------------------------------------------------------------------------- #
@pytest.fixture
def snapshot():
    return normalize(_fixture("league_inseason_2026.json"))


def test_the_format_is_read_from_the_league_payload(snapshot):
    fmt = LS.from_snapshot(snapshot)
    assert fmt.team_ids == (1, 2, 3, 4, 5, 6, 7, 8)
    assert fmt.regular_season_periods == tuple(range(1, 15))
    assert fmt.playoff_team_count == 4
    assert fmt.reseed is False
    assert fmt.matchup_tie_rule == "NONE"
    assert fmt.source_hashes  # provenance travels with the format


def test_the_starting_lineup_is_the_leagues_own_slots_including_k_and_dst(snapshot):
    fmt = LS.from_snapshot(snapshot)
    assert fmt.starting_slots == {"QB": 1, "RB": 2, "WR": 2, "TE": 1,
                                  "FLEX": 1, "K": 1, "D/ST": 1}
    assert "BE" not in fmt.starting_slots and "IR" not in fmt.starting_slots


def test_the_real_schedule_validates_as_a_complete_pairing(snapshot):
    fmt = LS.from_snapshot(snapshot)
    report = LS.validate_schedule(fmt.team_ids, fmt.schedule,
                                  fmt.regular_season_periods)
    assert report.ok
    assert set(report.games_per_team.values()) == {14}


def test_rosters_carry_espn_slot_eligibility(snapshot):
    rosters = LS.rosters_from_snapshot(snapshot)
    assert set(rosters) == set(range(1, 9))
    every = [spot for team in rosters.values() for spot in team]
    assert all(spot.eligible_slots for spot in every)
    assert any("FLEX" in spot.eligible_slots for spot in every)


def test_the_tiebreaker_order_is_derived_from_the_declared_seeding_rule(snapshot):
    fmt = LS.from_snapshot(snapshot)
    assert fmt.seeding_rule == "TOTAL_POINTS_SCORED"
    assert fmt.tiebreakers == ("points_for",)
    assert any("seeding rule" in note for note in fmt.notes)


def test_an_untranslatable_seeding_rule_fails_closed(snapshot):
    import dataclasses
    broken = dataclasses.replace(
        snapshot,
        playoffs=dataclasses.replace(snapshot.playoffs, seeding_rule="UNKNOWN"))
    with pytest.raises(LS.PlayoffSettingsError, match="seeding rule"):
        LS.from_snapshot(broken)
    # ...but the caller may state the order, and it is recorded as supplied
    fmt = LS.from_snapshot(broken, tiebreakers=("points_for",))
    assert any("supplied by the caller" in note for note in fmt.notes)


def test_a_contradictory_playoff_period_declaration_is_caught_at_build_time(snapshot):
    """The adapter derives playoff periods in scoring-period units. A two-week
    round declared as three periods is a contradiction, and it must surface
    when the format is built rather than after a season has been simulated."""
    import dataclasses
    broken = dataclasses.replace(
        snapshot,
        playoffs=dataclasses.replace(snapshot.playoffs,
                                     matchup_period_length=2,
                                     playoff_scoring_periods=(15, 16, 17)))
    with pytest.raises(LS.PlayoffSettingsError, match="contradict"):
        LS.from_snapshot(broken)


def test_two_week_rounds_from_the_live_settings_shape_are_accepted(snapshot):
    """The live league runs two-week rounds: four scoring periods, two rounds."""
    import dataclasses
    live = dataclasses.replace(
        snapshot,
        playoffs=dataclasses.replace(snapshot.playoffs,
                                     matchup_period_length=2,
                                     playoff_matchup_periods=(15, 16, 17, 18)))
    fmt = LS.from_snapshot(live)
    rounds = LS.playoff_rounds(
        playoff_team_count=fmt.playoff_team_count, first_period=15,
        matchup_period_length=fmt.matchup_period_length,
        declared_periods=fmt.declared_playoff_periods)
    assert [r.matchup_periods for r in rounds] == [(15, 16), (17, 18)]


def test_the_real_league_simulates_end_to_end(snapshot):
    fmt = LS.from_snapshot(snapshot)
    rosters = LS.rosters_from_snapshot(snapshot)
    means, sds = {}, {}
    for team_id, team in rosters.items():
        for i, spot in enumerate(team):
            means[spot.player_id] = 10.0 + (8 - team_id) * 0.5 + (i % 3)
            sds[spot.player_id] = 4.0
    playoff_periods = range(15, 15 + 2)
    result = LS.simulate_league(
        fmt, rosters=rosters, player_means=means, player_sds=sds,
        simulations=20, seed=6102026,
        period_basis={p: ("weekly_projection" if p == 1
                          else "rest_of_season_assumption")
                      for p in list(fmt.regular_season_periods) + list(playoff_periods)},
    )
    assert set(result.teams) == set(fmt.team_ids)
    assert sum(r.championship_probability for r in result.teams.values()) == pytest.approx(1.0)
    assert result.probability_kind == "model_relative_frequency"
    assert result.source_hashes == dict(snapshot.hashes)


def test_a_roster_that_cannot_fill_the_slots_scores_nothing_for_them(snapshot):
    """Honesty guard, and a documented limitation of the committed fixture.

    Only team 1 has a realistic roster in ``league_inseason_2026.json``; teams
    2-8 are filler, every player WR-eligible. A league model that quietly
    scored their empty QB/RB/TE/K/D-ST slots would produce plausible-looking
    standings from an impossible lineup. The optimiser reports the empty slots
    instead, so the shortfall is visible in the output rather than hidden in
    it -- and so a future fixture with real rosters trips this test.
    """
    fmt = LS.from_snapshot(snapshot)
    rosters = LS.rosters_from_snapshot(snapshot)
    flat = {spot.player_id: 10.0 for team in rosters.values() for spot in team}

    complete = LS.optimal_lineup(flat, rosters[1], fmt.starting_slots)
    assert complete.empty_slots == ()
    assert complete.total == pytest.approx(90.0)          # 9 slots x 10.0

    filler = LS.optimal_lineup(flat, rosters[2], fmt.starting_slots)
    assert filler.empty_slots == ("D/ST", "K", "QB", "RB", "RB", "TE")
    assert filler.total == pytest.approx(30.0)            # WR, WR, FLEX only
    assert len(rosters[2]) == 16, "a full roster that still cannot field a lineup"
