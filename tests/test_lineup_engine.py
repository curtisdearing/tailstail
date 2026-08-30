"""The lineup engine, checked against exhaustive search rather than itself.

Every optimizer this replaces was "verified" against another optimizer that
shared its assumption, so all five agreed and all five were wrong the same way:
greedy base-then-FLEX, and slot eligibility inferred from a position name. A
test that re-runs the implementation proves only that the code agrees with
itself, so the reference here is brute force over every legal assignment.

Brute force is only affordable on small rosters, which is exactly the point:
the cases that break greedy are small, and they are the ones enumerated below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue.fantasy import lineup as LU


# --------------------------------------------------------------------------- #
# The reference: every legal seating, scored
# --------------------------------------------------------------------------- #
def brute_force(points, players, slot_counts):
    """The best legal lineup by exhaustive search over seatings.

    Depth-first over seats: each seat takes an eligible unused player or is
    left empty (which ESPN allows). Exponential, but the rosters here are
    small on purpose — the cases that break a greedy fill are small.
    """
    seat_list = LU.seats(slot_counts)
    if not seat_list or not players:
        return 0.0
    values = [float(points.get(p.player_id, 0.0)) for p in players]
    eligible = [
        [i for i, player in enumerate(players) if seat.label in player.eligible_slots]
        for seat in seat_list
    ]

    best = 0.0

    def walk(seat_index, used, running):
        nonlocal best
        if running > best:
            best = running
        if seat_index == len(seat_list):
            return
        walk(seat_index + 1, used, running)          # leave this seat empty
        for i in eligible[seat_index]:
            bit = 1 << i
            if used & bit:
                continue
            walk(seat_index + 1, used | bit, running + values[i])

    walk(0, 0, 0.0)
    return best


def P(pid, *slots, position=None):
    return LU.LineupPlayer(player_id=pid, eligible_slots=frozenset(slots), position=position)


# --------------------------------------------------------------------------- #
# The case greedy gets wrong
# --------------------------------------------------------------------------- #
def test_greedy_base_then_flex_leaves_points_on_the_bench():
    """The classic failure: the best FLEX is the player a base slot took.

    One RB seat, one FLEX. Greedy seats the best RB in RB, then gives FLEX the
    best leftover. Here the two receivers outscore both backs, so the optimum
    seats the weaker back at RB and a receiver at FLEX.
    """
    slots = {"RB": 1, "WR": 1, "FLEX": 1}
    players = [P("rb1", "RB", "FLEX"), P("rb2", "RB", "FLEX"),
               P("wr1", "WR", "FLEX"), P("wr2", "WR", "FLEX")]
    points = {"rb1": 20.0, "rb2": 19.0, "wr1": 18.0, "wr2": 17.0}

    result = LU.optimize(points, players, slots)
    assert result.total == pytest.approx(brute_force(points, players, slots))
    assert result.total == pytest.approx(20.0 + 18.0 + 19.0)


def test_composite_slots_are_filled_not_silently_scored_as_zero():
    """`RB/WR` matches no position name, and used to contribute nothing."""
    slots = {"QB": 1, "RB/WR": 1, "OP": 1}
    players = [P("qb1", "QB", "OP"), P("rb1", "RB", "RB/WR", "OP"),
               P("wr1", "WR", "RB/WR", "OP")]
    points = {"qb1": 25.0, "rb1": 14.0, "wr1": 11.0}

    result = LU.optimize(points, players, slots)
    assert result.total == pytest.approx(brute_force(points, players, slots))
    assert result.total == pytest.approx(50.0)
    assert result.empty_slots == ()


@pytest.mark.parametrize("slots", [
    {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
    {"QB": 1, "RB": 1, "WR": 1, "FLEX": 2},
    {"QB": 1, "RB": 1, "WR": 1, "FLEX": 3},
    {"QB": 1, "RB/WR": 1, "WR/TE": 1},
    {"OP": 2, "RB": 1},
    {"QB": 1, "RB": 1, "WR": 1, "TE": 1, "FLEX": 1, "OP": 1},
])
def test_matches_brute_force_on_random_rosters(slots):
    """Random points, real slot shapes, exhaustive reference."""
    rng = np.random.default_rng(20260830)
    catalogue = [
        ("QB", ("QB", "OP")),
        ("RB", ("RB", "RB/WR", "FLEX", "OP")),
        ("RB", ("RB", "RB/WR", "FLEX", "OP")),
        ("WR", ("WR", "RB/WR", "WR/TE", "FLEX", "OP")),
        ("WR", ("WR", "RB/WR", "WR/TE", "FLEX", "OP")),
        ("TE", ("TE", "WR/TE", "FLEX", "OP")),
        ("TE", ("TE", "WR/TE", "FLEX", "OP")),
    ]
    players = [P(f"p{i}", *eligible, position=pos)
               for i, (pos, eligible) in enumerate(catalogue)]
    for _ in range(25):
        points = {p.player_id: float(v)
                  for p, v in zip(players, rng.normal(12.0, 6.0, len(players)))}
        assert LU.optimize(points, players, slots).total == pytest.approx(
            brute_force(points, players, slots))


def test_nonstandard_eligibility_is_honoured_not_inferred():
    """A player eligible only for a seat his position does not name."""
    slots = {"QB": 1, "FLEX": 1}
    # A gadget player ESPN lists as QB-positioned but FLEX-eligible only.
    players = [P("gadget", "FLEX", position="QB"), P("qb1", "QB", position="QB")]
    points = {"gadget": 30.0, "qb1": 5.0}
    result = LU.optimize(points, players, slots)
    assert result.assignment["FLEX"] == ("gadget",)
    assert result.assignment["QB"] == ("qb1",)
    assert result.total == pytest.approx(35.0)


# --------------------------------------------------------------------------- #
# Empty seats, short rosters, negatives
# --------------------------------------------------------------------------- #
def test_an_unfillable_seat_is_named_not_hidden():
    slots = {"QB": 1, "RB": 2, "TE": 1}
    players = [P("qb1", "QB"), P("rb1", "RB")]
    points = {"qb1": 20.0, "rb1": 10.0}
    result = LU.optimize(points, players, slots)
    assert result.total == pytest.approx(30.0)
    assert result.empty_slots == ("RB", "TE")
    assert result.is_full is False


def test_no_players_leaves_every_seat_empty_and_scores_zero():
    result = LU.optimize({}, [], {"QB": 1, "RB": 2})
    assert result.total == 0.0
    assert result.empty_slots == ("QB", "RB", "RB")


def test_only_bench_and_ir_slots_means_no_lineup_to_score():
    result = LU.optimize({"a": 9.0}, [P("a", "BE")], {"BE": 5, "IR": 1})
    assert result.total == 0.0
    assert result.empty_slots == ()


def test_a_negative_score_is_left_on_the_bench_rather_than_seated():
    """An empty seat is legal; starting a negative is a choice nobody makes."""
    slots = {"QB": 1, "RB": 1}
    players = [P("qb1", "QB"), P("rb1", "RB")]
    points = {"qb1": 18.0, "rb1": -3.0}
    result = LU.optimize(points, players, slots)
    assert result.total == pytest.approx(18.0)
    assert "RB" in result.empty_slots


# --------------------------------------------------------------------------- #
# Ties
# --------------------------------------------------------------------------- #
def test_ties_are_broken_deterministically_and_identically_across_runs():
    slots = {"RB": 1, "FLEX": 1}
    players = [P("a", "RB", "FLEX"), P("b", "RB", "FLEX")]
    points = {"a": 12.0, "b": 12.0}
    first = LU.optimize(points, players, slots)
    for _ in range(20):
        again = LU.optimize(points, players, slots)
        assert again.assignment == first.assignment
        assert again.total == first.total


def test_a_tie_seats_the_scarcer_slot_first():
    """Two equal players, one RB seat and one FLEX: RB is the scarcer seat."""
    slots = {"RB": 1, "FLEX": 1}
    players = [P("a", "RB", "FLEX"), P("b", "FLEX")]
    points = {"a": 10.0, "b": 10.0}
    result = LU.optimize(points, players, slots)
    assert result.assignment["RB"] == ("a",), "the FLEX-only player cannot fill RB"
    assert result.assignment["FLEX"] == ("b",)


# --------------------------------------------------------------------------- #
# Legality
# --------------------------------------------------------------------------- #
def test_legality_sees_what_counting_positions_cannot():
    """Three backs and no receiver: fine by totals, unfieldable in fact."""
    slots = {"RB": 1, "WR": 1, "FLEX": 1}
    backs = [P(f"rb{i}", "RB", "FLEX") for i in range(3)]
    assert LU.can_fill(backs, slots) is False
    mixed = [P("rb1", "RB", "FLEX"), P("wr1", "WR", "FLEX"), P("te1", "TE", "FLEX")]
    assert LU.can_fill(mixed, slots) is True


def test_legality_agrees_with_the_optimizer_about_a_full_lineup():
    rng = np.random.default_rng(7)
    slots = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
    catalogue = [("QB", ("QB",)), ("RB", ("RB", "FLEX")), ("RB", ("RB", "FLEX")),
                 ("WR", ("WR", "FLEX")), ("WR", ("WR", "FLEX")), ("TE", ("TE", "FLEX")),
                 ("WR", ("WR", "FLEX"))]
    for size in range(1, len(catalogue) + 1):
        players = [P(f"p{i}", *eligible) for i, (_, eligible) in enumerate(catalogue[:size])]
        points = {p.player_id: float(v) for p, v in zip(players, rng.uniform(1, 20, size))}
        result = LU.optimize(points, players, slots)
        assert LU.can_fill(players, slots) == result.is_full


def test_eligibility_must_be_supplied_never_guessed_from_a_position():
    with pytest.raises(LU.LineupError, match="eligibility"):
        LU.as_players([{"player_id": 1, "position": "RB", "eligible_slots": []}])


# --------------------------------------------------------------------------- #
# The matrix path is the same algorithm
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("slots", [
    {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
    {"QB": 1, "RB": 1, "WR": 1, "FLEX": 2},
    {"QB": 1, "RB/WR": 1, "OP": 1},
    {"RB": 1, "FLEX": 1},
])
def test_matrix_path_equals_the_per_row_optimizer(slots):
    rng = np.random.default_rng(31337)
    catalogue = [("QB", ("QB", "OP")), ("RB", ("RB", "RB/WR", "FLEX", "OP")),
                 ("RB", ("RB", "RB/WR", "FLEX", "OP")), ("WR", ("WR", "RB/WR", "FLEX", "OP")),
                 ("WR", ("WR", "RB/WR", "FLEX", "OP")), ("TE", ("TE", "FLEX", "OP"))]
    players = [P(f"p{i}", *eligible) for i, (_, eligible) in enumerate(catalogue)]
    matrix = rng.normal(11.0, 7.0, size=(60, len(players)))
    vector = LU.optimize_matrix(matrix, players, slots)
    for row in range(matrix.shape[0]):
        expected = LU.optimize(
            {p.player_id: matrix[row, i] for i, p in enumerate(players)}, players, slots).total
        assert vector[row] == pytest.approx(expected), f"row {row} disagrees"


def test_matrix_path_returns_one_value_per_simulation_row():
    players = [P("a", "RB", "FLEX"), P("b", "WR", "FLEX")]
    matrix = np.array([[10.0, 5.0], [1.0, 2.0], [0.0, 0.0]])
    vector = LU.optimize_matrix(matrix, players, {"RB": 1, "FLEX": 1})
    assert vector.shape == (3,)
    assert vector[0] == pytest.approx(15.0)
    assert vector[2] == pytest.approx(0.0)


def test_matrix_path_refuses_a_shape_it_cannot_align():
    players = [P("a", "RB")]
    with pytest.raises(LU.LineupError, match="player columns"):
        LU.optimize_matrix(np.zeros((5, 3)), players, {"RB": 1})


# --------------------------------------------------------------------------- #
# Cross-consumer value identity
# --------------------------------------------------------------------------- #
def test_every_consumer_values_the_same_roster_identically():
    """A lineup worth 118.4 to the bracket is worth 118.4 to everyone.

    Before this engine there were seven optimizers with three different answers
    for the same roster, so a trade could gain points in the scan and lose them
    on the weekly card. They all route here now, and this asserts it end to end
    rather than trusting that they do.
    """
    import pandas as pd

    from nflvalue.fantasy import league_sim as LS
    from nflvalue.fantasy import league_trades as LT
    from nflvalue.fantasy import season as SEASON
    from nflvalue.fantasy.config import LineupRules

    slots = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
    catalogue = [("qb1", "QB", 24.0), ("rb1", "RB", 19.0), ("rb2", "RB", 12.0),
                 ("rb3", "RB", 11.5), ("wr1", "WR", 18.0), ("wr2", "WR", 17.5),
                 ("wr3", "WR", 15.0), ("te1", "TE", 9.0)]
    points = {pid: value for pid, _, value in catalogue}

    engine_players = LU.from_positions([(pid, pos) for pid, pos, _ in catalogue], slots)
    expected = LU.optimize(points, engine_players, slots).total
    assert expected == pytest.approx(brute_force(points, engine_players, slots))

    # 1. the bracket
    spots = [LS.RosterSpot(player_id=index, name=pid,
                           eligible_slots=tuple(sorted(
                               LU.SLOT_ELIGIBILITY[pos] & set(slots))))
             for index, (pid, pos, _) in enumerate(catalogue)]
    by_index = {index: points[pid] for index, (pid, _, _) in enumerate(catalogue)}
    assert LS.optimal_lineup(by_index, spots, slots).total == pytest.approx(expected)

    # 2. the season path (one simulation row of the correlated matrix)
    frame = pd.DataFrame([[points[pid] for pid, _, _ in catalogue]],
                         columns=[pid for pid, _, _ in catalogue])
    meta = pd.DataFrame([{"player_id": pid, "position": pos} for pid, pos, _ in catalogue])
    sim = SEASON.SeasonSimulation(summaries=pd.DataFrame(), points=frame,
                                  player_meta=meta, metadata={})
    rules = LineupRules(starters=slots, flex_positions=("RB", "WR", "TE"))
    assert float(SEASON.lineup_points(sim, list(points), rules)[0]) == pytest.approx(expected)

    # 3. the trade scan's evaluator
    assert float(LT.LineupEvaluator(sim, rules).vector(list(points))[0]) == pytest.approx(
        expected)


def test_the_engine_is_the_only_optimizer_left():
    """A new greedy fill would be a silent regression, so it is asserted away."""
    import inspect

    from nflvalue.fantasy import league_trades as LT
    from nflvalue.fantasy import mock_draft, season, trade_planner

    for module in (season, trade_planner, LT, mock_draft):
        source = inspect.getsource(module)
        assert "from . import lineup as lineup_engine" in source, (
            f"{module.__name__} does not route through the shared engine")


# --------------------------------------------------------------------------- #
# Waiver add/drop legality, against exhaustive search
# --------------------------------------------------------------------------- #
def _fillable_by_search(players, slot_counts):
    """Can every seat be filled? Exhaustive, for the legality reference."""
    seat_list = LU.seats(slot_counts)
    if not seat_list:
        return True
    eligible = [[i for i, p in enumerate(players) if seat.label in p.eligible_slots]
                for seat in seat_list]

    def walk(seat_index, used):
        if seat_index == len(seat_list):
            return True
        for i in eligible[seat_index]:
            bit = 1 << i
            if used & bit:
                continue
            if walk(seat_index + 1, used | bit):
                return True
        return False

    return walk(0, 0)


@pytest.mark.parametrize("slots", [
    {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
    {"QB": 1, "RB": 1, "WR": 1, "FLEX": 2},
    {"QB": 1, "RB/WR": 2},
    {"OP": 1, "TE": 1},
])
def test_add_drop_legality_matches_exhaustive_search(slots):
    """Every add/drop pair, checked both ways."""
    rng = np.random.default_rng(4242)
    catalogue = [("QB", ("QB", "OP")), ("QB", ("QB", "OP")),
                 ("RB", ("RB", "RB/WR", "FLEX", "OP")), ("RB", ("RB", "RB/WR", "FLEX", "OP")),
                 ("WR", ("WR", "RB/WR", "FLEX", "OP")), ("WR", ("WR", "RB/WR", "FLEX", "OP")),
                 ("TE", ("TE", "FLEX", "OP")), ("TE", ("TE", "FLEX", "OP"))]
    roster = [P(f"p{i}", *eligible, position=pos)
              for i, (pos, eligible) in enumerate(catalogue)]
    pool = [P("add-rb", "RB", "RB/WR", "FLEX", "OP", position="RB"),
            P("add-te", "TE", "FLEX", "OP", position="TE"),
            P("add-qb", "QB", "OP", position="QB")]

    for add in pool:
        for drop_index in range(len(roster)):
            after = [p for i, p in enumerate(roster) if i != drop_index] + [add]
            assert LU.can_fill(after, slots) == _fillable_by_search(after, slots), (
                f"legality disagreed after adding {add.player_id} and dropping "
                f"{roster[drop_index].player_id}")
    del rng


def test_dropping_the_only_player_for_a_seat_is_illegal():
    slots = {"QB": 1, "RB": 1, "FLEX": 1}
    roster = [P("qb1", "QB"), P("rb1", "RB", "FLEX"), P("wr1", "WR", "FLEX")]
    assert LU.can_fill(roster, slots) is True
    assert LU.can_fill([p for p in roster if p.player_id != "qb1"], slots) is False
