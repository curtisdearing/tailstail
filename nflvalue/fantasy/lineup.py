"""One lineup optimizer, used for value and for legality everywhere.

There were seven. `season.lineup_points`, `trade_planner._FastLineup`,
`league_trades.LineupEvaluator`, `my_team._optimal_lineup` and
`mock_draft._best_lineup` all filled base slots greedily and then handed FLEX
whatever survived, which is not the optimum: the best FLEX is often the player
a base slot already took. All five also matched players to slots by comparing
position *names*, so a league with an `RB/WR`, `WR/TE` or `OP` seat scored that
seat as zero — silently, with no error — while `league_trades._can_fill_slots`
declared the same roster perfectly legal on the strength of the player's real
`eligible_slots`. Legality and value disagreed with each other by construction.

So: one engine. A slot is an *instance* to be filled, a player carries the
eligible-slot list ESPN publishes, and the assignment between them is solved
exactly. Nothing here knows what "FLEX" means, which is the point — the league
says which seats exist and which seats each player may occupy, and neither fact
is re-derived from a position name.

Two entry points, one algorithm:

* `optimize()` — one set of points, exact maximum-weight assignment.
* `optimize_matrix()` — a whole correlated sample matrix at once, still exact,
  returning one lineup total per simulation row so a delta can be taken
  *paired* rather than as a difference of two independently-summarised means.

An empty slot is legal in ESPN and is reported by name rather than hidden;
scoring a short roster as though it were full is how a lineup that cannot be
fielded looks fine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Mapping, Sequence

import numpy as np

#: Seats nobody starts from. Bench and IR are roster capacity, not lineup value.
NON_SCORING_SLOTS = frozenset({"BE", "IR"})

#: Below any real difference in fantasy points, and used only to order ties.
_TIE_EPSILON = 1e-9
_INELIGIBLE = -1e18

#: Above this many candidate count-vectors the closed-form enumeration stops
#: paying for itself and the per-row solver is used instead. Both are exact;
#: the tests assert they agree.
_MAX_ENUMERATION = 20_000


class LineupError(ValueError):
    """The lineup problem is malformed — never worked around silently."""


@dataclass(frozen=True)
class Seat:
    """One startable slot instance. Two FLEX seats are two Seats."""

    label: str
    index: int


@dataclass(frozen=True)
class LineupPlayer:
    """A player as the optimizer sees him: an id and where he may be seated."""

    player_id: object
    eligible_slots: frozenset[str]
    position: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.eligible_slots, frozenset):
            object.__setattr__(self, "eligible_slots", frozenset(self.eligible_slots))


@dataclass(frozen=True)
class Lineup:
    """The chosen assignment, its value, and what could not be filled."""

    total: float
    assignment: Mapping[str, tuple]
    empty_slots: tuple[str, ...]
    benched: tuple = field(default=())

    @property
    def is_full(self) -> bool:
        return not self.empty_slots


def seats(slot_counts: Mapping[str, int]) -> tuple[Seat, ...]:
    """Expand `{label: count}` into individual seats, bench and IR removed."""
    out: list[Seat] = []
    for label in sorted(slot_counts):
        count = int(slot_counts[label])
        if label in NON_SCORING_SLOTS or count <= 0:
            continue
        for index in range(count):
            out.append(Seat(label=str(label), index=index))
    return tuple(out)


def as_players(entries: Iterable[Mapping[str, object]], *,
               id_key: str = "player_id",
               slots_key: str = "eligible_slots",
               position_key: str = "position") -> tuple[LineupPlayer, ...]:
    """Adapt roster dicts to the optimizer's view. Eligibility is required.

    A player with no eligible-slot list cannot be seated by *this* engine, and
    guessing one from his position is the substitution that made composite
    slots vanish. He is refused here so the caller has to say what it knows.
    """
    players: list[LineupPlayer] = []
    for entry in entries:
        eligible = entry.get(slots_key)
        if not eligible:
            raise LineupError(
                f"player {entry.get(id_key)!r} carries no {slots_key}; refusing to infer "
                "eligibility from a position name, which is what silently dropped "
                "RB/WR, WR/TE and OP seats")
        players.append(LineupPlayer(
            player_id=entry.get(id_key),
            eligible_slots=frozenset(str(slot) for slot in eligible),
            position=(str(entry[position_key]) if entry.get(position_key) else None),
        ))
    return tuple(players)



# --------------------------------------------------------------------------- #
# Eligibility for callers that only know a position
# --------------------------------------------------------------------------- #
#: ESPN's standard slot eligibility, by position. Used only where a caller has
#: positions and no per-player eligible-slot list -- the season and trade paths
#: work from a projection frame, which carries a position and nothing else.
#: Snapshot-driven callers pass the league's real `eligible_slots` and never
#: consult this table, because a player's true eligibility is ESPN's answer,
#: not a lookup from his listed position.
SLOT_ELIGIBILITY: Mapping[str, frozenset[str]] = {
    "QB": frozenset({"QB", "OP"}),
    "RB": frozenset({"RB", "RB/WR", "FLEX", "OP"}),
    "WR": frozenset({"WR", "RB/WR", "WR/TE", "FLEX", "OP"}),
    "TE": frozenset({"TE", "WR/TE", "FLEX", "OP"}),
    "K": frozenset({"K"}),
    "D/ST": frozenset({"D/ST"}),
    "DST": frozenset({"D/ST"}),
}


def from_positions(rows: Iterable[tuple[object, str]], slot_counts: Mapping[str, int],
                   *, flex_positions: Sequence[str] | None = None
                   ) -> tuple[LineupPlayer, ...]:
    """Players built from `(player_id, position)` and the league's own seats.

    `flex_positions` overrides which positions may take a `FLEX` seat when a
    league declares something non-standard; everything else follows
    `SLOT_ELIGIBILITY`. A position with no entry gets the seat that shares its
    name if the league declares one, and otherwise no seat at all -- which
    surfaces as an empty slot rather than as a silent zero.
    """
    declared = {str(label) for label in slot_counts if label not in NON_SCORING_SLOTS}
    flex_override = None if flex_positions is None else {str(p) for p in flex_positions}
    players: list[LineupPlayer] = []
    for player_id, position in rows:
        name = str(position)
        eligible = set(SLOT_ELIGIBILITY.get(name, frozenset({name})))
        if flex_override is not None:
            eligible.discard("FLEX")
            if name in flex_override:
                eligible.add("FLEX")
        players.append(LineupPlayer(player_id=player_id,
                                    eligible_slots=frozenset(eligible & declared),
                                    position=name))
    return tuple(players)


def seat_breadth(label: str) -> int:
    """How many positions can fill this seat, league-wide.

    Used only to order ties. A base seat (breadth 1) is preferred to a `FLEX`
    (3) or `OP` (4) for the same player, and when the general seat must be
    used it takes the smaller score -- which is the lineup a person writes
    down, with the big numbers in the seats only they fit. Unknown labels are
    treated as specific, because guessing a seat is general is the assumption
    that lets a composite seat swallow a base starter.
    """
    return max(sum(1 for slots in SLOT_ELIGIBILITY.values() if label in slots), 1)


# --------------------------------------------------------------------------- #
# Legality
# --------------------------------------------------------------------------- #
def _reachable(signature: frozenset[str], seat_list: Sequence[Seat]) -> int:
    return sum(1 for seat in seat_list if seat.label in signature)


def can_fill(players: Sequence[LineupPlayer], slot_counts: Mapping[str, int]) -> bool:
    """Can these players fill every startable seat? Exact, via Hall's condition.

    Asked from the seat side: for every set of seat labels, at least as many
    players must be eligible for one of them as there are seats to fill.
    Counting per position is not enough once a seat accepts more than one
    position -- three running backs and no receiver can look fine by totals and
    still be unable to field a lineup. Having *more* players than seats is
    normal and is not a failure; the question is coverage, not consumption.
    """
    seat_list = seats(slot_counts)
    if not seat_list:
        return True
    labels = sorted({seat.label for seat in seat_list})
    demand = {label: sum(1 for seat in seat_list if seat.label == label) for label in labels}
    for mask in range(1, 1 << len(labels)):
        chosen = {labels[bit] for bit in range(len(labels)) if mask & (1 << bit)}
        needed = sum(demand[label] for label in chosen)
        available = sum(1 for player in players if player.eligible_slots & chosen)
        if available < needed:
            return False
    return True


def _signature_counts(players: Sequence[LineupPlayer]) -> dict[frozenset[str], int]:
    counts: dict[frozenset[str], int] = {}
    for player in players:
        counts[player.eligible_slots] = counts.get(player.eligible_slots, 0) + 1
    return counts


def _feasible(taken: Mapping[frozenset[str], int], seat_list: Sequence[Seat],
              *, required: int | None = None) -> bool:
    """Hall's condition for a bipartite b-matching of players onto seats.

    For every set of eligibility signatures, the players drawn from them must
    not outnumber the seats any of them can reach. Signatures are few (a real
    league has a handful), so checking every subset is exact and instant.
    """
    signatures = [sig for sig, count in taken.items() if count > 0]
    total = sum(taken.values())
    if required is not None and total < required:
        return False
    if total > len(seat_list):
        return False
    for mask in range(1, 1 << len(signatures)):
        union: set[str] = set()
        demand = 0
        for bit, signature in enumerate(signatures):
            if mask & (1 << bit):
                union |= signature
                demand += taken[signature]
        if demand > _reachable(frozenset(union), seat_list):
            return False
    return True


# --------------------------------------------------------------------------- #
# Value — one set of points
# --------------------------------------------------------------------------- #
def optimize(points: Mapping[object, float], players: Sequence[LineupPlayer],
             slot_counts: Mapping[str, int]) -> Lineup:
    """The best legal lineup for one set of points, solved exactly.

    A maximum-weight assignment between players and seats, so the answer is
    optimal by construction rather than by the luck of a fill order. Ties break
    toward the more restrictive seat, deterministically, so the same roster and
    the same points always produce the same lineup — the one a person would
    write down, with the big scores in the base seats.
    """
    from scipy.optimize import linear_sum_assignment

    seat_list = seats(slot_counts)
    ordered = sorted(players, key=lambda p: str(p.player_id))
    if not seat_list:
        return Lineup(total=0.0, assignment={}, empty_slots=(),
                      benched=tuple(p.player_id for p in ordered))
    if not ordered:
        return Lineup(total=0.0, assignment=dict.fromkeys(slot_counts, ()),
                      empty_slots=tuple(sorted(seat.label for seat in seat_list)))

    # Ties are ordered by seat breadth, scaled by the score, so a general seat
    # ends up with the smallest eligible number and the specific seats keep the
    # big ones. The nudge is ~1e-7 points against differences of ~0.01, so it
    # orders equal lineups without ever changing which lineup is optimal.
    penalty = {seat.label: seat_breadth(seat.label) for seat in seat_list}
    matrix = np.full((len(ordered), len(seat_list)), _INELIGIBLE, dtype=float)
    for i, player in enumerate(ordered):
        value = float(points.get(player.player_id, 0.0))
        for j, seat in enumerate(seat_list):
            if seat.label in player.eligible_slots:
                matrix[i, j] = value - _TIE_EPSILON * penalty[seat.label] * (1.0 + abs(value))

    rows, cols = linear_sum_assignment(matrix, maximize=True)

    assignment: dict[str, list] = {
        str(label): [] for label in slot_counts if label not in NON_SCORING_SLOTS}
    filled = [False] * len(seat_list)
    seated: set = set()
    total = 0.0
    for i, j in zip(rows, cols):
        if matrix[i, j] <= _INELIGIBLE / 2:
            continue
        value = float(points.get(ordered[i].player_id, 0.0))
        # An empty seat is legal; a seat filled by a negative score is a choice
        # nobody would make, so the solver is allowed to leave it open.
        if value < 0.0:
            continue
        assignment.setdefault(seat_list[j].label, []).append(ordered[i].player_id)
        seated.add(ordered[i].player_id)
        filled[j] = True
        total += value

    empty = tuple(sorted(seat.label for j, seat in enumerate(seat_list) if not filled[j]))
    return Lineup(
        total=total,
        assignment={label: tuple(ids) for label, ids in assignment.items()},
        empty_slots=empty,
        benched=tuple(p.player_id for p in ordered if p.player_id not in seated),
    )


# --------------------------------------------------------------------------- #
# Value — a whole correlated sample matrix, still exact
# --------------------------------------------------------------------------- #
def _feasible_vectors(signatures: Sequence[frozenset[str]],
                      supply: Sequence[int],
                      seat_list: Sequence[Seat]) -> list[tuple[int, ...]]:
    """Every count vector that can actually be seated.

    Players sharing an eligibility signature are interchangeable, so an optimal
    lineup is fully described by how many of each signature it seats plus each
    signature's own ranking. Real leagues have a handful of signatures, so the
    set of seatable count vectors is small and is computed once per roster
    shape rather than once per simulation.
    """
    ranges = [range(min(int(count), len(seat_list)) + 1) for count in supply]
    out: list[tuple[int, ...]] = []
    for combo in product(*ranges):
        if sum(combo) > len(seat_list):
            continue
        taken = {signatures[i]: combo[i] for i in range(len(signatures))}
        if _feasible(taken, seat_list):
            out.append(combo)
    return out


def optimize_matrix(points: np.ndarray, players: Sequence[LineupPlayer],
                    slot_counts: Mapping[str, int]) -> np.ndarray:
    """One optimal lineup total per simulation row, exactly.

    `points` is `(simulations, players)` aligned to `players`. The result is a
    vector of the same length as `points`, so a comparison between two rosters
    is a **paired** per-row difference: the same latent week, the same game
    script, both lineups. A difference of two independently summarised means
    throws that pairing away and cannot answer "how often is this actually
    better".
    """
    matrix = np.asarray(points, dtype=float)
    if matrix.ndim != 2:
        raise LineupError(f"points must be (simulations, players); got shape {matrix.shape}")
    if matrix.shape[1] != len(players):
        raise LineupError(
            f"points has {matrix.shape[1]} player columns but {len(players)} players")
    seat_list = seats(slot_counts)
    simulations = matrix.shape[0]
    if not seat_list or not players:
        return np.zeros(simulations, dtype=float)

    groups: dict[frozenset[str], list[int]] = {}
    for index, player in enumerate(players):
        groups.setdefault(player.eligible_slots, []).append(index)
    signatures = sorted(groups, key=lambda sig: sorted(sig))
    supply = [len(groups[sig]) for sig in signatures]

    vectors = _feasible_vectors(signatures, supply, seat_list)
    if not vectors:
        return np.zeros(simulations, dtype=float)
    if len(vectors) > _MAX_ENUMERATION:
        return np.array([
            optimize({players[i].player_id: matrix[row, i] for i in range(len(players))},
                     players, slot_counts).total
            for row in range(simulations)
        ], dtype=float)

    # Per signature: the sum of its top k scores, for every k, every row. A
    # negative score is never worth seating, so it is floored at zero before
    # the cumulative sum -- which is the same thing as leaving the seat empty.
    cumulative: list[np.ndarray] = []
    for signature in signatures:
        columns = matrix[:, groups[signature]]
        ordered = -np.sort(-np.maximum(columns, 0.0), axis=1)
        sums = np.concatenate(
            [np.zeros((simulations, 1)), np.cumsum(ordered, axis=1)], axis=1)
        cumulative.append(sums)

    best = np.full(simulations, -np.inf)
    for combo in vectors:
        total = np.zeros(simulations)
        for i, take in enumerate(combo):
            total += cumulative[i][:, take]
        np.maximum(best, total, out=best)
    return best


def lineup_value(points: Mapping[object, float], players: Sequence[LineupPlayer],
                 slot_counts: Mapping[str, int]) -> float:
    """Scalar convenience over `optimize` — the same algorithm, never a second one."""
    return optimize(points, players, slot_counts).total
