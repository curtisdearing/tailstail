"""Eight-team head-to-head league simulation.

This is the piece the fantasy engine did not have. ``simulation.py`` answers
"how many points will this player score"; ``season.py`` aggregates that into
rest-of-season value. Neither answers the question the league actually poses:
*given this schedule, these rosters, these lineup rules, this tiebreaker order
and this bracket, how often does my team win the thing?* Season totals do not
decide that -- a schedule does, one week at a time, against one opponent at a
time, and then a two-week bracket does the rest.

What this module reads, and from where
--------------------------------------
Every structural fact comes from the read-only ESPN league contract
(:mod:`nflvalue.fantasy.espn_league`), never from an assumption:

* team ids and count            -> ``LeagueSnapshot.teams``
* the weekly matchup schedule   -> ``LeagueSnapshot.schedule``
* regular-season length         -> ``Playoffs.regular_season_matchup_periods``
* legal lineup slots            -> ``RosterSettings.lineup_slot_counts``
* per-player slot eligibility   -> ``RosterPlayer.eligible_slots``
* standings + tiebreaker order  -> ``Standings.tiebreaker``
* qualifiers, seeding, reseed   -> ``Playoffs``
* round length (two-week rounds)-> ``Playoffs.matchup_period_length``
* tie rules and home bonus      -> ``Scoring``
* provenance                    -> ``LeagueSnapshot.hashes``

Where the payload is silent or self-contradictory this module raises. It never
picks a plausible default for a rule that decides who advances: a guessed
reseed flag or a guessed round length changes the champion, and a wrong answer
delivered confidently is worse than a refusal.

What is modelled, and therefore assumed
---------------------------------------
Three things are ours, not ESPN's, and each is named in the result so nobody
mistakes them for league facts:

1. **Weekly player outcomes.** Points are drawn from per-player mean/SD with a
   one-factor within-team correlation: teammates share a team-week factor
   (game script, pace, blowouts) on top of individual noise. Real fantasy
   weeks are correlated this way, and ignoring it understates both tails of a
   matchup margin. ``rho`` is a parameter, not a measurement.
2. **Rest-of-season weeks.** A week with no feature frame cannot have a
   projection; it has an assumption. Every period must be labelled
   ``weekly_projection`` or ``rest_of_season_assumption`` and the counts ride
   in the result.
3. **Lineup behaviour.** Every manager is assumed to start their optimal legal
   lineup with hindsight of that week's scores. That is an upper bound on real
   lineup-setting, and it is applied identically to all eight teams.

The probabilities this returns are **model-relative frequencies**: the share of
simulated seasons in which something happened, under the assumptions above.
They are not calibrated real-world probabilities and the payload says so.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Errors -- all fail closed
# --------------------------------------------------------------------------- #
class LeagueSimError(RuntimeError):
    """A league cannot be simulated as described."""


class ScheduleError(LeagueSimError):
    """The schedule is incomplete, non-reciprocal, or references unknown teams."""


class PlayoffSettingsError(LeagueSimError):
    """Playoff settings are incomplete or contradict each other."""


#: Regular-season tie rules we understand. Anything else stops the run.
MATCHUP_TIE_RULES = frozenset({"NONE", "HOME", "AWAY"})
#: Bracket tie rules. ``HIGHER_SEED`` resolves to the home side, because the
#: bracket always seats the better seed at home.
BRACKET_TIE_RULES = frozenset({"NONE", "HOME", "AWAY", "HIGHER_SEED"})

#: Tiebreakers this module can actually compute. An unknown key is refused
#: rather than skipped -- silently dropping a tiebreaker reorders a playoff field.
TIEBREAKERS = ("head_to_head", "points_for", "points_against")

WEEKLY_PROJECTION = "weekly_projection"
REST_OF_SEASON = "rest_of_season_assumption"
BASES = frozenset({WEEKLY_PROJECTION, REST_OF_SEASON})

PROBABILITY_KIND = "model_relative_frequency"
DISCLAIMER = (
    "Model-relative frequencies: the share of simulated seasons in which this "
    "happened, under this module's stated assumptions. These are NOT calibrated "
    "real-world probabilities and must not be presented as certainty."
)

#: Slot preference used only to break exact ties in lineup assignment, so a
#: bench-equivalent player lands in the specific slot rather than an arbitrary
#: one. It is far below any real point difference and never changes the total.
_TIE_EPSILON = 1e-9
_UNELIGIBLE = -1e18

_ROUND_NAMES = {1: ("final",),
                2: ("semifinal", "final"),
                3: ("quarterfinal", "semifinal", "final"),
                4: ("round_of_16", "quarterfinal", "semifinal", "final")}


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RosterSpot:
    """One rosterable player, carrying the slots ESPN says he may fill."""

    player_id: int
    name: str
    eligible_slots: tuple[str, ...]


@dataclass(frozen=True)
class Lineup:
    total: float
    assignment: Mapping[str, tuple[int, ...]]
    empty_slots: tuple[str, ...]


@dataclass(frozen=True)
class ScheduleReport:
    ok: bool
    games_per_team: Mapping[int, int]
    periods: tuple[int, ...]


@dataclass(frozen=True)
class TeamRecord:
    team_id: int
    wins: int
    losses: int
    ties: int
    points_for: float
    points_against: float

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def win_pct(self) -> float:
        return (self.wins + 0.5 * self.ties) / self.games if self.games else 0.0


@dataclass(frozen=True)
class StandingRow:
    team_id: int
    record: TeamRecord
    rank: int
    #: Teams this one was still level with after every declared tiebreaker was
    #: exhausted. A non-empty value means the final order was arbitrary and the
    #: reader should know it.
    unbroken_tie_with: tuple[int, ...]


@dataclass(frozen=True)
class PlayoffRound:
    name: str
    matchup_periods: tuple[int, ...]
    byes: tuple[int, ...] = ()


@dataclass(frozen=True)
class RoundOutcome:
    name: str
    pairings: tuple[tuple[int, int], ...]
    winners: tuple[int, ...]
    byes: tuple[int, ...]


@dataclass(frozen=True)
class BracketOutcome:
    rounds: tuple[RoundOutcome, ...]
    champion: int


@dataclass(frozen=True)
class LeagueFormat:
    """Everything structural, already validated, that a season needs."""

    team_ids: tuple[int, ...]
    schedule: Mapping[int, tuple[tuple[int, int], ...]]
    regular_season_periods: tuple[int, ...]
    starting_slots: Mapping[str, int]
    playoff_team_count: int
    matchup_period_length: int
    reseed: bool
    seeding_rule: str
    tiebreakers: tuple[str, ...]
    matchup_tie_rule: str
    playoff_tie_rule: str
    home_bonus: float = 0.0
    source_hashes: Mapping[str, str] = field(default_factory=dict)
    declared_playoff_periods: tuple[int, ...] | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TeamOutcome:
    team_id: int
    wins: float
    losses: float
    ties: float
    points_for: float
    made_playoffs: float
    seed_distribution: Mapping[int, float]
    round_advancement: Mapping[str, float]
    #: None when the run was not authorised to publish it: a championship
    #: probability is the most-quoted number this simulator produces and the
    #: one that most depends on weeks nobody has projected yet.
    championship_probability: float | None


@dataclass(frozen=True)
class LeagueSimResult:
    teams: Mapping[int, TeamOutcome]
    simulations: int
    seed: int
    config_hash: str
    source_hashes: Mapping[str, str]
    periods_by_basis: Mapping[str, int]
    probability_kind: str
    disclaimer: str
    notes: tuple[str, ...]


# --------------------------------------------------------------------------- #
# 1 · Schedule validation
# --------------------------------------------------------------------------- #
def validate_schedule(team_ids: Sequence[int],
                      schedule: Mapping[int, Sequence[tuple[int, int]]],
                      periods: Sequence[int]) -> ScheduleReport:
    """Refuse any schedule a season cannot actually be played on.

    Checks, in order, because the later ones are meaningless if an earlier one
    fails: every declared period exists; no unknown team id; nobody plays
    themselves; every team appears exactly once per period (which is what makes
    the pairing reciprocal by construction); and the per-team game count is
    uniform. A league with an odd team count would legitimately have byes --
    this function does not invent them, and a period that is short a game is
    reported rather than patched.
    """
    known = set(team_ids)
    if len(known) != len(team_ids):
        raise ScheduleError("duplicate team id in the league roster of teams")
    if not known:
        raise ScheduleError("a league with no teams cannot be simulated")

    games_per_team: dict[int, int] = dict.fromkeys(team_ids, 0)
    for period in periods:
        if period not in schedule:
            raise ScheduleError(f"period {period} is missing from the schedule")
        games = tuple(schedule[period])
        seen: dict[int, int] = {}
        for home, away in games:
            for side in (home, away):
                if side not in known:
                    raise ScheduleError(
                        f"period {period}: unknown team id {side} is not one of "
                        f"{sorted(known)}")
            if home == away:
                raise ScheduleError(
                    f"period {period}: team {home} is scheduled against itself")
            for side in (home, away):
                seen[side] = seen.get(side, 0) + 1
                games_per_team[side] += 1

        doubled = sorted(team for team, n in seen.items() if n > 1)
        missing = sorted(known - set(seen))
        if doubled or missing:
            raise ScheduleError(
                f"period {period} is not a complete pairing: "
                f"team(s) {doubled or '[]'} appear more than once and "
                f"team(s) {missing or '[]'} do not appear at all. "
                "A schedule whose two halves disagree cannot be simulated; "
                "it is not repaired here because guessing the lost game "
                "changes who makes the playoffs.")

    extra = sorted(set(schedule) - set(periods))
    if extra:
        raise ScheduleError(
            f"schedule contains period(s) {extra} outside the declared "
            f"regular season {tuple(periods)}")

    counts = set(games_per_team.values())
    if len(counts) > 1:
        raise ScheduleError(
            f"teams do not play the same number of games: {games_per_team}")
    return ScheduleReport(ok=True, games_per_team=games_per_team,
                          periods=tuple(periods))


# --------------------------------------------------------------------------- #
# 2 · Legal weekly lineups
# --------------------------------------------------------------------------- #
def _slot_instances(starting_slots: Mapping[str, int]) -> tuple[str, ...]:
    out: list[str] = []
    for slot in sorted(starting_slots):
        count = int(starting_slots[slot])
        if count < 0:
            raise LeagueSimError(f"slot {slot!r} has a negative count")
        out.extend([slot] * count)
    return tuple(out)


def optimal_lineup(points: Mapping[int, float],
                   roster: Sequence[RosterSpot],
                   starting_slots: Mapping[str, int]) -> Lineup:
    """The best legal lineup, from the one engine (see `lineup.optimize`).

    The maximum-weight assignment that used to live here is now shared with the
    weekly card, the waiver planner and the trade scan, so a lineup worth 118.4
    to the bracket is worth 118.4 to all of them.
    """
    from . import lineup as lineup_engine

    players = tuple(
        lineup_engine.LineupPlayer(player_id=spot.player_id,
                                   eligible_slots=frozenset(spot.eligible_slots))
        for spot in roster
    )
    result = lineup_engine.optimize(points, players, starting_slots)
    return Lineup(total=result.total,
                  assignment={slot: tuple(sorted(ids))
                              for slot, ids in result.assignment.items()},
                  empty_slots=result.empty_slots)


# --------------------------------------------------------------------------- #
# 3 · Matchups
# --------------------------------------------------------------------------- #
def score_matchup(home_points: float, away_points: float, *, tie_rule: str,
                  home_bonus: float = 0.0) -> str:
    """``"HOME"``, ``"AWAY"`` or ``"TIE"``. The bonus applies before comparison."""
    if tie_rule not in MATCHUP_TIE_RULES:
        raise LeagueSimError(
            f"unknown matchup tie rule {tie_rule!r}; this module understands "
            f"{sorted(MATCHUP_TIE_RULES)} and will not guess how the league "
            "breaks a tie")
    home = float(home_points) + float(home_bonus)
    away = float(away_points)
    if home > away:
        return "HOME"
    if away > home:
        return "AWAY"
    if tie_rule == "NONE":
        return "TIE"
    return tie_rule


# --------------------------------------------------------------------------- #
# 4 · Standings
# --------------------------------------------------------------------------- #
def _h2h_edge(head_to_head: Mapping[tuple[int, int], tuple[int, int, int]],
              a: int, b: int) -> float:
    """a's win share against b, or 0.5 when they are level or never met."""
    if (a, b) in head_to_head:
        wins, losses, ties = head_to_head[(a, b)]
    elif (b, a) in head_to_head:
        losses, wins, ties = head_to_head[(b, a)]
    else:
        return 0.5
    played = wins + losses + ties
    return (wins + 0.5 * ties) / played if played else 0.5


def standings(records: Sequence[TeamRecord], *, tiebreakers: Sequence[str],
              head_to_head: Mapping[tuple[int, int], tuple[int, int, int]]
              ) -> tuple[StandingRow, ...]:
    """Sort a league by record, then by the league's own tiebreaker order.

    The order is the league's, not ours: it arrives from
    ``Standings.tiebreaker`` in the ESPN payload and is applied exactly as
    given. An unknown key stops the sort, because quietly skipping a
    tiebreaker is indistinguishable from computing a different league.

    When two teams survive every declared tiebreaker the order falls back to
    ascending team id -- deterministic, reproducible, and reported on the row
    as ``unbroken_tie_with`` so it is never mistaken for a real separation.
    """
    unknown = [key for key in tiebreakers if key not in TIEBREAKERS]
    if unknown:
        raise LeagueSimError(
            f"unknown tiebreaker(s) {unknown}; this module implements "
            f"{list(TIEBREAKERS)}")

    ordered = sorted(records, key=lambda r: (-r.win_pct, r.team_id))
    groups: list[list[TeamRecord]] = []
    for rec in ordered:
        if groups and math.isclose(groups[-1][0].win_pct, rec.win_pct,
                                   rel_tol=0.0, abs_tol=1e-12):
            groups[-1].append(rec)
        else:
            groups.append([rec])

    rows: list[StandingRow] = []
    rank = 0
    for group in groups:
        resolved = _break_group(group, tiebreakers, head_to_head)
        for rec, tied_with in resolved:
            rank += 1
            rows.append(StandingRow(team_id=rec.team_id, record=rec, rank=rank,
                                    unbroken_tie_with=tied_with))
    return tuple(rows)


def _break_group(group: Sequence[TeamRecord], tiebreakers: Sequence[str],
                 head_to_head) -> list[tuple[TeamRecord, tuple[int, ...]]]:
    if len(group) == 1:
        return [(group[0], ())]

    def key(rec: TeamRecord):
        parts: list[float] = []
        for name in tiebreakers:
            if name == "head_to_head":
                parts.append(-sum(_h2h_edge(head_to_head, rec.team_id, other.team_id)
                                  for other in group if other is not rec))
            elif name == "points_for":
                parts.append(-rec.points_for)
            elif name == "points_against":
                parts.append(rec.points_against)
        parts.append(rec.team_id)          # deterministic last resort
        return tuple(parts)

    ranked = sorted(group, key=key)
    # anyone identical on every DECLARED tiebreaker is still tied
    out: list[tuple[TeamRecord, tuple[int, ...]]] = []
    for rec in ranked:
        same = tuple(sorted(other.team_id for other in ranked
                            if other is not rec and key(other)[:-1] == key(rec)[:-1]))
        out.append((rec, same))
    return out


# --------------------------------------------------------------------------- #
# 5 · Playoff structure
# --------------------------------------------------------------------------- #
def playoff_rounds(*, playoff_team_count: int, first_period: int,
                   matchup_period_length: int, team_count: int | None = None,
                   declared_periods: Sequence[int] | None = None
                   ) -> tuple[PlayoffRound, ...]:
    """Build the bracket's rounds, or refuse to.

    ``matchup_period_length`` is what makes a semifinal two weeks long: each
    round consumes that many consecutive periods and is decided on their
    aggregate.

    ``declared_periods`` is the adapter's derived period list. It is checked,
    not trusted: ESPN derives it in SCORING-period units while a bracket runs
    in MATCHUP periods, so a two-week final can arrive looking like two
    one-week rounds. When the declared list matches ``rounds x length`` it is
    accepted and grouped; when it matches neither ``rounds`` nor
    ``rounds x length`` the settings contradict each other and this raises,
    because picking one reading silently changes who is champion.
    """
    if matchup_period_length < 1:
        raise PlayoffSettingsError(
            f"playoff matchup period length must be at least 1, got "
            f"{matchup_period_length}")
    if playoff_team_count < 2:
        raise PlayoffSettingsError(
            f"a playoff field of {playoff_team_count} cannot produce a bracket")
    if team_count is not None and playoff_team_count > team_count:
        raise PlayoffSettingsError(
            f"playoff field of {playoff_team_count} exceeds the league's "
            f"{team_count} teams")

    n_rounds = int(math.ceil(math.log2(playoff_team_count)))
    if n_rounds not in _ROUND_NAMES:
        raise PlayoffSettingsError(
            f"a {playoff_team_count}-team bracket needs {n_rounds} rounds, "
            "which this module does not name or simulate")

    if declared_periods is not None:
        declared = tuple(declared_periods)
        expected_span = n_rounds * matchup_period_length
        if len(declared) not in (n_rounds, expected_span):
            raise PlayoffSettingsError(
                f"playoff settings contradict each other: {len(declared)} "
                f"declared playoff period(s) {declared} match neither "
                f"{n_rounds} matchup-period round(s) nor {expected_span} "
                f"scoring period(s) at length {matchup_period_length}")
        first_period = declared[0]

    bracket_size = 2 ** n_rounds
    byes = tuple(range(1, bracket_size - playoff_team_count + 1))

    rounds: list[PlayoffRound] = []
    period = first_period
    for index, name in enumerate(_ROUND_NAMES[n_rounds]):
        periods = tuple(range(period, period + matchup_period_length))
        rounds.append(PlayoffRound(name=name, matchup_periods=periods,
                                   byes=byes if index == 0 else ()))
        period += matchup_period_length
    return tuple(rounds)


def _bracket_order(size: int) -> list[int]:
    """Seed positions in true bracket order.

    Not simply 1 v N, 2 v N-1 down the page: that pairs the right teams in
    round one and the WRONG ones in round two, because the survivors are then
    adjacent in the wrong order. The recursive construction (1,8,4,5,2,7,3,6
    for eight) puts each sub-bracket's teams next to each other, so pairing
    adjacent winners in later rounds reproduces a real bracket.
    """
    order = [1]
    while len(order) < size:
        width = len(order) * 2 + 1
        order = [value for seed in order for value in (seed, width - seed)]
    return order


def _seed_pairs(seeds: Sequence[int], bracket_size: int
                ) -> list[tuple[int | None, int | None]]:
    """Round-one pairings, with ``None`` standing in for a bye."""
    positions = _bracket_order(bracket_size)
    slots: list[int | None] = [
        seeds[pos - 1] if pos - 1 < len(seeds) else None for pos in positions]
    return [(slots[i], slots[i + 1]) for i in range(0, bracket_size, 2)]


def run_bracket(*, seeds: Sequence[int], rounds: Sequence[PlayoffRound],
                team_period_points: Mapping[tuple[int, int], float],
                reseed: bool, tie_rule: str) -> BracketOutcome:
    """Play the bracket, aggregating each round over its matchup periods.

    A two-week round is decided on the SUM of its periods, not on week one --
    a team can win the first week and lose the round. Byes belong to the top
    seeds and are reported. Without reseeding the bracket is fixed by slot;
    with reseeding the best surviving seed draws the worst.
    """
    if tie_rule not in BRACKET_TIE_RULES:
        raise LeagueSimError(
            f"unknown playoff tie rule {tie_rule!r}; understood: "
            f"{sorted(BRACKET_TIE_RULES)}")
    effective_tie = "HOME" if tie_rule == "HIGHER_SEED" else tie_rule

    seeds = tuple(seeds)
    seed_of = {team: i + 1 for i, team in enumerate(seeds)}
    bracket_size = 2 ** len(rounds)
    if len(seeds) > bracket_size:
        raise PlayoffSettingsError(
            f"{len(seeds)} qualifiers do not fit a {len(rounds)}-round bracket")

    def aggregate(team: int, periods: Sequence[int]) -> float:
        total = 0.0
        for period in periods:
            key = (team, period)
            if key not in team_period_points:
                raise LeagueSimError(
                    f"team {team} has no points for playoff period {period}; "
                    "refusing to score a missing week as zero")
            total += float(team_period_points[key])
        return total

    outcomes: list[RoundOutcome] = []
    survivors: list[int | None] = []

    for index, rnd in enumerate(rounds):
        if index == 0:
            pairs = _seed_pairs(seeds, bracket_size)
        elif reseed:
            alive = sorted((t for t in survivors if t is not None),
                           key=lambda t: seed_of[t])
            pairs = [(alive[i], alive[len(alive) - 1 - i])
                     for i in range(len(alive) // 2)]
        else:
            pairs = [(survivors[i], survivors[i + 1])
                     for i in range(0, len(survivors), 2)]

        played: list[tuple[int, int]] = []
        winners: list[int | None] = []
        byes: list[int] = []
        for home, away in pairs:
            if home is None or away is None:
                advanced = home if home is not None else away
                if advanced is not None:
                    byes.append(advanced)
                winners.append(advanced)
                continue
            # the better seed is seated at home, which is also what makes
            # HIGHER_SEED resolvable as a home-side tie rule
            if seed_of[away] < seed_of[home]:
                home, away = away, home
            played.append((home, away))
            verdict = score_matchup(aggregate(home, rnd.matchup_periods),
                                    aggregate(away, rnd.matchup_periods),
                                    tie_rule=effective_tie)
            if verdict == "TIE":
                raise LeagueSimError(
                    f"{rnd.name}: {home} v {away} finished level and the league "
                    "declares no playoff tie rule; refusing to invent one")
            winners.append(home if verdict == "HOME" else away)

        outcomes.append(RoundOutcome(name=rnd.name, pairings=tuple(played),
                                     winners=tuple(w for w in winners if w is not None),
                                     byes=tuple(sorted(byes, key=lambda t: seed_of[t]))))
        survivors = winners

    alive = [t for t in survivors if t is not None]
    if len(alive) != 1:
        raise PlayoffSettingsError(
            f"the bracket ended with {len(alive)} teams alive, not one")
    return BracketOutcome(rounds=tuple(outcomes), champion=alive[0])


# --------------------------------------------------------------------------- #
# 5b · Building a format from the read-only ESPN contract
# --------------------------------------------------------------------------- #
#: ESPN's seeding rule is the one ordered tiebreaker the settings payload
#: actually states. Anything outside this map is refused rather than guessed,
#: because the tiebreaker order decides the playoff field.
SEEDING_RULE_TIEBREAKERS: Mapping[str, tuple[str, ...]] = {
    "TOTAL_POINTS_SCORED": ("points_for",),
    "HEAD_TO_HEAD_RECORD": ("head_to_head", "points_for"),
}

#: Slots that hold players but never score. They are excluded from the
#: starting lineup, not from the roster.
NON_SCORING_SLOTS = frozenset({"BE", "IR"})


def rosters_from_snapshot(snapshot) -> dict[int, tuple[RosterSpot, ...]]:
    """Every team's roster, carrying ESPN's own slot eligibility per player."""
    out: dict[int, tuple[RosterSpot, ...]] = {}
    for team_id, players in snapshot.rosters.items():
        out[int(team_id)] = tuple(
            RosterSpot(player_id=int(p.player_id), name=p.full_name,
                       eligible_slots=tuple(p.eligible_slots))
            for p in players)
    return out


def _wrap(value):
    if isinstance(value, dict):
        return _SnapshotView(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_wrap(item) for item in value)
    return value


class _SnapshotView(Mapping):
    """Attribute access over a canonical `espn-league/1` dict.

    The bracket used to take the frozen dataclass while the trade scan took
    the dict, so "the snapshot" meant two different objects depending on which
    consumer you asked. Both are accepted here, and both read the same fields.
    """

    __slots__ = ("_payload",)


    def __init__(self, payload):
        object.__setattr__(self, "_payload", payload)

    def __getattr__(self, name):
        payload = object.__getattribute__(self, "_payload")
        if name not in payload:
            raise AttributeError(f"snapshot has no {name!r}")
        return _wrap(payload[name])

    # `dict(view)` and `view["k"]` have to keep working: several call sites
    # copy a sub-block out with dict(), and a view that only answers attribute
    # access turns that into a confusing AttributeError about a dict method.
    def keys(self):
        return object.__getattribute__(self, "_payload").keys()

    def __getitem__(self, key):
        return _wrap(object.__getattribute__(self, "_payload")[key])

    def __iter__(self):
        return iter(object.__getattribute__(self, "_payload"))

    def __len__(self):
        return len(object.__getattribute__(self, "_payload"))


def from_snapshot(snapshot, *, tiebreakers: Sequence[str] | None = None
                  ) -> LeagueFormat:
    """Derive a simulation-ready format from a read-only league snapshot.

    Everything structural is taken from the payload: team ids, the weekly
    schedule, the regular-season length, the legal starting slots, the playoff
    field, the round length, the reseed flag, the tie rules and the home bonus.
    Nothing is defaulted.

    Two places refuse rather than assume:

    * **Tiebreaker order.** The settings payload states a *seeding rule*, not
      an ordered standings tiebreaker list. A known seeding rule is translated;
      an unknown one raises unless the caller passes ``tiebreakers`` explicitly,
      and that override is recorded in ``notes`` so a derived order is never
      mistaken for a read one.
    * **Playoff periods.** ``Playoffs.playoff_matchup_periods`` is derived by
      the adapter and can be expressed in scoring periods rather than matchup
      periods. It is passed through the consistency check in
      :func:`playoff_rounds`, which raises when the two readings disagree.
    """
    if isinstance(snapshot, Mapping):
        snapshot = _SnapshotView(snapshot)

    team_ids = tuple(sorted(int(t.team_id) for t in snapshot.teams))
    playoffs = snapshot.playoffs
    regular = int(playoffs.regular_season_matchup_periods)
    if regular < 1:
        raise PlayoffSettingsError(
            "the payload declares no regular-season matchup periods")
    periods = tuple(range(1, regular + 1))

    schedule: dict[int, list[tuple[int, int]]] = {p: [] for p in periods}
    for game in snapshot.schedule:
        period = int(game.matchup_period)
        if period in schedule and str(game.playoff_tier).upper() in ("NONE", ""):
            schedule[period].append((int(game.home_team_id), int(game.away_team_id)))
    frozen_schedule = {p: tuple(games) for p, games in schedule.items()}
    validate_schedule(team_ids, frozen_schedule, periods)

    starting = {slot: int(count)
                for slot, count in snapshot.roster_settings.lineup_slot_counts.items()
                if slot not in NON_SCORING_SLOTS and int(count) > 0}
    if not starting:
        raise LeagueSimError("the payload declares no starting lineup slots")

    notes: list[str] = []
    if tiebreakers is not None:
        order = tuple(tiebreakers)
        notes.append("standings tiebreaker order supplied by the caller, not "
                     "read from the league payload")
    else:
        rule = str(playoffs.seeding_rule).upper()
        if rule not in SEEDING_RULE_TIEBREAKERS:
            raise PlayoffSettingsError(
                f"seeding rule {playoffs.seeding_rule!r} is not one this module "
                f"can translate into a tiebreaker order "
                f"({sorted(SEEDING_RULE_TIEBREAKERS)}); pass tiebreakers "
                "explicitly rather than letting the order be guessed")
        order = SEEDING_RULE_TIEBREAKERS[rule]
        notes.append(f"tiebreaker order derived from seeding rule {rule}")

    fmt = LeagueFormat(
        team_ids=team_ids,
        schedule=frozen_schedule,
        regular_season_periods=periods,
        starting_slots=starting,
        playoff_team_count=int(playoffs.team_count),
        matchup_period_length=int(playoffs.matchup_period_length),
        reseed=bool(playoffs.reseed),
        seeding_rule=str(playoffs.seeding_rule),
        tiebreakers=order,
        matchup_tie_rule=str(snapshot.scoring.matchup_tie_rule),
        playoff_tie_rule=str(snapshot.scoring.playoff_matchup_tie_rule),
        home_bonus=float(snapshot.scoring.home_team_bonus),
        source_hashes=dict(snapshot.hashes),
        declared_playoff_periods=tuple(playoffs.playoff_scoring_periods) or None,
        notes=tuple(notes),
    )
    # surfaces a contradictory playoff-period declaration at build time rather
    # than three thousand simulated seasons later
    playoff_rounds(playoff_team_count=fmt.playoff_team_count,
                   first_period=regular + 1,
                   matchup_period_length=fmt.matchup_period_length,
                   team_count=len(team_ids),
                   declared_periods=fmt.declared_playoff_periods)
    return fmt


# --------------------------------------------------------------------------- #
# 6 · Correlated weekly outcomes
# --------------------------------------------------------------------------- #
def _draw_team_week(rng: np.random.Generator, roster: Sequence[RosterSpot],
                    means: Mapping[int, float], sds: Mapping[int, float],
                    rho: float) -> dict[int, float]:
    """QUARANTINED. Research only — not reachable from `simulate_league`.

    This is a two-parameter truncated Normal with a hand-set equicorrelation:
    every teammate pair correlated at exactly `rho`, every cross-team pair at
    zero, no skew, no zero-inflation, and `max(0, ...)` truncation that biases
    the realised mean above the mean it was given. `rho` defaulted to 0.35,
    which is a number somebody chose, not one anybody measured.

    It also discarded the correlated event simulation entirely: the projection
    layer produces a joint sample matrix per week, and this replaced it with
    two scalars per player. Championship probabilities computed this way are
    properties of the parametric assumption, not of the projections.

    Kept for the diagnostic in `team_week_correlation_check` and for nothing
    else; `simulate_league` now requires real per-period sample matrices.
    """
    # The original reasoning was right -- teammates share game script, pace and
    # blowout risk, so drawing them independently understates how often a team
    # posts an extreme total. A hand-set rho is not how to capture it.
    shared = float(rng.standard_normal())
    a, b = math.sqrt(rho), math.sqrt(1.0 - rho)
    out: dict[int, float] = {}
    for spot in roster:
        mean = float(means.get(spot.player_id, 0.0))
        sd = float(sds.get(spot.player_id, 0.0))
        noise = float(rng.standard_normal())
        out[spot.player_id] = max(0.0, mean + sd * (a * shared + b * noise))
    return out


def team_week_correlation_check(*, rho: float, seed: int,
                                players: int = 9, draws: int = 4000) -> float:
    """Spread of a team's weekly total at a given correlation.

    Exposed because the correlation assumption is load-bearing and otherwise
    invisible: higher ``rho`` must widen a team's weekly total, and a
    regression that silently drops the shared factor shows up here.
    """
    rng = np.random.default_rng(seed)
    roster = tuple(RosterSpot(player_id=i, name=f"p{i}", eligible_slots=("FLEX",))
                   for i in range(players))
    means = dict.fromkeys(range(players), 12.0)
    sds = dict.fromkeys(range(players), 5.0)
    totals = [sum(_draw_team_week(rng, roster, means, sds, rho).values())
              for _ in range(draws)]
    return float(np.std(totals))


# --------------------------------------------------------------------------- #
# 7 · The season
# --------------------------------------------------------------------------- #
def _sample_digest(period_samples: Mapping[int, Mapping[int, Any]]) -> str:
    """A digest of the actual draws, so a run cannot be reproduced from a label."""
    parts = []
    for period in sorted(period_samples):
        rows = period_samples[period]
        for player_id in sorted(rows):
            vector = np.asarray(rows[player_id], dtype=float)
            parts.append(f"{period}:{player_id}:{len(vector)}:"
                         f"{float(vector.sum()):.6f}:{float(vector.var()):.6f}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _config_hash(fmt: LeagueFormat, period_samples, simulations, seed, basis) -> str:
    payload = {
        "team_ids": list(fmt.team_ids),
        "schedule": {str(p): [list(g) for g in fmt.schedule[p]]
                     for p in sorted(fmt.schedule)},
        "regular_season_periods": list(fmt.regular_season_periods),
        "starting_slots": dict(sorted(fmt.starting_slots.items())),
        "playoff_team_count": fmt.playoff_team_count,
        "matchup_period_length": fmt.matchup_period_length,
        "reseed": fmt.reseed,
        "seeding_rule": fmt.seeding_rule,
        "tiebreakers": list(fmt.tiebreakers),
        "matchup_tie_rule": fmt.matchup_tie_rule,
        "playoff_tie_rule": fmt.playoff_tie_rule,
        "home_bonus": fmt.home_bonus,
        "source_hashes": dict(sorted(fmt.source_hashes.items())),
        "period_samples": _sample_digest(period_samples),
        "simulations": int(simulations),
        "seed": int(seed),
        "basis": {str(k): v for k, v in sorted(basis.items())},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _week_total(fmt: LeagueFormat, rosters, samples: Mapping[int, Mapping[int, Any]],
                team_period: dict, team: int, period: int, draw: int) -> float:
    """One team's optimal legal lineup for one matchup period, from real samples.

    `samples[period][player_id]` is that period's correlated draw vector from
    the projection layer, so row `draw` is one joint week: the same game
    script for every player in it. Memoised into `team_period` so the bracket
    scores the same week the season did.
    """
    period_samples = samples[period]
    points = {spot.player_id: float(period_samples[spot.player_id][draw])
              for spot in rosters[team] if spot.player_id in period_samples}
    total = optimal_lineup(points, rosters[team], fmt.starting_slots).total
    team_period[(team, period)] = total
    return total


def validate_period_samples(samples: Mapping[int, Mapping[int, Any]],
                            periods: Sequence[int], rosters, simulations: int,
                            period_basis: Mapping[int, str]) -> None:
    """Every scored period needs its own draws, of the right height.

    `period_basis` used to be an audit label and nothing else: a
    rest-of-season assumption week was drawn from the same two scalars as a
    fully projected one, so the label described a distinction the arithmetic
    did not make. It now has to point at real inputs — a period labelled
    `weekly_projection` must have that week's samples, and a period with no
    samples at all is refused rather than filled in from a neighbour.
    """
    missing = sorted(period for period in periods if period not in samples)
    if missing:
        raise LeagueSimError(
            f"period(s) {missing} have no sample matrix. A week with no projection is a "
            "modelling assumption, and this simulator will not manufacture one by "
            "reusing another week's mean and standard deviation.")
    for period in periods:
        rows = samples[period]
        if not rows:
            raise LeagueSimError(f"period {period} has an empty sample matrix")
        heights = {len(np.asarray(vector)) for vector in rows.values()}
        if heights != {simulations}:
            raise LeagueSimError(
                f"period {period} samples are {sorted(heights)} rows deep but the run asks "
                f"for {simulations}; a truncated or recycled matrix is not a joint sample")
        needed = {spot.player_id for roster in rosters.values() for spot in roster}
        absent = sorted(pid for pid in needed if pid not in rows)
        if absent:
            raise LeagueSimError(
                f"period {period} is missing samples for rostered player(s) {absent[:5]}"
                f"{' …' if len(absent) > 5 else ''}; those players would be scored as zero")


def simulate_league(fmt: LeagueFormat, *, rosters: Mapping[int, Sequence[RosterSpot]],
                    period_samples: Mapping[int, Mapping[int, Any]],
                    simulations: int, seed: int,
                    period_basis: Mapping[int, str],
                    publish_championship_probabilities: bool = False) -> LeagueSimResult:
    """Run the whole league: schedule, standings, seeding, bracket.

    Every simulated season replays the real schedule one matchup period at a
    time, optimises each team's legal lineup from correlated player draws,
    scores the head-to-head, sorts the standings on the league's own tiebreaker
    order, seeds the field and plays the bracket. What comes back are
    frequencies over those seasons -- see :data:`DISCLAIMER`.
    """
    validate_schedule(fmt.team_ids, fmt.schedule, fmt.regular_season_periods)

    rounds = playoff_rounds(playoff_team_count=fmt.playoff_team_count,
                            first_period=max(fmt.regular_season_periods) + 1,
                            matchup_period_length=fmt.matchup_period_length,
                            team_count=len(fmt.team_ids),
                            declared_periods=fmt.declared_playoff_periods)
    playoff_periods = tuple(p for rnd in rounds for p in rnd.matchup_periods)

    bad = sorted(set(period_basis.values()) - BASES)
    if bad:
        raise LeagueSimError(
            f"unknown projection basis {bad}; every period must be labelled "
            f"{sorted(BASES)}")
    all_periods = tuple(fmt.regular_season_periods) + playoff_periods
    unlabelled = sorted(set(all_periods) - set(period_basis))
    if unlabelled:
        raise LeagueSimError(
            f"period(s) {unlabelled} have no declared projection basis. A week "
            "with no feature frame is a modelling assumption, not a projection, "
            "and must be labelled as one.")
    if simulations < 1:
        raise LeagueSimError("at least one simulation is required")
    validate_period_samples(period_samples, all_periods, rosters, simulations, period_basis)

    future = [period for period in playoff_periods if period not in period_samples]
    if future and publish_championship_probabilities:
        raise LeagueSimError(
            f"playoff period(s) {future} have no samples, so a championship probability "
            "would be a property of whatever week was substituted for them. Run this "
            "research-only, or supply the samples.")

    missing_rosters = sorted(set(fmt.team_ids) - set(rosters))
    if missing_rosters:
        raise LeagueSimError(f"team(s) {missing_rosters} have no roster")

    # No RNG here any more: the randomness lives in the sample matrices the
    # projection layer produced, and `seed` is carried into the config hash so
    # a run is still identified by what produced its draws.
    teams = tuple(fmt.team_ids)
    wins = dict.fromkeys(teams, 0.0)
    losses = dict.fromkeys(teams, 0.0)
    ties = dict.fromkeys(teams, 0.0)
    points_for = dict.fromkeys(teams, 0.0)
    made = dict.fromkeys(teams, 0)
    champs = dict.fromkeys(teams, 0)
    seed_counts: dict[int, dict[int, int]] = {t: {} for t in teams}
    advanced: dict[int, dict[str, int]] = {
        t: {rnd.name: 0 for rnd in rounds} for t in teams}

    # Each pass reads one row of every period's sample matrix, so a season is a
    # coherent draw across weeks rather than a fresh parametric roll per week.
    for simulation in range(int(simulations)):
        w = dict.fromkeys(teams, 0)
        loss = dict.fromkeys(teams, 0)
        tie = dict.fromkeys(teams, 0)
        pf = dict.fromkeys(teams, 0.0)
        pa = dict.fromkeys(teams, 0.0)
        h2h: dict[tuple[int, int], tuple[int, int, int]] = {}
        team_period: dict[tuple[int, float], float] = {}

        for period in fmt.regular_season_periods:
            for home, away in fmt.schedule[period]:
                hp = _week_total(fmt, rosters, period_samples, team_period,
                                 home, period, simulation)
                ap = _week_total(fmt, rosters, period_samples, team_period,
                                 away, period, simulation)
                pf[home] += hp
                pf[away] += ap
                pa[home] += ap
                pa[away] += hp
                verdict = score_matchup(hp, ap, tie_rule=fmt.matchup_tie_rule,
                                        home_bonus=fmt.home_bonus)
                key = (home, away)
                hw, hl, ht = h2h.get(key, (0, 0, 0))
                if verdict == "HOME":
                    w[home] += 1
                    loss[away] += 1
                    h2h[key] = (hw + 1, hl, ht)
                elif verdict == "AWAY":
                    w[away] += 1
                    loss[home] += 1
                    h2h[key] = (hw, hl + 1, ht)
                else:
                    tie[home] += 1
                    tie[away] += 1
                    h2h[key] = (hw, hl, ht + 1)

        table = standings(
            [TeamRecord(t, w[t], loss[t], tie[t], pf[t], pa[t]) for t in teams],
            tiebreakers=fmt.tiebreakers, head_to_head=h2h)

        for t in teams:
            wins[t] += w[t]
            losses[t] += loss[t]
            ties[t] += tie[t]
            points_for[t] += pf[t]
        for row in table:
            seed_counts[row.team_id][row.rank] = \
                seed_counts[row.team_id].get(row.rank, 0) + 1

        qualifiers = tuple(row.team_id for row in table[:fmt.playoff_team_count])
        for t in qualifiers:
            made[t] += 1
        for period in playoff_periods:
            for t in qualifiers:
                _week_total(fmt, rosters, period_samples, team_period, t, period,
                            simulation)

        outcome = run_bracket(seeds=qualifiers, rounds=rounds,
                              team_period_points=team_period,
                              reseed=fmt.reseed, tie_rule=fmt.playoff_tie_rule)
        for rnd in outcome.rounds:
            for t in rnd.winners:
                advanced[t][rnd.name] += 1
        champs[outcome.champion] += 1

    n = float(simulations)
    results = {
        t: TeamOutcome(
            team_id=t,
            wins=wins[t] / n, losses=losses[t] / n, ties=ties[t] / n,
            points_for=points_for[t] / n,
            made_playoffs=made[t] / n,
            seed_distribution={k: v / n for k, v in sorted(seed_counts[t].items())},
            round_advancement={k: v / n for k, v in advanced[t].items()},
            championship_probability=(champs[t] / n
                                      if publish_championship_probabilities else None),
        )
        for t in teams
    }
    basis_counts = {WEEKLY_PROJECTION: 0, REST_OF_SEASON: 0}
    for period in all_periods:
        basis_counts[period_basis[period]] += 1

    return LeagueSimResult(
        teams=results, simulations=int(simulations), seed=int(seed),
        config_hash=_config_hash(fmt, period_samples, simulations, seed, period_basis),
        source_hashes=dict(fmt.source_hashes),
        periods_by_basis=basis_counts,
        probability_kind=PROBABILITY_KIND,
        disclaimer=DISCLAIMER,
        notes=tuple(fmt.notes) + (
            "weekly player outcomes read from the projection layer's own per-period "
            "correlated sample matrices; no parametric outcome model is used here",
            "managers assumed to start their optimal legal lineup every week",
        ) + ((
            "championship probabilities withheld: see published_championship_probabilities",
        ) if not publish_championship_probabilities else ()),
    )
