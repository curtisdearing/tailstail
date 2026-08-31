"""ID-based, recommendation-only trade analysis over a live ESPN league.

This is the upgrade of the name-keyed trade scan. The old path pulled roster
*names* from ESPN, matched them loosely against the board, and reasoned about
"Team1" and "Team4" with a hard-coded playoff window. Every one of those is a
place where the analysis could be confidently wrong about a real league, so
each is replaced here:

* **Identity is by id.** ESPN player ids are stable; display names are not.
  Names are used once, to build the id map, and a name that maps to two board
  players or to a player at a different position is recorded as ambiguous and
  dropped — never resolved by coin flip. The map is order-independent and
  deterministic.
* **Rules come from the league, not from defaults.** Starting slots, bench
  size, IR, roster cap and the playoff matchup periods are read out of the
  snapshot. Weeks 15-17 are not assumed; a league whose playoffs start in
  period 14 gets period 14.
* **Kickers and defenses are shadow.** The projection board carries QB/RB/WR/TE
  only, so K and D/ST have no distribution behind them. They are excluded from
  the optimizer and reported as shadow rather than silently valued at zero,
  which would make every K trade look free. Promotion requires supplying real
  projections; asking for it without them raises.
* **Every package is checked for legality after the swap** — roster cap, and an
  exact bipartite feasibility check that the remaining players can still fill
  every starting slot, including the shadow K and D/ST slots.
* **Locked players cannot move.** A player named in a pending transaction is
  locked, because a claim resolving underneath a proposal changes what the
  proposal was about.

What this module will not do: send, agree to, decline or cancel anything; act as
a commissioner; or predict the other manager's answer. It produces packages
with two-sided evidence and says plainly that plausibility is not consent. When
nothing clears the gate it publishes an explicit hold state, which is a
finding, not a failure.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import LineupRules
from .draft import normalize_name
from .season import SeasonSimulation

SCHEMA_VERSION = "trade-scan/2"

#: Positions the projection board does not carry. Valued as shadow, never zero.
SHADOW_POSITIONS = frozenset({"K", "D/ST"})

#: ESPN lineup slot id -> the position name this module reasons about.
SLOT_POSITION = {
    0: "QB", 2: "RB", 4: "WR", 6: "TE", 16: "D/ST", 17: "K", 23: "FLEX",
    3: "RB/WR", 5: "WR/TE", 7: "OP", 20: "BE", 21: "IR",
}
BENCH_SLOT_ID = 20
IR_SLOT_ID = 21

#: ESPN proTeamId -> the abbreviation the board and the bye table use.
ESPN_PRO_TEAM: Mapping[int, str] = {
    0: "FA", 1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN",
    8: "DET", 9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV", 14: "LA", 15: "MIA",
    16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ", 21: "PHI", 22: "ARI",
    23: "PIT", 24: "LAC", 25: "SF", 26: "SEA", 27: "TB", 28: "WAS", 29: "CAR",
    30: "JAX", 33: "BAL", 34: "HOU",
}

ACCEPTANCE_DISCLAIMER = (
    "Plausibility is not consent. Every number here is about roster fit under one "
    "projection model; none of it predicts what the other manager will do, or whether "
    "they will read this at all. Nothing has been sent to anyone."
)


class TradeScanError(Exception):
    """The scan cannot be run honestly on the inputs given."""


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IdentityMap:
    """ESPN player id -> board player id, plus everything that did not map."""

    espn_to_board: Mapping[int, str]
    board_to_espn: Mapping[str, int]
    unmatched: tuple[Mapping[str, Any], ...]
    ambiguous: tuple[Mapping[str, Any], ...]
    shadow: tuple[Mapping[str, Any], ...]

    def matched(self, espn_ids: Iterable[int]) -> list[str]:
        return [self.espn_to_board[pid] for pid in espn_ids if pid in self.espn_to_board]


def _board_index(board: pd.DataFrame) -> tuple[dict, dict]:
    """(exact normalized-name key, loose first-initial+surname key) -> ids.

    Both indexes map to *lists*, so a collision is visible as a collision
    rather than resolved by whichever row pandas happened to see last.
    """
    exact: dict[tuple[str, str], list[str]] = {}
    loose: dict[tuple[str, str], list[str]] = {}
    for name, pid, position in zip(board["player_name"], board["player_id"], board["position"]):
        key = normalize_name(name)
        exact.setdefault((key, str(position)), []).append(str(pid))
        parts = key.split()
        if len(parts) >= 2:
            loose.setdefault((f"{parts[0][0]} {parts[-1]}", str(position)), []).append(str(pid))
    return exact, loose


def _roster_players(snapshot: Mapping[str, Any]) -> list[dict]:
    """Flatten every rostered player with its team, slot and position."""
    players = []
    for team_id, entries in (snapshot.get("rosters") or {}).items():
        for entry in entries:
            players.append({
                "team_id": int(team_id),
                "espn_id": int(entry["player_id"]),
                "name": entry["full_name"],
                "position": entry["default_position"],
                "lineup_slot": entry["lineup_slot"],
                "eligible_slots": tuple(entry["eligible_slots"]),
                "injury_status": entry.get("injury_status", "UNKNOWN"),
                "pro_team": ESPN_PRO_TEAM.get(int(entry.get("pro_team_id") or 0), "UNK"),
                "on_ir": entry["lineup_slot"] == "IR",
            })
    return sorted(players, key=lambda item: (item["team_id"], item["espn_id"]))


def map_identities(snapshot: Mapping[str, Any], board: pd.DataFrame, *,
                   shadow_positions: Iterable[str] = SHADOW_POSITIONS) -> IdentityMap:
    """Map ESPN roster ids onto board player ids, deterministically.

    Exact normalized name plus position agreement first; then first-initial and
    surname, still requiring the position to agree. A key that resolves to more
    than one board row is ambiguous and is dropped with a record — a wrong join
    here silently prices the wrong player into every package downstream.
    """
    shadow_set = {str(value) for value in shadow_positions}
    exact, loose = _board_index(board)
    espn_to_board: dict[int, str] = {}
    board_to_espn: dict[str, int] = {}
    unmatched: list[dict] = []
    ambiguous: list[dict] = []
    shadow: list[dict] = []

    for player in _roster_players(snapshot):
        record = {"espn_id": player["espn_id"], "name": player["name"],
                  "position": player["position"], "team_id": player["team_id"],
                  "pro_team": player["pro_team"], "on_ir": bool(player.get("on_ir"))}
        if player["position"] in shadow_set:
            shadow.append({**record, "reason": "position carries no board projection"})
            continue
        key = normalize_name(player["name"])
        candidates = exact.get((key, player["position"]))
        rule = "exact_name+position"
        if not candidates:
            parts = key.split()
            loose_key = f"{parts[0][0]} {parts[-1]}" if len(parts) >= 2 else key
            candidates = loose.get((loose_key, player["position"]))
            rule = "initial+surname+position"
        if not candidates:
            unmatched.append({**record, "reason": "no board row for this name at this position"})
            continue
        if len(candidates) > 1:
            ambiguous.append({**record, "reason": f"{len(candidates)} board rows share this "
                                                  f"name key ({rule}): {sorted(candidates)}"})
            continue
        board_id = candidates[0]
        if board_id in board_to_espn and board_to_espn[board_id] != player["espn_id"]:
            ambiguous.append({**record, "reason": f"board row {board_id} already claimed by "
                                                  f"espn id {board_to_espn[board_id]}"})
            continue
        espn_to_board[player["espn_id"]] = board_id
        board_to_espn[board_id] = player["espn_id"]

    return IdentityMap(
        espn_to_board=dict(sorted(espn_to_board.items())),
        board_to_espn=dict(sorted(board_to_espn.items())),
        unmatched=tuple(unmatched), ambiguous=tuple(ambiguous), shadow=tuple(shadow))


def blocked_teams(identity: IdentityMap) -> dict[int, tuple[str, ...]]:
    """Teams whose lineup cannot be valued, because a player in it is unknown.

    An unmatched or ambiguous player is not a footnote. Every lineup total on
    that side is computed as though he does not exist, so a package can look
    like a gain purely because the roster it is measured against is missing a
    starter. A warning at the top of the scan does not travel with the package
    a reader is about to act on, so the team is excluded instead and named.

    Shadow positions are exempt: K and D/ST contribute nothing to a modelled
    lineup by design, so being unable to price one cannot move a delta.
    """
    blocked: dict[int, list[str]] = {}
    for row in list(identity.unmatched) + list(identity.ambiguous):
        if row.get("on_ir"):
            # An IR player cannot be seated, so not pricing him cannot move a
            # lineup total. This is the only exemption: everything else on a
            # roster is a player the optimizer might have started.
            continue
        team_id = int(row.get("team_id", 0))
        blocked.setdefault(team_id, []).append(
            f"{row.get('name')} ({row.get('position')}): {row.get('reason')}")
    return {team_id: tuple(sorted(reasons)) for team_id, reasons in sorted(blocked.items())}


# --------------------------------------------------------------------------- #
# League rules, read from the league
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LeagueRules:
    lineup: LineupRules
    modeled_slots: Mapping[str, int]
    shadow_slots: Mapping[str, int]
    roster_size: int
    bench_slots: int
    ir_slots: int
    flex_positions: tuple[str, ...]


def lineup_rules_from_snapshot(snapshot: Mapping[str, Any], *,
                               shadow_positions: Iterable[str] = SHADOW_POSITIONS
                               ) -> LeagueRules:
    """Build the optimizer's lineup contract from the league's own settings."""
    settings = snapshot.get("roster_settings") or {}
    counts = dict(settings.get("lineup_slot_counts") or {})
    if not counts:
        raise TradeScanError("snapshot carries no lineup slot counts")
    shadow_set = {str(value) for value in shadow_positions}

    modeled, shadow = {}, {}
    for slot, count in counts.items():
        if slot in ("BE", "IR"):
            continue
        (shadow if slot in shadow_set else modeled)[slot] = int(count)
    if not modeled:
        raise TradeScanError("no modeled starting slots remain after removing shadow positions")

    flex_positions = ("RB", "WR", "TE")
    return LeagueRules(
        lineup=LineupRules(starters=dict(modeled), flex_positions=flex_positions),
        modeled_slots=dict(sorted(modeled.items())),
        shadow_slots=dict(sorted(shadow.items())),
        roster_size=int(settings.get("roster_size") or 0),
        bench_slots=int(settings.get("bench_slots") or 0),
        ir_slots=int(settings.get("ir_slots") or 0),
        flex_positions=flex_positions,
    )


def playoff_scoring_periods(snapshot: Mapping[str, Any]) -> tuple[int, ...]:
    """The league's real playoff weeks, as published by the adapter.

    This used to expand `playoff_matchup_periods` by the round length here.
    The adapter now does that expansion once, so doing it again squares it: a
    two-week final at period 16 became weeks 31 and 32, and every playoff
    valuation quietly scored a slate that does not exist.
    """
    from .espn_league import playoff_scoring_periods as _periods

    return _periods(snapshot)


def locked_players(snapshot: Mapping[str, Any], *,
                   extra: Mapping[int, str] | None = None) -> dict[int, str]:
    """ESPN ids that must not appear in a package, and why.

    Pending transactions are the case this can see from the league endpoint: a
    claim resolving underneath a proposal changes what the proposal was about.
    Game-time locks are NOT visible here -- the league views carry no kickoff
    clock -- so a caller that knows them passes them in rather than this module
    pretending it checked.
    """
    locked: dict[int, str] = {}
    for transaction in ((snapshot.get("transactions") or {}).get("pending") or []):
        for item in transaction.get("items") or []:
            player_id = item.get("playerId") or item.get("player_id")
            if isinstance(player_id, int):
                locked[player_id] = (
                    f"named in pending {transaction.get('type', 'transaction')} "
                    f"{transaction.get('transaction_id') or ''}".strip())
    for player_id, reason in (extra or {}).items():
        locked[int(player_id)] = str(reason)
    return dict(sorted(locked.items()))


# --------------------------------------------------------------------------- #
# Legality
# --------------------------------------------------------------------------- #
def _can_fill_slots(players: Sequence[Mapping[str, Any]],
                    slots: Mapping[str, int]) -> bool:
    """Exact seatability, from the one engine (see `lineup.can_fill`)."""
    from . import lineup as lineup_engine

    try:
        seated = lineup_engine.as_players(players)
    except lineup_engine.LineupError:
        return False
    return lineup_engine.can_fill(seated, slots)


@dataclass(frozen=True)
class Legality:
    legal: bool
    violations: tuple[str, ...]
    roster_size_after: int
    roster_cap: int


def check_legality(roster_after: Sequence[Mapping[str, Any]], rules: LeagueRules, *,
                   label: str) -> Legality:
    """Roster cap and startable-lineup feasibility after a package."""
    active = [player for player in roster_after if not player.get("on_ir")]
    violations: list[str] = []
    if rules.roster_size and len(active) > rules.roster_size:
        violations.append(
            f"{label}: {len(active)} active players exceeds the {rules.roster_size}-man roster")
    all_slots = {**rules.modeled_slots, **rules.shadow_slots}
    if not _can_fill_slots(active, all_slots):
        violations.append(
            f"{label}: remaining players cannot fill every starting slot "
            f"({', '.join(f'{count}x{slot}' for slot, count in sorted(all_slots.items()))})")
    return Legality(legal=not violations, violations=tuple(violations),
                    roster_size_after=len(active), roster_cap=rules.roster_size)


# --------------------------------------------------------------------------- #
# Lineup value, as a distribution
# --------------------------------------------------------------------------- #
class LineupEvaluator:
    """Per-simulation optimal-lineup points for a roster, from the one engine.

    Returns the whole vector, not its mean: a package whose average gain is
    +1.2 with a 45% chance of being negative is a different proposition from
    one that gains +1.2 almost surely, and the mean hides that. The
    single-FLEX decomposition that used to live here is gone -- it was exact
    only for one FLEX seat, and it scored composite seats as zero.
    """

    def __init__(self, season: SeasonSimulation, rules: LineupRules) -> None:
        from . import lineup as lineup_engine

        self._engine = lineup_engine
        self.season = season
        self.rules = rules
        self.columns = list(season.points.columns)
        self._index = {column: i for i, column in enumerate(self.columns)}
        self._matrix = season.points.to_numpy(dtype=float)
        self.position = dict(zip(season.player_meta["player_id"], season.player_meta["position"]))
        self.simulations = len(season.points)
        self._cache: dict[frozenset, np.ndarray] = {}

    def vector(self, roster: Sequence[str]) -> np.ndarray:
        usable = frozenset(pid for pid in roster if pid in self._index)
        cached = self._cache.get(usable)
        if cached is not None:
            return cached
        if not usable:
            value = np.zeros(self.simulations)
        else:
            ordered = sorted(usable)
            players = self._engine.from_positions(
                [(pid, str(self.position.get(pid, ""))) for pid in ordered],
                self.rules.starters, flex_positions=self.rules.flex_positions)
            columns = [self._index[pid] for pid in ordered]
            value = self._engine.optimize_matrix(
                self._matrix[:, columns], players, self.rules.starters)
        self._cache[usable] = value
        return value


@dataclass(frozen=True)
class DeltaDistribution:
    """The three-way split matters more than the mean here.

    A roster change usually changes *nothing*: in most simulations neither
    player involved reaches the optimal lineup, so the delta is exactly zero.
    A swap can therefore be clearly good -- +0.8 on average, never worse in
    80% of simulations -- while P(delta > 0) sits near 0.20, because four
    simulations in five it simply does not come up. Gating on P(gain) would
    reject every marginal-but-favourable move ever proposed; the question worth
    asking is whether the change is *not worse*, and how the tails compare when
    it does bite.
    """

    mean: float
    sd: float
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float
    prob_gain: float
    prob_loss: float
    prob_no_change: float
    prob_not_worse: float
    simulations: int


def distribution(delta: np.ndarray) -> DeltaDistribution:
    quantiles = np.quantile(delta, [0.05, 0.25, 0.50, 0.75, 0.95])
    gain = float((delta > 0).mean())
    loss = float((delta < 0).mean())
    return DeltaDistribution(
        mean=round(float(delta.mean()), 3),
        sd=round(float(delta.std(ddof=1)), 3) if len(delta) > 1 else 0.0,
        p05=round(float(quantiles[0]), 3), p25=round(float(quantiles[1]), 3),
        p50=round(float(quantiles[2]), 3), p75=round(float(quantiles[3]), 3),
        p95=round(float(quantiles[4]), 3),
        prob_gain=round(gain, 4), prob_loss=round(loss, 4),
        prob_no_change=round(1.0 - gain - loss, 4), prob_not_worse=round(1.0 - loss, 4),
        simulations=int(len(delta)))


# --------------------------------------------------------------------------- #
# The scan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PackageSide:
    team_id: int
    team_name: str
    sends: tuple[Mapping[str, Any], ...]
    receives: tuple[Mapping[str, Any], ...]
    delta: DeltaDistribution
    legality: Legality
    roster_before: int
    roster_after: int
    starters_on_bye_before: int
    starters_on_bye_after: int


@dataclass(frozen=True)
class Package:
    package_id: str
    mine: PackageSide
    theirs: PackageSide
    playoff_delta: DeltaDistribution | None
    rationale: tuple[str, ...]
    plausibility: tuple[str, ...]
    uncertainty: tuple[str, ...]
    acceptance_claim: str = "none — this is a recommendation, not a negotiation"


@dataclass(frozen=True)
class TradeScan:
    schema_version: str
    state: str
    generated_for: Mapping[str, Any]
    league: Mapping[str, Any]
    rules: Mapping[str, Any]
    identity: Mapping[str, Any]
    locked: Mapping[str, str]
    packages: tuple[Package, ...]
    hold_reason: str | None
    gate: Mapping[str, Any]
    context: Mapping[str, Any]
    disclaimer: str = ACCEPTANCE_DISCLAIMER
    warnings: tuple[str, ...] = field(default=())


def _bye_weeks(pro_team: str, byes: Mapping[str, Sequence[int]]) -> set[int]:
    return {int(week) for week in byes.get(pro_team, ())}


def _starters_on_bye(players: Sequence[Mapping[str, Any]], byes: Mapping[str, Sequence[int]],
                     weeks: Sequence[int]) -> int:
    upcoming = {int(week) for week in weeks}
    return sum(1 for player in players
               if not player.get("on_ir") and _bye_weeks(player["pro_team"], byes) & upcoming)


def _describe(player: Mapping[str, Any], identity: IdentityMap,
              board_index: pd.DataFrame | None) -> dict:
    board_id = identity.espn_to_board.get(player["espn_id"])
    described = {
        "espn_player_id": player["espn_id"], "name": player["name"],
        "position": player["position"], "pro_team": player["pro_team"],
        "board_player_id": board_id, "lineup_slot": player["lineup_slot"],
    }
    if board_id is not None and board_index is not None and board_id in board_index.index:
        described["season_mean"] = round(float(board_index.loc[board_id, "season_mean"]), 2)
    return described


#: How old a sample set may be and still describe *this* week. A draft board is
#: a preseason artifact: using it in-season is not a smaller claim, it is a
#: different one.
MAX_SAMPLE_AGE_DAYS = 8


def _require_current_inputs(snapshot: Mapping[str, Any], season: Any, *,
                            on_demand: bool) -> None:
    """Everything a trade scan needs to be about today, checked before it runs."""
    if not on_demand:
        raise TradeScanError(
            "trade scans are on demand: pass on_demand=True. They are not part of the "
            "weekly card, because the card runs on a schedule and a scan run on a "
            "schedule will be run against whatever board happens to be on disk.")

    basis = getattr(season, "metadata", {}) or {}
    label = str(basis.get("basis") or basis.get("source") or "")
    if label and "draft" in label.lower():
        raise TradeScanError(
            f"the supplied samples are labelled {label!r}: a draft-board valuation is a "
            "preseason artifact and is not current weekly or rest-of-season advice")
    generated = basis.get("generated_at") or basis.get("information_as_of")
    retrieved = snapshot.get("retrieved_at")
    if generated and retrieved:
        from .espn_league import parse_timestamp

        sampled_at, snapshot_at = parse_timestamp(generated), parse_timestamp(retrieved)
        if sampled_at and snapshot_at:
            age = abs((snapshot_at - sampled_at).total_seconds()) / 86400.0
            if age > MAX_SAMPLE_AGE_DAYS:
                raise TradeScanError(
                    f"samples were generated {age:.1f} days from this snapshot, beyond the "
                    f"{MAX_SAMPLE_AGE_DAYS}-day window; refresh them rather than trading on "
                    "a stale projection")

    playoffs = (snapshot.get("playoffs") or {})
    if not playoffs.get("playoff_scoring_periods"):
        raise TradeScanError(
            "the snapshot publishes no playoff scoring periods, so a playoff-weighted "
            "valuation would be scoring weeks nobody has identified")


def scan_trades(snapshot: Mapping[str, Any], board: pd.DataFrame,
                season: SeasonSimulation, *,
                playoff_season: SeasonSimulation | None = None,
                byes: Mapping[str, Sequence[int]] | None = None,
                upcoming_weeks: Sequence[int] = (),
                extra_locked: Mapping[int, str] | None = None,
                promote_shadow: bool = False,
                shadow_projections: Mapping[int, Any] | None = None,
                max_package: int = 2, top_candidates: int = 12,
                min_my_gain: float = 0.5, min_prob_not_worse: float = 0.55,
                their_tolerance: float = 0.0, max_results: int = 12,
                on_demand: bool = False) -> TradeScan:
    """Scan every opponent for packages that clear a two-sided gate.

    On demand only. This is not part of the weekly card, and `on_demand=True`
    is a required acknowledgement rather than a default: a trade scan needs
    current weekly or rest-of-season samples, complete identity coverage for
    both teams involved, exact live rosters, current locks and the league's
    real playoff weeks. A card that runs it on a schedule will eventually run
    it against a stale draft board, and a draft-day valuation presented in
    week 9 is not advice, it is an artifact of when the file was written.
    """
    if snapshot.get("schema_version") != "espn-league/1":
        raise TradeScanError(
            f"unsupported snapshot schema {snapshot.get('schema_version')!r}; "
            "this scan reads espn-league/1")
    if snapshot.get("roster_state") != "populated":
        raise TradeScanError(
            f"league roster state is {snapshot.get('roster_state')!r}: there are no rosters to "
            "trade. A pre-draft league has intentions, not teams.")
    if promote_shadow and not shadow_projections:
        raise TradeScanError(
            "promote_shadow requires shadow_projections: the board carries no K or D/ST "
            "distribution, and promoting them without one would value every kicker at zero.")
    _require_current_inputs(snapshot, season, on_demand=on_demand)

    byes = dict(byes or {})
    rules = lineup_rules_from_snapshot(snapshot)
    identity = map_identities(snapshot, board)
    locked = locked_players(snapshot, extra=extra_locked)
    board_index = board.set_index("player_id")

    my_team_id = int(snapshot["my_team"]["team_id"])
    my_team_name = snapshot["my_team"]["name"]
    team_names = {int(team["team_id"]): team["name"] for team in snapshot["teams"]}

    rostered: dict[int, list[dict]] = {team_id: [] for team_id in team_names}
    for player in _roster_players(snapshot):
        rostered[player["team_id"]].append(player)

    playoff_weeks = playoff_scoring_periods(snapshot)
    upcoming = tuple(upcoming_weeks) or tuple(
        range(int(snapshot["league"]["current_scoring_period"]),
              int(snapshot["league"]["current_scoring_period"]) + 3))

    evaluator = LineupEvaluator(season, rules.lineup)
    playoff_evaluator = (LineupEvaluator(playoff_season, rules.lineup)
                         if playoff_season is not None else None)

    def tradeable(players: Sequence[Mapping[str, Any]]) -> list[dict]:
        return [player for player in players
                if not player["on_ir"]
                and player["espn_id"] not in locked
                and player["espn_id"] in identity.espn_to_board]

    def _raw(player: Mapping[str, Any]) -> float:
        return float(board_index.loc[identity.espn_to_board[player["espn_id"]], "season_mean"])

    def _board_id(player: Mapping[str, Any]) -> str:
        return identity.espn_to_board[player["espn_id"]]

    def _surplus(player: Mapping[str, Any], roster_ids: Sequence[str], base: float) -> float:
        """Market value minus what my own lineup would actually miss.

        A fourth running back who never cracks the FLEX costs me almost nothing
        and is worth a starter's points to a team that would play him. That gap
        is where positive-sum trades live.
        """
        without = [pid for pid in roster_ids if pid != _board_id(player)]
        return _raw(player) - (base - float(evaluator.vector(without).mean()))

    def _marginal_gain(player: Mapping[str, Any], roster_ids: Sequence[str],
                       base: float) -> float:
        """What adding this player would add to that roster's optimal lineup."""
        return float(evaluator.vector([*roster_ids, _board_id(player)]).mean()) - base

    def _blend(players: Sequence[Mapping[str, Any]], first, second) -> list[dict]:
        """Half the slate from each ranking, deduplicated, order deterministic.

        Ranking candidates one way only ever proposes one shape of trade.
        Ranking by value to the receiver proposes their best starters, which a
        counterparty gate then rejects every time; ranking by the sender's
        surplus proposes players nobody wants. A two-sided gate needs a
        two-sided candidate generator, so both rankings get half the slate.
        """
        half = max(top_candidates // 2, 1)
        picked: dict[int, dict] = {}
        for ranking in (first, second):
            scored = sorted(players, key=lambda player: (-ranking(player), player["espn_id"]))
            for player in scored[:half]:
                picked.setdefault(player["espn_id"], player)
        return [picked[key] for key in sorted(picked)][:top_candidates]

    mine_all = rostered[my_team_id]
    my_board_ids = identity.matched(player["espn_id"] for player in mine_all)
    my_base = evaluator.vector(my_board_ids)
    my_playoff_base = (playoff_evaluator.vector(my_board_ids)
                       if playoff_evaluator is not None else None)

    considered = 0
    rejected = {"my_gain": 0, "my_downside": 0, "their_gain": 0, "legality": 0}
    packages: list[tuple[float, Package]] = []

    unresolved = blocked_teams(identity)
    if my_team_id in unresolved:
        return TradeScan(
            schema_version=SCHEMA_VERSION, state="blocked",
            generated_for={"team_id": my_team_id, "team_name": my_team_name},
            league={"league_id": snapshot["league"]["league_id"],
                    "season": snapshot["league"]["season"],
                    "name": snapshot["league"]["name"],
                    "snapshot_retrieved_at": snapshot.get("retrieved_at")},
            rules={"modeled_slots": dict(rules.modeled_slots),
                   "shadow_slots": dict(rules.shadow_slots)},
            identity={"matched": len(identity.espn_to_board),
                      "unmatched": [dict(row) for row in identity.unmatched],
                      "ambiguous": [dict(row) for row in identity.ambiguous]},
            locked=dict(locked), packages=(),
            hold_reason=("your own roster carries a player this board cannot identify"),
            gate={"blocked": True},
            context={"blocked_teams": {str(my_team_id): list(unresolved[my_team_id])}},
            warnings=(f"your own roster has {len(unresolved[my_team_id])} player(s) that cannot "
                      "be identified; every lineup total here would be computed without them, "
                      "so no package is offered",))

    for opponent_id, opponent_players in sorted(rostered.items()):
        if opponent_id == my_team_id:
            continue
        if opponent_id in unresolved:
            rejected["unresolved_identity"] = rejected.get("unresolved_identity", 0) + 1
            continue
        their_board_ids = identity.matched(p["espn_id"] for p in opponent_players)
        their_base = evaluator.vector(their_board_ids)
        my_mean, their_mean = float(my_base.mean()), float(their_base.mean())
        # Bound as defaults rather than closed over: these change every
        # opponent, and a late-binding closure would rank one team's players
        # against another team's roster.
        my_candidates = _blend(
            tradeable(mine_all),
            lambda player, ids=my_board_ids, base=my_mean: _surplus(player, ids, base),
            lambda player, ids=their_board_ids, base=their_mean: _marginal_gain(
                player, ids, base))
        their_candidates = _blend(
            tradeable(opponent_players),
            lambda player, ids=my_board_ids, base=my_mean: _marginal_gain(player, ids, base),
            lambda player, ids=their_board_ids, base=their_mean: _surplus(player, ids, base))

        for out_count in range(1, max_package + 1):
            for in_count in range(1, max_package + 1):
                for send in itertools.combinations(my_candidates, out_count):
                    send_ids = {player["espn_id"] for player in send}
                    for receive in itertools.combinations(their_candidates, in_count):
                        considered += 1
                        receive_ids = {player["espn_id"] for player in receive}

                        my_after = [p for p in mine_all if p["espn_id"] not in send_ids] + list(receive)
                        their_after = ([p for p in opponent_players
                                        if p["espn_id"] not in receive_ids] + list(send))

                        my_delta = distribution(
                            evaluator.vector(identity.matched(p["espn_id"] for p in my_after))
                            - my_base)
                        if my_delta.mean < min_my_gain:
                            rejected["my_gain"] += 1
                            continue
                        if my_delta.prob_not_worse < min_prob_not_worse:
                            rejected["my_downside"] += 1
                            continue
                        their_delta = distribution(
                            evaluator.vector(identity.matched(p["espn_id"] for p in their_after))
                            - their_base)
                        if their_delta.mean < -abs(their_tolerance):
                            rejected["their_gain"] += 1
                            continue

                        my_legal = check_legality(my_after, rules, label=my_team_name)
                        their_legal = check_legality(their_after, rules,
                                                     label=team_names[opponent_id])
                        if not (my_legal.legal and their_legal.legal):
                            rejected["legality"] += 1
                            continue

                        playoff_delta = None
                        if playoff_evaluator is not None and my_playoff_base is not None:
                            playoff_delta = distribution(
                                playoff_evaluator.vector(
                                    identity.matched(p["espn_id"] for p in my_after))
                                - my_playoff_base)

                        mine_side = PackageSide(
                            team_id=my_team_id, team_name=my_team_name,
                            sends=tuple(_describe(p, identity, board_index) for p in send),
                            receives=tuple(_describe(p, identity, board_index) for p in receive),
                            delta=my_delta, legality=my_legal,
                            roster_before=sum(1 for p in mine_all if not p["on_ir"]),
                            roster_after=my_legal.roster_size_after,
                            starters_on_bye_before=_starters_on_bye(mine_all, byes, upcoming),
                            starters_on_bye_after=_starters_on_bye(my_after, byes, upcoming))
                        theirs_side = PackageSide(
                            team_id=opponent_id, team_name=team_names[opponent_id],
                            sends=tuple(_describe(p, identity, board_index) for p in receive),
                            receives=tuple(_describe(p, identity, board_index) for p in send),
                            delta=their_delta, legality=their_legal,
                            roster_before=sum(1 for p in opponent_players if not p["on_ir"]),
                            roster_after=their_legal.roster_size_after,
                            starters_on_bye_before=_starters_on_bye(opponent_players, byes, upcoming),
                            starters_on_bye_after=_starters_on_bye(their_after, byes, upcoming))

                        rationale = [
                            f"my optimal lineup gains {my_delta.mean:+.2f} over the simulated "
                            f"span on average; it is better in {my_delta.prob_gain:.0%} of "
                            f"simulations, unchanged in {my_delta.prob_no_change:.0%}, and worse "
                            f"in {my_delta.prob_loss:.0%}",
                            f"their optimal lineup moves {their_delta.mean:+.2f} over the same span, "
                            f"so the package is not one-sided on the model",
                        ]
                        relief = (mine_side.starters_on_bye_before
                                  - mine_side.starters_on_bye_after)
                        if relief:
                            rationale.append(
                                f"bye relief over weeks {list(upcoming)}: "
                                f"{relief:+d} starters available")
                        if playoff_delta is not None:
                            rationale.append(
                                f"playoff weeks {list(playoff_weeks)} delta "
                                f"{playoff_delta.mean:+.2f} (league's own playoff "
                                f"periods, not an assumed 15-17)")

                        plausibility = [
                            f"they receive {', '.join(p['name'] for p in send)} and the model "
                            f"has their lineup {their_delta.mean:+.2f}, better in "
                            f"{their_delta.prob_gain:.0%} of simulations and worse in "
                            f"{their_delta.prob_loss:.0%}",
                            f"their roster goes {theirs_side.roster_before} -> "
                            f"{theirs_side.roster_after} of {rules.roster_size} and still fields "
                            f"a legal lineup",
                        ]
                        their_relief = (theirs_side.starters_on_bye_before
                                        - theirs_side.starters_on_bye_after)
                        if their_relief > 0:
                            plausibility.append(
                                f"it also relieves {their_relief} of their bye-week holes")

                        uncertainty = [
                            f"my delta spans {my_delta.p05:+.2f} to {my_delta.p95:+.2f} "
                            f"(5th-95th percentile over {my_delta.simulations} simulations); "
                            f"it changes nothing at all in {my_delta.prob_no_change:.0%} of them",
                            f"their delta spans {their_delta.p05:+.2f} to {their_delta.p95:+.2f}",
                            "model variance only: it excludes injury news after the snapshot, "
                            "trade-deadline behaviour, and how the other manager values anyone",
                        ]
                        if identity.unmatched or identity.ambiguous:
                            uncertainty.append(
                                f"{len(identity.unmatched)} unmatched and "
                                f"{len(identity.ambiguous)} ambiguous roster identities are "
                                "excluded from every lineup here")
                        if rules.shadow_slots:
                            uncertainty.append(
                                f"shadow slots {dict(rules.shadow_slots)} contribute nothing to "
                                "these deltas; K and D/ST are checked for legality only")

                        package = Package(
                            package_id=f"{my_team_id}-{opponent_id}-"
                                       f"{'_'.join(str(p['espn_id']) for p in send)}-for-"
                                       f"{'_'.join(str(p['espn_id']) for p in receive)}",
                            mine=mine_side, theirs=theirs_side, playoff_delta=playoff_delta,
                            rationale=tuple(rationale), plausibility=tuple(plausibility),
                            uncertainty=tuple(uncertainty))
                        score = my_delta.mean + (
                            0.5 * playoff_delta.mean if playoff_delta is not None else 0.0)
                        packages.append((score, package))

    packages.sort(key=lambda item: (-item[0], item[1].package_id))
    chosen = tuple(package for _, package in packages[:max_results])

    warnings: list[str] = []
    if unresolved:
        warnings.append(
            f"{len(unresolved)} team(s) were excluded entirely because a rostered player "
            "could not be identified; a lineup missing a starter is not a comparison")
    if identity.unmatched:
        warnings.append(f"{len(identity.unmatched)} rostered player(s) have no board projection")
    if identity.ambiguous:
        warnings.append(f"{len(identity.ambiguous)} rostered player(s) are ambiguous and excluded")
    if locked:
        warnings.append(f"{len(locked)} player(s) are locked and cannot be packaged")

    hold_reason = None
    if not chosen:
        hold_reason = (
            f"No package cleared the two-sided gate. {considered} combinations were evaluated: "
            f"{rejected['my_gain']} did not gain me at least {min_my_gain:+.2f} over the simulated span, "
            f"{rejected['my_downside']} gained on average but were worse for me "
            f"more often than a {1 - min_prob_not_worse:.0%} downside budget allows, {rejected['their_gain']} made the other roster "
            f"worse than the {their_tolerance:+.2f} tolerance, and {rejected['legality']} of the "
            f"packages that got that far would have left a roster illegal. (The gates run in that "
            f"order, so a later count only sees what earlier gates admitted.) Hold.")

    return TradeScan(
        schema_version=SCHEMA_VERSION,
        state="proposals" if chosen else "hold",
        generated_for={"team_id": my_team_id, "team_name": my_team_name},
        league={"league_id": snapshot["league"]["league_id"],
                "season": snapshot["league"]["season"],
                "name": snapshot["league"]["name"],
                "current_scoring_period": snapshot["league"]["current_scoring_period"],
                "snapshot_retrieved_at": snapshot.get("retrieved_at"),
                "league_hash": (snapshot.get("hashes") or {}).get("league"),
                "scoring_hash": (snapshot.get("hashes") or {}).get("scoring"),
                "roster_hash": (snapshot.get("hashes") or {}).get("roster")},
        rules={"modeled_slots": dict(rules.modeled_slots),
               "shadow_slots": dict(rules.shadow_slots),
               "roster_size": rules.roster_size, "bench_slots": rules.bench_slots,
               "ir_slots": rules.ir_slots,
               "playoff_scoring_periods": list(playoff_weeks),
               "upcoming_weeks": list(upcoming)},
        identity={"matched": len(identity.espn_to_board),
                  "unmatched": [dict(row) for row in identity.unmatched],
                  "ambiguous": [dict(row) for row in identity.ambiguous],
                  "shadow": [dict(row) for row in identity.shadow]},
        locked=dict(locked),
        packages=chosen, hold_reason=hold_reason,
        gate={"min_my_gain": min_my_gain, "min_prob_not_worse": min_prob_not_worse,
              "their_tolerance": their_tolerance, "max_package": max_package,
              "top_candidates": top_candidates, "considered": considered,
              "rejected": dict(rejected)},
        context={"teams": {str(team_id): name for team_id, name in sorted(team_names.items())},
                 "blocked_teams": {str(team_id): list(reasons)
                                   for team_id, reasons in unresolved.items()},
                 "byes_source_teams": sorted(byes)},
        warnings=tuple(warnings))


def scan_to_dict(scan: TradeScan) -> dict[str, Any]:
    payload = asdict(scan)
    payload["locked"] = {str(key): value for key, value in payload["locked"].items()}
    return payload
