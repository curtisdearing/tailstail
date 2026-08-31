"""Waiver and transaction rules, read from the canonical league snapshot.

This module replaces `league_contract.py`, which parsed a *second* copy of the
league out of the raw ESPN payload and published its own 16-character
`scoring_hash` and `roster_hash` over uncanonicalised blobs. Two modules
answering "what are this league's scoring rules?" with two different digests
is not redundancy, it is a coin flip: the waiver planner stamped one on its
recommendations and the shadow projections stamped the other, and nothing
could tell you they described the same league.

So there is one snapshot now, and the rules travel inside it. Everything here
is a *view* over `espn-league/1` — no parsing, no defaults, and no hashing.
The hashes are the ones the adapter embedded from the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from .espn_league import (
    SCHEMA_VERSION,
    WAIVER_MODE_FAAB,
    WAIVER_MODE_INVERSE,
    WAIVER_MODE_ROLLING,
    parse_timestamp,
)

UTC = timezone.utc

# Kept as re-exports so existing callers and tests keep their vocabulary.
WAIVER_INVERSE_STANDINGS = WAIVER_MODE_INVERSE
WAIVER_FAAB = WAIVER_MODE_FAAB
WAIVER_CONTINUOUS = WAIVER_MODE_ROLLING

FLEX_ELIGIBLE = ("RB", "WR", "TE")
IR_LABEL = "IR"
BENCH_LABEL = "BE"
FLEX_LABEL = "FLEX"

#: A player may only occupy an IR slot with one of these ESPN statuses.
IR_ELIGIBLE_STATUSES = frozenset({"OUT", "INJURY_RESERVE", "IR", "NA", "SUSPENSION"})


class ContractError(RuntimeError):
    """Raised when the snapshot cannot be read as a rules source."""


@dataclass(frozen=True)
class RosterSlot:
    slot_id: int
    label: str
    count: int
    eligible_positions: tuple[str, ...]


@dataclass(frozen=True)
class PendingClaim:
    transaction_id: str
    team_id: int
    player_id: int
    kind: str


@dataclass(frozen=True)
class WaiverRules:
    """Everything a recommendation is allowed to rely on, and where it came from."""

    league_id: int
    season: int
    scoring_period: int
    slots: tuple[RosterSlot, ...]
    roster_limit: int
    ir_slots: int
    ir_eligible_statuses: frozenset[str]
    waiver_mode: str
    waiver_mode_assumed: bool
    uses_faab: bool
    faab_budget: float | None
    priority_order: tuple[int, ...]
    priority_tied_teams: frozenset[int]
    transaction_deadline: datetime | None
    acquisition_limit: int | None
    pending_claims: tuple[PendingClaim, ...]
    scoring_hash: str
    roster_hash: str
    payload_hash: str
    as_of: datetime
    notes: tuple[str, ...] = field(default=())

    # -- queries ---------------------------------------------------------- #
    def transactions_open(self, now: datetime) -> bool:
        if self.transaction_deadline is None:
            return True
        return now < self.transaction_deadline

    def pending_claims_for(self, player_id: int) -> tuple[int, ...]:
        return tuple(sorted(
            c.team_id for c in self.pending_claims
            if int(c.player_id) == int(player_id)))

    def slot_for(self, label: str) -> RosterSlot | None:
        for slot in self.slots:
            if slot.label == label:
                return slot
        return None

    def eligible_slots(self, position: str) -> tuple[str, ...]:
        return tuple(s.label for s in self.slots if position in s.eligible_positions)

    def priority_of(self, team_id: int) -> int | None:
        try:
            return self.priority_order.index(int(team_id)) + 1
        except ValueError:
            return None

    def to_dict(self) -> dict:
        return {
            "league_id": self.league_id, "season": self.season,
            "scoring_period": self.scoring_period,
            "roster_limit": self.roster_limit, "ir_slots": self.ir_slots,
            "waiver_mode": self.waiver_mode,
            "waiver_mode_assumed": self.waiver_mode_assumed,
            "uses_faab": self.uses_faab, "faab_budget": self.faab_budget,
            "priority_order": list(self.priority_order),
            "priority_tied_teams": sorted(self.priority_tied_teams),
            "transaction_deadline": (self.transaction_deadline.isoformat()
                                     if self.transaction_deadline else None),
            "scoring_hash": self.scoring_hash,
            "roster_hash": self.roster_hash,
            "payload_hash": self.payload_hash,
            "as_of": self.as_of.isoformat(),
            "notes": list(self.notes),
        }


def _slots_from_counts(counts: Mapping[str, Any]) -> tuple[tuple[RosterSlot, ...], int, int]:
    """Slots as the adapter named them. IR is excluded from the active limit."""
    slots: list[RosterSlot] = []
    limit = 0
    ir = 0
    for label, raw_count in sorted(counts.items()):
        count = int(raw_count)
        if count <= 0:
            continue
        if label == IR_LABEL:
            ir = count
            eligible: tuple[str, ...] = ()
        elif label == FLEX_LABEL:
            eligible = FLEX_ELIGIBLE
            limit += count
        elif label == BENCH_LABEL:
            eligible = ()
            limit += count
        else:
            eligible = (label,)
            limit += count
        slots.append(RosterSlot(slot_id=-1, label=str(label), count=count,
                                eligible_positions=eligible))
    return tuple(slots), limit, ir


def from_snapshot(snapshot: Mapping[str, Any], *,
                  as_of: datetime | None = None) -> WaiverRules:
    """Read waiver rules off a canonical `espn-league/1` snapshot.

    The waiver *order* is taken exactly as ESPN publishes it. ESPN already
    applies the league's rule when it fills `teams[].waiverRank`, so deriving
    the order again from the standings — and inverting it, because
    `waiverOrderReset` means "resets to inverse standings" — produces a
    confidently reversed priority list. The published rank is the answer.
    """
    if not isinstance(snapshot, Mapping):
        raise ContractError(f"snapshot is not a mapping ({type(snapshot).__name__})")
    version = snapshot.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ContractError(
            f"unsupported snapshot schema {version!r}; waiver rules read {SCHEMA_VERSION}")

    league = snapshot.get("league") or {}
    waivers = snapshot.get("waivers") or {}
    roster_settings = snapshot.get("roster_settings") or {}
    hashes = snapshot.get("hashes") or {}

    slots, limit, ir = _slots_from_counts(roster_settings.get("lineup_slot_counts") or {})

    # ESPN's own rank: 1 is first claim. Sorting by it gives the order; it is
    # not re-derived and not inverted.
    priority = {int(team): int(rank) for team, rank in (waivers.get("team_priority") or {}).items()}
    ranked = sorted((rank, team) for team, rank in priority.items() if rank > 0)
    order = tuple(team for _, team in ranked)
    counts: dict[int, int] = {}
    for rank, _ in ranked:
        counts[rank] = counts.get(rank, 0) + 1
    tied = frozenset(team for rank, team in ranked if counts[rank] > 1)

    pending = tuple(
        PendingClaim(transaction_id=str(txn.get("transaction_id") or ""),
                     team_id=int(txn.get("team_id") or 0),
                     player_id=int((item or {}).get("playerId") or 0),
                     kind=str(txn.get("type") or "UNKNOWN"))
        for txn in ((snapshot.get("transactions") or {}).get("pending") or [])
        for item in (txn.get("items") or [])
        if (item or {}).get("playerId") is not None
    )

    acquisition_limit = waivers.get("acquisition_limit")
    if acquisition_limit is not None and int(acquisition_limit) < 0:
        acquisition_limit = None

    retrieved = parse_timestamp(snapshot.get("retrieved_at"))
    if retrieved is None:
        raise ContractError("snapshot carries no readable retrieved_at")

    notes: list[str] = []
    mode = str(waivers.get("mode") or "")
    if not mode:
        raise ContractError("snapshot waivers block states no mode")
    if mode == WAIVER_MODE_INVERSE:
        notes.append("waiver order resets weekly to inverse standings (waiverOrderReset=true)")

    return WaiverRules(
        league_id=int(league.get("league_id") or 0),
        season=int(league.get("season") or 0),
        scoring_period=int(league.get("current_scoring_period") or 0),
        slots=slots,
        roster_limit=limit,
        ir_slots=ir,
        ir_eligible_statuses=IR_ELIGIBLE_STATUSES,
        waiver_mode=mode,
        waiver_mode_assumed=False,      # the snapshot states it; nothing is guessed
        uses_faab=bool(waivers.get("uses_acquisition_budget")),
        faab_budget=(float(waivers["acquisition_budget"])
                     if waivers.get("uses_acquisition_budget") else None),
        priority_order=order,
        priority_tied_teams=tied,
        transaction_deadline=parse_timestamp(waivers.get("transaction_deadline")),
        acquisition_limit=acquisition_limit,
        pending_claims=pending,
        scoring_hash=str(hashes.get("scoring") or ""),
        roster_hash=str(hashes.get("roster") or ""),
        payload_hash=str(hashes.get("league") or ""),
        as_of=as_of or retrieved,
        notes=tuple(notes),
    )
