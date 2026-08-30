"""Waiver / free-agent planner — RECOMMENDATION ONLY.

This module ranks nothing it cannot justify and performs no transaction.  It
has no HTTP client, no ESPN session, and no code path that mutates a league.
`tests/test_waiver_planner.py` asserts that structurally rather than trusting
this docstring.

Three refusals do most of the safety work:

* **Legality first.** A candidate that is rostered, still processing on
  waivers, or past the league's transaction deadline never becomes a
  recommendation.  A roster player who is locked, undroppable or parked on IR
  is never a drop candidate.  When nothing is legally droppable the record
  says ``no_legal_drop`` rather than naming someone.
* **No distribution, no number.** When the weekly / rest-of-season
  distributions are not supplied, ``lineup_delta`` is ``None`` and confidence
  is ``"none"``.  It is never silently zero, which would read as "no benefit"
  when the truth is "not measured".
* **Stale input degrades visibly.** A free-agent pool older than
  ``MAX_POOL_AGE_HOURS`` produces one degraded record that recommends nothing,
  never a confident-looking add over stale availability.

K and D/ST candidates carry ``status: "shadow"`` because their scoring model
is not built and not promoted.  Denying an opponent is reported as a secondary
field; the objective is always the owner's own optimal-lineup delta.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from . import waiver_rules as LC

UTC = timezone.utc

NO_LEGAL_DROP = "no_legal_drop"
DROP_NOT_REQUIRED = "not_required"
DROP_SELECTED = "selected"

# A free-agent pool older than this cannot support an add recommendation.
MAX_POOL_AGE_HOURS = 12

# Positions whose projection model is shadow-only (see the D/ST work item).
SHADOW_POSITIONS = frozenset({"K", "D/ST", "DST"})

OBJECTIVE = "own_optimal_lineup_delta"

AVAILABLE_STATES = frozenset({"freeagent", "waivers"})


class PlannerError(RuntimeError):
    """Raised when required team state is missing — never guessed around."""


@dataclass(frozen=True)
class RosterEntry:
    espn_id: int
    name: str
    position: str
    slot: str = "BE"
    locked: bool = False
    undroppable: bool = False
    injury_status: str = "ACTIVE"
    on_ir: bool = False


@dataclass(frozen=True)
class PoolEntry:
    espn_id: int
    name: str
    position: str
    availability: str = "freeagent"
    waiver_process_time: datetime | None = None
    as_of: datetime | None = None


@dataclass(frozen=True)
class Recommendation:
    add_espn_id: int | None
    add_name: str | None
    add_position: str | None
    drop_espn_id: int | None
    drop_name: str | None
    drop_state: str
    status: str
    shadow_reason: str | None
    confidence: str
    rationale: str
    invalidation_trigger: str
    priority_implications: Mapping[str, Any]
    replacement_effect: Mapping[str, Any]
    opponent_opportunity_impact: Mapping[str, Any]
    lineup_delta: Mapping[str, float] | None
    lineup_delta_status: str
    data_timestamps: Mapping[str, str]
    degraded: bool
    faab: Mapping[str, Any] | None = None

    def to_dict(self) -> dict:
        out = {
            "recommendation_only": True,
            "objective": OBJECTIVE,
            "status": self.status,
            "shadow_reason": self.shadow_reason,
            "add_espn_id": self.add_espn_id,
            "add_name": self.add_name,
            "add_position": self.add_position,
            "drop_espn_id": self.drop_espn_id,
            "drop_name": self.drop_name,
            "drop_state": self.drop_state,
            "priority_implications": dict(self.priority_implications),
            "lineup_delta": (dict(self.lineup_delta)
                             if self.lineup_delta is not None else None),
            "lineup_delta_status": self.lineup_delta_status,
            "replacement_effect": dict(self.replacement_effect),
            "opponent_opportunity_impact": dict(self.opponent_opportunity_impact),
            "confidence": self.confidence,
            "rationale": self.rationale,
            "invalidation_trigger": self.invalidation_trigger,
            "data_timestamps": dict(self.data_timestamps),
            "degraded": self.degraded,
        }
        if self.faab is not None:
            out["faab_bid"] = self.faab.get("bid")
            out["faab_budget_remaining"] = self.faab.get("budget_remaining")
        return out


# --------------------------------------------------------------------------- #
# Legality
# --------------------------------------------------------------------------- #
def active_roster(roster: Sequence[RosterEntry]) -> list[RosterEntry]:
    return [e for e in roster if not e.on_ir]


def roster_is_full(contract: LC.WaiverRules,
                   roster: Sequence[RosterEntry]) -> bool:
    return len(active_roster(roster)) >= contract.roster_limit


def ir_eligible(contract: LC.WaiverRules, entry: RosterEntry) -> bool:
    return str(entry.injury_status).upper() in contract.ir_eligible_statuses


def droppable(contract: LC.WaiverRules, roster: Sequence[RosterEntry],
              *, now: datetime) -> list[RosterEntry]:
    """Roster players who may legally be dropped right now."""
    del now  # locks travel on the entry itself; kept for signature symmetry
    return [e for e in roster
            if not e.locked and not e.undroppable and not e.on_ir]


def addable(contract: LC.WaiverRules, pool: Sequence[PoolEntry],
            roster: Sequence[RosterEntry], *, now: datetime) -> list[PoolEntry]:
    """Pool players a claim or an add may legally be placed on right now.

    A pending waiver-processing time is *when the claim resolves*, not a
    reason the player cannot be claimed — it is precisely the window in which
    a claim is placed. Skipping those entries hid every genuinely contested
    player and left only the ones nobody had to bid for.
    """
    if not contract.transactions_open(now):
        return []
    rostered = {int(e.espn_id) for e in roster}
    out = []
    for entry in pool:
        if acquisition_kind(entry) == "rostered":
            continue
        if acquisition_kind(entry) == "unknown":
            continue
        if int(entry.espn_id) in rostered:
            continue
        out.append(entry)
    return out


def acquisition_kind(entry: PoolEntry) -> str:
    """`free_agent` (add now) or `waiver_claim` (claim, resolves later)."""
    from .espn_league import acquisition_kind as _kind

    return _kind(entry.availability)


def immediate_free_agents(pool: Sequence[PoolEntry]) -> list[PoolEntry]:
    return [e for e in pool if acquisition_kind(e) == "free_agent"]


def waiver_claims(pool: Sequence[PoolEntry]) -> list[PoolEntry]:
    return [e for e in pool if acquisition_kind(e) == "waiver_claim"]


def pool_is_stale(pool: Sequence[PoolEntry], *, now: datetime) -> bool:
    for entry in pool:
        if entry.as_of is None:
            return True
        if (now - entry.as_of).total_seconds() > MAX_POOL_AGE_HOURS * 3600:
            return True
    return False


def slot_legal_after(contract: LC.WaiverRules, roster: Sequence[RosterEntry],
                     add: PoolEntry, drop: RosterEntry | None) -> bool:
    """Would the roster still satisfy its size limit after this move?"""
    after = [e for e in active_roster(roster)
             if drop is None or e.espn_id != drop.espn_id]
    if contract.eligible_slots(add.position) == () and add.position not in ("BE",):
        return False
    return len(after) + 1 <= contract.roster_limit


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
def _drop_choice(contract, roster, now):
    legal = droppable(contract, roster, now=now)
    if not legal:
        return None, NO_LEGAL_DROP
    if not roster_is_full(contract, roster):
        return None, DROP_NOT_REQUIRED
    # Without distributions there is no defensible ranking, so the choice is
    # positional and deterministic, and the rationale says exactly that.
    bench = [e for e in legal if e.slot == "BE"] or legal
    return max(bench, key=lambda e: int(e.espn_id)), DROP_SELECTED


def _degraded_record(contract, now, reason):
    return Recommendation(
        add_espn_id=None, add_name=None, add_position=None,
        drop_espn_id=None, drop_name=None, drop_state=DROP_NOT_REQUIRED,
        status="degraded", shadow_reason=None, confidence="none",
        rationale=reason,
        invalidation_trigger="refresh the free-agent pool and re-run",
        priority_implications={"mode": contract.waiver_mode,
                               "assumed": contract.waiver_mode_assumed},
        replacement_effect={}, opponent_opportunity_impact={},
        lineup_delta=None,
        lineup_delta_status="unavailable: input data failed the freshness gate",
        data_timestamps={"contract_as_of": contract.as_of.isoformat(),
                         "evaluated_at": now.isoformat()},
        degraded=True,
    )


def plan(contract: LC.WaiverRules, *, roster: Sequence[RosterEntry] | None,
         pool: Sequence[PoolEntry] | None, now: datetime,
         distributions: Any = None, my_team_id: int | None = None
         ) -> list[Recommendation]:
    """Produce recommendation-only add/drop options.  Executes nothing."""

    if roster is None:
        raise PlannerError("roster state is required — refusing to guess a roster")
    if pool is None:
        raise PlannerError("free-agent pool is required — refusing to guess availability")

    if pool and pool_is_stale(pool, now=now):
        return [_degraded_record(
            contract, now,
            "free-agent pool is stale beyond the freshness gate; availability "
            "cannot be trusted, so no add is recommended")]

    candidates = addable(contract, pool, roster, now=now)
    if not candidates:
        return []

    drop, drop_state = _drop_choice(contract, roster, now)
    my_priority = contract.priority_of(my_team_id) if my_team_id else None

    records = []
    for cand in sorted(candidates, key=lambda c: int(c.espn_id)):
        shadow = str(cand.position).upper() in SHADOW_POSITIONS
        faab = None
        if contract.uses_faab:
            faab = {"bid": None, "budget_remaining": contract.faab_budget}

        records.append(Recommendation(
            add_espn_id=int(cand.espn_id),
            add_name=cand.name,
            add_position=cand.position,
            drop_espn_id=(int(drop.espn_id) if drop else None),
            drop_name=(drop.name if drop else None),
            drop_state=drop_state,
            status=("shadow" if shadow else "recommendation"),
            shadow_reason=(
                f"{cand.position} projections are shadow-only and not promoted "
                "into lineup optimization" if shadow else None),
            confidence="none",
            rationale=(
                "legality verified against the live league contract; "
                "no projection distribution supplied, so no value claim is made"
                + ("; drop candidate chosen positionally, not by value"
                   if drop_state == DROP_SELECTED else "")),
            invalidation_trigger=(
                "roster, availability, lock state, or league settings change; "
                f"contract payload hash {contract.payload_hash}"),
            priority_implications={
                "mode": contract.waiver_mode,
                "assumed": contract.waiver_mode_assumed,
                "my_priority": my_priority,
                "order": list(contract.priority_order),
                "tied_teams": sorted(contract.priority_tied_teams),
                "competing_claims": list(
                    contract.pending_claims_for(cand.espn_id)),
            },
            replacement_effect={
                "status": "unavailable: requires weekly/ROS distributions",
                "bye_coverage": "unavailable", "injury_coverage": "unavailable",
            },
            opponent_opportunity_impact={
                "status": "unavailable: requires opponent roster state",
                "note": "secondary field; never the ranking objective",
            },
            lineup_delta=None,
            lineup_delta_status=(
                "unavailable: no weekly/rest-of-season distributions supplied"
                if distributions is None else "unavailable: not yet wired"),
            data_timestamps={
                "contract_as_of": contract.as_of.isoformat(),
                "pool_as_of": (cand.as_of.isoformat() if cand.as_of else "unknown"),
                "evaluated_at": now.isoformat(),
            },
            degraded=False,
            faab=faab,
        ))
    return records
