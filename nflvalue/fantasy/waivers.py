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

import numpy as np

from . import waiver_rules as LC

UTC = timezone.utc

NO_LEGAL_DROP = "no_legal_drop"
DROP_NOT_REQUIRED = "not_required"
DROP_SELECTED = "selected"
#: A legal drop exists but cannot be valued, so none is nominated.
NO_VALUED_DROP = "no_valued_drop"

# A free-agent pool older than this cannot support an add recommendation.
MAX_POOL_AGE_HOURS = 12

# Positions whose projection model is shadow-only (see the D/ST work item).
SHADOW_POSITIONS = frozenset({"K", "D/ST", "DST"})

OBJECTIVE = "own_optimal_lineup_delta"

#: What a waiver row has to clear before it exists at all. Declared here, ahead
#: of any run, so a thin result cannot be answered by moving the line.
#:
#: The planner used to emit a row for every legally addable player, ordered by
#: ESPN player id, with `lineup_delta` hardcoded to None and a `distributions`
#: argument it accepted and never read. That is a list of transactions the
#: rules permit, presented where a recommendation belongs. A row now requires
#: all four of: a legal claim or immediate add, an identified legal drop, a
#: computed joint-sample delta, and that delta clearing the gate.
WAIVER_GATE = {
    "min_mean_lineup_delta": 0.5,
    "min_prob_improves": 0.60,
    "min_simulations": 200,
}

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
def _lineup_delta(contract: LC.WaiverRules, roster: Sequence[RosterEntry],
                  add: PoolEntry, drop: RosterEntry | None,
                  samples: Mapping[int, Any] | None) -> dict | None:
    """The paired per-simulation change in this roster's own optimal lineup.

    Both lineups are solved on the same simulation row, so the difference is a
    real distribution over weeks rather than a gap between two summaries. No
    samples means no number: the planner reports that it cannot value the move
    instead of ranking by player id and calling it a recommendation.
    """
    from . import lineup as lineup_engine

    if not samples:
        return None
    before = [entry for entry in roster if not entry.on_ir]
    after = [entry for entry in before if drop is None or entry.espn_id != drop.espn_id]

    def matrix(entries, extra=None):
        rows = list(entries) + ([extra] if extra is not None else [])
        columns, players = [], []
        for entry in rows:
            draw = samples.get(int(entry.espn_id))
            if draw is None:
                return None, None
            columns.append(np.asarray(draw, dtype=float))
            players.append(lineup_engine.LineupPlayer(
                player_id=int(entry.espn_id),
                eligible_slots=frozenset(contract.eligible_slots(str(entry.position))),
                position=str(entry.position)))
        if not columns:
            return None, None
        widths = {column.shape for column in columns}
        if len(widths) != 1:
            return None, None
        return np.column_stack(columns), tuple(players)

    slots = {slot.label: slot.count for slot in contract.slots
             if slot.label not in ("BE", "IR")}
    base_matrix, base_players = matrix(before)
    after_matrix, after_players = matrix(after, add)
    if base_matrix is None or after_matrix is None:
        return None
    base = lineup_engine.optimize_matrix(base_matrix, base_players, slots)
    moved = lineup_engine.optimize_matrix(after_matrix, after_players, slots)
    delta = moved - base
    return {
        "own_optimal_lineup_delta": round(float(delta.mean()), 3),
        "median": round(float(np.median(delta)), 3),
        "p10": round(float(np.percentile(delta, 10)), 3),
        "p90": round(float(np.percentile(delta, 90)), 3),
        # Model-relative: the share of the model's own simulated weeks in which
        # the move helps. Not a calibrated probability.
        "model_relative_prob_improves": round(float((delta > 0).mean()), 4),
        "simulations": int(delta.size),
        "basis": "paired joint simulation rows, both lineups solved per row",
    }


def _clears_gate(delta: Mapping[str, Any] | None) -> tuple[bool, str]:
    if delta is None:
        return False, "no joint samples were supplied, so the move cannot be valued"
    if int(delta["simulations"]) < WAIVER_GATE["min_simulations"]:
        return False, (f"{delta['simulations']} simulation rows is below the declared minimum "
                       f"of {WAIVER_GATE['min_simulations']}")
    if delta["own_optimal_lineup_delta"] < WAIVER_GATE["min_mean_lineup_delta"]:
        return False, (f"mean lineup gain {delta['own_optimal_lineup_delta']} is below the "
                       f"declared gate of {WAIVER_GATE['min_mean_lineup_delta']}")
    if delta["model_relative_prob_improves"] < WAIVER_GATE["min_prob_improves"]:
        return False, (f"the move helps in {delta['model_relative_prob_improves']:.0%} of "
                       f"simulated weeks, below the declared "
                       f"{WAIVER_GATE['min_prob_improves']:.0%}")
    return True, ""


def _drop_choice(contract, roster, now, samples=None, add=None):
    """The legal drop that costs the least, valued rather than picked by id.

    This used to return the bench player with the highest ESPN player id, which
    is deterministic and meaningless. With joint samples the drop is the one
    whose removal costs this roster's own optimal lineup the least; without
    them there is no defensible ranking and the planner says so rather than
    dropping somebody arbitrary.
    """
    legal = droppable(contract, roster, now=now)
    if not legal:
        return None, NO_LEGAL_DROP
    if not roster_is_full(contract, roster):
        return None, DROP_NOT_REQUIRED
    if not samples or add is None:
        return None, NO_VALUED_DROP
    scored = []
    for candidate in legal:
        delta = _lineup_delta(contract, roster, add, candidate, samples)
        if delta is not None:
            scored.append((delta["own_optimal_lineup_delta"], int(candidate.espn_id), candidate))
    if not scored:
        return None, NO_VALUED_DROP
    scored.sort(key=lambda row: (-row[0], row[1]))
    return scored[0][2], DROP_SELECTED


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
         distributions: Mapping[int, Any] | None = None, my_team_id: int | None = None
         ) -> list[Recommendation]:
    """Add/drop options that clear the declared gate. Executes nothing.

    Four things have to be true before a row exists: the add is legally
    available now (as an immediate free agent or as a claim), a legal drop is
    identified, the change in this roster's own optimal lineup is computable
    from joint samples, and that change clears `WAIVER_GATE`. A row missing any
    of them is not a weaker recommendation, it is not a recommendation, and the
    planner returns the reason instead.
    """
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

    my_priority = contract.priority_of(my_team_id) if my_team_id else None
    records: list[Recommendation] = []
    for cand in sorted(candidates, key=lambda c: int(c.espn_id)):
        shadow = str(cand.position).upper() in SHADOW_POSITIONS
        drop, drop_state = _drop_choice(contract, roster, now, distributions, cand)
        delta = (None if drop_state in (NO_LEGAL_DROP, NO_VALUED_DROP)
                 else _lineup_delta(contract, roster, cand, drop, distributions))
        clears, blocked_reason = _clears_gate(delta)

        if shadow:
            status, reason = "shadow", (
                f"{cand.position} projections are shadow-only and never enter the lineup "
                "objective, so this add cannot be valued against it")
        elif drop_state == NO_LEGAL_DROP:
            status, reason = "no_current_pick", (
                "no legal drop exists, so this add cannot be made at all")
        elif drop_state == NO_VALUED_DROP:
            status, reason = "no_current_pick", (
                "a legal drop exists but cannot be valued without joint samples; "
                "nominating one by roster position would be an arbitrary cut")
        elif not clears:
            status, reason = "no_current_pick", blocked_reason
        else:
            status, reason = "recommendation", None

        if status != "recommendation":
            # Kept visible rather than dropped: "we looked and it did not clear"
            # is information, and an empty list is indistinguishable from an
            # engine that never ran.
            records.append(_no_pick_record(contract, now, cand, drop, drop_state,
                                           status, reason, delta, my_priority,
                                           shadow_reason=reason if shadow else None))
            continue

        records.append(Recommendation(
            add_espn_id=int(cand.espn_id), add_name=cand.name, add_position=cand.position,
            drop_espn_id=(int(drop.espn_id) if drop else None),
            drop_name=(drop.name if drop else None), drop_state=drop_state,
            status="recommendation", shadow_reason=None,
            confidence="none",
            rationale=(
                f"adding {cand.name} for {drop.name if drop else 'an open spot'} raises this "
                f"roster's own optimal lineup by {delta['own_optimal_lineup_delta']} points on "
                f"average, helping in {delta['model_relative_prob_improves']:.0%} of simulated "
                "weeks; the drop is the legal cut that costs the lineup least"),
            invalidation_trigger=(
                "roster, availability, lock state, or league settings change; "
                f"contract payload hash {contract.payload_hash}"),
            priority_implications={
                "mode": contract.waiver_mode, "assumed": contract.waiver_mode_assumed,
                "my_priority": my_priority, "order": list(contract.priority_order),
                "tied_teams": sorted(contract.priority_tied_teams),
                "competing_claims": list(contract.pending_claims_for(cand.espn_id)),
                "acquisition_kind": acquisition_kind(cand),
            },
            replacement_effect={"status": "unavailable: requires bye/injury coverage modelling",
                                "bye_coverage": "unavailable", "injury_coverage": "unavailable"},
            opponent_opportunity_impact={
                "status": "unavailable: requires opponent roster state",
                "note": "secondary field; never the ranking objective"},
            lineup_delta=delta,
            lineup_delta_status="ok",
            data_timestamps={
                "contract_as_of": contract.as_of.isoformat(),
                "pool_as_of": (cand.as_of.isoformat() if cand.as_of else "unknown"),
                "evaluated_at": now.isoformat()},
            degraded=False,
            faab=({"bid": None, "budget_remaining": contract.faab_budget}
                  if contract.uses_faab else None),
        ))

    ranked = [r for r in records if r.status == "recommendation"]
    rest = [r for r in records if r.status != "recommendation"]
    ranked.sort(key=lambda r: (-(r.lineup_delta or {}).get("own_optimal_lineup_delta", 0.0),
                               int(r.add_espn_id or 0)))
    return ranked + rest


def _no_pick_record(contract, now, cand, drop, drop_state, status, reason, delta,
                    my_priority, *, shadow_reason=None) -> Recommendation:
    """A candidate that was considered and did not clear, with the reason."""
    return Recommendation(
        add_espn_id=int(cand.espn_id), add_name=cand.name, add_position=cand.position,
        drop_espn_id=(int(drop.espn_id) if drop else None),
        drop_name=(drop.name if drop else None), drop_state=drop_state,
        status=status, shadow_reason=shadow_reason, confidence="none",
        rationale=reason or "did not clear the declared waiver gate",
        invalidation_trigger=(
            "roster, availability, lock state, or league settings change; "
            f"contract payload hash {contract.payload_hash}"),
        priority_implications={
            "mode": contract.waiver_mode, "assumed": contract.waiver_mode_assumed,
            "my_priority": my_priority, "order": list(contract.priority_order),
            "tied_teams": sorted(contract.priority_tied_teams),
            "competing_claims": list(contract.pending_claims_for(cand.espn_id)),
            "acquisition_kind": acquisition_kind(cand)},
        replacement_effect={"status": "unavailable", "bye_coverage": "unavailable",
                            "injury_coverage": "unavailable"},
        opponent_opportunity_impact={"status": "unavailable",
                                     "note": "secondary field; never the ranking objective"},
        lineup_delta=delta,
        lineup_delta_status=("ok" if delta else "unavailable"),
        data_timestamps={
            "contract_as_of": contract.as_of.isoformat(),
            "pool_as_of": (cand.as_of.isoformat() if cand.as_of else "unknown"),
            "evaluated_at": now.isoformat()},
        degraded=False,
        faab=({"bid": None, "budget_remaining": contract.faab_budget}
              if contract.uses_faab else None),
    )
