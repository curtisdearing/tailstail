"""Curtis-specific recommendation contract built from a read-only ESPN snapshot.

This is a Monitor surface.  Its job is to say what is true about one fantasy
team right now, and to say *nothing* — loudly, with a reason — when the inputs
cannot support a recommendation.  Every section therefore has two shapes:
``status="ok"`` with content, or ``status="no_current_pick"`` with a reason.
There is no third shape, and in particular there is no shape that renders a
plausible recommendation from inputs that do not belong to this league.

Fail-closed design
------------------
The guard is POSITIVE: a source must prove it belongs to the live league
(:func:`reject_source`).  It is deliberately not a blocklist of suspicious
names, because two managers in this league are genuinely called "Team 7" and
"Team 8" — a name-shaped guard would reject real people while still admitting a
12-team mock board whose teams have realistic names.  Requiring
``league_id`` + ``season`` + ``league_size`` + ``captured_at`` to match rejects
``data/draft_board_2026_6team.csv``, ``data/draft_board_2026_12team.csv`` and
``data/trade_scan.json`` (``my_team: "Team1"``) by construction.

Hashes
------
``scoring_hash`` and ``roster_slot_hash`` are NOT computed here.  They come from
:class:`nflvalue.fantasy.espn_contract.LeagueContract`, the single authority.
Three modules once each computed a "scoring_hash" for this league and returned
three different values — a hash whose canonicalization is not pinned to one
implementation is not evidence.  With no contract the hashes are null and say
why.

Freeze
------
Nothing here is a lever and nothing here grades anything.  Projections are
consumed as given; K and D/ST are labelled shadow until promoted through the
2026 protocol's normal gate.  See ``docs/PROTOCOL_FREEZE_2026.md`` §3, §5.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "my_team/1.0.0"

#: Sections that owe the reader a rationale, a confidence, and a statement of
#: what would make the recommendation wrong.
ACTIONABLE_SECTIONS = (
    "draft", "optimal_lineup", "start_sit", "waivers", "trades",
    "kicker_shadow", "dst_shadow",
)

class SnapshotRejected(ValueError):
    """The snapshot is not one this builder can read. Never degraded around.

    A wrong-schema snapshot is worse than no snapshot: every section below
    would render "no roster", "no picks", "no trades" — which is what a real
    empty league looks like — and nothing would say the reader and the writer
    disagreed about the shape.
    """


SLOT_NAMES: Mapping[int, str] = {
    0: "QB", 2: "RB", 4: "WR", 6: "TE", 16: "D/ST", 17: "K",
    20: "BE", 21: "IR", 23: "FLEX",
}
BENCH_SLOTS = frozenset({20, 21})
FLEX_SLOT = 23
FLEX_POSITIONS = frozenset({"RB", "WR", "TE"})
#: Statuses that make a player ineligible to start.  DOUBTFUL is deliberately
#: NOT here: it is a judgement call the reader makes, not one this file makes.
INELIGIBLE_INJURY = frozenset({"OUT", "IR", "INJURY_RESERVE", "SUSPENSION", "NA"})

FRESH_HOURS = 6.0
STALE_HOURS = 24.0

_NO_WAIVER_PLAN = (
    "no waiver plan was supplied for this snapshot — nflvalue.fantasy.waivers.plan() "
    "produces one from a live roster and free-agent pool, and nothing is guessed "
    "in its absence"
)
_NO_WAIVER_BENEFIT = (
    "the waiver planner returned no legal add that improves the lineup"
)


# --------------------------------------------------------------------------- #
# Hashes come from the league contract — never reimplemented here
# --------------------------------------------------------------------------- #
# A hash is only evidence if exactly one implementation produces it.  Three
# modules once computed a "scoring_hash" for this league and returned three
# different values, so this module no longer computes one at all:
# ``nflvalue.fantasy.espn_contract.LeagueContract`` is the single authority
# (it owns the stat registry, the D/ST bands and the field-goal buckets, and
# its digest is order-independent and value-sensitive under test).
#
# When no contract is supplied the hashes are ``None`` with a stated reason.
# A homegrown second-best hash would be worse than no hash: it would look like
# provenance while proving nothing.
HASH_SOURCE = "nflvalue.fantasy.espn_contract.LeagueContract"
SNAPSHOT_HASH_SOURCE = "nflvalue.fantasy.espn_league snapshot hashes"
NO_CONTRACT_REASON = (
    "no league contract was supplied, so scoring and roster-slot identity cannot "
    "be established; a locally computed hash would not be comparable to anything"
)


def contract_hashes(contract: Any | None, *,
                    snapshot: Mapping[str, Any] | None = None) -> dict:
    """Scoring and roster-slot identity — the snapshot's, or refused.

    The adapter embeds the contract's own hashes, so the snapshot and the
    contract cannot disagree unless one of them was built from a different
    league. If they do disagree, that is reported rather than resolved by
    preferring one: silently picking a winner is how a stale contract gets
    stamped on a current snapshot.
    """
    embedded = dict((snapshot or {}).get("hashes") or {})
    if contract is None:
        if embedded.get("scoring") and embedded.get("roster"):
            return {"scoring_hash": embedded["scoring"], "roster_slot_hash": embedded["roster"],
                    "hash_source": f"{SNAPSHOT_HASH_SOURCE} (embedded)", "hash_reason": None}
        return {"scoring_hash": None, "roster_slot_hash": None,
                "hash_source": None, "hash_reason": NO_CONTRACT_REASON}

    scoring = getattr(contract, "scoring_hash", None)
    roster = getattr(contract, "roster_slot_hash", None)
    if embedded and (embedded.get("scoring") != scoring or embedded.get("roster") != roster):
        return {"scoring_hash": None, "roster_slot_hash": None,
                "hash_source": None,
                "hash_reason": ("the supplied contract and the snapshot describe different "
                                f"rules (contract {scoring!r}/{roster!r} vs snapshot "
                                f"{embedded.get('scoring')!r}/{embedded.get('roster')!r}); "
                                "refusing to stamp either onto this output")}
    return {
        "scoring_hash": scoring,
        "roster_slot_hash": roster,
        "hash_source": f"{HASH_SOURCE} v{getattr(contract, 'contract_version', '?')}",
        "hash_reason": None,
    }


# --------------------------------------------------------------------------- #
# Provenance guard
# --------------------------------------------------------------------------- #
def reject_source(source: Any, snapshot: Mapping[str, Any], *, what: str) -> str | None:
    """Return why *source* may not be used for this league, or None if it may.

    Positive proof, in this order: it is a mapping; it names the same league;
    the same season; the same league size; it says when it was captured; and,
    if it claims a scoring period at all, the same one the snapshot is in.
    """
    league = snapshot.get("league", {})
    if not isinstance(source, Mapping):
        return f"{what}: not a structured source ({type(source).__name__})"

    league_id = str(league.get("league_id", ""))
    claimed = source.get("league_id")
    if claimed is None:
        return (f"{what}: carries no league_id, so it cannot be shown to describe "
                f"league {league_id}")
    if str(claimed) != league_id:
        return f"{what}: league_id {claimed!r} is not this league ({league_id})"

    season = league.get("season")
    if source.get("season") is not None and int(source["season"]) != int(season):
        return f"{what}: season {source['season']} is not the live season ({season})"

    size = league.get("size")
    claimed_size = source.get("league_size")
    if claimed_size is None:
        return f"{what}: carries no league_size, so it cannot be matched to a {size}-team league"
    if int(claimed_size) != int(size):
        return (f"{what}: built for a {claimed_size}-team league, but this league has "
                f"{size} teams — its values do not transfer")

    if not source.get("captured_at"):
        return f"{what}: carries no captured_at, so its age cannot be established"

    period = league.get("scoring_period_id")
    if source.get("scoring_period") is not None and int(source["scoring_period"]) != int(period):
        return (f"{what}: built for scoring_period {source['scoring_period']}, but the live "
                f"scoring_period is {period} — a prior card is not the current one")
    return None


# --------------------------------------------------------------------------- #
# Freshness
# --------------------------------------------------------------------------- #
def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def freshness(snapshot: Mapping[str, Any], *, now: str) -> dict:
    """Age from the snapshot's own `retrieved_at` — the time it was captured."""
    captured_at = snapshot.get("retrieved_at")
    captured = _parse_utc(captured_at)
    current = _parse_utc(now)
    if captured is None or current is None:
        return {"state": "missing", "age_hours": None, "captured_at": captured_at,
                "evaluated_at": now,
                "reason": "the snapshot carries no readable retrieved_at"}
    age = (current - captured).total_seconds() / 3600.0
    if age < FRESH_HOURS:
        state = "fresh"
    elif age < STALE_HOURS:
        state = "aging"
    else:
        state = "stale"
    return {
        "state": state,
        "age_hours": round(age, 3),
        "captured_at": captured_at,
        "evaluated_at": now,
        "fresh_under_hours": FRESH_HOURS,
        "stale_at_or_over_hours": STALE_HOURS,
        "reason": (f"snapshot is {round(age, 1)}h old" if state != "fresh" else None),
    }


# --------------------------------------------------------------------------- #
# Section helpers
# --------------------------------------------------------------------------- #
def _blocked(reason: str, *, rationale: str, invalidation: str, **extra) -> dict:
    section = {
        "status": "no_current_pick",
        "reason": reason,
        "rationale": rationale,
        "confidence": "none",
        "invalidation_trigger": invalidation,
    }
    section.update(extra)
    return section


#: Slot label -> the id this module orders and flexes by. The adapter already
#: named every slot; this is the inverse of `SLOT_NAMES`, not a second table.
SLOT_IDS: Mapping[str, int] = {label: slot_id for slot_id, label in SLOT_NAMES.items()}

#: Seats whose projections are a separate, unpromoted lane. A missing kicker
#: distribution is a fact about the kicker lane, not about the offence: it must
#: not take the QB/RB/WR/TE/FLEX lineup down with it, because the lineup those
#: seats describe is decidable and useful on its own.
SHADOW_SLOTS = frozenset({"K", "D/ST"})


def _slot_counts(snapshot: Mapping[str, Any]) -> dict[str, int]:
    return {str(label): int(count)
            for label, count in ((snapshot.get("roster_settings") or {})
                                 .get("lineup_slot_counts") or {}).items()}


def _starting_slots(snapshot: Mapping[str, Any]) -> list[tuple[int, str, int]]:
    """Startable slots, in slot-id order, read from the canonical snapshot."""
    slots = []
    for label, count in _slot_counts(snapshot).items():
        slot_id = SLOT_IDS.get(label)
        if slot_id is None or slot_id in BENCH_SLOTS or count <= 0:
            continue
        slots.append((slot_id, label, count))
    return sorted(slots, key=lambda row: row[0])


def _roster_capacity(snapshot: Mapping[str, Any]) -> int:
    """Every slot a player may legally occupy, IR included."""
    return sum(_slot_counts(snapshot).values())


def _mean(player: Mapping[str, Any]) -> float:
    projection = player.get("projection") or {}
    value = projection.get("mean")
    return float(value) if isinstance(value, (int, float)) else float("-inf")


def _split_identities(roster: Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    resolved, unresolved = [], []
    for player in roster:
        if player.get("player_id"):
            resolved.append(dict(player))
        else:
            unresolved.append({
                "espn_player_id": player.get("espn_player_id"),
                "name": player.get("name"),
                "position": player.get("position"),
                "reason": ("no identity crosswalk to a model player_id; excluded from the "
                           "lineup rather than scored as zero"),
            })
    return resolved, unresolved


def _players_from_snapshot(snapshot: Mapping[str, Any], team_id: int, *,
                           crosswalk: Mapping[int, str] | None,
                           projections: Mapping[str, Mapping[str, Any]] | None,
                           byes: Mapping[str, int] | None,
                           samples: Mapping[str, Any] | None = None
                           ) -> tuple[list[dict], list[dict]]:
    """Canonical roster entries -> the shape the lineup logic reads.

    Identity goes through `identity.resolve`, which is the ESPN comparison
    crosswalk. Without a crosswalk nothing is *guessed* into an id: every
    player arrives unresolved, and the lineup says so instead of scoring a
    roster of unknowns as zeroes.
    """
    from . import identity as identity_mod

    entries = list((snapshot.get("rosters") or {}).get(str(team_id), []))
    resolution = identity_mod.resolve(entries, crosswalk)
    projections = projections or {}
    byes = byes or {}

    resolved: list[dict] = []
    for entry in entries:
        espn_id = entry.get("player_id")
        model_id = resolution.matched.get(espn_id) if isinstance(espn_id, int) else None
        if model_id is None:
            continue
        projection = dict(projections.get(model_id) or {})
        resolved.append({
            "player_id": model_id,
            "espn_player_id": espn_id,
            "name": entry.get("full_name"),
            "position": entry.get("default_position"),
            "injury_status": entry.get("injury_status"),
            "lineup_slot": entry.get("lineup_slot"),
            "bye_week": byes.get(model_id),
            "eligible_slots": list(entry.get("eligible_slots") or []),
            "projection": projection,
            "samples": None if samples is None else samples.get(model_id),
        })

    unresolved = [{"position": None, **dict(row)} for row in resolution.unresolved]
    # Entries the adapter itself could not normalize never reached `rosters`;
    # they are still this team's players and still owed a mention.
    for row in snapshot.get("unmatched_players") or []:
        if int(row.get("team_id", -1)) != int(team_id):
            continue
        unresolved.append({
            "espn_player_id": row.get("player_id"), "name": None, "position": None,
            "reason": f"adapter could not normalize this roster entry: {row.get('reason')}",
        })
    return resolved, unresolved


def _availability(player: Mapping[str, Any], scoring_period: int) -> str | None:
    status = str(player.get("injury_status") or "ACTIVE").upper()
    if status in INELIGIBLE_INJURY:
        return f"injury status {status}"
    bye = player.get("bye_week")
    if bye is not None and int(bye) == int(scoring_period):
        return f"on bye in week {scoring_period}"
    if _mean(player) == float("-inf"):
        return "no projection available"
    return None


def _split_slots(snapshot: Mapping[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    """Modelled seats and shadow seats, from the league's own slot counts."""
    counts = _slot_counts(snapshot)
    modeled = {label: count for label, count in counts.items()
               if label not in SHADOW_SLOTS and label not in {"BE", "IR"} and count > 0}
    shadow = {label: count for label, count in counts.items()
              if label in SHADOW_SLOTS and count > 0}
    return modeled, shadow


def _optimal_lineup(snapshot, resolved, unresolved, scoring_period) -> dict:
    """The best legal offensive lineup, solved exactly by the shared engine.

    Only the modelled seats are decided here. A league with a K and a D/ST seat
    still gets a QB/RB/WR/TE/FLEX answer even though neither shadow lane has a
    promoted distribution -- those seats are reported separately as their own
    NO CURRENT PICK, because refusing the whole lineup over a kicker withholds
    six decisions that were never in doubt to avoid one that was.
    """
    from . import lineup as lineup_engine

    modeled, shadow = _split_slots(snapshot)
    rationale = ("Maximum-weight assignment of eligible players to the league's own "
                 "starting slots — exact, not a greedy fill, and using each player's "
                 "ESPN eligibility rather than his position name.")
    invalidation = ("Any inactive/injury designation change, a bye correction, or a projection "
                    "revision that reorders two players at the same slot.")

    available, excluded = [], []
    for player in resolved:
        blocker = _availability(player, scoring_period)
        if blocker:
            excluded.append({"player_id": player.get("player_id"),
                             "espn_player_id": player.get("espn_player_id"),
                             "name": player.get("name"), "position": player.get("position"),
                             "reason": blocker})
        else:
            available.append(player)
    for entry in unresolved:
        excluded.append({"player_id": None, "espn_player_id": entry["espn_player_id"],
                         "name": entry.get("name"), "position": entry.get("position"),
                         "reason": entry["reason"]})

    violations = []
    capacity = _roster_capacity(snapshot)
    if capacity and len(resolved) + len(unresolved) > capacity:
        violations.append(f"roster holds {len(resolved) + len(unresolved)} players, "
                          f"above the league capacity of {capacity}")
    seen = [p.get("espn_player_id") for p in resolved]
    if len(seen) != len(set(seen)):
        violations.append("the same player appears on the roster more than once")

    seatable, unseatable = [], []
    for player in available:
        eligible = {str(slot) for slot in (player.get("eligible_slots") or [])} & set(modeled)
        if eligible:
            seatable.append((player, frozenset(eligible)))
        elif str(player.get("position")) not in SHADOW_SLOTS:
            unseatable.append(player)

    engine_players = tuple(
        lineup_engine.LineupPlayer(player_id=player.get("espn_player_id"),
                                   eligible_slots=eligible,
                                   position=str(player.get("position")))
        for player, eligible in seatable)
    by_id = {player.get("espn_player_id"): player for player, _ in seatable}
    points = {pid: _mean(player) for pid, player in by_id.items()}
    points = {pid: (value if value != float("-inf") else 0.0) for pid, value in points.items()}

    solved = lineup_engine.optimize(points, engine_players, modeled)
    starters = [
        _starter(by_id[pid], label)
        for label in sorted(solved.assignment)
        for pid in solved.assignment[label]
    ]
    for label in solved.empty_slots:
        violations.append(f"cannot fill required slot {label} "
                          "(no eligible, available, projected player remains)")
    for player in unseatable:
        excluded.append({"player_id": player.get("player_id"),
                         "espn_player_id": player.get("espn_player_id"),
                         "name": player.get("name"), "position": player.get("position"),
                         "reason": "eligible for no modelled starting slot in this league"})

    if violations:
        return _blocked(
            "; ".join(violations),
            rationale=rationale, invalidation=invalidation,
            legal=False, violations=violations, starters=[], bench=[], excluded=excluded,
            projected_total=None, shadow_slots=sorted(shadow),
        )

    seated = {entry["espn_player_id"] for entry in starters}
    bench = [_starter(p, "BE") for p in available if p.get("espn_player_id") not in seated]
    return {
        "status": "ok",
        "legal": True,
        "violations": [],
        "starters": starters,
        "bench": sorted(bench, key=lambda e: -e["projected_mean"]),
        "excluded": excluded,
        "projected_total": round(sum(e["projected_mean"] for e in starters), 2),
        "shadow_slots": sorted(shadow),
        "rationale": rationale,
        "confidence": "medium",
        "invalidation_trigger": invalidation,
    }


def _starter(player: Mapping[str, Any], slot_name: str) -> dict:
    projection = player.get("projection") or {}
    return {
        "slot": slot_name,
        "player_id": player.get("player_id"),
        "espn_player_id": player.get("espn_player_id"),
        "name": player.get("name"),
        "position": player.get("position"),
        "projected_mean": float(projection.get("mean", 0.0)),
        "projected_p10": float(projection.get("p10", 0.0)),
        "projected_p90": float(projection.get("p90", 0.0)),
        "eligible_slots": [SLOT_NAMES.get(int(s), str(s))
                           for s in player.get("eligible_slot_ids", [])],
    }


def _delta_summary(start_samples, sit_samples, mean_delta: float) -> dict:
    """The distribution of a swap's effect, from paired simulation rows.

    Swapping one player for another in a fixed lineup changes the total by
    exactly (start - sit) in every simulation, so the delta is read off the
    same row for both players: the same latent week, the same game script.

    The old summary subtracted one player's p90 from the other's p10. Those are
    marginal corners of two separate distributions, and their difference is the
    worst case under perfect negative dependence — a scenario the simulation
    never draws. It systematically overstated the spread of a swap between two
    players on the same offence, who rise and fall together.
    """
    if start_samples is None or sit_samples is None:
        return {
            "status": "unavailable",
            "reason": ("no joint simulation rows were supplied for both players; a spread "
                       "built from separate marginal percentiles is not this swap's spread"),
            "mean_delta": round(mean_delta, 2),
        }
    import numpy as _np

    left = _np.asarray(start_samples, dtype=float)
    right = _np.asarray(sit_samples, dtype=float)
    if left.shape != right.shape or left.size == 0:
        return {
            "status": "unavailable",
            "reason": (f"sample rows do not align ({left.shape} vs {right.shape}); a delta "
                       "across mismatched draws is not a paired comparison"),
            "mean_delta": round(mean_delta, 2),
        }
    delta = left - right
    p10, p50, p90 = (float(v) for v in _np.percentile(delta, [10, 50, 90]))
    return {
        "status": "ok",
        "basis": "paired joint simulation rows",
        "simulations": int(delta.size),
        "mean_delta": round(float(delta.mean()), 2),
        "median_delta": round(p50, 2),
        "p10_delta": round(p10, 2),
        "p90_delta": round(p90, 2),
        # Deliberately not called confidence: this is the share of simulated
        # weeks in which the model's own draws favour the start. It inherits
        # every error in the model that produced them and has not been checked
        # against what actually happened.
        "model_relative_prob_start_scores_more": round(float((delta > 0).mean()), 4),
        "probability_note": ("model-relative frequency over the model's own draws; not a "
                             "calibrated probability and not validated against outcomes"),
    }


def _start_sit(lineup: Mapping[str, Any], resolved: Sequence[Mapping[str, Any]]) -> dict:
    rationale = ("Each row pairs the player the legal optimum starts with the player it "
                 "benches at the same slot; the delta is read from paired simulation rows.")
    invalidation = ("A projection revision smaller than the delta, or any status change to "
                    "either player, flips the row.")
    if lineup.get("status") != "ok":
        return _blocked("no legal lineup, so no start/sit comparison is possible",
                        rationale=rationale, invalidation=invalidation, decisions=[])

    by_id = {p.get("espn_player_id"): p for p in resolved}
    current_by_slot: dict[str, list[Mapping[str, Any]]] = {}
    for player in resolved:
        slot = str(player.get("lineup_slot") or "BE")
        if SLOT_IDS.get(slot) in BENCH_SLOTS:
            continue
        current_by_slot.setdefault(slot, []).append(player)

    started_now = {p.get("espn_player_id")
                   for players in current_by_slot.values() for p in players}
    decisions = []
    for entry in lineup["starters"]:
        if entry["espn_player_id"] in started_now:
            continue
        slot = entry["slot"]
        benched = [p for p in current_by_slot.get(slot, [])
                   if p.get("espn_player_id") not in
                   {s["espn_player_id"] for s in lineup["starters"]}]
        out = min(benched, key=_mean) if benched else None
        out_mean = _mean(out) if out is not None else 0.0
        delta = entry["projected_mean"] - out_mean
        incoming = by_id.get(entry["espn_player_id"]) or {}
        summary = _delta_summary(incoming.get("samples"),
                                 (out or {}).get("samples"), delta)
        decisions.append({
            "slot": slot,
            "start": {"name": entry["name"], "player_id": entry["player_id"],
                      "espn_player_id": entry["espn_player_id"],
                      "position": entry["position"], "projected_mean": entry["projected_mean"]},
            "sit": ({"name": out.get("name"), "player_id": out.get("player_id"),
                     "espn_player_id": out.get("espn_player_id"),
                     "position": out.get("position"), "projected_mean": out_mean}
                    if out is not None else None),
            "projected_delta": round(delta, 2),
            "uncertainty": summary,
            "rationale": (f"{entry['name']} projects {round(delta, 2)} points above "
                          f"{out.get('name') if out is not None else 'an empty slot'} at {slot}."),
            "invalidation_trigger": invalidation,
        })

    if not decisions:
        return _blocked("no lineup change beats the one already set",
                        rationale=rationale, invalidation=invalidation, decisions=[])
    return {
        "status": "ok",
        "decisions": sorted(decisions, key=lambda d: -d["projected_delta"]),
        "rationale": rationale,
        "confidence": "medium",
        "invalidation_trigger": invalidation,
    }


def _draft(snapshot: Mapping[str, Any], *, team_id: int) -> dict:
    draft = snapshot.get("draft", {})
    raw = draft.get("picks") or []
    selections = [
        {**dict(s), "espn_player_id": s.get("player_id")}
        for s in raw if int(s.get("player_id", -1) or -1) > 0
    ]
    selections.sort(key=lambda s: int(s.get("overall_pick", 0)))
    # The adapter states the draft's state outright; it is not re-derived from
    # two Booleans that can disagree with each other.
    status = str(draft.get("status") or "pre_draft")
    state = {"post_draft": "complete", "in_progress": "in_progress"}.get(status, "pre_draft")

    rationale = ("Draft state is read from ESPN's own drafted/inProgress flags; selections are "
                 "only those picks that carry a real player id.")
    invalidation = "Any new pick, or the draft flipping to in-progress or complete."

    if state == "complete":
        targets = _blocked(
            "the draft is complete, so a target list would be a fiction",
            rationale="Targets exist only before or during a draft.",
            invalidation="A new season or a re-draft.", entries=[])
    else:
        source = snapshot.get("targets_source")
        refusal = reject_source(source, snapshot, what="draft board") if source is not None else (
            "no draft board source is attached to the snapshot")
        if refusal:
            targets = _blocked(refusal, rationale="Targets require a board built for this league.",
                               invalidation="Attaching a board that matches this league.", entries=[])
        else:
            entries = []
            for item in source.get("entries", []):
                entry = {k: v for k, v in item.items() if k not in {"round", "overall_pick"}}
                entry["kind"] = "target"
                entry["note"] = "target only — not a pick, and not on any roster"
                entries.append(entry)
            targets = {"status": "ok", "entries": entries,
                       "rationale": ("Board reweighted to this league's size and slot; a target is a "
                                     "watchlist intention, never a selection."),
                       "confidence": "low",
                       "invalidation_trigger": "Any pick removing one of these players."}

    return {
        "status": "ok",
        "state": state,
        "selections": selections,
        "selection_count": len(selections),
        "pick_slot_count": int(draft.get("pick_slot_count", 0)),
        "empty_pick_slots": int(draft.get("pick_slot_count", 0)) - len(selections),
        "my_picks": [s for s in selections if int(s.get("team_id", 0)) == int(team_id)],
        "scheduled_at_utc": draft.get("date_utc"),
        "my_draft_slot": draft.get("my_draft_slot"),
        "rounds": draft.get("rounds"),
        "targets": targets,
        "rationale": rationale,
        "confidence": "high",
        "invalidation_trigger": invalidation,
        "note": (
            "ESPN pre-creates every pick slot before a draft with playerId -1. "
            f"{int(draft.get('pick_slot_count', 0))} slots exist; {len(selections)} are real "
            "selections."),
    }


def _trades(snapshot: Mapping[str, Any], *, team_id: int) -> dict:
    rationale = ("Trade opportunities are only computed from live rosters on both sides; a "
                 "counterparty with no roster cannot be valued.")
    invalidation = "Any roster change on either side, or a completed draft filling the league."
    rosters = snapshot.get("rosters", {})
    empty = [tid for tid, entries in sorted(rosters.items()) if not entries]
    if empty:
        return _blocked(
            f"{len(empty)} of {len(rosters)} team rosters are empty in this snapshot, so no "
            "counterparty roster can be valued",
            rationale=rationale, invalidation=invalidation, opportunities=[])
    return _blocked(
        "no trade engine is wired to the live-roster snapshot in this build",
        rationale=rationale, invalidation=invalidation, opportunities=[])


def _waivers(plan: Any | None) -> dict:
    """Render the waiver planner's output, or say precisely why there is none.

    This layer decides nothing about waivers.  It presents
    :func:`nflvalue.fantasy.waivers.plan` records, which are recommendation-only
    by construction, and refuses in three distinguishable ways: no plan was run,
    the plan found no benefit, or the plan itself degraded on stale inputs.
    Collapsing those three into one empty table is how a reader concludes
    "nothing to do" when the truth is "nothing was measured".
    """
    rationale = ("Waiver targets come from the planner's own legality pass; each is "
                 "shown beside the legal drop it requires, never alone.")
    invalidation = ("Any roster or free-agent pool change, a processed claim, or the "
                    "transaction deadline passing.")
    if plan is None:
        return _blocked(_NO_WAIVER_PLAN, rationale=rationale, invalidation=invalidation,
                        targets=[], drops=[])
    records = list(plan)
    if not records:
        return _blocked(_NO_WAIVER_BENEFIT, rationale=rationale,
                        invalidation=invalidation, targets=[], drops=[])
    live = [r for r in records if not getattr(r, "degraded", False)]
    if not live:
        reason = getattr(records[0], "rationale", None) or "waiver inputs are degraded"
        return _blocked(reason, rationale=rationale, invalidation=invalidation,
                        targets=[], drops=[], degraded=True)

    targets = []
    for record in live:
        targets.append({
            "add": {"espn_player_id": getattr(record, "add_espn_id", None),
                    "name": getattr(record, "add_name", None),
                    "position": getattr(record, "add_position", None)},
            "drop": {"espn_player_id": getattr(record, "drop_espn_id", None),
                     "name": getattr(record, "drop_name", None)},
            "drop_state": getattr(record, "drop_state", None),
            "status": getattr(record, "status", None),
            "shadow_reason": getattr(record, "shadow_reason", None),
            "confidence": getattr(record, "confidence", None),
            "rationale": getattr(record, "rationale", None),
            "invalidation_trigger": getattr(record, "invalidation_trigger", None),
            "priority_implications": dict(getattr(record, "priority_implications", {}) or {}),
            "lineup_delta": (dict(getattr(record, "lineup_delta", None) or {})
                             if getattr(record, "lineup_delta", None) is not None else None),
            "lineup_delta_status": getattr(record, "lineup_delta_status", None),
            "faab": (dict(getattr(record, "faab", None) or {})
                     if getattr(record, "faab", None) is not None else None),
            "data_timestamps": dict(getattr(record, "data_timestamps", {}) or {}),
            # The planner never executes anything; this says so on every row.
            "recommendation_only": True,
        })
    return {
        "status": "ok",
        "targets": targets,
        "drops": [t["drop"] for t in targets if t["drop"]["espn_player_id"] is not None],
        "rationale": rationale,
        "confidence": "medium",
        "invalidation_trigger": invalidation,
    }


def _shadow(position: str, lineup: Mapping[str, Any]) -> dict:
    """A shadow seat's own NO CURRENT PICK, stated separately from the offence.

    K and D/ST have no promoted distribution, so this seat has no
    recommendation — and that is the whole of the claim. It does not travel:
    the offensive lineup above is decided from projections that did pass the
    gate, and withholding six settled decisions to avoid one unsettled seat
    would be the more misleading answer.
    """
    seats = int((lineup.get("shadow_slots") or []).count(position)) or (
        1 if position in (lineup.get("shadow_slots") or []) else 0)
    return {
        "status": "no_current_pick",
        "shadow": True,
        "promoted": False,
        "label": f"{position} shadow — research only, never promoted",
        "reason": (f"{position} has no promoted projection, so this seat has no recommendation. "
                   "The offensive lineup is unaffected and is shown above."),
        "seats": seats,
        "starters": [],
        "rationale": (f"{position} projections have not passed the 2026 protocol's promotion "
                      "gate, so they never enter the lineup objective."),
        "confidence": "none",
        "invalidation_trigger": ("A passing season-forward audit promoting the position out of "
                                 "shadow."),
    }


# --------------------------------------------------------------------------- #
# Snapshot loading and projection attachment
# --------------------------------------------------------------------------- #
def load_latest_snapshot(directory: str | Any, *, now: str | None = None) -> dict | None:
    """The newest league snapshot in *directory* by its own retrieval time.

    Delegates to `espn_league.load_latest_snapshot`. The previous
    implementation sorted by `st_mtime`, so a checkout or an rsync — neither
    of which reads anything from ESPN — could promote a stale capture to "the
    current state of the league".
    """
    from .espn_league import load_latest_snapshot as _load

    return _load(directory, now=now)


def attach_projections(*_args, **_kwargs):
    """Removed: model output does not get written into an ESPN snapshot.

    This used to deep-copy the snapshot and splice a `projection` block onto
    each roster entry, keyed by a model id the snapshot did not carry. That is
    how the builder ended up reading a schema the adapter never produced —
    every consumer then needed the merged shape, and the real one drifted.

    Projections are the model's answer, so they are passed to `build()`
    alongside the snapshot and joined through the identity crosswalk.
    """
    raise NotImplementedError(
        "attach_projections is gone: pass projections=... to my_team.build() instead, "
        "so the ESPN snapshot stays a record of what ESPN said")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build(snapshot: Mapping[str, Any], *, now: str, contract: Any | None = None,
          waiver_plan: Any | None = None,
          crosswalk: Mapping[int, str] | None = None,
          projections: Mapping[str, Mapping[str, Any]] | None = None,
          byes: Mapping[str, int] | None = None,
          samples: Mapping[str, Any] | None = None) -> dict:
    """Assemble the versioned ``my_team`` contract from a league snapshot.

    *contract* is an :class:`espn_contract.LeagueContract`; it is the only
    source of scoring/roster identity.  Omitting it yields null hashes with a
    reason rather than a locally invented substitute.
    """
    from .espn_league import SCHEMA_VERSION as SNAPSHOT_SCHEMA

    if not isinstance(snapshot, Mapping):
        raise SnapshotRejected(f"snapshot is not a mapping ({type(snapshot).__name__})")
    version = snapshot.get("schema_version")
    if version != SNAPSHOT_SCHEMA:
        raise SnapshotRejected(
            f"unsupported snapshot schema {version!r}; this builder reads {SNAPSHOT_SCHEMA}. "
            "Degrading would render an empty team, which is what a real empty league looks "
            "like, and nothing would say the two sides disagreed about the shape.")

    league = snapshot.get("league", {})
    mine = snapshot.get("my_team", {})
    team_id = int(mine.get("team_id", 0))
    scoring_period = int(league.get("current_scoring_period", 0) or 0)
    fresh = freshness(snapshot, now=now)
    usable = fresh["state"] in {"fresh", "aging"}

    resolved, unresolved = _players_from_snapshot(
        snapshot, team_id, crosswalk=crosswalk, projections=projections, byes=byes,
        samples=samples)
    roster = resolved

    draft = _draft(snapshot, team_id=team_id)

    if not usable:
        blocked_reason = (f"league snapshot is {fresh['state']}"
                          + (f" ({fresh['age_hours']}h old)" if fresh["age_hours"] else ""))
        lineup = _blocked(blocked_reason,
                          rationale="A lineup drawn from stale inputs is not a lineup.",
                          invalidation="A fresh read-only snapshot.",
                          legal=False, violations=[], starters=[], bench=[], excluded=[],
                          projected_total=None)
        start_sit = _blocked(blocked_reason,
                             rationale="Start/sit needs current status and projections.",
                             invalidation="A fresh read-only snapshot.", decisions=[])
        trades = _blocked(blocked_reason, rationale="Trades need current rosters.",
                          invalidation="A fresh read-only snapshot.", opportunities=[])
    elif not roster:
        published = len((snapshot.get("rosters") or {}).get(str(team_id), []))
        if published:
            # The roster is there; nothing in it could be tied to a model
            # player. Reporting that as "no roster" would blame the league for
            # a crosswalk this build did not supply.
            empty_reason = (
                f"team {team_id} holds {published} player(s), but none resolved to a model "
                f"id ({len(unresolved)} unresolved); supply an ESPN-to-model crosswalk")
        else:
            empty_reason = (f"no roster exists for team {team_id} in this snapshot "
                            f"(draft state: {draft['state']})")
        lineup = _blocked(empty_reason,
                          rationale="There is no lineup before there is a roster.",
                          invalidation="The first completed pick.",
                          legal=False, violations=[], starters=[], bench=[], excluded=[],
                          projected_total=None)
        start_sit = _blocked(empty_reason,
                             rationale="Start/sit compares players you own.",
                             invalidation="The first completed pick.", decisions=[])
        trades = _trades(snapshot, team_id=team_id)
    else:
        lineup = _optimal_lineup(snapshot, resolved, unresolved, scoring_period)
        start_sit = _start_sit(lineup, resolved)
        trades = _trades(snapshot, team_id=team_id)

    waivers = _waivers(waiver_plan)

    source = snapshot.get("source") or {}
    abbrev = next((team.get("abbrev") for team in (snapshot.get("teams") or [])
                   if int(team.get("team_id", -1)) == team_id), None)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_schema_version": snapshot.get("schema_version"),
        "generated_at": now,
        "league": {
            "platform": "ESPN",
            "league_id": str(league.get("league_id")),
            "league_name": league.get("name"),
            "season": league.get("season"),
            "size": league.get("size"),
            "scoring_period": scoring_period,
            "team_id": team_id,
            "team_name": mine.get("name"),
            "team_abbrev": abbrev,
        },
        "freshness": fresh,
        **contract_hashes(contract, snapshot=snapshot),
        "sources": [{
            "name": "espn_league_snapshot",
            "schema_version": snapshot.get("schema_version"),
            "views": source.get("views"),
            "access": "read_only",
            "retrieved_at": snapshot.get("retrieved_at"),
            "league_hash": (snapshot.get("hashes") or {}).get("league"),
        }],
        "roster": resolved,
        "draft": draft,
        "optimal_lineup": lineup,
        "start_sit": start_sit,
        "waivers": waivers,
        "trades": trades,
        "kicker_shadow": _shadow("K", lineup),
        "dst_shadow": _shadow("D/ST", lineup),
        "unresolved_identities": {"count": len(unresolved), "entries": unresolved},
        "confidence": "none" if not usable else (
            "medium" if lineup.get("status") == "ok" else "low"),
        "espn_use": "display and grading only; no ESPN write is ever performed",
    }
