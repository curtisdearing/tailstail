"""Normalize an ESPN fantasy league into a versioned, redacted contract.

This module is the *pure* half of the read-only ESPN adapter: it takes raw
view payloads (whatever produced them) and turns them into one immutable
snapshot, or refuses. It performs no network access and reads no environment.
The network boundary — and the only place credentials are touched — is
:mod:`nflvalue.fantasy.espn_client`.

Three properties are worth stating outright, because each exists to stop a
specific way this goes wrong.

**Fail closed on identity.** A snapshot that quietly describes the wrong
league, season, team, or team count is worse than no snapshot: every
downstream number inherits the error without a symptom. Wrong league, wrong
season, wrong team, a size that disagrees with the team list, duplicate team
ids or names, an unknown lineup slot, and incomplete settings all raise.
League size is not cosmetic here — replacement level, and therefore every
value on the draft board, is priced off it.

**Pre-draft is a state, not an absence.** Before the draft there are no
rosters and no picks, and the honest snapshot says so: ``draft.status ==
"pre_draft"``, ``draft.picks is None``, ``roster_state ==
"empty_pre_draft"``. A payload that claims picks while reporting
``drafted: false``, or rosters in an undrafted league, is contradictory and
is refused rather than reconciled. Watchlist entries are intent; picks are
history, and this module will not turn one into the other.

**Secrets cannot reach a snapshot.** ESPN uses the SWID cookie value as
``members[].id``, so the raw payload contains a credential by construction.
Member ids are hashed to ``member:<digest>`` keys, and unrecognized blocks
are recorded by *name only* — never by value, because an unrecognized block
is exactly where an echoed cookie would hide.

IDs beat names everywhere: ESPN player and team ids are stable, display names
are not. Anything that cannot be resolved is preserved in ``unmatched_players``
or ``unsupported`` rather than dropped.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import espn_contract

SCHEMA_VERSION = "espn-league/1"

#: How the league decides who gets a contested claim. Named once, here.
WAIVER_MODE_FAAB = "faab"
WAIVER_MODE_INVERSE = "inverse_standings"
WAIVER_MODE_ROLLING = "rolling_priority"
REDACTED = "<redacted>"

#: Views this module needs before it will describe a league at all. Standings,
#: transactions and the player pool are optional and reported as ``None`` when
#: absent -- "not read" and "empty" are different claims.
REQUIRED_VIEWS = ("mSettings", "mTeam", "mRoster", "mMatchup", "mDraftDetail")
OPTIONAL_VIEWS = ("mStandings", "mTransactions2", "mPendingTransactions", "kona_player_info")

#: ESPN lineup slot ids. Unknown ids are a hard error: a slot we cannot name is
#: a roster rule we do not understand, and guessing it mis-prices the board.
SLOT_NAMES: Mapping[int, str] = {
    0: "QB", 1: "TQB", 2: "RB", 3: "RB/WR", 4: "WR", 5: "WR/TE", 6: "TE",
    7: "OP", 8: "DT", 9: "DE", 10: "LB", 11: "DL", 12: "CB", 13: "S",
    14: "DB", 15: "DP", 16: "D/ST", 17: "K", 18: "P", 19: "HC",
    20: "BE", 21: "IR", 23: "FLEX", 24: "EDR",
}
BENCH_SLOT = "BE"
IR_SLOT = "IR"

POSITION_NAMES: Mapping[int, str] = {
    1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "D/ST",
}

#: Top-level league keys this module consumes. Everything else is reported in
#: ``unsupported["league_keys"]`` so schema drift is visible rather than silent.
CONSUMED_LEAGUE_KEYS = frozenset({
    "id", "seasonId", "segmentId", "scoringPeriodId", "firstScoringPeriod",
    "finalScoringPeriod", "currentMatchupPeriod", "status", "settings",
    "members", "teams", "schedule", "draftDetail", "transactions",
    "pendingTransactions",
})
CONSUMED_SETTINGS_KEYS = frozenset({
    "name", "size", "isPublic", "draftSettings", "rosterSettings",
    "scheduleSettings", "scoringSettings", "acquisitionSettings",
    "tradeSettings", "financeSettings",
})
REQUIRED_SETTINGS_BLOCKS = (
    "rosterSettings", "scoringSettings", "scheduleSettings",
    "draftSettings", "acquisitionSettings",
)

_SWID_RE = re.compile(r"\{[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\}")
_SECRET_KEY_RE = re.compile(r"espn[_-]?s2|swid|cookie|authorization|auth|token|header", re.I)
_SECRET_VALUE_RE = re.compile(r"espn_s2\s*=|swid\s*=", re.I)


class EspnAdapterError(Exception):
    """Base for every refusal this adapter makes deliberately."""


class EspnIdentityError(EspnAdapterError):
    """The payload does not describe the league/season/team we asked for."""


class EspnSchemaError(EspnAdapterError):
    """The payload is internally inconsistent or has drifted out of contract."""


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExpectedIdentity:
    """What the caller asserts this league is. Every field is checked."""

    league_id: int
    season: int
    team_id: int
    team_name: str
    team_count: int


@dataclass(frozen=True)
class LeagueIdentity:
    league_id: int
    season: int
    segment_id: int
    name: str
    size: int
    is_public: bool
    current_scoring_period: int
    first_scoring_period: int
    final_scoring_period: int
    current_matchup_period: int
    is_active: bool
    teams_joined: int


@dataclass(frozen=True)
class Member:
    member_key: str
    display_name: str
    is_league_manager: bool


@dataclass(frozen=True)
class Team:
    team_id: int
    name: str
    abbrev: str
    owner_keys: tuple[str, ...]
    division_id: int
    waiver_rank: int
    playoff_seed: int
    acquisition_budget_spent: float


@dataclass(frozen=True)
class MyTeam:
    team_id: int
    name: str
    matched_by: str


@dataclass(frozen=True)
class RosterSettings:
    lineup_slot_counts: Mapping[str, int]
    starting_slots: int
    bench_slots: int
    ir_slots: int
    roster_size: int


@dataclass(frozen=True)
class RosterPlayer:
    player_id: int
    full_name: str
    default_position: str
    eligible_slots: tuple[str, ...]
    pro_team_id: int
    injury_status: str
    lineup_slot: str
    is_starter: bool
    acquisition_type: str
    acquisition_date: str | None


@dataclass(frozen=True)
class PoolPlayer:
    player_id: int
    full_name: str
    default_position: str
    eligible_slots: tuple[str, ...]
    pro_team_id: int
    injury_status: str
    availability: str
    acquisition_kind: str


@dataclass(frozen=True)
class ScheduleGame:
    matchup_period: int
    home_team_id: int
    away_team_id: int
    winner: str
    home_points: float | None
    away_points: float | None
    playoff_tier: str


@dataclass(frozen=True)
class DraftPick:
    overall_pick: int
    round: int
    round_pick: int
    team_id: int
    player_id: int
    keeper: bool
    bid_amount: float


@dataclass(frozen=True)
class Draft:
    status: str
    type: str
    rounds: int
    pick_order: tuple[int, ...]
    my_slot: int | None
    scheduled_at: str | None
    time_per_selection: int | None
    picks: tuple[DraftPick, ...] | None


@dataclass(frozen=True)
class Waivers:
    acquisition_type: str
    uses_acquisition_budget: bool
    acquisition_budget: float
    process_days: tuple[str, ...]
    waiver_hours: int
    order_reset: bool
    mode: str
    transaction_deadline: str | None
    acquisition_limit: int
    lock_policy: str
    team_priority: Mapping[int, int]
    team_budget_spent: Mapping[int, float]


@dataclass(frozen=True)
class StandingsRow:
    team_id: int
    wins: int
    losses: int
    ties: int
    percentage: float
    points_for: float
    points_against: float
    playoff_seed: int
    games_back: float


@dataclass(frozen=True)
class Standings:
    rows: tuple[StandingsRow, ...]
    tiebreaker: Mapping[str, Any]


@dataclass(frozen=True)
class Playoffs:
    team_count: int
    seeding_rule: str
    matchup_period_length: int
    regular_season_matchup_periods: int
    playoff_matchup_periods: tuple[int, ...]
    playoff_scoring_periods: tuple[int, ...]
    reseed: bool


@dataclass(frozen=True)
class Transaction:
    transaction_id: str
    type: str
    status: str
    team_id: int
    scoring_period: int | None
    proposed_at: str | None
    processed_at: str | None
    bid_amount: float
    is_pending: bool
    items: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class Transactions:
    completed: tuple[Transaction, ...]
    pending: tuple[Transaction, ...]


@dataclass(frozen=True)
class Scoring:
    scoring_type: str
    matchup_tie_rule: str
    playoff_matchup_tie_rule: str
    home_team_bonus: float
    items: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class SnapshotSource:
    views: tuple[str, ...]
    urls: tuple[str, ...]
    credentialed: bool


@dataclass(frozen=True)
class LeagueSnapshot:
    schema_version: str
    retrieved_at: str
    source: SnapshotSource
    league: LeagueIdentity
    members: tuple[Member, ...]
    teams: tuple[Team, ...]
    my_team: MyTeam
    roster_settings: RosterSettings
    rosters: Mapping[int, tuple[RosterPlayer, ...]]
    roster_state: str
    free_agents: tuple[PoolPlayer, ...] | None
    schedule: tuple[ScheduleGame, ...]
    draft: Draft
    waivers: Waivers
    standings: Standings
    playoffs: Playoffs
    transactions: Transactions
    scoring: Scoring
    rules: Mapping[str, Any]
    hashes: Mapping[str, str]
    unsupported: Mapping[str, list]
    unmatched_players: tuple[Mapping[str, Any], ...]
    eligibility_violations: tuple[Mapping[str, Any], ...] = field(default=())
    warnings: tuple[str, ...] = field(default=())


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
#: Fields inside the required blocks that carry meaning by their absence as
#: much as their value. `or 0` on any of these turns "ESPN did not say" into a
#: confident zero, and a zero here is a real league setting -- a league with
#: `playoffTeamCount` absent is not a league with no playoffs.
REQUIRED_SETTING_FIELDS: Mapping[str, tuple[str, ...]] = {
    "scheduleSettings": ("matchupPeriodCount", "matchupPeriodLength",
                         "playoffTeamCount", "playoffMatchupPeriodLength"),
    "acquisitionSettings": ("acquisitionType", "waiverOrderReset",
                            "isUsingAcquisitionBudget"),
    "rosterSettings": ("lineupSlotCounts",),
}

#: A capture may be a couple of minutes ahead of this clock and still be real;
#: anything further is a wrong clock or a fabricated timestamp, and either way
#: it must not be allowed to win a "which snapshot is newest" comparison.
FUTURE_TOLERANCE_SECONDS = 120


def parse_timestamp(value: Any) -> dt.datetime | None:
    """ISO-8601 (including a trailing Z) to an aware UTC datetime, or None."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _require_settings_fields(settings: Mapping[str, Any]) -> None:
    """Every required field must be present. Absence is not a value."""
    for block, fields in REQUIRED_SETTING_FIELDS.items():
        payload = settings.get(block)
        if not isinstance(payload, Mapping):
            raise EspnSchemaError(f"incomplete settings: {block} is missing or not an object")
        for name in fields:
            if payload.get(name) is None:
                raise EspnSchemaError(
                    f"incomplete settings: {block}.{name} is absent. Defaulting it would "
                    "publish a league rule ESPN never stated.")


def _validate_retrieved_at(retrieved_at: str, *, now: dt.datetime | None = None) -> dt.datetime:
    moment = parse_timestamp(retrieved_at)
    if moment is None:
        raise EspnSchemaError(
            f"retrieved_at {retrieved_at!r} is not an ISO-8601 timestamp; a snapshot that "
            "cannot say when it was taken cannot be aged or ordered")
    reference = now or dt.datetime.now(dt.timezone.utc)
    if (moment - reference).total_seconds() > FUTURE_TOLERANCE_SECONDS:
        raise EspnSchemaError(
            f"retrieved_at {retrieved_at!r} is in the future relative to {reference.isoformat()}; "
            "refusing to record a capture that has not happened")
    return moment


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def member_key(raw_member_id: str) -> str:
    """Pseudonymize an ESPN member id.

    ESPN's ``members[].id`` *is* the SWID cookie value, so it is a credential
    wearing an identifier's clothes. Hashing keeps it join-able across
    snapshots without ever storing it.
    """
    digest = hashlib.sha256(str(raw_member_id).encode("utf-8")).hexdigest()
    return f"member:{digest[:16]}"


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EspnSchemaError(f"{label} must be an integer, got {type(value).__name__}: {value!r}")
    return value


def _epoch_ms_to_iso(value: Any) -> str | None:
    if not isinstance(value, (int, float)) or value <= 0:
        return None
    moment = dt.datetime.fromtimestamp(float(value) / 1000.0, tz=dt.timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def redact_raw(payload: Any) -> Any:
    """Return ``payload`` with anything credential-shaped replaced.

    Keys are matched by name (``espn_s2``, ``swid``, ``cookie``,
    ``authorization``, ``headers``) and string values by shape (a braced GUID,
    or a ``name=value`` cookie fragment). Everything else is returned
    unchanged, so a redacted payload is still readable.
    """
    if isinstance(payload, Mapping):
        cleaned: dict[str, Any] = {}
        for key, value in payload.items():
            if _SECRET_KEY_RE.search(str(key)):
                cleaned[str(key)] = REDACTED
            else:
                cleaned[str(key)] = redact_raw(value)
        return cleaned
    if isinstance(payload, (list, tuple)):
        return [redact_raw(item) for item in payload]
    if isinstance(payload, str):
        if _SWID_RE.search(payload) or _SECRET_VALUE_RE.search(payload):
            return REDACTED
        return payload
    return payload


def _slot_name(slot_id: Any, *, where: str) -> str:
    try:
        numeric = int(slot_id)
    except (TypeError, ValueError) as exc:
        raise EspnSchemaError(f"{where}: lineup slot id is not numeric: {slot_id!r}") from exc
    if numeric not in SLOT_NAMES:
        raise EspnSchemaError(
            f"{where}: unsupported lineup slot id {numeric}. Refusing to guess what it means; "
            f"add it to SLOT_NAMES once its meaning is confirmed.")
    return SLOT_NAMES[numeric]


def merge_views(views: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """Merge per-view league envelopes into one payload.

    ESPN answers every ``?view=`` request with the same league envelope and
    different sub-blocks populated, so merging is a top-level update with
    ``settings`` merged one level deeper. View order is sorted for determinism.
    """
    merged: dict[str, Any] = {}
    contributing: list[str] = []
    for name in sorted(views):
        payload = views[name]
        if name == "kona_player_info":
            contributing.append(name)
            continue
        if not isinstance(payload, Mapping):
            raise EspnSchemaError(f"view {name!r} did not return an object")
        contributing.append(name)
        for key, value in payload.items():
            if key == "settings" and isinstance(value, Mapping):
                base = dict(merged.get("settings") or {})
                base.update(value)
                merged["settings"] = base
            else:
                merged[key] = value
    return merged, contributing


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def _validate_identity(raw: Mapping[str, Any], expected: ExpectedIdentity) -> None:
    league_id = _require_int(raw.get("id"), "league id")
    season = _require_int(raw.get("seasonId"), "seasonId")
    if league_id != expected.league_id:
        raise EspnIdentityError(
            f"wrong league: payload describes league {league_id}, expected {expected.league_id}")
    if season != expected.season:
        raise EspnIdentityError(
            f"wrong season: payload describes season {season}, expected {expected.season}")


def _validate_teams(raw: Mapping[str, Any], expected: ExpectedIdentity) -> list[Mapping[str, Any]]:
    teams = raw.get("teams")
    if not isinstance(teams, list) or not teams:
        raise EspnSchemaError("payload carries no team list")

    ids = [_require_int(team.get("id"), "team id") for team in teams]
    if len(set(ids)) != len(ids):
        raise EspnIdentityError(f"ambiguous league: duplicate team ids {sorted(ids)}")

    names = [str(team.get("name") or "").strip() for team in teams]
    if len(set(names)) != len(names):
        raise EspnIdentityError(f"ambiguous league: duplicate team names {sorted(names)}")

    if len(teams) != expected.team_count:
        raise EspnIdentityError(
            f"team count mismatch: expected {expected.team_count}, league reports {len(teams)} teams")

    settings_size = (raw.get("settings") or {}).get("size")
    if settings_size is not None and int(settings_size) != len(teams):
        raise EspnIdentityError(
            f"league size is inconsistent: settings.size={settings_size} but {len(teams)} team records")
    return teams


def _resolve_my_team(teams: Sequence[Mapping[str, Any]], expected: ExpectedIdentity) -> MyTeam:
    by_id = {int(team["id"]): team for team in teams}
    mine = by_id.get(expected.team_id)
    if mine is None:
        raise EspnIdentityError(
            f"wrong team: no team with team_id={expected.team_id} in league {expected.league_id} "
            f"(ids present: {sorted(by_id)})")
    actual = str(mine.get("name") or "").strip()
    if actual != expected.team_name:
        raise EspnIdentityError(
            f"wrong team: team_id={expected.team_id} is named {actual!r}, "
            f"expected name {expected.team_name!r}")
    return MyTeam(team_id=expected.team_id, name=actual, matched_by="team_id+name")


def _roster_settings(settings: Mapping[str, Any]) -> RosterSettings:
    counts_raw = (settings.get("rosterSettings") or {}).get("lineupSlotCounts")
    if not isinstance(counts_raw, Mapping) or not counts_raw:
        raise EspnSchemaError("rosterSettings.lineupSlotCounts is missing or empty")
    counts: dict[str, int] = {}
    for slot_id, count in counts_raw.items():
        name = _slot_name(slot_id, where="rosterSettings.lineupSlotCounts")
        counts[name] = int(count)
    bench = counts.get(BENCH_SLOT, 0)
    ir_slots = counts.get(IR_SLOT, 0)
    starting = sum(value for name, value in counts.items() if name not in (BENCH_SLOT, IR_SLOT))
    return RosterSettings(
        lineup_slot_counts=dict(sorted(counts.items())),
        starting_slots=starting,
        bench_slots=bench,
        ir_slots=ir_slots,
        roster_size=starting + bench,
    )


def _normalize_roster_entry(entry: Mapping[str, Any], *, team_id: int,
                            unmatched: list, violations: list) -> RosterPlayer | None:
    pool = entry.get("playerPoolEntry") or {}
    player = pool.get("player") or {}
    player_id = entry.get("playerId", player.get("id"))
    name = str(player.get("fullName") or "").strip()
    if not isinstance(player_id, int) or not name:
        unmatched.append({
            "team_id": team_id,
            "reason": "missing player id or name",
            "player_id": player_id if isinstance(player_id, int) else None,
            "lineup_slot_id": entry.get("lineupSlotId"),
        })
        return None

    slot = _slot_name(entry.get("lineupSlotId"), where=f"team {team_id} roster entry {player_id}")
    eligible_ids = [int(value) for value in (player.get("eligibleSlots") or [])]
    eligible = tuple(
        _slot_name(value, where=f"player {player_id} eligibleSlots") for value in eligible_ids)
    if slot not in (BENCH_SLOT, IR_SLOT) and int(entry.get("lineupSlotId")) not in eligible_ids:
        violations.append({
            "team_id": team_id, "player_id": player_id, "player_name": name,
            "lineup_slot": slot, "eligible_slots": list(eligible),
        })
    position_id = player.get("defaultPositionId")
    return RosterPlayer(
        player_id=player_id,
        full_name=name,
        default_position=POSITION_NAMES.get(position_id, f"POS{position_id}"),
        eligible_slots=eligible,
        pro_team_id=int(player.get("proTeamId") or 0),
        injury_status=str(entry.get("injuryStatus") or player.get("injuryStatus") or "UNKNOWN"),
        lineup_slot=slot,
        is_starter=slot not in (BENCH_SLOT, IR_SLOT),
        acquisition_type=str(entry.get("acquisitionType") or "UNKNOWN"),
        acquisition_date=_epoch_ms_to_iso(entry.get("acquisitionDate")),
    )


def _normalize_draft(raw: Mapping[str, Any], settings: Mapping[str, Any], *,
                     roster_size: int, my_team_id: int) -> Draft:
    detail = raw.get("draftDetail") or {}
    draft_settings = settings.get("draftSettings") or {}
    drafted = bool(detail.get("drafted"))
    in_progress = bool(detail.get("inProgress"))
    raw_picks = detail.get("picks") or []

    if not drafted and raw_picks:
        raise EspnSchemaError(
            f"contradictory draft state: draftDetail reports drafted=false but carries "
            f"{len(raw_picks)} pick(s). A pick list on an undrafted league is intent "
            f"(a watchlist), not history, and will not be recorded as selections.")

    pick_order = tuple(int(value) for value in (draft_settings.get("pickOrder") or []))
    my_slot = pick_order.index(my_team_id) + 1 if my_team_id in pick_order else None

    if drafted:
        status = "complete"
    elif in_progress:
        status = "in_progress"
    else:
        status = "pre_draft"

    picks: tuple[DraftPick, ...] | None = None
    if drafted:
        picks = tuple(
            DraftPick(
                overall_pick=int(pick["overallPickNumber"]),
                round=int(pick["roundId"]),
                round_pick=int(pick["roundPickNumber"]),
                team_id=int(pick["teamId"]),
                player_id=int(pick["playerId"]),
                keeper=bool(pick.get("keeper")),
                bid_amount=float(pick.get("bidAmount") or 0.0),
            )
            for pick in sorted(raw_picks, key=lambda item: int(item["overallPickNumber"]))
        )

    return Draft(
        status=status,
        type=str(draft_settings.get("type") or "UNKNOWN"),
        rounds=roster_size,
        pick_order=pick_order,
        my_slot=my_slot,
        scheduled_at=_epoch_ms_to_iso(draft_settings.get("date")),
        time_per_selection=draft_settings.get("timePerSelection"),
        picks=picks,
    )


def _deadline_iso(status: Mapping[str, Any] | None) -> str | None:
    """The league's transaction deadline, ISO-8601, or None if it states none."""
    raw = (status or {}).get("transactionDeadline")
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return _epoch_ms_to_iso(raw)
    moment = parse_timestamp(raw)
    return moment.isoformat().replace("+00:00", "Z") if moment else None


def _normalize_waivers(settings: Mapping[str, Any],
                       teams: Sequence[Mapping[str, Any]],
                       status: Mapping[str, Any] | None = None) -> Waivers:
    acquisition = settings.get("acquisitionSettings") or {}
    process_days = tuple(str(day) for day in (acquisition.get("waiverProcessDays") or []))
    waiver_hours = int(acquisition.get("waiverHours") or 0)
    uses_budget = bool(acquisition.get("isUsingAcquisitionBudget"))
    # `waiverOrderReset` is a Boolean, and it means the order is rebuilt from
    # the standings every week -- i.e. inverse standings. Reading it as a mode
    # string and hunting for the word INVERSE lands on the opposite answer for
    # every league that sets it, which is every league that leaves the default.
    order_reset = bool(acquisition.get("waiverOrderReset"))
    mode = (WAIVER_MODE_FAAB if uses_budget
            else WAIVER_MODE_INVERSE if order_reset
            else WAIVER_MODE_ROLLING)
    lock_policy = (
        "continuous waivers; " if not process_days
        else f"waivers process {', '.join(process_days)} at {waiver_hours:02d}:00 league time; ")
    lock_policy += {
        WAIVER_MODE_FAAB: "FAAB bidding",
        WAIVER_MODE_INVERSE: "priority resets weekly to inverse standings",
        WAIVER_MODE_ROLLING: "rolling waiver priority (moves to last after a claim)",
    }[mode]
    lock_policy += "; player lock policy is not exposed by the league endpoint (see unsupported)"
    return Waivers(
        acquisition_type=str(acquisition.get("acquisitionType") or "UNKNOWN"),
        uses_acquisition_budget=uses_budget,
        acquisition_budget=float(acquisition.get("acquisitionBudget") or 0.0),
        process_days=process_days,
        waiver_hours=waiver_hours,
        order_reset=order_reset,
        mode=mode,
        transaction_deadline=_deadline_iso(status),
        acquisition_limit=int(acquisition.get("acquisitionLimit", -1)),
        lock_policy=lock_policy,
        team_priority={int(team["id"]): int(team.get("waiverRank") or 0) for team in teams},
        team_budget_spent={
            int(team["id"]): float((team.get("transactionCounter") or {}).get(
                "acquisitionBudgetSpent") or 0.0)
            for team in teams
        },
    )


def _normalize_playoffs(settings: Mapping[str, Any]) -> Playoffs:
    """The playoff calendar, in both units, derived exactly once.

    Two units are in play and conflating them is how a two-week final becomes
    four rounds. A *matchup period* is one bracket round; a *scoring period* is
    one NFL week. `playoffMatchupPeriodLength` is how many weeks a round spans.

    Both views are published here so no consumer has to derive the second from
    the first -- which is what produced two disagreeing answers in the tree.
    Downstream reads `playoff_scoring_periods` and multiplies nothing.
    """
    schedule_settings = settings.get("scheduleSettings") or {}
    regular = _require_int(schedule_settings.get("matchupPeriodCount"),
                           "scheduleSettings.matchupPeriodCount")
    regular_length = _require_int(schedule_settings.get("matchupPeriodLength"),
                                  "scheduleSettings.matchupPeriodLength")
    playoff_teams = _require_int(schedule_settings.get("playoffTeamCount"),
                                 "scheduleSettings.playoffTeamCount")
    length = _require_int(schedule_settings.get("playoffMatchupPeriodLength"),
                          "scheduleSettings.playoffMatchupPeriodLength")
    if length < 1 or regular_length < 1:
        raise EspnSchemaError(
            f"a matchup period cannot span {min(length, regular_length)} scoring periods")
    rounds = int(math.ceil(math.log2(playoff_teams))) if playoff_teams > 1 else 0
    periods = tuple(range(regular + 1, regular + 1 + rounds))
    # The regular season occupies `regular` periods of `regular_length` weeks,
    # so the bracket opens the week after those, and each round takes `length`.
    first_week = regular * regular_length + 1
    weeks = tuple(
        week
        for index in range(rounds)
        for week in range(first_week + index * length, first_week + (index + 1) * length)
    )
    return Playoffs(
        team_count=playoff_teams,
        seeding_rule=str(schedule_settings.get("playoffSeedingRule") or "UNKNOWN"),
        matchup_period_length=length,
        regular_season_matchup_periods=regular,
        playoff_matchup_periods=periods,
        playoff_scoring_periods=weeks,
        reseed=bool(schedule_settings.get("playoffReseed")),
    )


def _normalize_transaction(raw: Mapping[str, Any], *, pending: bool) -> Transaction:
    return Transaction(
        transaction_id=str(raw.get("id") or ""),
        type=str(raw.get("type") or "UNKNOWN"),
        status=str(raw.get("status") or ("PENDING" if pending else "UNKNOWN")),
        team_id=int(raw.get("teamId") or 0),
        scoring_period=raw.get("scoringPeriodId"),
        proposed_at=_epoch_ms_to_iso(raw.get("proposedDate")),
        processed_at=_epoch_ms_to_iso(raw.get("processDate")),
        bid_amount=float(raw.get("bidAmount") or 0.0),
        is_pending=bool(raw.get("isPending", pending)),
        items=tuple(dict(item) for item in (raw.get("items") or [])),
    )


#: ESPN's pool `status`, translated into what a manager can actually do. The
#: distinction matters: a free agent is added now and is yours; a player on
#: waivers is *claimed*, and the claim resolves at the next processing time.
#: Collapsing the two either invents an add that will not happen or refuses a
#: claim that is perfectly legal to place.
ACQUISITION_FREE_AGENT = "free_agent"
ACQUISITION_WAIVER_CLAIM = "waiver_claim"
ACQUISITION_ROSTERED = "rostered"

_AVAILABILITY_KINDS: Mapping[str, str] = {
    "FREEAGENT": ACQUISITION_FREE_AGENT,
    "FREE_AGENT": ACQUISITION_FREE_AGENT,
    "WAIVERS": ACQUISITION_WAIVER_CLAIM,
    "ONWAIVERS": ACQUISITION_WAIVER_CLAIM,
    "ONTEAM": ACQUISITION_ROSTERED,
}


def acquisition_kind(availability: Any) -> str:
    """What a manager may do with a pool entry, or ``"unknown"``."""
    return _AVAILABILITY_KINDS.get(str(availability).strip().upper(), "unknown")


def _normalize_pool(payload: Mapping[str, Any]) -> tuple[PoolPlayer, ...]:
    players = []
    for entry in payload.get("players") or []:
        player = entry.get("player") or {}
        player_id = entry.get("id", player.get("id"))
        if not isinstance(player_id, int):
            continue
        availability = str(entry.get("status") or "UNKNOWN")
        players.append(PoolPlayer(
            player_id=player_id,
            full_name=str(player.get("fullName") or ""),
            default_position=POSITION_NAMES.get(player.get("defaultPositionId"),
                                                f"POS{player.get('defaultPositionId')}"),
            eligible_slots=tuple(
                _slot_name(value, where=f"free agent {player_id}")
                for value in (player.get("eligibleSlots") or [])),
            pro_team_id=int(player.get("proTeamId") or 0),
            injury_status=str(player.get("injuryStatus") or "UNKNOWN"),
            availability=availability,
            acquisition_kind=acquisition_kind(availability),
        ))
    return tuple(players)


def normalize_league(views: Mapping[str, Mapping[str, Any]], *, expected: ExpectedIdentity,
                     retrieved_at: str, source_urls: Sequence[str],
                     credentialed: bool = False) -> LeagueSnapshot:
    """Turn raw ESPN view payloads into one validated snapshot, or refuse."""
    missing = [name for name in REQUIRED_VIEWS if name not in views]
    if missing:
        raise EspnSchemaError(
            f"missing required view(s): {', '.join(missing)}. A partial league is not a league.")

    _validate_retrieved_at(retrieved_at)

    raw, contributing = merge_views(views)
    settings = raw.get("settings")
    if not isinstance(settings, Mapping):
        raise EspnSchemaError("payload carries no settings block")
    for block in REQUIRED_SETTINGS_BLOCKS:
        if not isinstance(settings.get(block), Mapping):
            raise EspnSchemaError(f"incomplete settings: {block} is missing or not an object")
    _require_settings_fields(settings)

    _validate_identity(raw, expected)
    teams_raw = _validate_teams(raw, expected)
    my_team = _resolve_my_team(teams_raw, expected)
    roster_settings = _roster_settings(settings)
    # The one rules contract. Scoring and roster identity are ITS answer, not a
    # second derivation here -- two modules each hashing "the scoring rules"
    # and returning different digests is precisely the drift this replaces.
    contract = espn_contract.from_settings_payload(raw)

    status = raw.get("status") or {}
    league = LeagueIdentity(
        league_id=int(raw["id"]),
        season=int(raw["seasonId"]),
        segment_id=int(raw.get("segmentId") or 0),
        name=str(settings.get("name") or ""),
        size=len(teams_raw),
        is_public=bool(settings.get("isPublic")),
        current_scoring_period=_require_int(raw.get("scoringPeriodId"), "scoringPeriodId"),
        first_scoring_period=int(raw.get("firstScoringPeriod") or 0),
        final_scoring_period=int(raw.get("finalScoringPeriod") or 0),
        current_matchup_period=int(raw.get("currentMatchupPeriod")
                                   or status.get("currentMatchupPeriod") or 0),
        is_active=bool(status.get("isActive")),
        teams_joined=int(status.get("teamsJoined") or len(teams_raw)),
    )

    members = tuple(
        Member(
            member_key=member_key(member.get("id")),
            display_name=str(member.get("displayName") or ""),
            is_league_manager=bool(member.get("isLeagueManager")),
        )
        for member in (raw.get("members") or [])
    )

    teams = tuple(
        Team(
            team_id=int(team["id"]),
            name=str(team.get("name") or "").strip(),
            abbrev=str(team.get("abbrev") or ""),
            owner_keys=tuple(member_key(owner) for owner in (team.get("owners") or [])),
            division_id=int(team.get("divisionId") or 0),
            waiver_rank=int(team.get("waiverRank") or 0),
            playoff_seed=int(team.get("playoffSeed") or 0),
            acquisition_budget_spent=float(
                (team.get("transactionCounter") or {}).get("acquisitionBudgetSpent") or 0.0),
        )
        for team in teams_raw
    )

    unmatched: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    rosters: dict[int, tuple[RosterPlayer, ...]] = {}
    for team in teams_raw:
        team_id = int(team["id"])
        entries = ((team.get("roster") or {}).get("entries")) or []
        players = [
            normalized for normalized in (
                _normalize_roster_entry(entry, team_id=team_id, unmatched=unmatched,
                                        violations=violations)
                for entry in entries
            ) if normalized is not None
        ]
        rosters[team_id] = tuple(players)

    # The same player on two rosters is not a tie to break later: every value
    # downstream double-counts him, and the trade scan will happily offer a
    # player his owner does not have.
    seen: dict[int, int] = {}
    for team_id, players in rosters.items():
        for player in players:
            first = seen.get(player.player_id)
            if first is not None:
                raise EspnSchemaError(
                    f"duplicate roster player {player.player_id} ({player.full_name}) appears "
                    f"on team {first} and team {team_id}; this league payload is inconsistent")
            seen[player.player_id] = team_id

    draft = _normalize_draft(raw, settings, roster_size=roster_settings.roster_size,
                             my_team_id=expected.team_id)

    populated = any(players for players in rosters.values())
    if draft.status == "pre_draft" and populated:
        raise EspnSchemaError(
            "contradictory league state: draftDetail reports the league is not drafted, but "
            "teams carry roster entries. Refusing to publish a roster the draft cannot explain.")
    roster_state = "populated" if populated else (
        "empty_pre_draft" if draft.status == "pre_draft" else "empty")

    schedule = tuple(
        ScheduleGame(
            matchup_period=int(game.get("matchupPeriodId") or 0),
            home_team_id=int((game.get("home") or {}).get("teamId") or 0),
            away_team_id=int((game.get("away") or {}).get("teamId") or 0),
            winner=str(game.get("winner") or "UNDECIDED"),
            home_points=(game.get("home") or {}).get("totalPoints"),
            away_points=(game.get("away") or {}).get("totalPoints"),
            playoff_tier=str(game.get("playoffTierType") or "NONE"),
        )
        for game in (raw.get("schedule") or [])
    )

    schedule_settings = settings.get("scheduleSettings") or {}
    scoring_settings = settings.get("scoringSettings") or {}
    standings = Standings(
        rows=tuple(
            StandingsRow(
                team_id=int(team["id"]),
                wins=int(((team.get("record") or {}).get("overall") or {}).get("wins") or 0),
                losses=int(((team.get("record") or {}).get("overall") or {}).get("losses") or 0),
                ties=int(((team.get("record") or {}).get("overall") or {}).get("ties") or 0),
                percentage=float(
                    ((team.get("record") or {}).get("overall") or {}).get("percentage") or 0.0),
                points_for=float(
                    ((team.get("record") or {}).get("overall") or {}).get("pointsFor") or 0.0),
                points_against=float(
                    ((team.get("record") or {}).get("overall") or {}).get("pointsAgainst") or 0.0),
                playoff_seed=int(team.get("playoffSeed") or 0),
                games_back=float(
                    ((team.get("record") or {}).get("overall") or {}).get("gamesBack") or 0.0),
            )
            for team in teams_raw
        ),
        tiebreaker={
            "playoff_seeding_rule": str(schedule_settings.get("playoffSeedingRule") or "UNKNOWN"),
            "matchup_tie_rule": str(scoring_settings.get("matchupTieRule") or "UNKNOWN"),
            "playoff_matchup_tie_rule": str(
                scoring_settings.get("playoffMatchupTieRule") or "UNKNOWN"),
            "home_team_bonus": float(scoring_settings.get("homeTeamBonus") or 0.0),
            "playoff_reseed": bool(schedule_settings.get("playoffReseed")),
        },
    )

    transactions = Transactions(
        completed=tuple(
            _normalize_transaction(item, pending=False) for item in (raw.get("transactions") or [])),
        pending=tuple(
            _normalize_transaction(item, pending=True)
            for item in (raw.get("pendingTransactions") or [])),
    )

    scoring = Scoring(
        scoring_type=str(scoring_settings.get("scoringType") or "UNKNOWN"),
        matchup_tie_rule=str(scoring_settings.get("matchupTieRule") or "UNKNOWN"),
        playoff_matchup_tie_rule=str(scoring_settings.get("playoffMatchupTieRule") or "UNKNOWN"),
        home_team_bonus=float(scoring_settings.get("homeTeamBonus") or 0.0),
        items=tuple(dict(item) for item in (scoring_settings.get("scoringItems") or [])),
    )

    pool_payload = views.get("kona_player_info")
    free_agents = _normalize_pool(pool_payload) if isinstance(pool_payload, Mapping) else None

    hashes = {
        "league": _sha256({
            "league_id": league.league_id, "season": league.season,
            "segment_id": league.segment_id, "size": league.size, "name": league.name,
        }),
        # Taken from the contract, never recomputed. A local digest over the
        # same rules would be a second opinion nobody asked for, and the two
        # would drift the first time either canonicalisation changed.
        "scoring": contract.scoring_hash,
        "roster": contract.roster_slot_hash,
    }

    # Names only, never values: an unrecognized block is exactly where an
    # echoed cookie would sit.
    unsupported = {
        "league_keys": sorted(set(raw) - CONSUMED_LEAGUE_KEYS),
        "settings_keys": sorted(set(settings) - CONSUMED_SETTINGS_KEYS),
        "views_not_requested": sorted(set(OPTIONAL_VIEWS) - set(views)),
    }

    warnings: list[str] = []
    if violations:
        warnings.append(f"{len(violations)} roster entr(ies) sit in a slot they are not eligible for")
    if unmatched:
        warnings.append(f"{len(unmatched)} roster entr(ies) could not be normalized")
    if roster_state == "empty_pre_draft":
        warnings.append(
            f"league has not drafted (scheduled {draft.scheduled_at}); no rosters and no picks exist")

    return LeagueSnapshot(
        schema_version=SCHEMA_VERSION,
        retrieved_at=retrieved_at,
        source=SnapshotSource(views=tuple(contributing), urls=tuple(source_urls),
                              credentialed=bool(credentialed)),
        league=league,
        members=members,
        teams=teams,
        my_team=my_team,
        roster_settings=roster_settings,
        rosters=rosters,
        roster_state=roster_state,
        free_agents=free_agents,
        schedule=schedule,
        draft=draft,
        waivers=_normalize_waivers(settings, teams_raw, status),
        standings=standings,
        playoffs=_normalize_playoffs(settings),
        transactions=transactions,
        scoring=scoring,
        rules={
            "contract_version": contract.contract_version,
            "scoring": contract.canonical_scoring(),
            "roster": contract.canonical_roster(),
            "league_rules": contract.rules.canonical(),
        },
        hashes=hashes,
        unsupported=unsupported,
        unmatched_players=tuple(unmatched),
        eligibility_violations=tuple(violations),
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def snapshot_to_dict(snapshot: LeagueSnapshot) -> dict[str, Any]:
    """Plain-JSON view of a snapshot, with integer roster keys stringified."""
    payload = asdict(snapshot)
    payload["rosters"] = {str(team_id): players for team_id, players in payload["rosters"].items()}
    payload["waivers"]["team_priority"] = {
        str(team_id): rank for team_id, rank in payload["waivers"]["team_priority"].items()}
    payload["waivers"]["team_budget_spent"] = {
        str(team_id): spent for team_id, spent in payload["waivers"]["team_budget_spent"].items()}
    return payload


# --------------------------------------------------------------------------- #
# Reading a snapshot back
# --------------------------------------------------------------------------- #
def playoff_scoring_periods(snapshot: Mapping[str, Any]) -> tuple[int, ...]:
    """The league's real playoff weeks, as the adapter already expanded them.

    This is the single definition. A consumer that takes
    `playoff_matchup_periods` and multiplies by the round length is redoing
    work that has been done, and squares it: with a two-week final, period 16
    becomes weeks 31 and 32.
    """
    playoffs = snapshot.get("playoffs") or {}
    weeks = playoffs.get("playoff_scoring_periods")
    if weeks is None:
        raise EspnSchemaError(
            "snapshot publishes no playoff_scoring_periods; it predates the single "
            "playoff-calendar derivation and its periods cannot be interpreted safely")
    return tuple(int(week) for week in weeks)


def waiver_rules_from_snapshot(snapshot: Mapping[str, Any], **kwargs) -> Any:
    """Waiver rules as a view over this snapshot (see `waiver_rules`)."""
    from . import waiver_rules  # local: waiver_rules reads this module's vocabulary

    return waiver_rules.from_snapshot(snapshot, **kwargs)


def snapshot_is_usable(payload: Any, *, now: dt.datetime) -> str | None:
    """Why this payload may not be treated as a league snapshot, or None."""
    if not isinstance(payload, Mapping):
        return f"not a snapshot object ({type(payload).__name__})"
    if payload.get("schema_version") != SCHEMA_VERSION:
        return f"schema {payload.get('schema_version')!r} is not {SCHEMA_VERSION}"
    retrieved = parse_timestamp(payload.get("retrieved_at"))
    if retrieved is None:
        return f"retrieved_at {payload.get('retrieved_at')!r} is not a timestamp"
    if (retrieved - now).total_seconds() > FUTURE_TOLERANCE_SECONDS:
        return f"retrieved_at {payload['retrieved_at']!r} is in the future"
    source = payload.get("source") or {}
    started = parse_timestamp(source.get("request_started"))
    received = parse_timestamp(source.get("response_received"))
    if started and received and received < started:
        return (f"capture clock runs backwards: response {source['response_received']} "
                f"precedes request {source['request_started']}")
    return None


def load_latest_snapshot(directory: str | Path, *,
                         now: str | dt.datetime | None = None) -> dict | None:
    """The most recently *retrieved* valid snapshot in *directory*, or None.

    Selection is by the timestamp the snapshot carries, never by the file's
    mtime. A checkout, an rsync or a `cp -r` rewrites every mtime without
    re-reading anything from ESPN, and a stale capture that wins that
    comparison becomes "the current state of the league".
    """
    moment = (dt.datetime.now(dt.timezone.utc) if now is None
              else now if isinstance(now, dt.datetime) else parse_timestamp(now))
    if moment is None:
        raise EspnSchemaError(f"now {now!r} is not a timestamp")
    folder = Path(directory)
    if not folder.is_dir():
        return None
    best: tuple[dt.datetime, str, dict] | None = None
    for path in sorted(folder.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if snapshot_is_usable(payload, now=moment) is not None:
            continue
        retrieved = parse_timestamp(payload["retrieved_at"])
        candidate = (retrieved, path.name, payload)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        return None
    payload = best[2]
    payload.setdefault("_snapshot_path", str(folder / best[1]))
    return payload

def content_digest(payload: Mapping[str, Any]) -> str:
    """sha256 over the snapshot body, excluding the digest field itself."""
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    return _sha256(body)


def write_snapshot(snapshot: LeagueSnapshot, directory: str | Path) -> Path:
    """Write an immutable snapshot file and return its path.

    Immutable in two senses: the filename carries the league, season and
    retrieval instant, and the write refuses to clobber an existing file
    (``FileExistsError``). A snapshot is evidence of what the league was at one
    moment; rewriting one silently rewrites history.
    """
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = re.sub(r"[^0-9A-Za-z]", "", snapshot.retrieved_at)
    path = target_dir / (
        f"espn-league-{snapshot.league.league_id}-{snapshot.league.season}-{stamp}.json")

    payload = snapshot_to_dict(snapshot)
    payload["content_sha256"] = content_digest(payload)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, default=str)
        handle.write("\n")
    path.chmod(0o444)
    return path
