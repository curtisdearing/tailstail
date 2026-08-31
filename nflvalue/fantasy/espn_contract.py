"""Versioned ESPN league contract, imported from the live settings payload.

Why this module exists
----------------------
The league's scoring is NOT ESPN's default.  Receiving two-point conversions
pay 4.0 while passing and rushing pay 2.0; the D/ST return-touchdown values
are amplified (20 for an interception or fumble return, 30 for a blocked-kick
return, 12 for a kick or punt return) against 6 for the same play by a
position player; and both the points-allowed and yards-allowed ladders are
customised.  None of that survives a "close enough" reconstruction, so this
module reads the live payload and refuses anything it cannot represent
exactly.

The hard rule
-------------
Every value here comes from the league settings payload.  Nothing is defaulted
from ESPN's standard scoring and nothing is inferred from prose notes.  A
category the registry does not know, or a lineup slot with no representation,
raises :class:`UnsupportedEspnSetting` rather than being approximated away --
silently dropping a category the league actually scores is the single most
expensive failure this contract can have, because every downstream number
would look plausible and be wrong.

Scope
-----
Contracts and pure scoring functions only.  No projection or modelling of
kickers or defenses lives here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

CONTRACT_VERSION = 1

#: ESPN lineup slot ids -> the slot names this project uses.
LINEUP_SLOT_NAMES: Mapping[int, str] = {
    0: "QB", 1: "TQB", 2: "RB", 3: "RB/WR", 4: "WR", 5: "WR/TE", 6: "TE",
    7: "OP", 8: "DT", 9: "DE", 10: "LB", 11: "DL", 12: "CB", 13: "S",
    14: "DB", 15: "DP", 16: "D/ST", 17: "K", 18: "P", 19: "HC",
    20: "BE", 21: "IR", 22: "UNKNOWN22", 23: "FLEX", 24: "EDR",
}

#: ESPN PLAYER POSITION ids. A different id space from the lineup slots above
#: -- ``positionLimits`` and ``pointsOverrides`` are keyed by these, while
#: ``lineupSlotCounts`` is keyed by slot. Conflating the two silently reads
#: "max 4 quarterbacks" as a limit on some other position, so only the ids
#: reconciled against the league's own settings page are named here.
PLAYER_POSITION_NAMES: Mapping[int, str] = {
    1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "D/ST",
}

#: ESPN position id used for the team defense in ``pointsOverrides``.
DST_POSITION_ID = 16


@dataclass(frozen=True)
class StatDefinition:
    """What an ESPN ``statId`` means, in this project's vocabulary."""

    stat_id: int
    key: str
    label: str
    group: str


def _defs(*rows: tuple[int, str, str, str]) -> Mapping[int, StatDefinition]:
    out: dict[int, StatDefinition] = {}
    for stat_id, key, label, group in rows:
        if stat_id in out:                                  # pragma: no cover
            raise ValueError(f"duplicate statId {stat_id} in registry")
        out[stat_id] = StatDefinition(stat_id, key, label, group)
    return out


#: The statIds this contract can represent.  Labels are ESPN's own, read from
#: the league's scoring settings page; the reconciliation test asserts that
#: every id here matches the live payload and vice versa, so an ESPN change
#: surfaces as a failure instead of a silent mis-score.
STAT_REGISTRY: Mapping[int, StatDefinition] = _defs(
    # ---- offense -----------------------------------------------------------
    (3, "passing_yards", "Passing yards", "offense"),
    (4, "passing_td", "TD Pass", "offense"),
    (19, "passing_2pt", "2pt Passing Conversion", "offense"),
    (20, "interception_thrown", "Interceptions Thrown", "offense"),
    (24, "rushing_yards", "Rushing yards", "offense"),
    (25, "rushing_td", "TD Rush", "offense"),
    (26, "rushing_2pt", "2pt Rushing Conversion", "offense"),
    (42, "receiving_yards", "Receiving yards", "offense"),
    (43, "receiving_td", "TD Reception", "offense"),
    (44, "receiving_2pt", "2pt Receiving Conversion", "offense"),
    (53, "reception", "Each reception", "offense"),
    (63, "fumble_recovered_td", "Fumble Recovered for TD", "offense"),
    (72, "fumble_lost", "Total Fumbles Lost", "offense"),
    (155, "team_win", "Team Win", "team"),
    # ---- kicking -----------------------------------------------------------
    (86, "pat_made", "Each PAT Made", "kicking"),
    (88, "pat_missed", "Each PAT Missed", "kicking"),
    (85, "fg_missed_total", "Total FG Missed", "kicking"),
    (80, "fg_made_0_39", "FG Made (0-39 yards)", "kicking"),
    (77, "fg_made_40_49", "FG Made (40-49 yards)", "kicking"),
    (198, "fg_made_50_59", "FG Made (50-59 yards)", "kicking"),
    (201, "fg_made_60_plus", "FG Made (60+ yards)", "kicking"),
    # ---- touchdowns scored by a return or a defense -------------------------
    (101, "kickoff_return_td", "Kickoff Return TD", "return"),
    (102, "punt_return_td", "Punt Return TD", "return"),
    (103, "fumble_return_td", "Fumble Return TD", "return"),
    (104, "interception_return_td", "Interception Return TD", "return"),
    (93, "blocked_kick_return_td", "Blocked Punt or FG return for TD", "return"),
    (206, "two_point_return", "2pt Return", "return"),
    (209, "one_point_safety", "1pt Safety", "return"),
    # ---- team defense ------------------------------------------------------
    (99, "defensive_sack", "Each Sack", "defense"),
    (95, "defensive_interception", "Each Interception", "defense"),
    (96, "defensive_fumble_recovery", "Each Fumble Recovered", "defense"),
    (98, "defensive_safety", "Each Safety", "defense"),
    (97, "blocked_kick", "Blocked Punt, PAT or FG", "defense"),
    # ---- points allowed ----------------------------------------------------
    (89, "points_allowed_0", "0 points allowed", "points_allowed"),
    (90, "points_allowed_1_6", "1-6 points allowed", "points_allowed"),
    (91, "points_allowed_7_13", "7-13 points allowed", "points_allowed"),
    (92, "points_allowed_14_17", "14-17 points allowed", "points_allowed"),
    (123, "points_allowed_28_34", "28-34 points allowed", "points_allowed"),
    (124, "points_allowed_35_45", "35-45 points allowed", "points_allowed"),
    (125, "points_allowed_46_plus", "46+ points allowed", "points_allowed"),
    # ---- yards allowed -----------------------------------------------------
    (128, "yards_allowed_under_100", "Less than 100 total yards allowed", "yards_allowed"),
    (129, "yards_allowed_100_199", "100-199 total yards allowed", "yards_allowed"),
    (130, "yards_allowed_200_299", "200-299 total yards allowed", "yards_allowed"),
    (132, "yards_allowed_350_399", "350-399 total yards allowed", "yards_allowed"),
    (133, "yards_allowed_400_449", "400-449 total yards allowed", "yards_allowed"),
    (134, "yards_allowed_450_499", "450-499 total yards allowed", "yards_allowed"),
    (135, "yards_allowed_500_549", "500-549 total yards allowed", "yards_allowed"),
    (136, "yards_allowed_550_plus", "550+ total yards allowed", "yards_allowed"),
)

#: Tier ladders are declared here, not derived from which categories happen to
#: carry a non-zero value.  ESPN omits a band from the payload when it scores
#: zero, so a ladder built only from the payload would have HOLES: a defense
#: allowing 20 points, or 320 yards, would match nothing and score nothing
#: rather than the zero the league actually intends.  The bands below are
#: exhaustive and non-overlapping over ``[0, inf)``, and a test proves it.
POINTS_ALLOWED_BANDS: Sequence[tuple[int, float, str | None]] = (
    (0, 0, "points_allowed_0"),
    (1, 6, "points_allowed_1_6"),
    (7, 13, "points_allowed_7_13"),
    (14, 17, "points_allowed_14_17"),
    (18, 21, None),                     # scored zero; absent from the payload
    (22, 27, None),                     # scored zero; absent from the payload
    (28, 34, "points_allowed_28_34"),
    (35, 45, "points_allowed_35_45"),
    (46, float("inf"), "points_allowed_46_plus"),
)

YARDS_ALLOWED_BANDS: Sequence[tuple[int, float, str | None]] = (
    (0, 99, "yards_allowed_under_100"),
    (100, 199, "yards_allowed_100_199"),
    (200, 299, "yards_allowed_200_299"),
    (300, 349, None),                   # scored zero; absent from the payload
    (350, 399, "yards_allowed_350_399"),
    (400, 449, "yards_allowed_400_449"),
    (450, 499, "yards_allowed_450_499"),
    (500, 549, "yards_allowed_500_549"),
    (550, float("inf"), "yards_allowed_550_plus"),
)

#: ESPN's field-goal distance buckets for THIS league, as its settings page
#: names them.  Boundaries are inclusive on both ends.
FIELD_GOAL_BUCKETS: Sequence[tuple[int, float, str]] = (
    (0, 39, "fg_made_0_39"),
    (40, 49, "fg_made_40_49"),
    (50, 59, "fg_made_50_59"),
    (60, float("inf"), "fg_made_60_plus"),
)


class UnsupportedEspnSetting(ValueError):
    """A live setting that cannot be represented exactly.

    Raised instead of approximating.  The message names the offending id so a
    reviewer can decide whether to extend the registry or change the league.
    """


@dataclass(frozen=True)
class ScoringCategory:
    """One league scoring category, exactly as the payload states it."""

    stat_id: int
    key: str
    label: str
    group: str
    points: float
    position_overrides: Mapping[str, float] = field(default_factory=dict)

    def value_for(self, position_id: int | None = None) -> float:
        """Points for a scorer at ``position_id``.

        ESPN expresses "6 for a wide receiver, 20 for a defense" as a base
        value plus a per-position override, and both are league-configurable.
        """
        if position_id is not None:
            override = self.position_overrides.get(str(int(position_id)))
            if override is not None:
                return float(override)
        return float(self.points)

    def canonical(self) -> dict[str, Any]:
        return {
            "stat_id": self.stat_id,
            "key": self.key,
            "points": _num(self.points),
            "position_overrides": {
                str(k): _num(v) for k, v in sorted(
                    self.position_overrides.items(), key=lambda kv: int(kv[0]))
            },
        }


def _num(value: Any) -> float | int:
    """Normalise numerics so 6 and 6.0 hash the same.

    ESPN returns ``6.0`` in one payload and ``6`` in another for the same
    setting; without this the hash would change when nothing about the league
    did, and a hash that moves on its own is worse than no hash.
    """
    number = float(value)
    if number == int(number):
        return int(number)
    return round(number, 6)


@dataclass(frozen=True)
class RosterContract:
    """Starting slots, bench and IR, exactly as configured."""

    slot_counts: Mapping[str, int]
    position_limits: Mapping[str, int]

    @property
    def starters(self) -> dict[str, int]:
        skip = {"BE", "IR"}
        return {k: v for k, v in self.slot_counts.items() if v > 0 and k not in skip}

    @property
    def bench(self) -> int:
        return int(self.slot_counts.get("BE", 0))

    @property
    def injured_reserve(self) -> int:
        return int(self.slot_counts.get("IR", 0))

    @property
    def total_starters(self) -> int:
        return sum(self.starters.values())

    def canonical(self) -> dict[str, Any]:
        return {
            "slot_counts": {k: int(v) for k, v in sorted(self.slot_counts.items())},
            "position_limits": {k: int(v) for k, v in sorted(self.position_limits.items())},
        }


@dataclass(frozen=True)
class LeagueRules:
    """Waiver, regular-season, matchup-period and playoff rules."""

    waiver_type: str
    waiver_hours: int
    waiver_process_days: tuple[str, ...]
    waiver_order_reset: bool
    uses_acquisition_budget: bool
    acquisition_budget: int
    matchup_period_count: int
    matchup_period_length: int
    regular_season_matchups: int
    playoff_team_count: int
    playoff_matchup_period_length: int
    playoff_seeding_rule: str
    playoff_reseed: bool
    first_scoring_period: int
    final_scoring_period: int

    def canonical(self) -> dict[str, Any]:
        return {
            "waiver_type": self.waiver_type,
            "waiver_hours": int(self.waiver_hours),
            # sorted: ESPN returns this list in an arbitrary order, and an
            # arbitrary order must not move the hash
            "waiver_process_days": sorted(self.waiver_process_days),
            "waiver_order_reset": bool(self.waiver_order_reset),
            "uses_acquisition_budget": bool(self.uses_acquisition_budget),
            "acquisition_budget": int(self.acquisition_budget),
            "matchup_period_count": int(self.matchup_period_count),
            "matchup_period_length": int(self.matchup_period_length),
            "regular_season_matchups": int(self.regular_season_matchups),
            "playoff_team_count": int(self.playoff_team_count),
            "playoff_matchup_period_length": int(self.playoff_matchup_period_length),
            "playoff_seeding_rule": self.playoff_seeding_rule,
            "playoff_reseed": bool(self.playoff_reseed),
            "first_scoring_period": int(self.first_scoring_period),
            "final_scoring_period": int(self.final_scoring_period),
        }


@dataclass(frozen=True)
class LeagueContract:
    """A league's scoring and roster rules, versioned and hashable."""

    contract_version: int
    season: int
    scoring_type: str
    categories: Mapping[str, ScoringCategory]
    roster: RosterContract
    rules: LeagueRules

    # -- lookups ------------------------------------------------------------
    def points(self, key: str, position_id: int | None = None) -> float:
        """Points for ``key``; 0.0 when the league does not score it.

        Absence is a real answer: ESPN omits zero-valued categories, so a key
        the registry knows but the payload never mentions is genuinely worth
        nothing and must not raise.
        """
        category = self.categories.get(key)
        if category is None:
            return 0.0
        return category.value_for(position_id)

    def dst_points(self, key: str) -> float:
        return self.points(key, DST_POSITION_ID)

    # -- canonical serialisation + hashes -----------------------------------
    def canonical_scoring(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "scoring_type": self.scoring_type,
            "categories": [
                self.categories[k].canonical() for k in sorted(self.categories)
            ],
        }

    def canonical_roster(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "roster": self.roster.canonical(),
        }

    @property
    def scoring_hash(self) -> str:
        return _hash(self.canonical_scoring())

    @property
    def roster_slot_hash(self) -> str:
        return _hash(self.canonical_roster())

    def to_scoring_rules(self):
        """Project the offensive categories onto :class:`ScoringRules`.

        Only the offensive event types the simulator produces cross over; the
        league contract stays the authority for everything else.  This is what
        applies league scoring to the scoring-independent football components
        without touching how those components are produced.
        """
        from .config import ScoringRules

        return ScoringRules(
            reception=self.points("reception"),
            passing_yard=self.points("passing_yards"),
            passing_td=self.points("passing_td"),
            interception=self.points("interception_thrown"),
            rushing_yard=self.points("rushing_yards"),
            rushing_td=self.points("rushing_td"),
            receiving_yard=self.points("receiving_yards"),
            receiving_td=self.points("receiving_td"),
            passing_two_point=self.points("passing_2pt"),
            rushing_two_point=self.points("rushing_2pt"),
            receiving_two_point=self.points("receiving_2pt"),
            fumble_lost=self.points("fumble_lost"),
        )


def _hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def from_settings_payload(payload: Mapping[str, Any]) -> LeagueContract:
    """Build a contract from a raw ESPN ``view=mSettings`` response.

    Rejects rather than approximates: an unknown ``statId``, an unknown lineup
    slot id, or a populated slot this project has no name for all raise
    :class:`UnsupportedEspnSetting`.
    """
    try:
        settings = payload["settings"]
        scoring = settings["scoringSettings"]
        roster = settings["rosterSettings"]
        schedule = settings["scheduleSettings"]
        acquisition = settings["acquisitionSettings"]
        status = payload["status"]
        season = int(payload["seasonId"])
    except (KeyError, TypeError) as exc:
        raise UnsupportedEspnSetting(f"settings payload missing {exc}") from exc

    categories: dict[str, ScoringCategory] = {}
    unknown: list[int] = []
    items = scoring.get("scoringItems")
    if not items:
        raise UnsupportedEspnSetting(
            "settings carry no scoringItems: a league that prices nothing cannot be "
            "scored exactly, and an empty ruleset hashes to a stable digest over nothing")
    for item in items:
        stat_id = int(item["statId"])
        definition = STAT_REGISTRY.get(stat_id)
        if definition is None:
            unknown.append(stat_id)
            continue
        overrides = {
            str(int(k)): float(v)
            for k, v in (item.get("pointsOverrides") or {}).items()
        }
        categories[definition.key] = ScoringCategory(
            stat_id=stat_id, key=definition.key, label=definition.label,
            group=definition.group, points=float(item.get("points", 0.0)),
            position_overrides=overrides,
        )
    if unknown:
        raise UnsupportedEspnSetting(
            "league scores categories this contract cannot represent exactly: "
            f"statId(s) {sorted(unknown)}. Add them to STAT_REGISTRY with their "
            "ESPN label -- do not approximate or drop them."
        )

    slot_counts: dict[str, int] = {}
    unnamed: list[int] = []
    for raw_id, count in (roster.get("lineupSlotCounts") or {}).items():
        slot_id = int(raw_id)
        name = LINEUP_SLOT_NAMES.get(slot_id)
        if name is None:
            if int(count) > 0:
                unnamed.append(slot_id)
            continue
        if int(count) > 0:
            slot_counts[name] = int(count)
    if unnamed:
        raise UnsupportedEspnSetting(
            f"league uses lineup slot id(s) {sorted(unnamed)} with no representation"
        )

    # Roster maximums, keyed by PLAYER POSITION id. -1 means "no limit" and is
    # dropped; an id with a real limit that has no verified name is kept under
    # its numeric id rather than guessed at or discarded, so the value survives
    # into the hash without inventing a meaning for it.
    position_limits: dict[str, int] = {}
    for raw_id, limit in (roster.get("positionLimits") or {}).items():
        if int(limit) < 0:
            continue
        position_id = int(raw_id)
        name = PLAYER_POSITION_NAMES.get(position_id, f"position_{position_id}")
        position_limits[name] = int(limit)

    rules = LeagueRules(
        waiver_type=str(acquisition.get("acquisitionType", "")),
        waiver_hours=int(acquisition.get("waiverHours", 0)),
        waiver_process_days=tuple(acquisition.get("waiverProcessDays", ()) or ()),
        waiver_order_reset=bool(acquisition.get("waiverOrderReset", False)),
        uses_acquisition_budget=bool(acquisition.get("isUsingAcquisitionBudget", False)),
        acquisition_budget=int(acquisition.get("acquisitionBudget", 0)),
        matchup_period_count=int(schedule.get("matchupPeriodCount", 0)),
        matchup_period_length=int(schedule.get("matchupPeriodLength", 0)),
        regular_season_matchups=int(schedule.get("matchupPeriodCount", 0)),
        playoff_team_count=int(schedule.get("playoffTeamCount", 0)),
        playoff_matchup_period_length=int(schedule.get("playoffMatchupPeriodLength", 0)),
        playoff_seeding_rule=str(schedule.get("playoffSeedingRule", "")),
        playoff_reseed=bool(schedule.get("playoffReseed", False)),
        first_scoring_period=int(status.get("firstScoringPeriod", 0)),
        final_scoring_period=int(status.get("finalScoringPeriod", 0)),
    )

    return LeagueContract(
        contract_version=CONTRACT_VERSION,
        season=season,
        scoring_type=str(scoring.get("scoringType", "")),
        categories=categories,
        roster=RosterContract(slot_counts=slot_counts, position_limits=position_limits),
        rules=rules,
    )


def load_contract(path: str) -> LeagueContract:
    with open(path, encoding="utf-8") as handle:
        return from_settings_payload(json.load(handle))
