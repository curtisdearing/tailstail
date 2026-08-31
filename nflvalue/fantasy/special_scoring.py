"""Pure scoring functions for kickers and team defenses.

Scoring only.  Nothing here projects, simulates or ranks a kicker or a
defense -- these functions turn an already-known stat line into the league's
points, using the imported :class:`LeagueContract` as the only source of
values.  Modelling comes later and is deliberately not started here.

Every value is read from the contract.  No constant in this module encodes a
point value, so a league settings change flows through by re-import rather
than by editing code.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .espn_contract import (
    DST_POSITION_ID,
    FIELD_GOAL_BUCKETS,
    POINTS_ALLOWED_BANDS,
    YARDS_ALLOWED_BANDS,
    LeagueContract,
)


class StatLineError(ValueError):
    """A stat line that cannot be scored without guessing."""


def field_goal_bucket(distance: float) -> str:
    """The contract key for a field goal of ``distance`` yards.

    Bounds are inclusive, matching how ESPN's settings page names the buckets
    ("FG Made (40-49 yards)").  A 49-yarder and a 50-yarder are worth
    different points in this league, so the edges are tested explicitly.
    """
    if math.isnan(distance) or distance < 0:
        raise StatLineError(f"field-goal distance {distance!r} is not a real attempt")
    for low, high, key in FIELD_GOAL_BUCKETS:
        if low <= distance <= high:
            return key
    raise StatLineError(f"no field-goal bucket covers {distance}")   # pragma: no cover


def _band_key(value: float, bands: Sequence[tuple[int, float, str | None]], what: str) -> str | None:
    if math.isnan(value) or value < 0:
        raise StatLineError(f"{what} {value!r} is not a real total")
    for low, high, key in bands:
        if low <= value <= high:
            return key
    raise StatLineError(f"no {what} band covers {value}")            # pragma: no cover


def points_allowed_key(points: float) -> str | None:
    """Contract key for a points-allowed total, or ``None`` for a zero band."""
    return _band_key(points, POINTS_ALLOWED_BANDS, "points allowed")


def yards_allowed_key(yards: float) -> str | None:
    """Contract key for a yards-allowed total, or ``None`` for a zero band."""
    return _band_key(yards, YARDS_ALLOWED_BANDS, "yards allowed")


def _count(line: Mapping[str, Any], key: str) -> float:
    value = line.get(key, 0) or 0
    return float(value)


def score_kicker(line: Mapping[str, Any], contract: LeagueContract) -> float:
    """Score a kicker's stat line.

    ``line`` accepts either ``field_goals_made`` as a sequence of distances --
    the honest input, since the league pays by distance -- or pre-bucketed
    counts under the contract keys.  Supplying both for the same bucket is an
    error rather than a silent double count.

    Recognised keys: ``field_goals_made`` (distances), ``fg_made_0_39``,
    ``fg_made_40_49``, ``fg_made_50_59``, ``fg_made_60_plus``,
    ``field_goals_missed``, ``pat_made``, ``pat_missed``.
    """
    bucket_counts: dict[str, float] = {key: 0.0 for _, _, key in FIELD_GOAL_BUCKETS}

    distances = line.get("field_goals_made")
    if distances is not None and not isinstance(distances, (list, tuple)):
        raise StatLineError("field_goals_made must be a sequence of distances")
    for distance in distances or ():
        bucket_counts[field_goal_bucket(float(distance))] += 1.0

    for key in list(bucket_counts):
        if key in line:
            if distances:
                raise StatLineError(
                    f"{key} given alongside field_goals_made; supply one or the "
                    "other so makes are not counted twice"
                )
            bucket_counts[key] += _count(line, key)

    total = 0.0
    for key, count in bucket_counts.items():
        total += count * contract.points(key)
    total += _count(line, "field_goals_missed") * contract.points("fg_missed_total")
    total += _count(line, "pat_made") * contract.points("pat_made")
    total += _count(line, "pat_missed") * contract.points("pat_missed")
    return total


#: Defensive/return event keys that are counted per occurrence.  Values come
#: from the contract at the D/ST position, which is where this league's
#: amplified return-touchdown numbers live.
DST_EVENT_KEYS: tuple[str, ...] = (
    "defensive_sack",
    "defensive_interception",
    "defensive_fumble_recovery",
    "defensive_safety",
    "blocked_kick",
    "interception_return_td",
    "fumble_return_td",
    "kickoff_return_td",
    "punt_return_td",
    "blocked_kick_return_td",
    "two_point_return",
    "one_point_safety",
)


def score_dst(line: Mapping[str, Any], contract: LeagueContract) -> float:
    """Score a team defense / special teams stat line.

    ``points_allowed`` and ``yards_allowed`` are required: they are the two
    largest terms in a D/ST score, and defaulting a missing one to zero would
    hand out the best tier in the league for absent data.
    """
    for required in ("points_allowed", "yards_allowed"):
        if line.get(required) is None:
            raise StatLineError(
                f"{required} is required to score a defense; refusing to assume 0, "
                "which would award the top tier for missing data"
            )

    total = 0.0
    for key in DST_EVENT_KEYS:
        count = _count(line, key)
        if count:
            total += count * contract.points(key, DST_POSITION_ID)

    pa_key = points_allowed_key(float(line["points_allowed"]))
    if pa_key is not None:
        total += contract.dst_points(pa_key)
    ya_key = yards_allowed_key(float(line["yards_allowed"]))
    if ya_key is not None:
        total += contract.dst_points(ya_key)
    return total
