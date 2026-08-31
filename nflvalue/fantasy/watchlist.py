"""League-specific draft-board and ESPN watchlist helpers.

The production draft board is serialized after season simulation.  These
stdlib-only helpers let a league adapter re-price that board for the live
league size and turn the result into a slot-aware watchlist without rerunning
or mutating the frozen QB/RB/WR/TE model.
"""

from __future__ import annotations

from copy import deepcopy
from math import ceil, erf, isfinite, sqrt
from statistics import median
from typing import Iterable, Mapping, Sequence

DEFAULT_STARTERS: Mapping[str, int] = {
    "QB": 1,
    "RB": 2,
    "WR": 2,
    "TE": 1,
    "FLEX": 1,
}
DEFAULT_FLEX_POSITIONS: tuple[str, ...] = ("RB", "WR", "TE")
DEFAULT_BENCH_MULTIPLIERS: Mapping[str, float] = {
    "QB": 1.2,
    "TE": 1.1,
    "RB": 1.6,
    "WR": 1.6,
}


def _positive_int(value: int, name: str) -> int:
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be at least 1")
    return number


def snake_pick_numbers(*, slot: int, league_teams: int, rounds: int) -> list[int]:
    """Return a 1-indexed snake-draft pick sequence for one slot."""

    teams = _positive_int(league_teams, "league_teams")
    total_rounds = _positive_int(rounds, "rounds")
    draft_slot = int(slot)
    if not 1 <= draft_slot <= teams:
        raise ValueError("slot must be within league size")

    picks: list[int] = []
    for round_number in range(1, total_rounds + 1):
        if round_number % 2:
            picks.append((round_number - 1) * teams + draft_slot)
        else:
            picks.append(round_number * teams - draft_slot + 1)
    return picks


def _tier_numbers(scores: Sequence[float], tier_gap_fraction: float) -> list[int]:
    if not scores:
        return []
    tiers = [1] * len(scores)
    if len(scores) == 1:
        return tiers

    gaps = [scores[index] - scores[index + 1] for index in range(len(scores) - 1)]
    tier = 1
    window = 8
    for index, gap in enumerate(gaps):
        if index >= 80:
            tiers[index + 1] = tier
            continue
        local = gaps[max(0, index - window) : index + window + 1]
        positive = [candidate for candidate in local if candidate > 0]
        local_scale = max(float(median(positive)) if positive else 0.0, 1e-9)
        if gap > max(2.0 * local_scale, float(tier_gap_fraction)):
            tier += 1
        tiers[index + 1] = tier
    return tiers


def reweight_board_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    league_teams: int,
    starters: Mapping[str, int] | None = None,
    flex_positions: Sequence[str] = DEFAULT_FLEX_POSITIONS,
    bench_multipliers: Mapping[str, float] | None = None,
    ceiling_weight: float = 0.55,
    tier_gap_fraction: float = 0.35,
) -> list[dict[str, object]]:
    """Recompute VOR, ranks, and tiers from serialized season summaries.

    This mirrors :func:`nflvalue.fantasy.draft.draft_board` but accepts plain
    dictionaries, so a live league integration does not need pandas, the model,
    or the season-sample matrix.  Only replacement-level pricing changes; the
    frozen season distributions remain untouched.
    """

    teams = _positive_int(league_teams, "league_teams")
    if not 0.0 <= float(ceiling_weight) <= 1.0:
        raise ValueError("ceiling_weight must be in [0, 1]")

    lineup = dict(DEFAULT_STARTERS if starters is None else starters)
    if any(int(value) < 0 for value in lineup.values()):
        raise ValueError("starter counts cannot be negative")
    flex_set = tuple(flex_positions)
    if not flex_set and int(lineup.get("FLEX", 0)):
        raise ValueError("flex_positions cannot be empty when FLEX is used")

    multipliers = dict(DEFAULT_BENCH_MULTIPLIERS)
    if bench_multipliers:
        multipliers.update({str(key): float(value) for key, value in bench_multipliers.items()})
    if any(value <= 0 for value in multipliers.values()):
        raise ValueError("bench multipliers must be positive")

    copied: list[dict[str, object]] = [deepcopy(dict(row)) for row in rows]
    if not copied:
        return []

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in copied:
        position = str(row.get("position", "")).strip()
        if not position:
            raise ValueError("every row needs a position")
        for field in ("season_mean", "season_p90"):
            value = float(row[field])
            if not isfinite(value):
                raise ValueError(f"{field} must be finite")
            row[field] = value
        row["position"] = position
        grouped.setdefault(position, []).append(row)

    flex_slots = int(lineup.get("FLEX", 0))
    output: list[dict[str, object]] = []
    for position, group in grouped.items():
        direct = int(lineup.get(position, 0))
        flex_share = (
            flex_slots / len(flex_set) if position in flex_set and flex_set else 0.0
        )
        bench_multiplier = float(multipliers.get(position, 1.5))
        replacement_rank = max(
            int(ceil(teams * (direct + flex_share) * bench_multiplier)), 1
        )
        ordered = sorted(group, key=lambda row: float(row["season_mean"]), reverse=True)
        replacement = ordered[min(replacement_rank - 1, len(ordered) - 1)]
        replacement_mean = float(replacement["season_mean"])
        replacement_p90 = float(replacement["season_p90"])
        for row in ordered:
            row["replacement_rank"] = replacement_rank
            row["vor_mean"] = float(row["season_mean"]) - replacement_mean
            row["vor_p90"] = float(row["season_p90"]) - replacement_p90
            row["draft_score"] = (
                (1.0 - float(ceiling_weight)) * float(row["vor_mean"])
                + float(ceiling_weight) * float(row["vor_p90"])
            )
            output.append(row)

    output.sort(key=lambda row: float(row["draft_score"]), reverse=True)
    tiers = _tier_numbers(
        [float(row["draft_score"]) for row in output], float(tier_gap_fraction)
    )
    position_counts: dict[str, int] = {}
    for overall_rank, (row, tier) in enumerate(zip(output, tiers), start=1):
        position = str(row["position"])
        position_counts[position] = position_counts.get(position, 0) + 1
        row["overall_rank"] = overall_rank
        row["tier"] = tier
        row["position_rank"] = position_counts[position]
        raw_adp = row.get("adp")
        try:
            adp = float(raw_adp) if isinstance(raw_adp, (int, float, str)) else float("nan")
        except ValueError:
            adp = float("nan")
        if isfinite(adp) and adp > 0:
            row["adp_round"] = int((adp - 1) // teams + 1)
            row["value_gap"] = adp - overall_rank
        else:
            row["adp_round"] = None
            row["value_gap"] = None
    return output


def _availability_probability(row: Mapping[str, object], pick_number: int) -> float:
    adp = row.get("adp")
    if adp is None:
        return 1.0
    try:
        adp_value = float(adp)
    except (TypeError, ValueError):
        return 1.0
    if not isfinite(adp_value):
        return 1.0
    try:
        spread = float(row.get("adp_sd", 6.0))
    except (TypeError, ValueError):
        spread = 6.0
    if not isfinite(spread):
        spread = 6.0
    spread = max(spread, 2.0)
    z = (adp_value - pick_number) / spread
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _turn_starts(picks: Sequence[int]) -> list[int]:
    """Collapse consecutive snake picks into one decision window."""

    starts: list[int] = []
    for pick in picks:
        if not starts or pick != starts[-1] + 1:
            starts.append(pick)
    return starts


def watchlist_targets(
    rows: Iterable[Mapping[str, object]],
    *,
    slot: int,
    league_teams: int,
    rounds: int,
    candidates_per_pick: int = 4,
    minimum_availability: float = 0.10,
    maximum_players: int | None = None,
) -> list[dict[str, object]]:
    """Build a unique, slot-aware list of candidates for each draft turn.

    A candidate must have at least ``minimum_availability`` probability of
    reaching the turn under the serialized ADP distribution.  Candidates are
    then ranked by the same progressive mean/P90 weighting used by the live
    draft assistant.  Consecutive snake picks are one turn so the same market
    window is not counted twice.
    """

    per_turn = _positive_int(candidates_per_pick, "candidates_per_pick")
    if not 0.0 <= float(minimum_availability) <= 1.0:
        raise ValueError("minimum_availability must be in [0, 1]")
    if maximum_players is not None:
        maximum_players = _positive_int(maximum_players, "maximum_players")

    board = [deepcopy(dict(row)) for row in rows]
    picks = snake_pick_numbers(slot=slot, league_teams=league_teams, rounds=rounds)
    turn_starts = _turn_starts(picks)
    selected: set[str] = set()
    targets: list[dict[str, object]] = []

    for pick in turn_starts:
        round_number = (pick - 1) // int(league_teams) + 1
        ceiling_weight = min(0.30 + 0.08 * (round_number - 1), 0.80)
        candidates: list[tuple[float, int, dict[str, object], float]] = []
        for row in board:
            name = str(row.get("player_name", "")).strip()
            if not name or name in selected:
                continue
            position = str(row.get("position", ""))
            if position not in {"QB", "RB", "WR", "TE"}:
                continue
            availability = _availability_probability(row, pick)
            if availability < float(minimum_availability):
                continue
            score = (
                (1.0 - ceiling_weight) * float(row["vor_mean"])
                + ceiling_weight * float(row["vor_p90"])
            )
            overall_rank = int(row.get("overall_rank", 10**9))
            candidates.append((score, -overall_rank, row, availability))

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for score, _rank_key, row, availability in candidates[:per_turn]:
            name = str(row["player_name"])
            selected.add(name)
            target = deepcopy(row)
            target["target_pick"] = pick
            target["target_round"] = round_number
            target["availability_probability"] = availability
            target["round_score"] = score
            targets.append(target)
            if maximum_players is not None and len(targets) >= maximum_players:
                return targets
    return targets
