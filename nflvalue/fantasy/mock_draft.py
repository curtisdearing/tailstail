"""Mock-draft simulator: our progressive-ceiling strategy vs ADP drafters.

Simulates full snake drafts in an N-team league where the 11 opponents pick
the best available player by *noisy* ADP (each opponent's private board is
``adp + Normal(0, adp_sd)``, redrawn per draft) under light positional
sanity caps, while our seat picks by the progressive-ceiling round score
(``round_ceiling_weight``) subject to the same caps and a
fill-your-starters-first constraint.

Rosters are then evaluated on the shared season Monte Carlo sample matrix:
for every simulation draw, each team fields its best starting lineup by
season totals and the draw's title goes to the top-scoring roster.  Scoring
lineups on season totals (rather than week-by-week) is a documented
approximation — it ignores start/sit variance, which affects every team
symmetrically.

Outputs per draft slot: expected starting-lineup points, title probability,
and most-frequent early-round picks — the "which seat should I hope for"
table.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import LineupRules
from .draft import round_ceiling_weight

#: Per-position roster caps applied to every team (ours and opponents).
POSITION_CAPS: dict[str, int] = {"QB": 2, "RB": 6, "WR": 7, "TE": 2}


@dataclass
class MockDraftResult:
    """Summary of repeated mock drafts from one seat."""

    slot: int
    expected_points: float
    title_probability: float
    round_targets: dict[int, list[tuple[str, float]]] = field(default_factory=dict)


def _positional_ok(position: str, counts: Counter, cap_scale: float = 1.0) -> bool:
    return counts[position] < int(POSITION_CAPS.get(position, 3) * cap_scale)


def _starters_needed(counts: Counter, rules: LineupRules, rounds_left: int) -> set[str]:
    """Positions we must still draft to be able to field a legal lineup."""

    deficits: dict[str, int] = {}
    for position, needed in rules.starters.items():
        if position == "FLEX":
            continue
        deficits[position] = max(needed - counts[position], 0)
    total_deficit = sum(deficits.values())
    if total_deficit >= rounds_left:
        return {p for p, d in deficits.items() if d > 0}
    return set()


def _our_pick(
    pool: pd.DataFrame,
    round_number: int,
    counts: Counter,
    rules: LineupRules,
    rounds_left: int,
) -> int:
    weight = round_ceiling_weight(round_number)
    score = (1 - weight) * pool["vor_mean"] + weight * pool["vor_p90"]
    eligible = pool["position"].map(lambda p: _positional_ok(p, counts))
    must_fill = _starters_needed(counts, rules, rounds_left)
    if must_fill:
        eligible &= pool["position"].isin(must_fill)
    candidates = score[eligible]
    if candidates.empty:  # caps exhausted the pool; take best available
        candidates = score
    return int(candidates.idxmax())


def _opponent_pick(
    pool: pd.DataFrame,
    noisy_board: pd.Series,
    counts: Counter,
    rules: LineupRules,
    rounds_left: int,
) -> int:
    """Best available by the opponent's noisy ADP board, but sane: respects
    positional caps and fills outstanding starter slots when running out of
    rounds (nobody finishes a real draft with zero TEs)."""

    must_fill = _starters_needed(counts, rules, rounds_left)
    order = noisy_board.loc[pool.index].sort_values()
    for idx in order.index:
        position = pool.at[idx, "position"]
        if must_fill and position not in must_fill:
            continue
        if _positional_ok(position, counts):
            return int(idx)
    return int(order.index[0])


def simulate_draft(
    board: pd.DataFrame,
    my_slot: int,
    *,
    league_teams: int = 12,
    rounds: int = 14,
    rules: LineupRules | None = None,
    rng: np.random.Generator,
) -> dict[int, list[int]]:
    """Run one snake draft; returns {team_slot: [board row indices]}."""

    rules = rules or LineupRules()
    pool = board.reset_index(drop=True).copy()
    # Undrafted-in-ADP players go to the back of opponents' boards.
    adp = pool["adp"].fillna(pool["adp"].max() + pool["overall_rank"])
    adp_sd = pool["adp_sd"].fillna(12.0).clip(lower=2.0)
    noisy = {
        slot: pd.Series(adp + rng.normal(0.0, adp_sd), index=pool.index)
        for slot in range(1, league_teams + 1)
        if slot != my_slot
    }
    rosters: dict[int, list[int]] = {slot: [] for slot in range(1, league_teams + 1)}
    counts: dict[int, Counter] = {slot: Counter() for slot in rosters}
    available = pd.Series(True, index=pool.index)

    for overall in range(1, league_teams * rounds + 1):
        round_number = (overall - 1) // league_teams + 1
        pick_in_round = (overall - 1) % league_teams + 1
        slot = (
            pick_in_round
            if round_number % 2 == 1
            else league_teams - pick_in_round + 1
        )
        open_pool = pool[available]
        if open_pool.empty:
            break
        if slot == my_slot:
            choice = _our_pick(
                open_pool, round_number, counts[slot], rules, rounds - round_number + 1
            )
        else:
            choice = _opponent_pick(
                open_pool, noisy[slot], counts[slot], rules, rounds - round_number + 1
            )
        rosters[slot].append(choice)
        counts[slot][pool.at[choice, "position"]] += 1
        available.at[choice] = False
    return rosters


def _lineup_totals(
    rosters: dict[int, list[int]],
    pool: pd.DataFrame,
    samples: np.ndarray,
    sample_cols: dict[str, int],
    rules: LineupRules,
) -> np.ndarray:
    """(teams,) x (sims,) matrix of best-lineup season totals per draw."""

    totals = np.zeros((len(rosters), samples.shape[0]))
    for row, (_slot, picks) in enumerate(sorted(rosters.items())):
        players = pool.loc[picks]
        cols = [sample_cols.get(pid, -1) for pid in players["player_id"]]
        points = np.column_stack([
            samples[:, c] if c >= 0 else np.zeros(samples.shape[0]) for c in cols
        ])
        positions = players["position"].to_numpy()
        for sim in range(samples.shape[0]):
            totals[row, sim] = _best_lineup(points[sim], positions, rules)
    return totals


def _best_lineup(points: np.ndarray, positions: np.ndarray, rules: LineupRules) -> float:
    used = np.zeros(len(points), dtype=bool)
    total = 0.0
    for position, needed in rules.starters.items():
        if position == "FLEX":
            continue
        mask = (positions == position) & ~used
        idx = np.argsort(points * mask - (~mask) * 1e9)[::-1][:needed]
        for i in idx:
            if mask[i]:
                used[i] = True
                total += points[i]
    flex_slots = rules.starters.get("FLEX", 0)
    mask = np.isin(positions, rules.flex_positions) & ~used
    idx = np.argsort(points * mask - (~mask) * 1e9)[::-1][:flex_slots]
    for i in idx:
        if mask[i]:
            total += points[i]
    return total


def evaluate_slot(
    board: pd.DataFrame,
    samples: pd.DataFrame,
    my_slot: int,
    *,
    league_teams: int = 12,
    rounds: int = 14,
    n_drafts: int = 100,
    sims_per_draft: int = 400,
    rules: LineupRules | None = None,
    random_seed: int = 7,
) -> MockDraftResult:
    """Repeated mock drafts from one seat, scored on the season MC samples."""

    rules = rules or LineupRules()
    rng = np.random.default_rng(random_seed + my_slot)
    pool = board.reset_index(drop=True)
    sample_cols = {pid: i for i, pid in enumerate(samples.columns)}
    matrix = samples.to_numpy()

    titles = 0
    trials = 0
    my_points: list[float] = []
    early_picks: dict[int, Counter] = {r: Counter() for r in range(1, 4)}

    for _ in range(n_drafts):
        rosters = simulate_draft(
            board, my_slot, league_teams=league_teams, rounds=rounds,
            rules=rules, rng=rng,
        )
        draw = rng.choice(matrix.shape[0], size=sims_per_draft, replace=False)
        totals = _lineup_totals(rosters, pool, matrix[draw], sample_cols, rules)
        my_row = sorted(rosters).index(my_slot)
        titles += int((totals.argmax(axis=0) == my_row).sum())
        trials += sims_per_draft
        my_points.append(float(totals[my_row].mean()))
        for round_number in early_picks:
            picks = rosters[my_slot]
            if len(picks) >= round_number:
                early_picks[round_number][pool.at[picks[round_number - 1], "player_name"]] += 1

    round_targets = {
        r: [(name, count / n_drafts) for name, count in counter.most_common(3)]
        for r, counter in early_picks.items()
    }
    return MockDraftResult(
        slot=my_slot,
        expected_points=float(np.mean(my_points)),
        title_probability=titles / trials,
        round_targets=round_targets,
    )
