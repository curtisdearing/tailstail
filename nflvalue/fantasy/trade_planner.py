"""Weekly rotate/trade planner: who to trade, for whom, and when.

Takes the league's rosters (ESPN API or a plain JSON file), a rest-of-season
outlook (from ``draft.simulate_season`` restricted to remaining weeks), and
proposes packages that pass a two-sided realism gate:

* **my gate** — the package must improve MY optimal lineup's mean weekly
  points over the remaining fantasy season (``season.lineup_points`` on the
  simulated sample matrix, never sum-of-projections);
* **their gate** — the same package must not make the counterparty's optimal
  lineup worse (they won't accept a strictly bad trade), evaluated against
  THEIR roster context (positional surplus/deficit is what makes trades
  positive-sum).

Timing signals ride along with each proposal:

* ``bye_relief`` — the package reduces my starters lost to byes in the next
  three weeks;
* ``playoff_tilt`` — points delta concentrated in the fantasy playoff weeks
  (default 15-17);
* ``sell_high`` / ``buy_low`` — realized recent points minus model
  expectation, flagged when the market likely over/under-prices a player.

ESPN access: for private leagues supply ``espn_s2``/``swid`` cookies; public
leagues need only the league id.  Everything degrades to a rosters JSON of
``{"teams": [{"name": ..., "players": ["player name", ...]}, ...]}``.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import LineupRules
from .draft import normalize_name
from .season import SeasonSimulation, lineup_points

FANTASY_PLAYOFF_WEEKS: tuple[int, ...] = (15, 16, 17)


@dataclass
class LeagueRosters:
    """Normalized league rosters keyed by fantasy-team name."""

    teams: dict[str, list[str]]  # team name -> list of player names
    my_team: str

    def opponents(self) -> list[str]:
        return [name for name in self.teams if name != self.my_team]


# ---------------------------------------------------------------------------
# Roster ingestion
# ---------------------------------------------------------------------------

def load_rosters_json(path: str | Path, my_team: str) -> LeagueRosters:
    payload = json.loads(Path(path).read_text())
    teams = {team["name"]: list(team["players"]) for team in payload["teams"]}
    if my_team not in teams:
        raise ValueError(f"my_team {my_team!r} not in rosters file: {sorted(teams)}")
    return LeagueRosters(teams=teams, my_team=my_team)


def load_rosters_from_snapshot(snapshot) -> LeagueRosters:
    """Build rosters from a validated ESPN league snapshot.

    This is the honest path. The snapshot has already proved it is the right
    league, season and team, carries stable ESPN player ids, and knows whether
    a draft has happened -- so a caller cannot silently plan trades against a
    league that has not drafted, or against somebody else's league.

    Names are produced here only because ``LeagueRosters`` is a name-keyed
    structure the rest of the planner already speaks; the ids remain in
    ``snapshot.rosters`` and should be preferred by anything new.
    """

    if snapshot.roster_state != "populated":
        raise ValueError(
            f"league {snapshot.league.league_id} is {snapshot.roster_state} "
            f"(draft status {snapshot.draft.status}); there are no rosters to plan against. "
            "A pre-draft league has intentions, not teams.")
    id_to_name = {team.team_id: team.name for team in snapshot.teams}
    teams = {
        id_to_name[team_id]: [player.full_name for player in players]
        for team_id, players in snapshot.rosters.items()
    }
    return LeagueRosters(teams=teams, my_team=snapshot.my_team.name)


def load_rosters_espn(
    league_id: int,
    season: int,
    my_team: str,
    *,
    espn_s2: str | None = None,
    swid: str | None = None,
) -> LeagueRosters:
    """Pull roster player NAMES from ESPN. **This is not a full league sync.**

    A name-only loader, kept for the legacy ``espn_api`` path. It returns
    player names and nothing else -- no player ids, no lineup slots, no
    eligibility, no bench/IR split, no free agents, no schedule, no draft
    state, and no identity check beyond "a team with this name exists". It
    cannot tell a pre-draft league from a league whose rosters failed to load;
    both come back as empty name lists.

    For anything that matters, use :mod:`nflvalue.fantasy.espn_client` +
    :mod:`nflvalue.fantasy.espn_league` to build a validated snapshot and pass
    it to :func:`load_rosters_from_snapshot`, which fails closed on all of the
    above. Credentials there come from the environment; the parameters here are
    part of the legacy signature and should not be filled from a CLI argument.
    """

    try:
        from espn_api.football import League
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "pip install espn_api for the legacy name-only path, or build a snapshot with "
            "nflvalue.fantasy.espn_client and use load_rosters_from_snapshot"
        ) from exc
    league = League(league_id=league_id, year=season, espn_s2=espn_s2, swid=swid)
    teams = {
        team.team_name: [player.name for player in team.roster]
        for team in league.teams
    }
    if my_team not in teams:
        raise ValueError(f"my_team {my_team!r} not found; teams: {sorted(teams)}")
    return LeagueRosters(teams=teams, my_team=my_team)


def match_players(
    names: Sequence[str], board: pd.DataFrame
) -> tuple[list[str], list[str]]:
    """Map roster names to board player_ids (exact normalized match, then
    last-name + first-initial).  Returns (player_ids, unmatched_names)."""

    norm = normalize_name

    index = {norm(n): pid for n, pid in zip(board["player_name"], board["player_id"])}
    loose: dict[str, str] = {}
    for name, pid in zip(board["player_name"], board["player_id"]):
        parts = norm(name).split()
        if len(parts) >= 2:
            loose.setdefault(f"{parts[0][0]} {parts[-1]}", pid)
    ids, unmatched = [], []
    for name in names:
        key = norm(name)
        if key in index:
            ids.append(index[key])
            continue
        parts = key.split()
        loose_key = f"{parts[0][0]} {parts[-1]}" if len(parts) >= 2 else key
        if loose_key in loose:
            ids.append(loose[loose_key])
        else:
            unmatched.append(name)
    return ids, unmatched


# ---------------------------------------------------------------------------
# Timing signals
# ---------------------------------------------------------------------------

def bye_pressure(
    player_ids: Sequence[str],
    board: pd.DataFrame,
    byes: Mapping[str, Sequence[int]],
    upcoming_weeks: Sequence[int],
) -> int:
    """How many of these players sit out during ``upcoming_weeks``."""

    teams = board.set_index("player_id").loc[list(player_ids), "team"]
    return int(sum(
        any(week in set(byes.get(str(team), ())) for week in upcoming_weeks)
        for team in teams
    ))


def market_temperature(
    board: pd.DataFrame, recent_actual: Mapping[str, float]
) -> pd.Series:
    """Recent realized points-per-game minus model per-game expectation.
    Positive = hot (sell-high candidate); negative = cold (buy-low)."""

    mu = board.set_index("player_id")["mu_pergame"]
    actual = pd.Series(recent_actual, dtype=float)
    return (actual - mu).dropna()


# ---------------------------------------------------------------------------
# Trade search
# ---------------------------------------------------------------------------

class _FastLineup:
    """Vectorized optimal-lineup mean for single-FLEX rules.

    With exactly one FLEX slot the greedy optimum decomposes exactly: sum the
    top ``k_p`` per base position, then add the best (k_p+1)-th-ranked player
    across flex-eligible positions.  Verified against ``season.lineup_points``
    in tests/test_draft_and_trades.py; falls back to the reference loop for
    multi-FLEX rules.
    """

    def __init__(self, season: SeasonSimulation, rules: LineupRules) -> None:
        self.season = season
        self.rules = rules
        self.points = {c: season.points[c].to_numpy() for c in season.points.columns}
        self.position = dict(zip(
            season.player_meta["player_id"], season.player_meta["position"]
        ))
        self.n = len(season.points)
        self.fast = int(rules.starters.get("FLEX", 0)) <= 1
        self._cache: dict[frozenset, float] = {}

    def mean(self, roster: Sequence[str]) -> float:
        usable = frozenset(pid for pid in roster if pid in self.points)
        if not usable:
            return 0.0
        cached = self._cache.get(usable)
        if cached is not None:
            return cached
        if not self.fast:
            value = float(lineup_points(self.season, sorted(usable), self.rules).mean())
            self._cache[usable] = value
            return value
        total = np.zeros(self.n)
        flex_pool: list[np.ndarray] = []
        flex_positions = set(self.rules.flex_positions)
        flex_count = int(self.rules.starters.get("FLEX", 0))
        for position, count in self.rules.starters.items():
            if position == "FLEX":
                continue
            candidates = [pid for pid in usable if self.position.get(pid) == position]
            if not candidates:
                continue
            matrix = np.column_stack([self.points[pid] for pid in candidates])
            matrix.sort(axis=1)  # ascending
            k = min(int(count), matrix.shape[1])
            if k:
                total += matrix[:, -k:].sum(axis=1)
            if position in flex_positions and matrix.shape[1] > int(count):
                flex_pool.append(matrix[:, -(int(count) + 1)])  # (k+1)-th best
        # Flex-eligible positions with no dedicated starter slot still feed FLEX.
        for position in flex_positions - set(self.rules.starters):
            candidates = [pid for pid in usable if self.position.get(pid) == position]
            if candidates:
                matrix = np.column_stack([self.points[pid] for pid in candidates])
                flex_pool.append(matrix.max(axis=1))
        if flex_count and flex_pool:
            total += np.max(np.column_stack(flex_pool), axis=1)
        value = float(total.mean())
        self._cache[usable] = value
        return value


def _lineup_mean(
    season: SeasonSimulation, roster: Sequence[str], rules: LineupRules
) -> float:
    usable = [pid for pid in roster if pid in set(season.points.columns)]
    if not usable:
        return 0.0
    return float(lineup_points(season, usable, rules).mean())


def propose_trades(
    season: SeasonSimulation,
    rosters: LeagueRosters,
    board: pd.DataFrame,
    *,
    rules: LineupRules | None = None,
    byes: Mapping[str, Sequence[int]] | None = None,
    upcoming_weeks: Sequence[int] = (),
    playoff_season: SeasonSimulation | None = None,
    max_package: int = 2,
    top_candidates: int = 9,
    min_my_gain: float = 0.5,
    their_tolerance: float = 0.0,
    max_results: int = 12,
) -> pd.DataFrame:
    """Scan all opponents for packages passing the two-sided gate.

    ``max_package=2`` considers 1-for-1, 2-for-1, 1-for-2 and 2-for-2 swaps
    among each side's ``top_candidates`` by rest-of-season mean — bounded on
    purpose; a 6-team league has enormous rosters and the tails of the
    combinatorics are never where accepted trades live.
    """

    lineup = rules or LineupRules()
    fast = _FastLineup(season, lineup)
    fast_playoff = _FastLineup(playoff_season, lineup) if playoff_season is not None else None
    my_ids, _ = match_players(rosters.teams[rosters.my_team], board)
    my_base = fast.mean(my_ids)
    board_indexed = board.set_index("player_id")

    def top(ids: Sequence[str]) -> list[str]:
        known = [pid for pid in ids if pid in board_indexed.index]
        return sorted(
            known, key=lambda pid: float(board_indexed.loc[pid, "season_mean"]),
            reverse=True,
        )[:top_candidates]

    proposals = []
    playoff_base = fast_playoff.mean(my_ids) if fast_playoff is not None else 0.0
    for opponent in rosters.opponents():
        their_ids, _ = match_players(rosters.teams[opponent], board)
        their_base = fast.mean(their_ids)
        mine, theirs = top(my_ids), top(their_ids)
        for k_out in range(1, max_package + 1):
            for k_in in range(1, max_package + 1):
                for send in itertools.combinations(mine, k_out):
                    for receive in itertools.combinations(theirs, k_in):
                        my_after = [p for p in my_ids if p not in send] + list(receive)
                        their_after = [p for p in their_ids if p not in receive] + list(send)
                        my_gain = fast.mean(my_after) - my_base
                        if my_gain < min_my_gain:
                            continue
                        their_gain = fast.mean(their_after) - their_base
                        if their_gain < -abs(their_tolerance):
                            continue
                        row = {
                            "opponent": opponent,
                            "send": [board_indexed.loc[p, "player_name"] for p in send],
                            "receive": [board_indexed.loc[p, "player_name"] for p in receive],
                            "my_gain_per_sim": round(my_gain, 2),
                            "their_gain_per_sim": round(their_gain, 2),
                        }
                        if byes and upcoming_weeks:
                            before = bye_pressure(send, board, byes, upcoming_weeks)
                            after = bye_pressure(receive, board, byes, upcoming_weeks)
                            row["bye_relief"] = before - after
                        if fast_playoff is not None:
                            playoff_after = fast_playoff.mean(my_after)
                            row["playoff_tilt"] = round(playoff_after - playoff_base, 2)
                        proposals.append(row)
    result = pd.DataFrame(proposals)
    if result.empty:
        return result
    result["score"] = result["my_gain_per_sim"] + 0.5 * result.get(
        "playoff_tilt", pd.Series(0.0, index=result.index)
    ).fillna(0.0)
    return (
        result.sort_values("score", ascending=False)
        .head(max_results)
        .reset_index(drop=True)
    )
