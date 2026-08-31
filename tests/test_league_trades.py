"""Contract tests for ID-based, recommendation-only trade analysis.

The old scan matched roster *names* against the board, reasoned about "Team1"
and "Team4", and assumed the fantasy playoffs were weeks 15-17. Each of those
is a way to be confidently wrong about a real league, and each has a test here.

What this file pins down:

* **Identity is by id, and a doubtful match is not a match.** A name that maps
  to two board rows, or to a board row at a different position, is recorded as
  ambiguous and excluded. Silently joining the wrong player prices him into
  every package downstream, where nobody would ever see it.
* **Rules come from the league.** Starting slots, roster cap and the playoff
  periods are read from the snapshot. A league whose playoffs run in matchup
  periods 15-16 gets 15-16; a two-week final gets both of its weeks.
* **K and D/ST are shadow.** The board has no kicker distribution, so a kicker
  cannot be valued -- only checked for legality. Promoting one without a real
  projection would value it at zero and make every kicker trade look free, so
  asking for that raises instead.
* **Legality is checked after the swap**, including the case counting cannot
  catch: enough bodies, wrong positions.
* **Nothing implies acceptance.** Every package carries a two-sided
  distribution and an explicit statement that plausibility is not consent.
* **No trade is an answer.** When nothing clears the gate the scan says so and
  explains what it rejected, rather than returning an empty list that reads
  like a bug.

Every test here is offline and mutates nothing.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue.fantasy import espn_league, league_trades
from nflvalue.fantasy.season import SeasonSimulation, lineup_points

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "espn"
BOARD_CSV = ROOT / "data" / "draft_board_2026.csv"
BYES_JSON = ROOT / "data" / "byes_2026.json"

LEAGUE_ID = 1111111111
SEASON = 2026
TEAM_ID = 1
TEAM_NAME = "Team One"

PLACEHOLDER_TOKENS = ("Team1", "Team2", "Team3", "Team4", "Team5",
                      "Team6", "Team7", "Team8", "Player1", "Opponent1")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _scan(*args, **kwargs):
    """Every test is an on-demand caller: a person asked for this scan, now.

    `scan_trades` requires the acknowledgement because the weekly card must not
    call it on a schedule and end up trading against whatever board is on disk.
    """
    kwargs.setdefault("on_demand", True)
    return league_trades.scan_trades(*args, **kwargs)


def _expected():
    return espn_league.ExpectedIdentity(
        league_id=LEAGUE_ID, season=SEASON, team_id=TEAM_ID,
        team_name=TEAM_NAME, team_count=8)


def _snapshot_dict(raw, *, retrieved_at="2026-08-29T12:00:00Z"):
    views = dict.fromkeys(("mSettings", "mTeam", "mRoster", "mMatchup",
                           "mDraftDetail", "mStandings", "mTransactions2"), raw)
    snap = espn_league.normalize_league(
        views, expected=_expected(), retrieved_at=retrieved_at,
        source_urls=["https://lm-api-reads.fantasy.espn.com/<redacted>"])
    return espn_league.snapshot_to_dict(snap)


@pytest.fixture(scope="module")
def raw_league():
    return json.loads((FIXTURES / "league_trade_scan_2026.json").read_text())


@pytest.fixture
def snapshot(raw_league):
    return _snapshot_dict(deepcopy(raw_league))


@pytest.fixture(scope="module")
def board():
    assert BOARD_CSV.exists(), "data/draft_board_2026.csv is tracked and required here"
    return pd.read_csv(BOARD_CSV)


@pytest.fixture(scope="module")
def byes():
    return json.loads(BYES_JSON.read_text())


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(6102026)


def _season_for(snapshot_dict, board_df, *, simulations=600, seed=6102026, scale=1.0):
    """A deterministic season sample matrix over exactly the rostered players.

    Small and synthetic on purpose: these tests are about identity, legality
    and the shape of the deltas, not about the projection model.
    """
    identity = league_trades.map_identities(snapshot_dict, board_df)
    board_ids = sorted(set(identity.espn_to_board.values()))
    rows = board_df.set_index("player_id").loc[board_ids]
    generator = np.random.default_rng(seed)
    points = pd.DataFrame(
        {pid: generator.normal(float(rows.loc[pid, "season_mean"]) * scale,
                               max(float(rows.loc[pid, "season_mean"]) * 0.25, 1.0),
                               simulations)
         for pid in board_ids},
        index=np.arange(simulations))
    meta = rows.reset_index()[["player_id", "player_name", "position", "team"]]
    summaries = rows.reset_index()[
        ["player_id", "player_name", "position", "team", "season_mean"]
    ].rename(columns={"season_mean": "mean"})
    return SeasonSimulation(summaries=summaries, points=points, player_meta=meta,
                            metadata={"source": "test"})


@pytest.fixture
def season(snapshot, board):
    return _season_for(snapshot, board)


# =========================================================================== #
# 1. Identity mapping
# =========================================================================== #
def test_rostered_players_map_to_board_ids_by_espn_id(snapshot, board):
    identity = league_trades.map_identities(snapshot, board)
    assert len(identity.espn_to_board) >= 100
    assert all(isinstance(espn_id, int) for espn_id in identity.espn_to_board)
    # Board ids are nflverse GSIS ids, plus a handful of `adp:<name>` rows for
    # rookies who exist only in the ADP feed. Both are board ids; neither is a name.
    known = set(board["player_id"].astype(str))
    assert set(identity.espn_to_board.values()) <= known
    # A board row may back exactly one ESPN player.
    assert len(set(identity.espn_to_board.values())) == len(identity.espn_to_board)


def test_identity_mapping_is_deterministic_and_order_independent(snapshot, board):
    first = league_trades.map_identities(snapshot, board)
    shuffled = board.sample(frac=1.0, random_state=17).reset_index(drop=True)
    second = league_trades.map_identities(snapshot, shuffled)
    assert first.espn_to_board == second.espn_to_board


def test_an_off_board_player_is_reported_unmatched_not_dropped(snapshot, board):
    identity = league_trades.map_identities(snapshot, board)
    names = {row["name"] for row in identity.unmatched}
    assert "Dontae Whitfield" in names, "a rostered player with no projection must be visible"
    row = next(row for row in identity.unmatched if row["name"] == "Dontae Whitfield")
    assert row["espn_id"] and row["team_id"] == TEAM_ID and row["reason"]


def test_a_name_matching_two_board_rows_is_ambiguous_not_guessed(snapshot, board):
    identity_before = league_trades.map_identities(snapshot, board)
    target = next(iter(identity_before.espn_to_board))
    name = next(player["name"] for player in league_trades._roster_players(snapshot)
                if player["espn_id"] == target)
    position = next(player["position"] for player in league_trades._roster_players(snapshot)
                    if player["espn_id"] == target)

    twin = board[board["player_name"] == name].iloc[0].copy()
    twin["player_id"] = "00-9999999"
    duplicated = pd.concat([board, pd.DataFrame([twin])], ignore_index=True)

    identity = league_trades.map_identities(snapshot, duplicated)
    assert target not in identity.espn_to_board
    row = next(row for row in identity.ambiguous if row["espn_id"] == target)
    assert row["position"] == position and "share this name key" in row["reason"]


def test_position_must_agree_for_a_match(snapshot, board):
    """A WR who happens to share a name with an RB is not that RB."""
    mutated = board.copy()
    identity_before = league_trades.map_identities(snapshot, board)
    espn_id, board_id = next(iter(identity_before.espn_to_board.items()))
    mutated.loc[mutated["player_id"] == board_id, "position"] = "QB" \
        if mutated.loc[mutated["player_id"] == board_id, "position"].iloc[0] != "QB" else "TE"
    identity = league_trades.map_identities(snapshot, mutated)
    assert espn_id not in identity.espn_to_board


def test_kickers_and_defenses_are_shadow_not_unmatched(snapshot, board):
    identity = league_trades.map_identities(snapshot, board)
    shadow_positions = {row["position"] for row in identity.shadow}
    assert shadow_positions == {"K", "D/ST"}
    assert len(identity.shadow) == 16, "one kicker and one defense on each of eight teams"
    assert not any(row["position"] in ("K", "D/ST") for row in identity.unmatched), (
        "a kicker is not an identity failure; it is a position the board does not model")
    assert all(row["reason"] for row in identity.shadow)


# =========================================================================== #
# 2. Rules read from the live league
# =========================================================================== #
def test_lineup_rules_come_from_the_snapshot(snapshot):
    rules = league_trades.lineup_rules_from_snapshot(snapshot)
    assert rules.modeled_slots == {"FLEX": 1, "QB": 1, "RB": 2, "TE": 1, "WR": 2}
    assert rules.shadow_slots == {"D/ST": 1, "K": 1}
    assert rules.roster_size == 16
    assert rules.bench_slots == 7 and rules.ir_slots == 1
    assert "K" not in rules.lineup.starters and "D/ST" not in rules.lineup.starters


def test_playoff_weeks_come_from_the_league_not_a_hard_coded_15_17(snapshot):
    assert league_trades.playoff_scoring_periods(snapshot) == (15, 16)

    from nflvalue.fantasy import trade_planner
    assert trade_planner.FANTASY_PLAYOFF_WEEKS == (15, 16, 17)
    assert league_trades.playoff_scoring_periods(snapshot) != trade_planner.FANTASY_PLAYOFF_WEEKS


def test_a_two_week_final_resolves_to_both_of_its_weeks(raw_league):
    longer = deepcopy(raw_league)
    # 14 one-week regular-season periods, then two two-week rounds: weeks
    # 15-18. `matchupPeriodCount` counts periods, not weeks, so the old 7 here
    # only produced 15-18 by multiplying the round length in twice.
    longer["settings"]["scheduleSettings"]["playoffMatchupPeriodLength"] = 2
    longer["settings"]["scheduleSettings"]["matchupPeriodCount"] = 14
    snapshot = _snapshot_dict(longer)
    assert league_trades.playoff_scoring_periods(snapshot) == (15, 16, 17, 18)


def test_a_league_with_different_playoff_settings_gets_different_weeks(raw_league):
    early = deepcopy(raw_league)
    early["settings"]["scheduleSettings"]["matchupPeriodCount"] = 13
    early["settings"]["scheduleSettings"]["playoffTeamCount"] = 8
    snapshot = _snapshot_dict(early)
    assert league_trades.playoff_scoring_periods(snapshot) == (14, 15, 16)


# =========================================================================== #
# 3. Lock state and pending transactions
# =========================================================================== #
def test_a_player_in_a_pending_transaction_is_locked(snapshot):
    locked = league_trades.locked_players(snapshot)
    assert locked, "the fixture carries a pending waiver"
    reason = next(iter(locked.values()))
    assert "pending" in reason.lower()


def test_locked_players_never_appear_in_a_package(snapshot, board, season, byes):
    locked = league_trades.locked_players(snapshot)
    scan = _scan(snapshot, board, season, byes=byes,
                                     min_my_gain=0.0, min_prob_not_worse=0.0)
    for package in scan.packages:
        moved = {p["espn_player_id"] for p in package.mine.sends + package.mine.receives}
        assert not (moved & set(locked)), "a locked player was packaged"


def test_a_caller_supplied_lock_is_honoured(snapshot, board, season):
    identity = league_trades.map_identities(snapshot, board)
    mine = [p for p in league_trades._roster_players(snapshot) if p["team_id"] == TEAM_ID]
    target = next(p["espn_id"] for p in mine if p["espn_id"] in identity.espn_to_board)
    locked = league_trades.locked_players(
        snapshot, extra={target: "kickoff has passed for this player's NFL game"})
    assert locked[target].startswith("kickoff")

    scan = _scan(snapshot, board, season, min_my_gain=0.0,
                                     min_prob_not_worse=0.0, extra_locked={target: "locked by kickoff"})
    for package in scan.packages:
        assert target not in {p["espn_player_id"] for p in package.mine.sends}


def test_ir_players_are_not_tradeable_by_this_tool(snapshot, board, season):
    ir_ids = {player["espn_id"] for player in league_trades._roster_players(snapshot)
              if player["on_ir"]}
    assert ir_ids, "the fixture stashes a player on IR"
    scan = _scan(snapshot, board, season, min_my_gain=0.0, min_prob_not_worse=0.0)
    for package in scan.packages:
        moved = {p["espn_player_id"] for p in package.mine.sends + package.mine.receives}
        assert not (moved & ir_ids)


# =========================================================================== #
# 4. Legality after the package
# =========================================================================== #
def _player(position, eligible, *, on_ir=False, espn_id=1):
    return {"espn_id": espn_id, "name": f"p{espn_id}", "position": position,
            "eligible_slots": tuple(eligible), "on_ir": on_ir, "pro_team": "KC",
            "lineup_slot": "BE"}


def _rules(snapshot):
    return league_trades.lineup_rules_from_snapshot(snapshot)


def test_a_package_over_the_roster_cap_is_illegal(snapshot):
    rules = _rules(snapshot)
    roster = [_player("WR", ["WR", "FLEX", "BE"], espn_id=i) for i in range(rules.roster_size + 1)]
    result = league_trades.check_legality(roster, rules, label="test")
    assert not result.legal
    assert any("exceeds" in violation for violation in result.violations)


def test_enough_bodies_at_the_wrong_positions_is_still_illegal(snapshot):
    """The case counting cannot catch: 16 players, zero of them a quarterback."""
    rules = _rules(snapshot)
    roster = ([_player("RB", ["RB", "FLEX", "BE"], espn_id=i) for i in range(14)]
              + [_player("K", ["K", "BE"], espn_id=90),
                 _player("D/ST", ["D/ST", "BE"], espn_id=91)])
    result = league_trades.check_legality(roster, rules, label="test")
    assert not result.legal
    assert any("cannot fill every starting slot" in violation for violation in result.violations)


def test_a_legal_roster_passes(snapshot):
    rules = _rules(snapshot)
    roster = [
        _player("QB", ["QB", "BE"], espn_id=1),
        _player("RB", ["RB", "FLEX", "BE"], espn_id=2),
        _player("RB", ["RB", "FLEX", "BE"], espn_id=3),
        _player("WR", ["WR", "FLEX", "BE"], espn_id=4),
        _player("WR", ["WR", "FLEX", "BE"], espn_id=5),
        _player("TE", ["TE", "FLEX", "BE"], espn_id=6),
        _player("WR", ["WR", "FLEX", "BE"], espn_id=7),
        _player("K", ["K", "BE"], espn_id=8),
        _player("D/ST", ["D/ST", "BE"], espn_id=9),
    ]
    assert league_trades.check_legality(roster, rules, label="test").legal


def test_losing_the_only_kicker_is_illegal_even_though_kickers_are_shadow(snapshot):
    """Shadow means 'not valued', not 'not required'."""
    rules = _rules(snapshot)
    roster = [
        _player("QB", ["QB", "BE"], espn_id=1),
        _player("RB", ["RB", "FLEX", "BE"], espn_id=2),
        _player("RB", ["RB", "FLEX", "BE"], espn_id=3),
        _player("WR", ["WR", "FLEX", "BE"], espn_id=4),
        _player("WR", ["WR", "FLEX", "BE"], espn_id=5),
        _player("TE", ["TE", "FLEX", "BE"], espn_id=6),
        _player("WR", ["WR", "FLEX", "BE"], espn_id=7),
        _player("D/ST", ["D/ST", "BE"], espn_id=9),
    ]
    result = league_trades.check_legality(roster, rules, label="test")
    assert not result.legal


def test_ir_players_do_not_count_against_the_roster_cap(snapshot):
    rules = _rules(snapshot)
    roster = ([_player("QB", ["QB", "BE"], espn_id=1)]
              + [_player("RB", ["RB", "FLEX", "BE"], espn_id=i) for i in range(2, 5)]
              + [_player("WR", ["WR", "FLEX", "BE"], espn_id=i) for i in range(5, 9)]
              + [_player("TE", ["TE", "FLEX", "BE"], espn_id=9),
                 _player("K", ["K", "BE"], espn_id=10),
                 _player("D/ST", ["D/ST", "BE"], espn_id=11)]
              + [_player("RB", ["RB", "FLEX", "BE"], espn_id=99, on_ir=True)])
    result = league_trades.check_legality(roster, rules, label="test")
    assert result.legal and result.roster_size_after == 11


# =========================================================================== #
# 5. Roster-space effects on 2-for-1 shapes
# =========================================================================== #
def test_receiving_two_for_one_costs_a_roster_spot(snapshot, board, season, byes):
    scan = _scan(snapshot, board, season, byes=byes,
                                     min_my_gain=0.0, min_prob_not_worse=0.0, max_package=2)
    for package in scan.packages:
        net = len(package.mine.receives) - len(package.mine.sends)
        assert package.mine.roster_after == package.mine.roster_before + net
        assert package.mine.roster_after <= package.mine.legality.roster_cap
        assert package.theirs.roster_after <= package.theirs.legality.roster_cap


def test_a_full_roster_cannot_take_two_for_one(snapshot):
    """Curtis is at the 16-man cap: receiving two for one would be 17."""
    rules = _rules(snapshot)
    mine = [player for player in league_trades._roster_players(snapshot)
            if player["team_id"] == TEAM_ID and not player["on_ir"]]
    assert len(mine) == rules.roster_size
    after = mine[1:] + [_player("WR", ["WR", "FLEX", "BE"], espn_id=8001),
                        _player("WR", ["WR", "FLEX", "BE"], espn_id=8002)]
    assert not league_trades.check_legality(after, rules, label="mine").legal


def test_a_roster_with_an_open_spot_can_take_two_for_one(snapshot):
    rules = _rules(snapshot)
    theirs = [player for player in league_trades._roster_players(snapshot)
              if player["team_id"] == 3 and not player["on_ir"]]
    assert len(theirs) == rules.roster_size - 1, "Kupp of Joe sits one under the cap"
    after = theirs[1:] + [_player("WR", ["WR", "FLEX", "BE"], espn_id=8001),
                          _player("WR", ["WR", "FLEX", "BE"], espn_id=8002)]
    assert league_trades.check_legality(after, rules, label="theirs").legal


# =========================================================================== #
# 6. Delta distributions
# =========================================================================== #
def test_lineup_evaluator_matches_the_reference_optimizer(snapshot, board, season):
    rules = league_trades.lineup_rules_from_snapshot(snapshot).lineup
    evaluator = league_trades.LineupEvaluator(season, rules)
    identity = league_trades.map_identities(snapshot, board)
    roster = identity.matched(
        player["espn_id"] for player in league_trades._roster_players(snapshot)
        if player["team_id"] == TEAM_ID)
    reference = np.asarray(lineup_points(season, roster, rules), dtype=float)
    assert np.allclose(evaluator.vector(roster), reference, atol=1e-9)


def test_packages_report_a_distribution_not_just_a_mean(snapshot, board, season, byes):
    scan = _scan(snapshot, board, season, byes=byes,
                                     min_my_gain=0.0, min_prob_not_worse=0.0)
    assert scan.packages
    delta = scan.packages[0].mine.delta
    assert delta.p05 <= delta.p25 <= delta.p50 <= delta.p75 <= delta.p95
    assert 0.0 <= delta.prob_gain <= 1.0
    # Better, unchanged and worse are the whole sample space, and "unchanged"
    # is usually the biggest of the three for a single roster swap.
    assert abs(delta.prob_gain + delta.prob_loss + delta.prob_no_change - 1.0) < 1e-6
    assert delta.prob_not_worse == pytest.approx(1.0 - delta.prob_loss)
    assert delta.sd > 0 and delta.simulations == len(season.points)
    assert scan.packages[0].theirs.delta.simulations == delta.simulations


def test_the_gate_is_two_sided(snapshot, board, season):
    scan = _scan(snapshot, board, season,
                                     min_my_gain=0.5, min_prob_not_worse=0.55, their_tolerance=0.0)
    for package in scan.packages:
        assert package.mine.delta.mean >= 0.5
        # The gate is downside, not upside: a swap that changes nothing in most
        # simulations can still be clearly worth making.
        assert package.mine.delta.prob_not_worse >= 0.55
        assert package.theirs.delta.mean >= 0.0, "counterparty protection was not applied"


def test_an_unreachable_gate_produces_an_explicit_hold(snapshot, board, season):
    scan = _scan(snapshot, board, season, min_my_gain=10_000.0)
    assert scan.state == "hold"
    assert scan.packages == ()
    assert scan.hold_reason and "Hold." in scan.hold_reason
    assert str(scan.gate["considered"]) in scan.hold_reason
    assert sum(scan.gate["rejected"].values()) > 0


# =========================================================================== #
# 7. Output shape and honesty
# =========================================================================== #
def test_every_package_shows_exact_sends_and_receives(snapshot, board, season, byes):
    scan = _scan(snapshot, board, season, byes=byes,
                                     min_my_gain=0.0, min_prob_not_worse=0.0)
    assert scan.packages
    for package in scan.packages:
        assert package.mine.sends and package.mine.receives
        assert package.mine.sends == package.theirs.receives
        assert package.mine.receives == package.theirs.sends
        for player in package.mine.sends + package.mine.receives:
            assert isinstance(player["espn_player_id"], int)
            assert player["name"] and player["position"] and player["board_player_id"]


def test_no_package_claims_the_other_manager_would_accept(snapshot, board, season, byes):
    scan = _scan(snapshot, board, season, byes=byes,
                                     min_my_gain=0.0, min_prob_not_worse=0.0)
    payload = json.dumps(league_trades.scan_to_dict(scan)).lower()
    for claim in ("would accept", "will accept", "they'll take", "guaranteed",
                  "sure thing", "easy yes"):
        assert claim not in payload
    assert "plausibility is not consent" in payload
    for package in scan.packages:
        assert package.acceptance_claim.startswith("none")


def test_every_package_carries_rationale_uncertainty_and_plausibility(snapshot, board,
                                                                     season, byes):
    scan = _scan(snapshot, board, season, byes=byes,
                                     min_my_gain=0.0, min_prob_not_worse=0.0)
    for package in scan.packages:
        assert package.rationale and package.plausibility and package.uncertainty
        assert any("percentile" in line for line in package.uncertainty)
        assert any("model variance only" in line for line in package.uncertainty)
        assert any("shadow" in line for line in package.uncertainty)


def test_output_contains_no_placeholder_team_or_player_artifacts(snapshot, board, season, byes):
    scan = _scan(snapshot, board, season, byes=byes,
                                     min_my_gain=0.0, min_prob_not_worse=0.0)
    payload = json.dumps(league_trades.scan_to_dict(scan))
    for token in PLACEHOLDER_TOKENS:
        assert token not in payload, f"placeholder {token!r} reached the output"
    assert scan.generated_for["team_name"] == TEAM_NAME
    real_names = {team["name"] for team in
                  json.loads((FIXTURES / "league_trade_scan_2026.json").read_text())["teams"]}
    for package in scan.packages:
        assert package.theirs.team_name in real_names
        assert not package.theirs.team_name.startswith("Team ")


def test_the_scan_reports_unmatched_identities_at_the_top_level(snapshot, board, season):
    scan = _scan(snapshot, board, season, min_my_gain=0.0, min_prob_not_worse=0.0)
    assert scan.identity["unmatched"], "unmatched players must be visible in the artifact"
    assert any(row["name"] == "Dontae Whitfield" for row in scan.identity["unmatched"])
    assert scan.identity["shadow"]
    assert any("no board projection" in warning for warning in scan.warnings)


def test_the_scan_carries_the_snapshot_hashes_and_provenance(snapshot, board, season):
    scan = _scan(snapshot, board, season, min_my_gain=0.0, min_prob_not_worse=0.0)
    assert scan.league["league_id"] == LEAGUE_ID and scan.league["season"] == SEASON
    for name in ("league_hash", "scoring_hash", "roster_hash"):
        assert len(scan.league[name]) == 64
    assert scan.league["snapshot_retrieved_at"] == "2026-08-29T12:00:00Z"
    assert scan.schema_version == "trade-scan/2"


def test_scan_is_json_serializable(snapshot, board, season, byes):
    scan = _scan(snapshot, board, season, byes=byes,
                                     min_my_gain=0.0, min_prob_not_worse=0.0)
    payload = json.dumps(league_trades.scan_to_dict(scan), indent=1)
    assert json.loads(payload)["state"] in ("proposals", "hold")


# =========================================================================== #
# 8. Shadow promotion, pre-draft, and refusals
# =========================================================================== #
def test_promoting_shadow_positions_without_projections_is_refused(snapshot, board, season):
    with pytest.raises(league_trades.TradeScanError, match="shadow_projections"):
        _scan(snapshot, board, season, promote_shadow=True)


def test_a_pre_draft_league_cannot_be_scanned(board):
    raw = json.loads((FIXTURES / "league_predraft_2026.json").read_text())
    snapshot = _snapshot_dict(raw)
    season = SeasonSimulation(
        summaries=pd.DataFrame(columns=["player_id", "player_name", "position", "team", "mean"]),
        points=pd.DataFrame(index=range(10)),
        player_meta=pd.DataFrame(columns=["player_id", "player_name", "position", "team"]),
        metadata={})
    with pytest.raises(league_trades.TradeScanError, match=r"pre-draft|empty_pre_draft"):
        _scan(snapshot, board, season)


def test_an_unknown_snapshot_schema_is_refused(snapshot, board, season):
    drifted = dict(snapshot, schema_version="espn-league/99")
    with pytest.raises(league_trades.TradeScanError, match="schema"):
        _scan(drifted, board, season)


# =========================================================================== #
# 9. Bye and schedule context
# =========================================================================== #
def test_bye_context_uses_real_pro_team_abbreviations(snapshot, board, byes):
    identity = league_trades.map_identities(snapshot, board)
    players = league_trades._roster_players(snapshot)
    abbrevs = {player["pro_team"] for player in players}
    assert abbrevs <= set(byes) | {"UNK", "FA"}
    assert "UNK" not in abbrevs, "every rostered player resolved to a real NFL team"
    assert identity.espn_to_board


def test_packages_report_bye_pressure_on_both_sides(snapshot, board, season, byes):
    scan = _scan(snapshot, board, season, byes=byes,
                                     upcoming_weeks=(6, 7, 8),
                                     min_my_gain=0.0, min_prob_not_worse=0.0)
    assert scan.rules["upcoming_weeks"] == [6, 7, 8]
    for package in scan.packages:
        for side in (package.mine, package.theirs):
            assert side.starters_on_bye_before >= 0
            assert side.starters_on_bye_after >= 0


# =========================================================================== #
# 10. The scan mutates nothing
# =========================================================================== #
def test_the_module_has_no_write_path():
    source = (ROOT / "nflvalue" / "fantasy" / "league_trades.py").read_text()
    body = source.split('"""', 2)[-1]          # skip the module docstring
    for forbidden in ("urlopen", "requests.", "http", "POST(", "PUT(", "PATCH(",
                      "submitTrade", "acceptTrade", "proposeTrade", "os.environ"):
        assert forbidden not in body, f"{forbidden!r} suggests a mutation path"


def test_the_scan_does_not_mutate_its_inputs(snapshot, board, season, byes):
    before = json.dumps(snapshot, sort_keys=True)
    board_before = board.copy()
    _scan(snapshot, board, season, byes=byes,
                              min_my_gain=0.0, min_prob_not_worse=0.0)
    assert json.dumps(snapshot, sort_keys=True) == before
    pd.testing.assert_frame_equal(board, board_before)


def test_the_trade_scan_cli_takes_no_credentials():
    source = (ROOT / "scripts" / "trade_scan.py").read_text()
    for flag in ("--espn-s2", "--swid", "--espn_s2", "espn_s2=", "swid="):
        assert flag not in source, f"{flag!r} would put a session cookie in shell history"


# --------------------------------------------------------------------------- #
# An unidentifiable starter blocks the team, not just the footnotes
# --------------------------------------------------------------------------- #
def test_a_startable_unmatched_player_blocks_that_team_entirely(raw_league, board, season):
    """A warning at the top of the scan does not travel with the package.

    Every lineup total for that side is computed as though the player does not
    exist, so a package can look like a gain purely because the roster it is
    measured against is missing a starter.
    """
    broken = deepcopy(raw_league)
    for team in broken["teams"]:
        if team["id"] != 2:
            continue
        for entry in team["roster"]["entries"]:
            if entry.get("lineupSlotId") == 20:          # a bench skill player
                entry["playerPoolEntry"]["player"]["fullName"] = "Nobody Whoexists"
                break
    scan = _scan(_snapshot_dict(broken), board, season, min_my_gain=0.0,
                 min_prob_not_worse=0.0)
    assert "2" in scan.context["blocked_teams"]
    assert all(p.theirs.team_id != 2 for p in scan.packages), (
        "a team whose lineup cannot be valued must not appear in a package")
    assert any("excluded entirely" in warning for warning in scan.warnings)


def test_an_unmatched_player_on_ir_does_not_block_his_team(raw_league, board, season):
    """He cannot be seated, so failing to price him cannot move a lineup."""
    scan = _scan(_snapshot_dict(raw_league), board, season, min_my_gain=0.0,
                 min_prob_not_worse=0.0)
    assert scan.identity["unmatched"], "the fixture should still report him"
    assert "1" not in scan.context["blocked_teams"]


def test_a_scan_is_refused_unless_the_caller_asked_for_one(snapshot, board, season):
    """It is not part of the weekly card, and the default says so."""
    with pytest.raises(league_trades.TradeScanError, match="on demand"):
        league_trades.scan_trades(snapshot, board, season)
