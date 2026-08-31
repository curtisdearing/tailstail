"""One snapshot, one rules contract, one set of hashes — proved end to end.

This file exists because the tree had grown four parallel answers to the same
questions. Two modules each computed a `scoring_hash` and disagreed (one over
canonicalised categories, one over the raw settings blob, one 64 chars and one
16). Two modules each decided what a playoff period was and disagreed (one
handed out matchup periods, the other multiplied them into weeks). The private
decision builder read a schema nothing produced, so its own tests fed it a
hand-shaped fixture and proved only that the fixture matched the reader.

So every test here starts from the real path — `normalize_league()` over the
recorded ESPN views, then `snapshot_to_dict()` — and feeds *that* to whatever
is under test. A hand-shaped input is not allowed to be the integration
boundary anywhere in this file: if the adapter and a consumer disagree about
the schema, that is precisely the bug these tests exist to catch, and a
bespoke fixture would hide it.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue.fantasy import espn_contract, espn_league, my_team, waivers

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "espn"

LEAGUE_ID = 1111111111
SEASON = 2026
TEAM_ID = 1
TEAM_NAME = "Team One"
TEAM_COUNT = 8
# Fixed points in the past. `normalize_league` refuses a capture dated in the
# future, so a constant here that drifts ahead of the wall clock would fail the
# whole file for a reason that has nothing to do with what it tests.
RETRIEVED_AT = "2026-08-29T12:00:00Z"
NOW = "2026-08-29T13:00:00Z"
LATER = "2026-08-29T23:00:00Z"


def _fixture(name):
    return json.loads((FIXTURES / name).read_text())


def expected(**overrides):
    fields = {"league_id": LEAGUE_ID, "season": SEASON, "team_id": TEAM_ID,
              "team_name": TEAM_NAME, "team_count": TEAM_COUNT}
    fields.update(overrides)
    return espn_league.ExpectedIdentity(**fields)


def normalize(raw, *, players=None, exp=None, retrieved_at=RETRIEVED_AT):
    views = {"mSettings": raw, "mTeam": raw, "mRoster": raw, "mMatchup": raw,
             "mDraftDetail": raw, "mStandings": raw, "mTransactions2": raw}
    if players is not None:
        views["kona_player_info"] = players
    return espn_league.normalize_league(
        views, expected=exp or expected(), retrieved_at=retrieved_at,
        source_urls=["https://lm-api-reads.fantasy.espn.com/<redacted>"])


@pytest.fixture
def inseason():
    return _fixture("league_inseason_2026.json")


@pytest.fixture
def free_agents():
    return _fixture("players_free_agents_2026.json")


@pytest.fixture
def snapshot(inseason, free_agents):
    """The canonical snapshot dict — the only integration boundary in this file."""
    return espn_league.snapshot_to_dict(normalize(inseason, players=free_agents))


@pytest.fixture
def contract(inseason):
    return espn_contract.from_settings_payload(inseason)


@pytest.fixture
def custom_contract():
    """The real league's recorded rules — the ones with priced 2pt conversions.

    The synthetic in-season fixture prices nine plain categories, so a coverage
    audit over it is honestly `exact`. The coverage question only bites on a
    league that pays for events the producer never emits, which is this one.
    """
    recorded = ROOT / "tests" / "fixtures" / "espn_league_settings_2026_recorded.json"
    return espn_contract.from_settings_payload(json.loads(recorded.read_text()))


# =========================================================================== #
# 1. The real snapshot is what the private decision builder consumes
# =========================================================================== #
def test_decision_builder_consumes_the_adapter_output_directly(snapshot, contract):
    """No translation layer, no bespoke Monitor schema, no reshaping."""
    result = my_team.build(snapshot, now=NOW, contract=contract)

    assert result["league"]["league_id"] == str(LEAGUE_ID)
    assert result["league"]["season"] == SEASON
    assert result["league"]["team_id"] == TEAM_ID
    assert result["league"]["team_name"] == TEAM_NAME
    # It read the real roster out of the real snapshot, not an empty degrade.
    assert result["optimal_lineup"]["status"] in {"ok", "no_current_pick"}
    assert result["source_schema_version"] == espn_league.SCHEMA_VERSION


def test_decision_builder_sees_the_real_roster_not_an_empty_degrade(snapshot, contract):
    """A populated snapshot must not arrive as 'no roster exists'."""
    result = my_team.build(snapshot, now=NOW, contract=contract)
    reason = str(result["optimal_lineup"].get("reason") or "")
    assert "no roster exists" not in reason, (
        "the builder could not find the roster the adapter published — the two "
        f"disagree about the snapshot schema; reason was {reason!r}")


# =========================================================================== #
# 2. Fail closed, every way the input can be wrong
# =========================================================================== #
def test_wrong_schema_version_is_refused(snapshot, contract):
    bad = deepcopy(snapshot)
    bad["schema_version"] = "espn-league/999"
    with pytest.raises(my_team.SnapshotRejected, match="schema"):
        my_team.build(bad, now=NOW, contract=contract)


def test_missing_schema_version_is_refused(snapshot, contract):
    bad = deepcopy(snapshot)
    del bad["schema_version"]
    with pytest.raises(my_team.SnapshotRejected):
        my_team.build(bad, now=NOW, contract=contract)


def test_wrong_league_is_refused(inseason):
    wrong = deepcopy(inseason)
    wrong["id"] = 999999
    with pytest.raises(espn_league.EspnIdentityError):
        normalize(wrong)


def test_wrong_season_is_refused(inseason):
    wrong = deepcopy(inseason)
    wrong["seasonId"] = 2025
    with pytest.raises(espn_league.EspnIdentityError):
        normalize(wrong)


def test_wrong_team_is_refused(inseason):
    with pytest.raises(espn_league.EspnIdentityError):
        normalize(inseason, exp=expected(team_name="Somebody Else"))


def test_team_count_drift_is_refused(inseason):
    drifted = deepcopy(inseason)
    drifted["teams"] = drifted["teams"][:-1]
    with pytest.raises(espn_league.EspnIdentityError):
        normalize(drifted)


def test_duplicate_roster_player_is_refused(inseason):
    """The same player on two rosters is a corrupt league, not a tie."""
    doubled = deepcopy(inseason)
    first = doubled["teams"][0]["roster"]["entries"][0]
    doubled["teams"][1]["roster"]["entries"].append(deepcopy(first))
    with pytest.raises(espn_league.EspnSchemaError, match=r"(?i)duplicate"):
        normalize(doubled)


def test_unresolved_identity_is_reported_never_silently_scored(inseason, free_agents,
                                                               contract):
    """A player with no id is visible as unresolved, and is not a zero."""
    broken = deepcopy(inseason)
    entry = broken["teams"][0]["roster"]["entries"][0]
    entry.pop("playerId", None)
    entry.get("playerPoolEntry", {}).get("player", {}).pop("id", None)
    snap = espn_league.snapshot_to_dict(normalize(broken, players=free_agents))
    assert snap["unmatched_players"], "an unresolvable roster entry vanished silently"
    result = my_team.build(snap, now=NOW, contract=contract)
    assert result["unresolved_identities"], (
        "the decision builder scored or dropped an unresolved player instead of "
        "reporting it")


def test_unsupported_scoring_category_is_refused(inseason):
    unknown = deepcopy(inseason)
    unknown["settings"]["scoringSettings"]["scoringItems"].append(
        {"statId": 424242, "points": 1.0})
    with pytest.raises(espn_contract.UnsupportedEspnSetting, match="424242"):
        normalize(unknown)


def test_unsupported_lineup_slot_is_refused(inseason):
    unknown = deepcopy(inseason)
    unknown["settings"]["rosterSettings"]["lineupSlotCounts"]["99"] = 1
    with pytest.raises((espn_league.EspnSchemaError,
                        espn_contract.UnsupportedEspnSetting)):
        normalize(unknown)


# =========================================================================== #
# 3. Snapshot selection uses the embedded retrieval time, not the filesystem
# =========================================================================== #
def _write(directory: Path, name: str, snapshot: dict, retrieved_at: str,
           *, mtime: float | None = None) -> Path:
    payload = deepcopy(snapshot)
    payload["retrieved_at"] = retrieved_at
    path = directory / name
    path.write_text(json.dumps(payload))
    if mtime is not None:
        import os
        os.utime(path, (mtime, mtime))
    return path


def test_latest_snapshot_is_chosen_by_embedded_time_not_mtime(tmp_path, snapshot):
    """A checkout or an rsync rewrites mtimes; it does not re-retrieve anything."""
    _write(tmp_path, "older.json", snapshot, "2026-08-29T12:00:00Z", mtime=9_000_000_000)
    _write(tmp_path, "newer.json", snapshot, "2026-08-29T18:00:00Z", mtime=1_000_000_000)
    chosen = espn_league.load_latest_snapshot(tmp_path, now=LATER)
    assert chosen["retrieved_at"] == "2026-08-29T18:00:00Z", (
        "selection followed filesystem mtime, so touching a stale file makes it "
        "the league's current state")


def test_future_retrieval_time_is_rejected(tmp_path, snapshot):
    _write(tmp_path, "good.json", snapshot, "2026-08-29T12:00:00Z")
    _write(tmp_path, "future.json", snapshot,
           (datetime.now(timezone.utc) + timedelta(days=120)).isoformat())
    chosen = espn_league.load_latest_snapshot(tmp_path, now=NOW)
    assert chosen["retrieved_at"] == "2026-08-29T12:00:00Z", (
        "a snapshot retrieved in the future was accepted as the newest state")


def test_a_snapshot_retrieved_in_the_future_is_refused_outright(inseason):
    ahead = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    with pytest.raises(espn_league.EspnSchemaError, match=r"(?i)future"):
        normalize(inseason, retrieved_at=ahead)


def test_invalid_clock_order_is_refused(tmp_path, snapshot):
    """response_received before request_started is a broken clock, not a capture."""
    bad = deepcopy(snapshot)
    bad["source"] = dict(bad["source"])
    bad["source"]["request_started"] = "2026-08-29T12:00:05Z"
    bad["source"]["response_received"] = "2026-08-29T12:00:00Z"
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    assert espn_league.load_latest_snapshot(tmp_path, now=NOW) is None, (
        "a snapshot whose own clock runs backwards was accepted")


# =========================================================================== #
# 4. Waivers: the real Boolean, the real rank, the real claim window
# =========================================================================== #
def test_waiver_order_reset_boolean_means_inverse_standings(snapshot):
    """`waiverOrderReset: true` is ESPN for 'resets to inverse standings weekly'.

    It is a Boolean, not a mode string. Reading it as text and looking for the
    word INVERSE lands on the opposite answer for every league that sets it.
    """
    assert snapshot["waivers"]["order_reset"] is True
    assert snapshot["waivers"]["mode"] == "inverse_standings"


def test_team_priority_is_the_explicit_waiver_rank_not_an_inversion(snapshot, inseason):
    """ESPN already publishes the resulting order; deriving it again inverts it."""
    published = {int(t["id"]): int(t["waiverRank"]) for t in inseason["teams"]}
    assert {int(k): v for k, v in snapshot["waivers"]["team_priority"].items()} == published


def test_immediate_free_agents_are_distinguished_from_waiver_claims(snapshot):
    pool = snapshot["free_agents"]
    assert pool, "the fixture pool did not survive normalization"
    kinds = {entry["acquisition_kind"] for entry in pool}
    assert kinds <= {"free_agent", "waiver_claim"}
    assert "free_agent" in kinds, "no immediately addable player was identified"


def test_a_future_waiver_processing_time_still_permits_a_claim(snapshot):
    """A pending process time is when the claim resolves, not a refusal to take it."""
    rules = espn_league.waiver_rules_from_snapshot(snapshot)
    now = datetime(2026, 9, 24, 13, 0, tzinfo=timezone.utc)
    pool = [waivers.PoolEntry(
        espn_id=999001, name="Waiver Guy", position="WR", availability="ONWAIVERS",
        as_of=now - timedelta(hours=1), waiver_process_time=now + timedelta(days=2))]
    claimable = waivers.addable(rules, pool, roster=[], now=now)
    assert [e.espn_id for e in claimable] == [999001], (
        "a player on waivers until Wednesday was treated as unclaimable, which is "
        "backwards: that window is exactly when a claim is placed")


# =========================================================================== #
# 5. Playoff periods: derived once, never multiplied twice
# =========================================================================== #
@pytest.fixture
def two_week_playoffs(inseason):
    """14-game season, two-week semifinal, two-week final: weeks 15-18."""
    raw = deepcopy(inseason)
    schedule = raw["settings"]["scheduleSettings"]
    schedule["matchupPeriodCount"] = 14
    schedule["playoffTeamCount"] = 4
    schedule["playoffMatchupPeriodLength"] = 2
    raw["status"]["finalScoringPeriod"] = 18
    return raw


def test_two_week_rounds_expand_to_the_real_scoring_periods(two_week_playoffs,
                                                            free_agents):
    snap = espn_league.snapshot_to_dict(normalize(two_week_playoffs, players=free_agents))
    playoffs = snap["playoffs"]
    assert tuple(playoffs["playoff_matchup_periods"]) == (15, 16)
    assert tuple(playoffs["playoff_scoring_periods"]) == (15, 16, 17, 18)


def test_downstream_reads_the_published_weeks_and_does_not_multiply_again(
        two_week_playoffs, free_agents):
    from nflvalue.fantasy import league_trades

    snap = espn_league.snapshot_to_dict(normalize(two_week_playoffs, players=free_agents))
    assert league_trades.playoff_scoring_periods(snap) == tuple(
        snap["playoffs"]["playoff_scoring_periods"]), (
        "the trade scan re-expanded periods the adapter had already expanded, "
        "which squares the round length")


def test_one_week_rounds_are_unchanged_by_the_expansion(snapshot):
    playoffs = snapshot["playoffs"]
    assert tuple(playoffs["playoff_matchup_periods"]) == (15, 16)
    assert tuple(playoffs["playoff_scoring_periods"]) == (15, 16)


# =========================================================================== #
# 6. Required settings are validated, not defaulted to zero/false
# =========================================================================== #
@pytest.mark.parametrize("block,key", [
    ("scheduleSettings", "matchupPeriodCount"),
    ("scheduleSettings", "playoffTeamCount"),
    ("scheduleSettings", "playoffMatchupPeriodLength"),
    ("acquisitionSettings", "acquisitionType"),
    ("acquisitionSettings", "waiverOrderReset"),
    ("acquisitionSettings", "isUsingAcquisitionBudget"),
])
def test_a_missing_required_setting_refuses_instead_of_defaulting(inseason, block, key):
    stripped = deepcopy(inseason)
    stripped["settings"][block].pop(key, None)
    with pytest.raises(espn_league.EspnSchemaError, match=key):
        normalize(stripped)


def test_an_empty_required_block_is_not_silently_accepted(inseason):
    stripped = deepcopy(inseason)
    stripped["settings"]["scoringSettings"] = {}
    with pytest.raises((espn_league.EspnSchemaError,
                        espn_contract.UnsupportedEspnSetting)):
        normalize(stripped)


# =========================================================================== #
# 7. One scoring hash and one roster-slot hash, everywhere
# =========================================================================== #
def test_the_adapter_embeds_the_contract_rather_than_deriving_its_own(snapshot, contract):
    assert snapshot["hashes"]["scoring"] == contract.scoring_hash
    assert snapshot["hashes"]["roster"] == contract.roster_slot_hash
    assert snapshot["rules"]["contract_version"] == espn_contract.CONTRACT_VERSION


def test_every_consumer_reports_the_same_two_hashes(snapshot, contract):
    """Adapter, private output, waivers, trades, shadow and bracket must agree."""
    from nflvalue.fantasy import league_sim

    scoring = snapshot["hashes"]["scoring"]
    roster = snapshot["hashes"]["roster"]

    private = my_team.build(snapshot, now=NOW, contract=contract)
    assert private["scoring_hash"] == scoring
    assert private["roster_slot_hash"] == roster

    rules = espn_league.waiver_rules_from_snapshot(snapshot)
    assert rules.scoring_hash == scoring
    assert rules.roster_hash == roster

    fmt = league_sim.from_snapshot(snapshot)
    assert fmt.source_hashes["scoring"] == scoring
    assert fmt.source_hashes["roster"] == roster


def test_no_consumer_computes_a_second_scoring_hash(snapshot, contract):
    """A 16-char digest over a raw blob is not the same promise as this one."""
    scoring = snapshot["hashes"]["scoring"]
    assert len(scoring) == 64, "the canonical scoring hash is a full sha256"
    assert scoring == contract.scoring_hash
    assert espn_league.content_digest(snapshot) != scoring


# =========================================================================== #
# 8. Custom-scoring honesty: coverage is audited, absence is never a zero
# =========================================================================== #
def test_simulated_components_do_not_claim_exact_custom_scoring(custom_contract):
    from nflvalue.fantasy import coverage, simulation

    report = coverage.audit(custom_contract, emitted=simulation.EMITTED_COMPONENTS)
    assert not report.exact, (
        "the weekly simulator emits no two-point-conversion components, so an "
        "'exact custom scoring' label over its output is a claim about events "
        "it never produced")
    assert "passing_2pt" in report.unsupported
    assert "rushing_2pt" in report.unsupported
    assert "receiving_2pt" in report.unsupported


def test_team_win_is_named_unsupported_rather_than_scored_as_zero(custom_contract):
    from nflvalue.fantasy import coverage, simulation

    report = coverage.audit(custom_contract, emitted=simulation.EMITTED_COMPONENTS)
    if "team_win" in custom_contract.categories:
        assert "team_win" in report.unsupported


def test_a_producer_that_emits_everything_is_allowed_the_exact_label(custom_contract):
    from nflvalue.fantasy import coverage

    everything = coverage.required_components(custom_contract)
    report = coverage.audit(custom_contract, emitted=everything)
    # `team_win` has no modelled event anywhere on the fantasy path, so no
    # producer can back it: it stays unsupported however complete the rest is,
    # which is the honest answer rather than a component nobody computes.
    assert set(report.unsupported) <= set(coverage.NO_MODELLED_EVENT)
    assert report.exact is (not report.unsupported)
    assert "passing_2pt" in report.covered
    assert "receiving_2pt" in report.covered


# =========================================================================== #
# 9. The frozen generic-PPR outputs did not move
# =========================================================================== #
def _legacy_ppr_points(c):
    """Standard PPR, written out longhand from the pre-split arithmetic.

    Deliberately not a call into `score_components`: a re-run of the
    implementation would only prove the code agrees with itself, which is
    exactly the evidence a neutrality claim may not rest on. One two-point
    multiplier, as it was before passing/rushing/receiving were priced apart.
    """
    two_point = 2.0
    return (
        c.get("passing_yards", 0.0) * 0.04
        + c.get("passing_tds", 0.0) * 4.0
        + c.get("passing_interceptions", 0.0) * -2.0
        + c.get("rushing_yards", 0.0) * 0.1
        + c.get("rushing_tds", 0.0) * 6.0
        + c.get("receptions", 0.0) * 1.0
        + c.get("receiving_yards", 0.0) * 0.1
        + c.get("receiving_tds", 0.0) * 6.0
        + (c.get("passing_2pt_conversions", 0.0)
           + c.get("rushing_2pt_conversions", 0.0)
           + c.get("receiving_2pt_conversions", 0.0)) * two_point
        + c.get("fumbles_lost", 0.0) * -2.0
    )


@pytest.mark.parametrize("line", [
    {"passing_yards": 312.0, "passing_tds": 3.0, "passing_interceptions": 1.0,
     "rushing_yards": 22.0, "rushing_tds": 1.0, "passing_2pt_conversions": 1.0},
    {"rushing_yards": 118.0, "rushing_tds": 2.0, "receptions": 4.0,
     "receiving_yards": 31.0, "fumbles_lost": 1.0, "rushing_2pt_conversions": 1.0},
    {"receptions": 9.0, "receiving_yards": 142.0, "receiving_tds": 2.0,
     "receiving_2pt_conversions": 1.0},
    {"receptions": 5.0, "receiving_yards": 47.0},
])
def test_generic_ppr_scoring_is_numerically_unchanged_by_the_two_point_split(line):
    """Splitting one 2pt multiplier into three must not move a generic score.

    The three categories are priced apart because this league pays 4.0 for a
    receiving conversion. Where a league prices them equally — every generic
    QB/RB/WR/TE output this project has frozen — the arithmetic has to come out
    where it came out before, to the bit.
    """
    from nflvalue.fantasy.config import ScoringRules
    from nflvalue.fantasy.scoring import score_components

    scored = float(score_components(line, ScoringRules()))
    assert scored == pytest.approx(_legacy_ppr_points(line), abs=0.0, rel=0.0)


def test_the_simulator_emits_exactly_what_it_publishes():
    """`EMITTED_COMPONENTS` is the audit's evidence; it must not become a wish."""
    from nflvalue.fantasy import simulation

    assert simulation.EMITTED_COMPONENTS == (
        "completions", "attempts", "passing_yards", "passing_tds",
        "passing_interceptions", "carries", "rushing_yards", "rushing_tds",
        "targets", "receptions", "receiving_yards", "receiving_tds",
        "fumbles_lost",
    )
