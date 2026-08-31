"""Contract tests for the read-only ESPN league adapter.

The adapter's job is to say what Curtis's league *is*, exactly, or to refuse.
Everything asserted here follows from that:

* **Fail closed on identity.** A snapshot that quietly describes the wrong
  league, the wrong season, the wrong team, or the wrong number of teams is
  worse than no snapshot, because every downstream decision inherits it
  silently. Wrong league, wrong season, wrong team, size != expected,
  ambiguous team match, an unknown lineup slot and incomplete settings all
  raise rather than degrade.
* **Pre-draft is a state, not a gap.** The league does not draft until
  2026-09-05. Until then there are no rosters and no picks, and the adapter
  must say exactly that. It may not invent a roster, and it may not accept
  watchlist targets as draft selections -- a watchlist is a list of players
  someone is *thinking* about; a pick is a player they *have*.
* **Secrets never land.** ESPN's `members[].id` IS the SWID cookie value, so
  the raw payload contains a credential by construction. Nothing
  credential-shaped may survive into a snapshot, and no fixture may contain a
  real one.
* **IDs over names.** ESPN player ids are stable; names are not. Anything the
  adapter cannot resolve stays visible as unmatched rather than disappearing.

Every test here is offline. No test in this file may perform a network call.
"""

from __future__ import annotations

import json
import re
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue.fantasy import espn_client, espn_league

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "espn"

LEAGUE_ID = 1111111111
SEASON = 2026
TEAM_ID = 1
TEAM_NAME = "Team One"
TEAM_COUNT = 8
DRAFT_ROUNDS = 16
DRAFT_SLOT = 1

# Anything shaped like an ESPN credential. `espn_s2` values are long opaque
# strings; SWID is a braced GUID and is also what ESPN uses as a member id.
CREDENTIAL_PATTERNS = (
    re.compile(r"espn_s2", re.I),
    re.compile(r"\bswid\b", re.I),
    re.compile(r"\{[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\}"),
    re.compile(r"\bcookie\b", re.I),
)


def _fixture(name):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def predraft():
    return _fixture("league_predraft_2026.json")


@pytest.fixture
def inseason():
    return _fixture("league_inseason_2026.json")


@pytest.fixture
def free_agents():
    return _fixture("players_free_agents_2026.json")


def expected(**overrides):
    fields = {
        "league_id": LEAGUE_ID, "season": SEASON, "team_id": TEAM_ID,
        "team_name": TEAM_NAME, "team_count": TEAM_COUNT,
    }
    fields.update(overrides)
    return espn_league.ExpectedIdentity(**fields)


def normalize(raw, *, players=None, exp=None, retrieved_at="2026-08-29T12:00:00Z"):
    views = {"mSettings": raw, "mTeam": raw, "mRoster": raw, "mMatchup": raw,
             "mDraftDetail": raw, "mStandings": raw, "mTransactions2": raw}
    if players is not None:
        views["kona_player_info"] = players
    return espn_league.normalize_league(
        views, expected=exp or expected(), retrieved_at=retrieved_at,
        source_urls=["https://lm-api-reads.fantasy.espn.com/<redacted>"])


# =========================================================================== #
# 1. Identity — the gate everything else stands behind
# =========================================================================== #
def test_public_predraft_league_normalizes_to_the_expected_identity(predraft):
    snap = normalize(predraft)
    assert snap.league.league_id == LEAGUE_ID
    assert snap.league.season == SEASON
    assert snap.league.size == TEAM_COUNT
    assert snap.league.current_scoring_period == 1
    assert snap.my_team.team_id == TEAM_ID
    assert snap.my_team.name == TEAM_NAME
    assert len(snap.teams) == TEAM_COUNT
    assert snap.schema_version == espn_league.SCHEMA_VERSION


def test_wrong_league_is_refused(predraft):
    with pytest.raises(espn_league.EspnIdentityError, match="league"):
        normalize(predraft, exp=expected(league_id=99999999))


def test_wrong_season_is_refused(predraft):
    with pytest.raises(espn_league.EspnIdentityError, match="season"):
        normalize(predraft, exp=expected(season=2025))


def test_wrong_team_id_is_refused(predraft):
    with pytest.raises(espn_league.EspnIdentityError, match="team"):
        normalize(predraft, exp=expected(team_id=7))


def test_wrong_team_name_is_refused(predraft):
    """The id and the name must agree. One of them being right is not enough:
    a renamed team is a different claim about who we are."""
    with pytest.raises(espn_league.EspnIdentityError, match="name"):
        normalize(predraft, exp=expected(team_name="Somebody Else"))


def test_team_count_mismatch_is_refused(predraft):
    """An 8-team league that answers with 7 teams has either lost a team or
    is not the league we asked for. Replacement-level pricing depends on the
    count, so guessing here poisons the board."""
    shrunk = deepcopy(predraft)
    shrunk["teams"] = shrunk["teams"][:7]
    with pytest.raises(espn_league.EspnIdentityError, match="8"):
        normalize(shrunk)


def test_settings_size_disagreeing_with_the_team_list_is_refused(predraft):
    """Two sources of truth for league size inside one payload. If they
    disagree the payload is not trustworthy for anything."""
    inconsistent = deepcopy(predraft)
    inconsistent["settings"]["size"] = 10
    with pytest.raises(espn_league.EspnIdentityError):
        normalize(inconsistent)


def test_duplicate_team_ids_are_ambiguous_and_refused(predraft):
    ambiguous = deepcopy(predraft)
    ambiguous["teams"][1]["id"] = TEAM_ID
    with pytest.raises(espn_league.EspnIdentityError, match=r"ambiguous|duplicate"):
        normalize(ambiguous)


def test_duplicate_team_names_are_ambiguous_and_refused(predraft):
    ambiguous = deepcopy(predraft)
    ambiguous["teams"][3]["name"] = TEAM_NAME
    with pytest.raises(espn_league.EspnIdentityError, match=r"ambiguous|duplicate"):
        normalize(ambiguous)


def test_incomplete_settings_are_refused(predraft):
    for missing in ("rosterSettings", "scoringSettings", "scheduleSettings",
                    "draftSettings", "acquisitionSettings"):
        broken = deepcopy(predraft)
        del broken["settings"][missing]
        with pytest.raises(espn_league.EspnSchemaError, match=missing):
            normalize(broken)


def test_a_missing_required_view_is_refused(predraft):
    with pytest.raises(espn_league.EspnSchemaError, match="view"):
        espn_league.normalize_league(
            {"mTeam": predraft}, expected=expected(),
            retrieved_at="2026-08-29T12:00:00Z", source_urls=[])


# =========================================================================== #
# 2. Pre-draft is represented, never filled in
# =========================================================================== #
def test_predraft_state_is_explicit_and_rosters_are_empty(predraft):
    snap = normalize(predraft)
    assert snap.draft.status == "pre_draft"
    assert snap.draft.picks is None, "there are no picks before a draft"
    assert snap.roster_state == "empty_pre_draft"
    assert all(entries == () for entries in snap.rosters.values())
    assert snap.rosters.keys() == {team.team_id for team in snap.teams}


def test_predraft_snapshot_carries_the_draft_plan_without_inventing_results(predraft):
    snap = normalize(predraft)
    assert snap.draft.type == "SNAKE"
    assert snap.draft.rounds == DRAFT_ROUNDS
    assert snap.draft.my_slot == DRAFT_SLOT
    assert snap.draft.pick_order == (1, 2, 3, 4, 5, 6, 7, 8)
    assert snap.draft.scheduled_at.startswith("2026-09-05")


def test_watchlist_targets_can_never_become_draft_picks(predraft):
    """A watchlist is intent; a pick is history. `reports/espn_watchlist_*.json`
    exists in this repo and is shaped temptingly like a pick list."""
    contaminated = deepcopy(predraft)
    contaminated["draftDetail"]["picks"] = [
        {"playerId": 4262921, "teamId": 1, "roundId": 1, "roundPickNumber": 1,
         "overallPickNumber": 1, "source": "watchlist"},
    ]
    with pytest.raises(espn_league.EspnSchemaError, match=r"drafted|pick"):
        normalize(contaminated)


def test_an_undrafted_league_reporting_rosters_is_refused(predraft):
    """Rosters without a draft is a contradiction, not a shortcut to a board."""
    contradictory = deepcopy(predraft)
    contradictory["teams"][0]["roster"]["entries"] = [
        {"playerId": 4262921, "lineupSlotId": 2,
         "playerPoolEntry": {"id": 4262921, "onTeamId": 1, "player": {
             "id": 4262921, "fullName": "Saquon Barkley", "defaultPositionId": 2,
             "eligibleSlots": [2, 3, 23, 20], "proTeamId": 21}}},
    ]
    with pytest.raises(espn_league.EspnSchemaError, match=r"drafted|roster"):
        normalize(contradictory)


# =========================================================================== #
# 3. Rosters, slots and eligibility
# =========================================================================== #
def test_inseason_rosters_are_keyed_by_stable_player_id(inseason):
    snap = normalize(inseason)
    mine = snap.rosters[TEAM_ID]
    assert len(mine) == 17          # 16 active + 1 on IR
    assert snap.roster_state == "populated"
    barkley = next(p for p in mine if p.player_id == 4262921)
    assert barkley.full_name == "Saquon Barkley"
    assert barkley.lineup_slot == "RB"
    assert "FLEX" in barkley.eligible_slots
    assert all(isinstance(p.player_id, int) for p in mine)


def test_roster_settings_split_starters_bench_and_ir(inseason):
    snap = normalize(inseason)
    slots = snap.roster_settings
    assert slots.lineup_slot_counts["QB"] == 1
    assert slots.lineup_slot_counts["RB"] == 2
    assert slots.lineup_slot_counts["FLEX"] == 1
    assert slots.bench_slots == 7
    assert slots.ir_slots == 1
    assert slots.starting_slots == 9
    assert slots.roster_size == DRAFT_ROUNDS, "16 non-IR slots => a 16-round draft"


def test_ir_players_are_marked_and_not_counted_as_starters(inseason):
    snap = normalize(inseason)
    ir = [p for p in snap.rosters[TEAM_ID] if p.lineup_slot == "IR"]
    assert len(ir) == 1
    assert ir[0].player_id == 4239996
    assert ir[0].injury_status == "INJURY_RESERVE"
    assert not ir[0].is_starter
    starters = [p for p in snap.rosters[TEAM_ID] if p.is_starter]
    assert len(starters) == snap.roster_settings.starting_slots


def test_an_unknown_lineup_slot_is_refused_not_guessed(inseason):
    unknown = deepcopy(inseason)
    unknown["settings"]["rosterSettings"]["lineupSlotCounts"]["99"] = 1
    with pytest.raises(espn_league.EspnSchemaError, match="slot"):
        normalize(unknown)


def test_a_player_in_a_slot_they_are_not_eligible_for_is_surfaced(inseason):
    """ESPN can be edited by a league manager. A roster that violates its own
    eligibility rules is reported, not silently normalized away."""
    illegal = deepcopy(inseason)
    entry = illegal["teams"][0]["roster"]["entries"][0]   # a QB
    entry["lineupSlotId"] = 2                             # ...in an RB slot
    snap = normalize(illegal)
    assert snap.eligibility_violations
    violation = snap.eligibility_violations[0]
    assert violation["player_id"] == entry["playerId"]
    assert violation["lineup_slot"] == "RB"


def test_players_that_cannot_be_normalized_stay_visible(inseason):
    broken = deepcopy(inseason)
    broken["teams"][0]["roster"]["entries"].append(
        {"playerId": None, "lineupSlotId": 20, "playerPoolEntry": {"player": {}}})
    snap = normalize(broken)
    assert snap.unmatched_players, "a player we could not read must not vanish"
    assert len(snap.rosters[TEAM_ID]) == 17


# =========================================================================== #
# 4. Free agents, schedule, playoffs, waivers, standings, transactions
# =========================================================================== #
def test_free_agent_pool_is_normalized_with_availability(predraft, free_agents):
    snap = normalize(predraft, players=free_agents)
    assert snap.free_agents is not None
    assert len(snap.free_agents) == 3
    by_id = {p.player_id: p for p in snap.free_agents}
    assert by_id[800001].availability == "FREEAGENT"
    assert by_id[800003].availability == "WAIVERS"
    assert by_id[800003].injury_status == "QUESTIONABLE"


def test_free_agents_absent_is_none_not_empty(predraft):
    """None means 'not read'. [] would claim the pool is empty."""
    snap = normalize(predraft)
    assert snap.free_agents is None


def test_schedule_covers_every_matchup_period(inseason):
    snap = normalize(inseason)
    periods = {game.matchup_period for game in snap.schedule}
    assert periods == set(range(1, 15))
    assert len(snap.schedule) == 14 * (TEAM_COUNT // 2)
    played = [game for game in snap.schedule if game.winner != "UNDECIDED"]
    assert len(played) == 3 * (TEAM_COUNT // 2)
    assert all(game.home_points is not None for game in played)


def test_playoff_structure_is_captured(inseason):
    snap = normalize(inseason)
    assert snap.playoffs.team_count == 4
    assert snap.playoffs.seeding_rule == "TOTAL_POINTS_SCORED"
    assert snap.playoffs.matchup_period_length == 1
    assert snap.playoffs.regular_season_matchup_periods == 14
    assert snap.playoffs.playoff_matchup_periods == (15, 16)


def test_waiver_system_is_captured(inseason):
    snap = normalize(inseason)
    assert snap.waivers.acquisition_type == "WAIVERS_TRADITIONAL"
    assert snap.waivers.uses_acquisition_budget is False
    assert snap.waivers.process_days == ("WEDNESDAY",)
    assert snap.waivers.order_reset is True
    assert snap.waivers.team_priority[TEAM_ID] == 8
    assert snap.waivers.lock_policy


def test_standings_carry_records_and_tiebreaker_metadata(inseason):
    snap = normalize(inseason)
    mine = next(row for row in snap.standings.rows if row.team_id == TEAM_ID)
    assert (mine.wins, mine.losses, mine.ties) == (3, 0, 0)
    assert mine.points_for > 0
    assert snap.standings.tiebreaker["playoff_seeding_rule"] == "TOTAL_POINTS_SCORED"
    assert "matchup_tie_rule" in snap.standings.tiebreaker


def test_transactions_separate_completed_from_pending(inseason):
    snap = normalize(inseason)
    assert len(snap.transactions.completed) == 2
    assert len(snap.transactions.pending) == 1
    assert snap.transactions.pending[0].status == "PENDING"
    types = {tx.type for tx in snap.transactions.completed}
    assert types == {"WAIVER", "TRADE_ACCEPT"}
    assert snap.transactions.completed[0].items


def test_draft_selections_are_read_when_the_draft_has_happened(inseason):
    snap = normalize(inseason)
    assert snap.draft.status == "complete"
    assert snap.draft.picks is not None
    assert len(snap.draft.picks) == DRAFT_ROUNDS * TEAM_COUNT
    first = snap.draft.picks[0]
    assert (first.overall_pick, first.round, first.round_pick, first.team_id) == (1, 1, 1, 1)
    # Snake: round 2 reverses, so overall pick 9 belongs to team 8.
    ninth = next(p for p in snap.draft.picks if p.overall_pick == 9)
    assert ninth.team_id == 8 and ninth.round == 2


# =========================================================================== #
# 5. Hashes, versioning and the immutable snapshot
# =========================================================================== #
def test_snapshot_carries_league_scoring_and_roster_hashes(predraft):
    snap = normalize(predraft)
    for name in ("league", "scoring", "roster"):
        digest = snap.hashes[name]
        assert re.fullmatch(r"[0-9a-f]{64}", digest), f"{name} hash is not sha256"
    assert len({snap.hashes[k] for k in ("league", "scoring", "roster")}) == 3


def test_hashes_are_deterministic_and_change_only_with_their_subject(predraft):
    first = normalize(predraft)
    again = normalize(predraft, retrieved_at="2026-08-29T18:00:00Z")
    assert first.hashes == again.hashes, "hashes must not depend on fetch time"

    rescored = deepcopy(predraft)
    rescored["settings"]["scoringSettings"]["scoringItems"][7]["points"] = 0.5  # PPR -> half
    changed = normalize(rescored)
    assert changed.hashes["scoring"] != first.hashes["scoring"]
    assert changed.hashes["roster"] == first.hashes["roster"]

    reslotted = deepcopy(predraft)
    reslotted["settings"]["rosterSettings"]["lineupSlotCounts"]["20"] = 6
    assert normalize(reslotted).hashes["roster"] != first.hashes["roster"]


def test_snapshot_records_provenance(predraft):
    snap = normalize(predraft)
    assert snap.retrieved_at == "2026-08-29T12:00:00Z"
    assert snap.source.views
    assert snap.source.urls
    assert snap.source.credentialed is False


def test_snapshot_file_is_immutable(tmp_path, predraft):
    snap = normalize(predraft)
    path = espn_league.write_snapshot(snap, tmp_path)
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == espn_league.SCHEMA_VERSION
    assert payload["content_sha256"]
    with pytest.raises(FileExistsError):
        espn_league.write_snapshot(snap, tmp_path)


def test_snapshot_round_trips_as_plain_json(tmp_path, inseason, free_agents):
    snap = normalize(inseason, players=free_agents)
    path = espn_league.write_snapshot(snap, tmp_path)
    payload = json.loads(path.read_text())
    assert payload["league"]["league_id"] == LEAGUE_ID
    assert payload["my_team"]["team_id"] == TEAM_ID
    assert len(payload["draft"]["picks"]) == DRAFT_ROUNDS * TEAM_COUNT
    assert payload["content_sha256"] == espn_league.content_digest(payload)


# =========================================================================== #
# 6. Schema drift
# =========================================================================== #
def test_unrecognized_top_level_keys_are_reported_not_dropped(predraft):
    drifted = deepcopy(predraft)
    drifted["someNewEspnBlock"] = {"a": 1}
    drifted["settings"]["someNewSetting"] = True
    snap = normalize(drifted)
    assert "someNewEspnBlock" in snap.unsupported["league_keys"]
    assert "someNewSetting" in snap.unsupported["settings_keys"]


def test_a_changed_type_on_a_required_field_is_refused(predraft):
    drifted = deepcopy(predraft)
    drifted["seasonId"] = "2026"      # string where an int is required
    with pytest.raises((espn_league.EspnSchemaError, espn_league.EspnIdentityError)):
        normalize(drifted)


def test_schema_version_is_pinned():
    """Bumping this is a deliberate act: consumers key off it."""
    assert espn_league.SCHEMA_VERSION == "espn-league/1"


# =========================================================================== #
# 7. Secrets
# =========================================================================== #
def test_fixtures_contain_no_real_credentials():
    """The fake ones are allowed; a real one would be a leak in git history."""
    for path in sorted(FIXTURES.glob("*.json")):
        text = path.read_text()
        for guid in CREDENTIAL_PATTERNS[2].findall(text):
            body = guid.strip("{}").replace("-", "")
            # Every synthetic GUID here is one repeated nibble followed by a
            # zero-padded counter. A real SWID looks nothing like that.
            assert len(set(body[:8])) == 1 and body.count("0") >= 8, (
                f"{path.name} contains a GUID that does not look synthetic: {guid}")
        if "espn_s2" in text:
            assert "FAKEs2VALUE" in text, f"{path.name} names espn_s2 with a non-fake value"


def test_member_ids_are_hashed_because_espn_uses_the_swid_as_one(predraft):
    snap = normalize(predraft)
    raw_ids = {member["id"] for member in predraft["members"]}
    serialized = json.dumps(espn_league.snapshot_to_dict(snap))
    for raw in raw_ids:
        assert raw not in serialized, "a SWID-shaped member id reached the snapshot"
    assert all(member.member_key.startswith("member:") for member in snap.members)
    assert len({member.member_key for member in snap.members}) == TEAM_COUNT


def test_no_credential_shaped_string_survives_into_a_snapshot():
    raw = _fixture("raw_view_with_fake_secrets.json")
    snap = normalize(raw)
    serialized = json.dumps(espn_league.snapshot_to_dict(snap))
    for pattern in CREDENTIAL_PATTERNS:
        assert not pattern.search(serialized), (
            f"{pattern.pattern} survived into the snapshot")
    assert "FAKEs2VALUE" not in serialized


def test_redactor_strips_credentials_from_an_arbitrary_payload():
    raw = _fixture("raw_view_with_fake_secrets.json")
    cleaned = json.dumps(espn_league.redact_raw(raw))
    assert "FAKEs2VALUE" not in cleaned
    assert espn_league.REDACTED in cleaned
    # Non-secret content must survive the redactor untouched.
    assert "Test League" in cleaned


def test_credentials_are_read_only_from_the_environment(monkeypatch):
    monkeypatch.delenv("ESPN_S2", raising=False)
    monkeypatch.delenv("SWID", raising=False)
    assert espn_client.credentials_from_env() is None

    monkeypatch.setenv("ESPN_S2", "FAKEs2VALUE0000000000000000000000000000")
    monkeypatch.setenv("SWID", "{AAAAAAAA-1111-4000-8000-000000000001}")
    creds = espn_client.credentials_from_env()
    assert creds is not None


def test_credentials_never_render_their_values(monkeypatch):
    monkeypatch.setenv("ESPN_S2", "FAKEs2VALUE0000000000000000000000000000")
    monkeypatch.setenv("SWID", "{AAAAAAAA-1111-4000-8000-000000000001}")
    creds = espn_client.credentials_from_env()
    for rendering in (repr(creds), str(creds), f"{creds}"):
        assert "FAKEs2VALUE" not in rendering
        assert "AAAAAAAA" not in rendering
        assert espn_league.REDACTED in rendering


def test_the_client_module_exposes_no_credential_cli_surface():
    """`--espn-s2 <value>` would put the cookie in shell history and `ps`."""
    for path in (ROOT / "nflvalue" / "fantasy" / "espn_client.py",
                 ROOT / "scripts" / "espn_league_snapshot.py"):
        source = path.read_text()
        assert "--espn-s2" not in source and "--espn_s2" not in source
        assert "--swid" not in source
        assert "add_argument" not in source or "espn_s2" not in source.split("add_argument", 1)[1]


def test_the_adapter_never_calls_a_write_endpoint():
    source = (ROOT / "nflvalue" / "fantasy" / "espn_client.py").read_text()
    for forbidden in ("POST", "PUT", "PATCH", "DELETE", "method=", "data="):
        assert forbidden not in source, f"{forbidden!r} suggests a write path"
    assert "transactions/" not in source


# =========================================================================== #
# 8. The old name-only loader
# =========================================================================== #
def test_the_legacy_name_only_loader_says_what_it_is():
    """`load_rosters_espn` returns names only. Left unmarked, a caller reads it
    as a league sync and inherits every gap silently."""
    from nflvalue.fantasy import trade_planner
    doc = trade_planner.load_rosters_espn.__doc__ or ""
    assert "not a full" in doc.lower() or "name-only" in doc.lower()
    assert hasattr(trade_planner, "load_rosters_from_snapshot")


def test_rosters_can_be_built_from_a_snapshot_without_espn(inseason):
    from nflvalue.fantasy import trade_planner
    snap = normalize(inseason)
    rosters = trade_planner.load_rosters_from_snapshot(snap)
    assert rosters.my_team == TEAM_NAME
    assert len(rosters.teams) == TEAM_COUNT
    assert "Saquon Barkley" in rosters.teams[TEAM_NAME]


def test_a_predraft_snapshot_cannot_be_turned_into_rosters(predraft):
    """Silently returning eight empty rosters would let the trade planner
    'work' on a league that has not drafted."""
    from nflvalue.fantasy import trade_planner
    snap = normalize(predraft)
    with pytest.raises(ValueError, match=r"pre-draft|pre_draft"):
        trade_planner.load_rosters_from_snapshot(snap)


# =========================================================================== #
# 9. This suite stays offline
# =========================================================================== #
def test_no_test_in_this_file_performs_a_network_call():
    source = Path(__file__).read_text()
    for forbidden in ("urlopen(", "fetch_league_views(", "requests.get"):
        assert forbidden not in source.split("def test_no_test_in_this_file")[0], (
            f"{forbidden} would make this suite depend on ESPN being up")
