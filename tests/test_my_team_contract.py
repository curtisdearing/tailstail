"""Curtis-specific recommendation contract: identity, legality, and fail-closed.

This is a Monitor surface. Every assertion here exists because the honest
answer to "what should I do with my team right now" is frequently "nothing
trustworthy can be said", and the expensive failure is a page that says
something anyway.

The four traps this file was written against, all of which are real artifacts
sitting in this repository right now:

  * ESPN pre-creates all 128 pick slots before a draft, each with
    ``playerId -1``.  A reader that treats ``draftDetail.picks`` as selections
    reports 128 phantom picks for a draft that has not happened.
  * ``data/draft_board_2026_6team.csv`` (the league's old size) and
    ``data/draft_board_2026_12team.csv`` (a mock) will happily load for an
    8-team league and produce plausible, wrong targets.
  * ``data/trade_scan.json`` carries ``my_team: "Team1"`` against
    ``opponent: "Team4"`` — smoke-test output, not live rosters.
  * A prior week's card is structurally valid for the current week.

Note the guard is POSITIVE — a source must prove it matches the live league —
never a blocklist of names.  Teams 7 and 8 in this league are genuinely named
"Team 7" and "Team 8", so a name-pattern guard would reject real managers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nflvalue.fantasy import my_team  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "my_team"
NOW = "2026-08-29T03:00:00+00:00"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


#: The fixtures are canonical `espn-league/1` snapshots — what ESPN said, and
#: nothing else. Model ids, projections and byes are the *model's* answer and
#: travel beside the snapshot rather than inside it, which is what stopped the
#: builder and the adapter drifting into two schemas in the first place.
def model(name):
    return json.loads((FIXTURES / f"{name}.model.json").read_text())


def built(name, **kwargs):
    side = model(name)
    return my_team.build(
        fixture(name), now=NOW,
        crosswalk={int(k): v for k, v in side["crosswalk"].items()},
        projections=side["projections"], byes=side["byes"], **kwargs)



# --------------------------------------------------------------------------- #
# Hashes: one implementation, or none — never a second-best local one
# --------------------------------------------------------------------------- #
SETTINGS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "espn_league_settings_2026_recorded.json"


def league_contract():
    from nflvalue.fantasy import espn_contract

    return espn_contract.from_settings_payload(json.loads(SETTINGS_FIXTURE.read_text()))


def test_my_team_no_longer_implements_its_own_hash():
    """The duplication that made three modules disagree must not come back."""
    assert not hasattr(my_team, "scoring_hash")
    assert not hasattr(my_team, "roster_slot_hash")


def test_scoring_hash_is_stable_under_input_reordering():
    """Pinned against espn_contract, the one authority."""
    from nflvalue.fantasy import espn_contract

    raw = json.loads(SETTINGS_FIXTURE.read_text())
    shuffled = json.loads(SETTINGS_FIXTURE.read_text())
    items = shuffled["settings"]["scoringSettings"]["scoringItems"]
    shuffled["settings"]["scoringSettings"]["scoringItems"] = list(reversed(items))
    assert (espn_contract.from_settings_payload(raw).scoring_hash
            == espn_contract.from_settings_payload(shuffled).scoring_hash)


def test_scoring_hash_changes_when_a_point_value_changes():
    from nflvalue.fantasy import espn_contract

    raw = json.loads(SETTINGS_FIXTURE.read_text())
    before = espn_contract.from_settings_payload(raw).scoring_hash
    bumped = json.loads(SETTINGS_FIXTURE.read_text())
    item = bumped["settings"]["scoringSettings"]["scoringItems"][0]
    item["points"] = (item.get("points") or 0) + 1
    assert espn_contract.from_settings_payload(bumped).scoring_hash != before


def test_hashes_are_taken_from_the_contract_and_named_as_such():
    contract = league_contract()
    payload = built("pre_draft", contract=contract)
    assert payload["scoring_hash"] == contract.scoring_hash
    assert payload["roster_slot_hash"] == contract.roster_slot_hash
    assert "espn_contract" in payload["hash_source"]
    assert payload["hash_reason"] is None


def test_without_a_contract_the_snapshots_embedded_pair_is_used():
    """The adapter embeds the contract's hashes, so they travel with the read.

    This used to return nulls, because the builder had no way to learn the
    league's identity except from a contract handed to it separately. Now the
    snapshot carries the same two values the contract produced, so provenance
    survives without a second object — and it is still never invented locally.
    """
    payload = built("pre_draft")
    snapshot = fixture("pre_draft")
    assert payload["scoring_hash"] == snapshot["hashes"]["scoring"]
    assert payload["roster_slot_hash"] == snapshot["hashes"]["roster"]
    assert "embedded" in payload["hash_source"]
    assert payload["hash_reason"] is None


def test_a_contract_that_describes_another_league_is_refused_not_preferred():
    """Disagreement is reported; picking a winner would stamp the wrong rules."""
    snapshot = fixture("pre_draft")
    snapshot["hashes"] = dict(snapshot["hashes"], scoring="not-this-league")
    side = model("pre_draft")
    payload = my_team.build(
        snapshot, now=NOW, contract=league_contract(),
        crosswalk={int(k): v for k, v in side["crosswalk"].items()},
        projections=side["projections"], byes=side["byes"])
    assert payload["scoring_hash"] is None
    assert payload["roster_slot_hash"] is None
    assert "different rules" in payload["hash_reason"]


# --------------------------------------------------------------------------- #
# Draft state
# --------------------------------------------------------------------------- #
def test_pre_draft_reports_zero_selections_not_128_phantom_picks():
    payload = built("pre_draft")
    draft = payload["draft"]
    assert draft["state"] == "pre_draft"
    assert draft["selections"] == []
    assert draft["selection_count"] == 0
    assert draft["pick_slot_count"] == 128, "the empty slots are still reported, as slots"


def test_pre_draft_exposes_targets_that_are_never_labelled_picks():
    payload = built("pre_draft")
    targets = payload["draft"]["targets"]
    assert targets["status"] in {"ok", "no_current_pick"}
    if targets["status"] == "ok":
        for entry in targets["entries"]:
            assert entry["kind"] == "target"
            assert "overall_pick" not in entry
            assert "round" not in entry


def test_draft_in_progress_reports_only_real_selections():
    payload = built("draft_in_progress")
    draft = payload["draft"]
    assert draft["state"] == "in_progress"
    assert draft["selection_count"] == len(draft["selections"]) == 3
    first = draft["selections"][0]
    for field in ("round", "overall_pick", "team_id", "espn_player_id"):
        assert field in first, f"selection is missing {field}"
    assert all(s["espn_player_id"] > 0 for s in draft["selections"])


def test_post_draft_state_and_full_selection_ledger():
    payload = built("post_draft")
    assert payload["draft"]["state"] == "complete"
    assert payload["draft"]["selection_count"] == 16
    assert payload["draft"]["targets"]["status"] == "no_current_pick"
    assert "draft is complete" in payload["draft"]["targets"]["reason"].lower()


# --------------------------------------------------------------------------- #
# Fail closed: a source must PROVE it belongs to this league
# --------------------------------------------------------------------------- #
def test_six_team_board_is_refused_for_an_eight_team_league():
    snap = fixture("pre_draft")
    board = {"league_id": "1111111111", "season": 2026, "league_size": 6,
             "captured_at": "2026-08-29T00:00:00+00:00", "entries": [{"name": "X", "position": "RB"}]}
    reason = my_team.reject_source(board, snap, what="draft board")
    assert reason is not None
    assert "6-team" in reason and "8 teams" in reason


def test_twelve_team_mock_is_refused_for_an_eight_team_league():
    snap = fixture("pre_draft")
    board = {"league_id": "1111111111", "season": 2026, "league_size": 12,
             "captured_at": "2026-08-29T00:00:00+00:00", "entries": []}
    assert my_team.reject_source(board, snap, what="draft board") is not None


def test_placeholder_trade_scan_is_refused_for_missing_live_provenance():
    snap = fixture("pre_draft")
    placeholder = {"my_team": "Team1", "opportunities": [{"opponent": "Team4"}]}
    reason = my_team.reject_source(placeholder, snap, what="trade source")
    assert reason is not None
    assert "league_id" in reason


def test_real_teams_named_team_7_and_team_8_are_not_rejected():
    """The guard is provenance, not name shape — these managers are real."""
    snap = fixture("pre_draft")
    names = {t["name"] for t in snap["teams"]}
    assert {"Team 7", "Team 8"} <= names
    good = {"league_id": "1111111111", "season": 2026, "league_size": 8,
            "captured_at": "2026-08-29T02:55:45.998Z", "entries": []}
    assert my_team.reject_source(good, snap, what="draft board") is None


def test_prior_week_card_is_refused_as_current():
    """A week-1 card is structurally valid in week 2 — only the period catches it."""
    snap = fixture("post_draft")
    snap["league"]["scoring_period_id"] = 2
    week_one_card = {"league_id": "1111111111", "season": 2026, "league_size": 8,
                     "scoring_period": 1, "captured_at": "2026-08-29T00:00:00+00:00",
                     "entries": []}
    reason = my_team.reject_source(week_one_card, snap, what="lineup card")
    assert reason is not None
    assert "scoring_period 1" in reason and "scoring_period is 2" in reason

    snap["league"]["scoring_period_id"] = 1
    assert my_team.reject_source(week_one_card, snap, what="lineup card") is None


# --------------------------------------------------------------------------- #
# Freshness
# --------------------------------------------------------------------------- #
def test_stale_snapshot_degrades_every_actionable_section():
    payload = built("stale_snapshot")
    assert payload["freshness"]["state"] == "stale"
    for section in ("optimal_lineup", "start_sit", "waivers", "trades"):
        assert payload[section]["status"] == "no_current_pick", section
        assert payload[section]["reason"]
    assert payload["confidence"] == "none"


def test_fresh_snapshot_is_labelled_fresh():
    payload = built("pre_draft")
    assert payload["freshness"]["state"] == "fresh"


# --------------------------------------------------------------------------- #
# Lineup legality
# --------------------------------------------------------------------------- #
def test_optimal_lineup_is_legal_and_fills_flex_from_remaining_eligible():
    payload = built("post_draft")
    lineup = payload["optimal_lineup"]
    assert lineup["status"] == "ok"
    assert lineup["legal"] is True
    counts: dict[str, int] = {}
    for entry in lineup["starters"]:
        counts[entry["slot"]] = counts.get(entry["slot"], 0) + 1
    assert counts == {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "D/ST": 1, "K": 1}
    flex = next(e for e in lineup["starters"] if e["slot"] == "FLEX")
    assert flex["position"] in {"RB", "WR", "TE"}
    started = [e["player_id"] for e in lineup["starters"]]
    assert len(started) == len(set(started)), "a player was started in two slots"


def test_illegal_roster_refuses_to_emit_a_lineup():
    payload = built("illegal_roster")
    lineup = payload["optimal_lineup"]
    assert lineup["status"] == "no_current_pick"
    assert lineup["legal"] is False
    assert lineup["violations"], "an illegal roster must say what is wrong"


def test_bye_week_player_is_excluded_with_a_stated_reason():
    payload = built("bye")
    lineup = payload["optimal_lineup"]
    excluded = {e["player_id"]: e["reason"] for e in lineup["excluded"]}
    assert "bye" in " ".join(excluded.values()).lower()
    assert all(e["player_id"] not in excluded for e in lineup["starters"])


def test_injured_player_is_excluded_with_a_stated_reason():
    payload = built("injury")
    lineup = payload["optimal_lineup"]
    reasons = " ".join(e["reason"] for e in lineup["excluded"]).lower()
    assert "out" in reasons or "injur" in reasons


def test_unmatched_player_is_surfaced_not_scored_as_zero():
    payload = built("unmatched_player")
    unresolved = payload["unresolved_identities"]
    assert unresolved["count"] >= 1
    entry = unresolved["entries"][0]
    assert entry["espn_player_id"]
    assert entry["reason"]
    started = {e["player_id"] for e in payload["optimal_lineup"]["starters"]}
    assert entry["espn_player_id"] not in started


# --------------------------------------------------------------------------- #
# Start/sit, waivers, trades, shadow positions
# --------------------------------------------------------------------------- #
def test_start_sit_carries_delta_and_uncertainty():
    payload = built("post_draft")
    start_sit = payload["start_sit"]
    assert start_sit["status"] == "ok"
    for decision in start_sit["decisions"]:
        assert "projected_delta" in decision
        assert "uncertainty" in decision
        assert decision["uncertainty"]["p10_delta"] <= decision["uncertainty"]["p90_delta"]
        assert decision["confidence"] in {"low", "medium", "high"}
        assert decision["invalidation_trigger"]


def test_no_action_state_says_so_rather_than_inventing_a_move():
    payload = built("no_action")
    assert payload["start_sit"]["status"] == "no_current_pick"
    assert payload["start_sit"]["decisions"] == []
    assert "no" in payload["start_sit"]["reason"].lower()


def _recommendation(**over):
    """A real waivers.Recommendation, so this pins the actual integration."""
    from nflvalue.fantasy import waivers

    base = dict(
        add_espn_id=901, add_name="F901", add_position="RB",
        drop_espn_id=25, drop_name="R. Frost", drop_state="selected",
        status="ok", shadow_reason=None, confidence="medium",
        rationale="projects above the worst legal drop", 
        invalidation_trigger="a status change to either player",
        priority_implications={"waiver_mode": "inverse_standings", "position": 3},
        replacement_effect={"slot": "RB"}, opponent_opportunity_impact={},
        lineup_delta={"own_optimal_lineup_delta": 2.3}, lineup_delta_status="ok",
        data_timestamps={"pool_as_of": NOW}, degraded=False, faab=None,
    )
    base.update(over)
    return waivers.Recommendation(**base)


def test_waivers_no_longer_claim_the_engine_is_missing():
    """A planner now ships; saying otherwise would be a stale untruth."""
    payload = built("post_draft")
    waivers = payload["waivers"]
    assert waivers["status"] == "no_current_pick"
    assert "waiver engine" not in waivers["reason"].lower()
    assert "no waiver plan" in waivers["reason"].lower()
    assert waivers["targets"] == []


def test_waiver_plan_records_are_surfaced_with_add_drop_and_invalidation():
    payload = my_team.build(fixture("post_draft"), now=NOW,
                            waiver_plan=[_recommendation()])
    waivers = payload["waivers"]
    assert waivers["status"] == "ok"
    target = waivers["targets"][0]
    assert target["add"]["espn_player_id"] == 901
    assert target["drop"]["name"] == "R. Frost"
    assert target["drop_state"] == "selected"
    assert target["confidence"] == "medium"
    assert target["rationale"]
    assert target["invalidation_trigger"]
    assert target["recommendation_only"] is True


def test_planner_finding_nothing_is_reported_as_no_benefit_not_as_no_engine():
    payload = built("post_draft", waiver_plan=[])
    waivers = payload["waivers"]
    assert waivers["status"] == "no_current_pick"
    assert "no legal add" in waivers["reason"].lower()


def test_a_degraded_waiver_record_is_never_presented_as_a_recommendation():
    degraded = _recommendation(degraded=True, add_espn_id=None, add_name=None,
                               drop_espn_id=None, drop_name=None,
                               drop_state="not_required", status="degraded",
                               confidence="none", lineup_delta=None,
                               lineup_delta_status="unavailable",
                               rationale="free-agent pool is stale")
    payload = built("post_draft", waiver_plan=[degraded])
    waivers = payload["waivers"]
    assert waivers["status"] == "no_current_pick"
    assert waivers["targets"] == []
    assert "stale" in waivers["reason"].lower()


def test_shadow_waiver_candidates_stay_labelled_shadow():
    shadow = _recommendation(add_position="K", status="shadow",
                             shadow_reason="K is not promoted")
    payload = built("post_draft", waiver_plan=[shadow])
    target = payload["waivers"]["targets"][0]
    assert target["status"] == "shadow"
    assert target["shadow_reason"]


def test_trades_require_live_rosters_for_every_counterparty():
    payload = built("post_draft")
    trades = payload["trades"]
    assert trades["status"] == "no_current_pick"
    assert "roster" in trades["reason"].lower()
    assert trades["opportunities"] == []


def test_kicker_and_dst_sections_are_labelled_shadow():
    payload = built("post_draft")
    for key in ("kicker_shadow", "dst_shadow"):
        section = payload[key]
        assert section["shadow"] is True
        assert section["promoted"] is False
        assert "shadow" in section["label"].lower()


# --------------------------------------------------------------------------- #
# Contract shape
# --------------------------------------------------------------------------- #
def test_payload_is_versioned_and_carries_league_identity():
    payload = built("pre_draft")
    assert payload["schema_version"] == my_team.SCHEMA_VERSION
    league = payload["league"]
    for field in ("platform", "league_id", "season", "scoring_period", "team_id", "team_name"):
        assert league[field] is not None, field
    assert payload["sources"]
    assert "scoring_hash" in payload and "hash_source" in payload


def test_every_section_carries_rationale_and_invalidation():
    payload = built("post_draft")
    for key in my_team.ACTIONABLE_SECTIONS:
        section = payload[key]
        assert "rationale" in section, key
        assert "invalidation_trigger" in section, key
        assert "confidence" in section, key


def test_payload_is_json_serialisable_and_deterministic():
    first = json.dumps(built("post_draft"), sort_keys=True)
    second = json.dumps(built("post_draft"), sort_keys=True)
    assert first == second


@pytest.mark.parametrize("name", [
    "pre_draft", "draft_in_progress", "post_draft", "stale_snapshot",
    "unmatched_player", "bye", "injury", "illegal_roster", "no_action",
])
def test_every_fixture_builds_without_raising(name):
    payload = my_team.build(fixture(name), now=NOW)
    assert payload["schema_version"] == my_team.SCHEMA_VERSION
    json.dumps(payload)


# --------------------------------------------------------------------------- #
# Weekly-pipeline wiring: the publish must survive every snapshot state
# --------------------------------------------------------------------------- #
def _weekly():
    sys.path.insert(0, str(ROOT / "scripts"))
    import fantasy_weekly

    return fantasy_weekly


def _summaries(name=None):
    """Model output, as the weekly run produces it: one row per model player.

    The pipeline joins these onto the snapshot through the identity crosswalk,
    so a test that wants a fillable lineup has to supply projections for the
    players actually on the roster — which is the point. The two hand-written
    rows below stand in for a run that projected almost nobody.
    """
    import pandas as pd

    rows = [
        {"player_id": "00-0021", "mean": 18.9, "p10": 10.4, "p90": 27.4},
        {"player_id": "00-0031", "mean": 17.8, "p10": 9.8, "p90": 25.8},
    ]
    if name is not None:
        known = {r["player_id"] for r in rows}
        rows += [{"player_id": pid, **values}
                 for pid, values in model(name)["projections"].items()
                 if pid not in known]
    return pd.DataFrame(rows)


def test_missing_snapshot_directory_yields_no_current_pick_not_a_crash(tmp_path):
    weekly = _weekly()
    result = weekly.run_my_team(_summaries(), generated_at=NOW,
                                snapshot_dir=str(tmp_path / "absent"))
    assert result["status"] == "no_current_pick"
    assert "no ESPN league snapshot" in result["reason"]


def test_unreadable_snapshot_yields_no_current_pick_not_a_crash(tmp_path):
    (tmp_path / "broken.json").write_text("{not json")
    weekly = _weekly()
    result = weekly.run_my_team(_summaries(), generated_at=NOW, snapshot_dir=str(tmp_path))
    assert result["status"] == "no_current_pick"


def test_pipeline_builds_the_contract_from_a_snapshot_directory(tmp_path):
    import shutil

    shutil.copy(FIXTURES / "post_draft.json", tmp_path / "snap.json")
    weekly = _weekly()
    side = model("post_draft")
    result = weekly.run_my_team(
        _summaries("post_draft"), generated_at=NOW, snapshot_dir=str(tmp_path),
        espn_crosswalk={int(k): v for k, v in side["crosswalk"].items()})
    assert result["schema_version"] == my_team.SCHEMA_VERSION
    assert result["league"]["league_id"] == "1111111111"
    assert result["optimal_lineup"]["status"] == "ok"


def test_pipeline_without_a_crosswalk_says_so_rather_than_fielding_unknowns(tmp_path):
    """No identity map means no lineup — not a lineup of players it cannot name."""
    import shutil

    shutil.copy(FIXTURES / "post_draft.json", tmp_path / "snap.json")
    result = _weekly().run_my_team(_summaries(), generated_at=NOW, snapshot_dir=str(tmp_path))
    assert result["optimal_lineup"]["status"] == "no_current_pick"
    assert "crosswalk" in result["optimal_lineup"]["reason"]


def test_projections_join_by_player_id_and_absence_is_not_scored_as_zero():
    """Projections arrive beside the snapshot; a gap stays a gap.

    They used to be merged *into* the snapshot, which is how the builder came
    to read a schema the adapter never produced. Now the join happens through
    the identity crosswalk at build time, and a player the model did not
    project is excluded with a reason rather than scored as a zero.
    """
    side = model("post_draft")
    thinned = {k: v for k, v in side["projections"].items() if k != "00-0022"}
    payload = my_team.build(
        fixture("post_draft"), now=NOW,
        crosswalk={int(k): v for k, v in side["crosswalk"].items()},
        projections=thinned, byes=side["byes"])
    reasons = " ".join(e["reason"] for e in payload["optimal_lineup"]["excluded"])
    assert "no projection available" in reasons


def test_merging_the_model_into_the_snapshot_is_refused():
    with pytest.raises(NotImplementedError, match="projections="):
        my_team.attach_projections(fixture("post_draft"), {})


def test_latest_snapshot_is_chosen_by_its_own_retrieval_time(tmp_path):
    """mtime is a property of the file; retrieved_at is a property of the read."""
    import json as _json
    import os

    older = _json.loads((FIXTURES / "pre_draft.json").read_text())
    newer = _json.loads((FIXTURES / "post_draft.json").read_text())
    older["retrieved_at"] = "2026-08-29T02:00:00Z"
    newer["retrieved_at"] = "2026-08-29T09:00:00Z"
    (tmp_path / "a_old.json").write_text(_json.dumps(older))
    (tmp_path / "b_new.json").write_text(_json.dumps(newer))
    # Make the stale capture look freshest on disk, as a checkout would.
    os.utime(tmp_path / "a_old.json", (9_000_000_000, 9_000_000_000))
    os.utime(tmp_path / "b_new.json", (1_000_000_000, 1_000_000_000))

    loaded = my_team.load_latest_snapshot(str(tmp_path), now="2026-08-29T12:00:00Z")
    assert loaded["retrieved_at"] == "2026-08-29T09:00:00Z"
    assert loaded["draft"]["status"] == "post_draft"


def test_pipeline_passes_the_contract_through_so_hashes_are_the_contract_s(tmp_path):
    import shutil

    shutil.copy(FIXTURES / "post_draft.json", tmp_path / "snap.json")
    weekly = _weekly()
    contract = league_contract()
    result = weekly.run_my_team(_summaries(), generated_at=NOW,
                                snapshot_dir=str(tmp_path), contract=contract)
    assert result["scoring_hash"] == contract.scoring_hash
    assert result["roster_slot_hash"] == contract.roster_slot_hash


def test_pipeline_without_a_contract_uses_the_snapshots_embedded_hashes(tmp_path):
    import shutil

    shutil.copy(FIXTURES / "post_draft.json", tmp_path / "snap.json")
    result = _weekly().run_my_team(_summaries(), generated_at=NOW, snapshot_dir=str(tmp_path))
    # The snapshot carries the contract's own pair, so provenance survives a
    # run that was handed no contract object — and is still never invented.
    assert result["scoring_hash"] == fixture("post_draft")["hashes"]["scoring"]
    assert "embedded" in result["hash_source"]
    assert result["hash_reason"] is None
