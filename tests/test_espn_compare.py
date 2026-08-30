"""Data-independent tests for the ESPN external-challenger comparison.

Covers the four protocol-critical behaviours:
scoring-basis conversion, identity mapping (unmatched reporting), snapshot /
ledger immutability (regrading never alters stored projections), and the
who-was-closer grader.  No network, no historical parquet files.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from nflvalue.fantasy import espn_compare
from nflvalue.fantasy.config import ScoringRules
from nflvalue.fantasy.dashboard import render_fantasy_dashboard
from nflvalue.sources import espn_projections

PPR = ScoringRules.preset("ppr")


# ---------------------------------------------------------------------------
# Synthetic ESPN payload helpers
# ---------------------------------------------------------------------------

GIBBS_RAW = {  # shape mirrors the live kona_player_info raw stat map
    "23": 17.27, "24": 91.94, "25": 0.77, "26": 0.02,
    "42": 35.75, "43": 0.23, "44": 0.01, "53": 4.26, "58": 5.09, "72": 0.06,
}


def espn_player(espn_id, name, position_id, week, raw_stats, applied=None, season=2026):
    return {
        "player": {
            "id": espn_id,
            "fullName": name,
            "defaultPositionId": position_id,
            "proTeamId": 8,
            "stats": [
                {  # decoy: season split, must be ignored
                    "seasonId": season, "scoringPeriodId": 0,
                    "statSourceId": 1, "statSplitTypeId": 0,
                    "appliedTotal": 999.0, "stats": {},
                },
                {  # decoy: actuals, must be ignored
                    "seasonId": season, "scoringPeriodId": week,
                    "statSourceId": 0, "statSplitTypeId": 1,
                    "appliedTotal": 999.0, "stats": {},
                },
                {
                    "seasonId": season, "scoringPeriodId": week,
                    "statSourceId": 1, "statSplitTypeId": 1,
                    "appliedTotal": applied, "stats": raw_stats,
                },
            ],
        }
    }


def synthetic_payload(count=160, week=1):
    players = []
    for index in range(count):
        players.append(
            espn_player(1000 + index, f"Player {index:03d}", (index % 4) + 1, week, dict(GIBBS_RAW))
        )
    return {"players": players}


# ---------------------------------------------------------------------------
# Scoring-basis conversion
# ---------------------------------------------------------------------------

def test_stat_id_conversion_scores_full_ppr_with_the_models_scorer():
    raw = {"24": 100.0, "25": 1.0, "53": 5.0, "42": 50.0, "43": 1.0, "72": 1.0}
    # 100*0.1 + 1*6 + 5*1 + 50*0.1 + 1*6 - 2 = 30.0 under full PPR
    assert espn_projections.rescore_full_ppr(raw, PPR) == pytest.approx(30.0)


def test_conversion_reception_weight_distinguishes_ppr_from_standard():
    raw = {"53": 8.0}
    assert espn_projections.rescore_full_ppr(raw, PPR) == pytest.approx(8.0)
    assert espn_projections.rescore_full_ppr(raw, ScoringRules.preset("standard")) == pytest.approx(0.0)


def test_conversion_ignores_unmapped_espn_stat_ids():
    # ids like 39 (rushYds/game) and 210 (games played) are display stats, not events
    raw = {"24": 50.0, "39": 5.32, "210": 1.0, "212": 4.85}
    assert espn_projections.rescore_full_ppr(raw, PPR) == pytest.approx(5.0)


def test_snapshot_records_applied_vs_rescored_delta_explicitly():
    week = 3
    payload = {"players": [
        espn_player(1000 + i, f"P{i}", (i % 4) + 1, week, dict(GIBBS_RAW), applied=20.0)
        for i in range(160)
    ]}
    players = espn_projections.parse_players(payload, season=2026, week=week, rules=PPR)
    snapshot = espn_projections.build_snapshot(
        players, season=2026, week=week, endpoint="test://espn", auth_mode="public", rules=PPR
    )
    rescored = espn_projections.rescore_full_ppr(GIBBS_RAW, PPR)
    assert snapshot["scoring"]["rescored_players"] == 160
    assert snapshot["scoring"]["rescored_vs_applied_mean_abs_delta"] == pytest.approx(
        abs(rescored - 20.0)
    )
    # all five protocol-required provenance fields are present
    assert snapshot["source"]["name"]
    assert snapshot["retrieved_at"]
    assert snapshot["scoring"]["rules"] == PPR.to_dict()
    assert snapshot["coverage"]["players"] == 160
    assert snapshot["redistribution_rights"]


def test_applied_total_fallback_is_labelled_when_raw_stats_absent():
    week = 1
    payload = synthetic_payload(week=week)
    payload["players"].append(espn_player(9999, "Applied Only", 2, week, {}, applied=11.5))
    players = espn_projections.parse_players(payload, season=2026, week=week, rules=PPR)
    fallback = [p for p in players if p["espn_id"] == 9999]
    assert fallback and fallback[0]["points_basis"] == "espn_applied_total"
    assert fallback[0]["espn_ppr_points"] == pytest.approx(11.5)


# ---------------------------------------------------------------------------
# Fail-loud pull contract
# ---------------------------------------------------------------------------

def test_empty_payload_raises():
    with pytest.raises(espn_projections.EspnProjectionsError, match="no players"):
        espn_projections.parse_players({"players": []}, season=2026, week=1)


def test_thin_pull_raises_instead_of_returning_quietly():
    with pytest.raises(espn_projections.EspnProjectionsError, match="thin/empty"):
        espn_projections.parse_players(synthetic_payload(count=20), season=2026, week=1)


def test_wrong_week_projections_do_not_count():
    payload = synthetic_payload(count=160, week=7)
    with pytest.raises(espn_projections.EspnProjectionsError):
        espn_projections.parse_players(payload, season=2026, week=8)


# ---------------------------------------------------------------------------
# Snapshot immutability
# ---------------------------------------------------------------------------

def make_snapshot(week=1, retrieved_at="2026-09-09T12:00:00+00:00"):
    players = espn_projections.parse_players(
        synthetic_payload(week=week), season=2026, week=week, rules=PPR
    )
    return espn_projections.build_snapshot(
        players, season=2026, week=week, endpoint="test://espn",
        auth_mode="public", retrieved_at=retrieved_at, rules=PPR,
    )


def test_snapshot_write_refuses_overwrite(tmp_path):
    snapshot = make_snapshot()
    path = espn_projections.write_snapshot(snapshot, tmp_path)
    assert path.exists()
    with pytest.raises(FileExistsError, match="immutable"):
        espn_projections.write_snapshot(snapshot, tmp_path)


def test_snapshot_load_detects_tampering(tmp_path):
    snapshot = make_snapshot()
    path = espn_projections.write_snapshot(snapshot, tmp_path)
    assert espn_projections.load_snapshot(path)["players_sha256"] == snapshot["players_sha256"]
    tampered = json.loads(path.read_text())
    tampered["players"][0]["espn_ppr_points"] += 5.0
    path.write_text(json.dumps(tampered))
    with pytest.raises(espn_projections.EspnProjectionsError, match="hash"):
        espn_projections.load_snapshot(path)


# ---------------------------------------------------------------------------
# Identity mapping
# ---------------------------------------------------------------------------

def rosters_frame():
    return pd.DataFrame({
        "season": [2026, 2026, 2026, 2025],
        "week": [1, 1, 1, 18],
        "gsis_id": ["00-001", "00-002", "00-003", "00-009"],
        "espn_id": [1000, 1001, 1002, 9009],
        "position": ["RB", "WR", "QB", "TE"],
        "team": ["DET", "DET", "DET", "DET"],
    })


def test_identity_map_requires_espn_id_column():
    with pytest.raises(ValueError, match="espn_id"):
        espn_compare.build_identity_map(
            pd.DataFrame({"season": [2026], "gsis_id": ["00-001"]}), 2026
        )


def test_unmatched_players_are_reported_never_dropped():
    identity = espn_compare.build_identity_map(rosters_frame(), 2026)
    assert 9009 not in set(identity["espn_id"])  # other season excluded
    espn_players = [
        {"espn_id": 1000, "player_name": "Matched Model", "position": "RB"},
        {"espn_id": 1001, "player_name": "Not Projected", "position": "WR"},
        {"espn_id": 5555, "player_name": "No Crosswalk", "position": "QB"},
    ]
    matched, report = espn_compare.match_players(espn_players, identity, {"00-001"})
    assert matched == {1000: "00-001"}
    assert report["espn_players"] == 3
    assert report["matched"] == 1
    assert report["coverage_pct"] == pytest.approx(33.3)
    assert report["unmatched_no_crosswalk_names"] == ["No Crosswalk"]
    assert report["unmatched_model_not_projected_names"] == ["Not Projected"]


# ---------------------------------------------------------------------------
# Ledger: prospective locking, grading, immutability
# ---------------------------------------------------------------------------

KICKOFFS = {"2026_01_DET_GB": "2026-09-13T17:00:00+00:00"}


def record_args(espn_retrieved, model_generated, espn_pts=20.0, model_pts=18.0):
    return dict(
        week=1,
        espn_players=[{
            "espn_id": 1000, "player_name": "Matched Model", "position": "RB",
            "team": "DET", "espn_ppr_points": espn_pts, "points_basis": "full_ppr_rescored",
        }],
        espn_retrieved_at=espn_retrieved,
        espn_snapshot_sha256="0" * 64,
        matched={1000: "00-001"},
        model_points={"00-001": model_pts},
        model_meta={"00-001": {"team": "DET"}},
        model_generated_at=model_generated,
        player_games={"00-001": "2026_01_DET_GB"},
        kickoffs_utc=KICKOFFS,
    )


def test_prospective_rows_recorded_and_post_kickoff_refresh_locked():
    ledger = espn_compare.new_ledger(2026)
    entry = espn_compare.record_week(
        ledger, **record_args("2026-09-10T12:00:00+00:00", "2026-09-10T13:00:00+00:00")
    )
    assert len(entry["rows"]) == 1
    assert entry["rows"][0]["espn_pts"] == pytest.approx(20.0)
    # pre-kickoff refresh updates the row
    espn_compare.record_week(
        ledger, **record_args("2026-09-13T12:00:00+00:00", "2026-09-13T12:30:00+00:00",
                              espn_pts=22.0)
    )
    assert entry["rows"][0]["espn_pts"] == pytest.approx(22.0)
    # post-kickoff refresh cannot touch the locked row
    espn_compare.record_week(
        ledger, **record_args("2026-09-13T18:00:00+00:00", "2026-09-13T18:30:00+00:00",
                              espn_pts=99.0)
    )
    assert entry["rows"][0]["espn_pts"] == pytest.approx(22.0)
    assert entry["sources"][-1]["skipped_post_kickoff"] == 1


def test_row_requires_both_sides_before_kickoff():
    ledger = espn_compare.new_ledger(2026)
    entry = espn_compare.record_week(  # model snapshot AFTER kickoff: no row
        ledger, **record_args("2026-09-10T12:00:00+00:00", "2026-09-13T18:00:00+00:00")
    )
    assert entry["rows"] == []
    assert entry["sources"][-1]["skipped_post_kickoff"] == 1


def test_who_was_closer_all_outcomes():
    assert espn_compare.who_was_closer(20.0, 18.0, 17.0) == "model"
    assert espn_compare.who_was_closer(20.0, 18.0, 21.0) == "espn"
    assert espn_compare.who_was_closer(16.0, 18.0, 17.0) == "tie"


def graded_ledger():
    ledger = espn_compare.new_ledger(2026)
    espn_compare.record_week(
        ledger, **record_args("2026-09-10T12:00:00+00:00", "2026-09-10T13:00:00+00:00")
    )
    grading = espn_compare.grade_week(
        ledger, week=1, actual_points={"00-001": 17.0}, graded_at="2026-09-15T12:00:00+00:00"
    )
    return ledger, grading


def test_grading_never_alters_stored_projections(tmp_path):
    ledger, grading = graded_ledger()
    entry = ledger["weeks"]["1"]
    hash_before_grading = entry["projections_sha256"]
    assert espn_compare._rows_sha256(entry["rows"]) == hash_before_grading
    assert entry["rows"][0]["espn_pts"] == pytest.approx(20.0)  # untouched
    assert grading["rows"][0]["who_was_closer"] == "model"  # |18-17| < |20-17|
    # round-trips through disk with the hash re-verified on load
    path = tmp_path / "ledger.json"
    espn_compare.save_ledger(ledger, path)
    reloaded = espn_compare.load_ledger(path, 2026)
    assert reloaded["weeks"]["1"]["projections_sha256"] == hash_before_grading


def test_regrading_is_idempotent_and_returns_the_original_block():
    ledger, grading = graded_ledger()
    again = espn_compare.grade_week(
        ledger, week=1, actual_points={"00-001": 999.0}  # would change everything if applied
    )
    assert again == grading


def test_recording_into_a_graded_week_is_refused():
    ledger, _ = graded_ledger()
    with pytest.raises(ValueError, match="immutable"):
        espn_compare.record_week(
            ledger, **record_args("2026-09-13T12:00:00+00:00", "2026-09-13T12:30:00+00:00")
        )


def test_grading_cannot_backfill_an_unrecorded_week():
    ledger = espn_compare.new_ledger(2026)
    with pytest.raises(ValueError, match="backfill"):
        espn_compare.grade_week(ledger, week=5, actual_points={"00-001": 10.0})


def test_tampered_ledger_rows_fail_the_hash_check_on_load(tmp_path):
    ledger, _ = graded_ledger()
    path = tmp_path / "ledger.json"
    espn_compare.save_ledger(ledger, path)
    raw = json.loads(path.read_text())
    raw["weeks"]["1"]["rows"][0]["model_pts"] = 0.0
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="immutable"):
        espn_compare.load_ledger(path, 2026)


def test_dnp_players_are_flagged_and_reported_separately():
    ledger = espn_compare.new_ledger(2026)
    args = record_args("2026-09-10T12:00:00+00:00", "2026-09-10T13:00:00+00:00")
    args["espn_players"].append({
        "espn_id": 1001, "player_name": "Sat Out", "position": "WR",
        "team": "DET", "espn_ppr_points": 12.0, "points_basis": "full_ppr_rescored",
    })
    args["matched"][1001] = "00-002"
    args["model_points"]["00-002"] = 10.0
    args["player_games"]["00-002"] = "2026_01_DET_GB"
    espn_compare.record_week(ledger, **args)
    grading = espn_compare.grade_week(ledger, week=1, actual_points={"00-001": 17.0})
    aggregate = grading["aggregate"]
    assert aggregate["n"] == 2 and aggregate["n_played"] == 1 and aggregate["n_dnp"] == 1
    assert aggregate["mae_model"] == pytest.approx(1.0)  # played-only headline
    assert aggregate["mae_espn"] == pytest.approx(3.0)
    assert aggregate["mae_model_incl_dnp"] == pytest.approx((1.0 + 10.0) / 2)


def test_payload_ranks_by_abs_delta_and_builds_season_series():
    ledger, _ = graded_ledger()
    payload = espn_compare.build_payload(
        ledger, current_week=1, espn_provenance={"retrieved_at": "t"},
        identity_report={"matched": 1, "espn_players": 1},
    )
    assert payload["current_week_rows"][0]["abs_delta_rank"] == 1
    assert payload["current_week_rows"][0]["delta"] == pytest.approx(-2.0)
    assert payload["season_series"][0]["week"] == 1
    assert payload["latest_graded_week"]["mae_model"] == pytest.approx(1.0)
    assert "not a betting edge" in payload["disclaimer"]


# ---------------------------------------------------------------------------
# Kickoff parsing and dashboard rendering
# ---------------------------------------------------------------------------

def test_kickoffs_convert_eastern_to_utc():
    schedules = pd.DataFrame({
        "season": [2026], "week": [1], "game_id": ["2026_01_DET_GB"],
        "home_team": ["GB"], "away_team": ["DET"],
        "gameday": ["2026-09-13"], "gametime": ["13:00"],
    })
    kickoffs = espn_compare.game_kickoffs_utc(schedules, 2026, 1)
    assert kickoffs["2026_01_DET_GB"] == "2026-09-13T17:00:00+00:00"  # EDT = UTC-4


def test_dashboard_renders_espn_section_with_honest_labels(tmp_path):
    ledger, _ = graded_ledger()
    payload = espn_compare.build_payload(
        ledger, current_week=1,
        espn_provenance={
            "retrieved_at": "2026-09-10T12:00:00+00:00",
            "source": {"name": "ESPN Fantasy API"},
            "scoring": {"rescored_vs_applied_mean_abs_delta": 0.02},
        },
        identity_report={
            "matched": 1, "espn_players": 1, "coverage_pct": 100.0,
            "unmatched_no_crosswalk_count": 0, "unmatched_model_not_projected_count": 0,
        },
    )
    summaries = pd.DataFrame([{
        "player_id": "00-001", "player_name": "Matched Model", "position": "RB",
        "team": "DET", "mean": 18.0, "median": 17.5, "event_simulator_mean": 17.0,
        "p10": 8.0, "p90": 28.0, "prob_15_plus": 0.6, "prob_20_plus": 0.4,
        "availability_probability": 0.95, "component_model_disagreement": False,
    }])
    target = tmp_path / "fantasy.html"
    render_fantasy_dashboard(
        summaries, target, season=2026, week=1,
        generated_at="2026-09-10T13:00:00+00:00", espn_comparison=payload,
    )
    document = target.read_text()
    assert "ESPN vs model" in document
    assert "2026-09-10T12:00:00+00:00" in document          # retrieval timestamp label
    assert "not a betting edge" in document                  # honesty line
    assert "Season grading" in document
    # the pre-existing projections table is intact
    assert "fantasy projections" in document and "Matched Model" in document


def test_dashboard_renders_honest_failure_when_espn_unavailable(tmp_path):
    ledger = espn_compare.new_ledger(2026)
    payload = espn_compare.build_payload(
        ledger, current_week=1, espn_provenance=None, identity_report=None,
        status="espn_unavailable", error="SourceTimeout: espn_projections",
    )
    summaries = pd.DataFrame([{
        "player_id": "00-001", "player_name": "Matched Model", "position": "RB",
        "team": "DET", "mean": 18.0, "median": 17.5, "event_simulator_mean": 17.0,
        "p10": 8.0, "p90": 28.0, "prob_15_plus": 0.6, "prob_20_plus": 0.4,
        "availability_probability": 0.95, "component_model_disagreement": False,
    }])
    target = tmp_path / "fantasy.html"
    render_fantasy_dashboard(
        summaries, target, season=2026, week=1,
        generated_at="now", espn_comparison=payload,
    )
    document = target.read_text()
    assert "unavailable this run" in document
    assert "SourceTimeout" in document
    assert "No comparison is fabricated" in document
