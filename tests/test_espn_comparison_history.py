"""The public grading history: aggregates only, immutable, and independent.

`season_series` used to be derived from the row-level ledger every run, which
tied the published season history to a file that carries every ESPN and model
per-player projection. That is the file that may not be published, so the
series it fed could survive only by publishing it too, or by keeping the raw
rows somewhere a public job could read them.

`espn-comparison-history/1` breaks that tie: one immutable aggregate per graded
week, carrying nothing that could be a player, and durable in the public state
archive on its own.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nflvalue.fantasy import espn_compare  # noqa: E402

SEASON = 2026


def aggregate(**overrides) -> dict:
    """The shape `espn_compare._aggregate` produces, with nothing else in it."""
    fields = {
        "n": 300, "n_played": 280, "n_dnp": 20,
        "mae_espn": 5.412, "mae_model": 5.104,
        "mae_espn_incl_dnp": 5.9, "mae_model_incl_dnp": 5.6,
        "model_closer": 150, "espn_closer": 128, "ties": 2,
        "by_position": {"RB": {"n": 80, "mae_espn": 5.5, "mae_model": 5.2}},
    }
    fields.update(overrides)
    return fields


# --------------------------------------------------------------------------- #
# A1. Nothing that could be a player survives validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("smuggled", [
    {"rows": [{"player_id": "00-0011", "espn_pts": 12.0}]},
    {"player_name": "R. Bell"},
    {"player_id": "00-0011"},
    {"espn_player_id": 4262921},
    {"espn_pts": 12.0},
    {"model_pts": 13.0},
    {"actual_pts": 11.0},
    {"kind": "espn_comparison_ledger"},
    {"league_id": 1111111111},
    {"rosters": {"1": []}},
    {"members": [{"id": "x"}]},
])
def test_history_refuses_anything_that_could_be_a_player(smuggled):
    history = espn_compare.new_history(SEASON)
    history["weeks"]["1"] = {**espn_compare.public_aggregate(aggregate()),
                             "week": 1, "graded_at": "2026-09-12T00:00:00+00:00",
                             "projections_sha256": "a" * 64, **smuggled}
    with pytest.raises(espn_compare.HistoryRejected):
        espn_compare.validate_history(history)


# --------------------------------------------------------------------------- #
# A2. The history is loaded, validated and kept independently of the raw ledger
# --------------------------------------------------------------------------- #
def graded_history(*weeks: int, season: int = SEASON) -> dict:
    history = espn_compare.new_history(season)
    for week in weeks:
        espn_compare.record_graded_week(
            history, week=week, aggregate=aggregate(n=300 + week),
            graded_at=f"2026-09-{10 + week:02d}T00:00:00+00:00",
            projections_sha256=f"{week:064d}")
    return history


def test_a_missing_history_starts_empty_rather_than_raising(tmp_path):
    history = espn_compare.load_history(tmp_path / "nothing.json", SEASON)
    assert history == espn_compare.new_history(SEASON)


def test_history_round_trips_through_disk(tmp_path):
    path = tmp_path / "history.json"
    espn_compare.save_history(graded_history(1, 2), path)
    loaded = espn_compare.load_history(path, SEASON)
    assert sorted(loaded["weeks"]) == ["1", "2"]
    assert loaded["weeks"]["2"]["n"] == 302


def test_a_corrupt_history_fails_closed_rather_than_starting_fresh(tmp_path):
    """Silently starting over would erase a season of published history."""
    path = tmp_path / "history.json"
    espn_compare.save_history(graded_history(1), path)
    broken = path.read_text().replace('"n": 301', '"n": "three hundred"')
    path.write_text(broken)
    with pytest.raises(espn_compare.HistoryRejected):
        espn_compare.load_history(path, SEASON)


def test_prior_graded_weeks_survive_a_missing_raw_ledger(tmp_path):
    """The whole point: the public series does not depend on the private file."""
    history_path, ledger_path = tmp_path / "history.json", tmp_path / "ledger.json"
    espn_compare.save_history(graded_history(1, 2), history_path)

    ledger = espn_compare.load_ledger(ledger_path, SEASON)      # never written
    assert ledger["weeks"] == {}

    history = espn_compare.load_history(history_path, SEASON)
    espn_compare.sync_history_from_ledger(history, ledger)
    assert sorted(history["weeks"]) == ["1", "2"]
    assert [row["week"] for row in espn_compare.history_series(history)] == [1, 2]


def test_a_graded_ledger_week_is_appended_to_public_history():
    history = espn_compare.new_history(SEASON)
    ledger = espn_compare.new_ledger(SEASON)
    ledger["weeks"]["3"] = {
        "rows": [], "sources": [], "projections_sha256": "c" * 64,
        "grading": {"graded_at": "2026-09-20T00:00:00+00:00", "rows": [],
                    "aggregate": aggregate()},
    }
    added = espn_compare.sync_history_from_ledger(history, ledger)

    assert added == [3]
    espn_compare.validate_history(history)
    week = history["weeks"]["3"]
    assert week["mae_model"] == 5.104
    assert week["projections_sha256"] == "c" * 64
    assert "rows" not in week


# --------------------------------------------------------------------------- #
# A3. A graded week is immutable, and a disagreement is loud
# --------------------------------------------------------------------------- #
def test_re_recording_an_identical_week_is_a_no_op():
    history = graded_history(1)
    before = dict(history["weeks"]["1"])
    espn_compare.record_graded_week(
        history, week=1, aggregate=aggregate(n=301),
        graded_at="2026-09-11T00:00:00+00:00", projections_sha256=f"{1:064d}")
    assert history["weeks"]["1"] == before


def test_a_conflicting_rewrite_of_a_graded_week_fails_closed():
    history = graded_history(1)
    with pytest.raises(espn_compare.HistoryConflict):
        espn_compare.record_graded_week(
            history, week=1, aggregate=aggregate(n=301, mae_model=4.0),
            graded_at="2026-09-11T00:00:00+00:00", projections_sha256=f"{1:064d}")
    assert history["weeks"]["1"]["mae_model"] == 5.104


def test_a_ledger_that_disagrees_with_published_history_fails_closed():
    """Silently skipping the week would hide a regrade that moved a number."""
    history = graded_history(1)
    ledger = espn_compare.new_ledger(SEASON)
    ledger["weeks"]["1"] = {
        "rows": [], "sources": [], "projections_sha256": f"{1:064d}",
        "grading": {"graded_at": "2026-09-11T00:00:00+00:00", "rows": [],
                    "aggregate": aggregate(n=301, mae_espn=9.9)},
    }
    with pytest.raises(espn_compare.HistoryConflict):
        espn_compare.sync_history_from_ledger(history, ledger)


def test_a_ledger_that_agrees_with_published_history_adds_nothing():
    history = graded_history(1)
    ledger = espn_compare.new_ledger(SEASON)
    ledger["weeks"]["1"] = {
        "rows": [], "sources": [], "projections_sha256": f"{1:064d}",
        "grading": {"graded_at": "2026-09-11T00:00:00+00:00", "rows": [],
                    "aggregate": aggregate(n=301)},
    }
    assert espn_compare.sync_history_from_ledger(history, ledger) == []


def test_a_conflict_message_names_the_week_and_never_the_numbers():
    """The values on the private side of this boundary do not go in a log line."""
    history = graded_history(1)
    with pytest.raises(espn_compare.HistoryConflict) as caught:
        espn_compare.record_graded_week(
            history, week=1, aggregate=aggregate(n=301, mae_model=4.0),
            graded_at="2026-09-11T00:00:00+00:00", projections_sha256=f"{1:064d}")
    message = str(caught.value)
    assert "week" in message and "1" in message
    assert "4.0" not in message and "5.104" not in message


# --------------------------------------------------------------------------- #
# A4. What gets published reads the durable history, not the private ledger
# --------------------------------------------------------------------------- #
def test_build_payload_requires_the_durable_history():
    """No silent fall back to the ledger-derived series: that is the coupling."""
    with pytest.raises(TypeError):
        espn_compare.build_payload(
            espn_compare.new_ledger(SEASON), current_week=1,
            espn_provenance=None, identity_report=None)


def test_the_published_series_survives_an_empty_ledger():
    payload = espn_compare.build_payload(
        espn_compare.new_ledger(SEASON), current_week=3,
        espn_provenance=None, identity_report=None, history=graded_history(1, 2))

    assert [row["week"] for row in payload["season_series"]] == [1, 2]
    assert payload["latest_graded_week"]["week"] == 2
    assert payload["current_week_rows"] == []


def test_the_published_series_carries_no_row_level_field():
    payload = espn_compare.build_payload(
        espn_compare.new_ledger(SEASON), current_week=3,
        espn_provenance=None, identity_report=None, history=graded_history(1))
    for row in payload["season_series"]:
        for forbidden in ("rows", "player_id", "player_name", "espn_pts",
                          "model_pts", "actual_pts"):
            assert forbidden not in row


def test_the_history_passes_the_public_boundary_guard():
    from nflvalue.fantasy import private_boundary

    private_boundary.assert_public_safe(graded_history(1, 2), what="grading history")


# --------------------------------------------------------------------------- #
# A6. The history's only durable copy is a public artifact, so it needs a floor
# --------------------------------------------------------------------------- #
def test_a_season_rollover_archives_the_prior_season_instead_of_overwriting(tmp_path):
    """The file is gitignored; the release asset is its only copy."""
    path = tmp_path / "history.json"
    espn_compare.save_history(graded_history(1, 2, 3), path)

    rolled = espn_compare.load_history(path, SEASON + 1)
    assert rolled["weeks"] == {}
    assert rolled["season"] == SEASON + 1

    archived = path.with_name(f"history.{SEASON}.json")
    assert archived.exists(), "the prior season was dropped on the floor"
    assert sorted(json.loads(archived.read_text())["weeks"]) == ["1", "2", "3"]


def test_saving_fewer_weeks_over_a_published_season_fails_closed(tmp_path):
    """One transient restore failure must not republish an empty season."""
    path = tmp_path / "history.json"
    espn_compare.save_history(graded_history(1, 2, 3), path)

    with pytest.raises(espn_compare.HistoryConflict):
        espn_compare.save_history(espn_compare.new_history(SEASON), path)
    assert sorted(json.loads(path.read_text())["weeks"]) == ["1", "2", "3"]

    # Growing is fine; that is the normal weekly path.
    espn_compare.save_history(graded_history(1, 2, 3, 4), path)
    assert sorted(json.loads(path.read_text())["weeks"]) == ["1", "2", "3", "4"]


def test_a_history_for_a_different_season_may_replace_the_file(tmp_path):
    path = tmp_path / "history.json"
    espn_compare.save_history(graded_history(1, 2), path)
    espn_compare.save_history(graded_history(1, season=SEASON + 1), path)
    assert json.loads(path.read_text())["season"] == SEASON + 1


# --------------------------------------------------------------------------- #
# A7. The allow-list checks values, not only key names
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("smuggled", [
    {"n": "R. Bell (00-0011)"},
    {"mae_espn": "espn_pts=18.4"},
    {"mae_model": {"player_id": "00-0011"}},
    {"n": True},
])
def test_a_position_group_value_that_is_not_a_number_is_refused(smuggled):
    history = espn_compare.new_history(SEASON)
    entry = {**espn_compare.public_aggregate(aggregate()), "week": 1,
             "graded_at": "2026-09-12T00:00:00+00:00", "projections_sha256": "a" * 64}
    entry["by_position"]["RB"].update(smuggled)
    history["weeks"]["1"] = entry
    with pytest.raises(espn_compare.HistoryRejected):
        espn_compare.validate_history(history)


def test_a_rejected_week_does_not_survive_in_the_callers_history():
    """A caught rejection must not leave the bad week behind in the object."""
    history = espn_compare.new_history(SEASON)
    with pytest.raises(espn_compare.HistoryRejected):
        espn_compare.record_graded_week(
            history, week=1, aggregate=aggregate(n="three hundred"),
            graded_at="2026-09-12T00:00:00+00:00", projections_sha256="a" * 64)
    assert history["weeks"] == {}


def test_a_grading_block_with_no_aggregate_raises_the_modules_own_error():
    history = espn_compare.new_history(SEASON)
    ledger = espn_compare.new_ledger(SEASON)
    ledger["weeks"]["1"] = {"rows": [], "sources": [], "projections_sha256": "a" * 64,
                            "grading": {"graded_at": "2026-09-12T00:00:00+00:00"}}
    with pytest.raises(espn_compare.HistoryRejected):
        espn_compare.sync_history_from_ledger(history, ledger)


# --------------------------------------------------------------------------- #
# A8. A grading-bookkeeping problem must not take the projections down
# --------------------------------------------------------------------------- #
def test_a_corrupt_history_degrades_the_comparison_and_leaves_the_file_alone(tmp_path):
    """Failing closed on WRITING history is right; failing the whole site is not."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import fantasy_weekly

    path = tmp_path / "history.json"
    espn_compare.save_history(graded_history(1, 2), path)
    original = path.read_text()
    path.write_text(original.replace('"n": 301', '"n": "three hundred"'))
    corrupt = path.read_text()

    history, available, reason = fantasy_weekly.load_history_or_degrade(path, SEASON)

    assert available is False
    assert reason and "history" in reason.lower()
    assert history["weeks"] == {}
    # Untouched on disk: the next run, with a good restore, still has the season.
    assert path.read_text() == corrupt


def test_a_readable_history_loads_normally(tmp_path):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import fantasy_weekly

    path = tmp_path / "history.json"
    espn_compare.save_history(graded_history(1, 2), path)
    history, available, reason = fantasy_weekly.load_history_or_degrade(path, SEASON)
    assert available is True and reason is None
    assert sorted(history["weeks"]) == ["1", "2"]


# ---------------------------------------------------------------------------
# Registered freeze lever 3 (ESPN shrinkage blend) — measurement arm
# ---------------------------------------------------------------------------

def _blend_rows():
    """Two played rows and one DNP, with model and ESPN on opposite sides."""
    return [
        {"player_id": "00-1", "player_name": "A", "position": "RB", "espn_pts": 10.0,
         "model_pts": 20.0, "actual_pts": 10.0, "played": True,
         "abs_err_espn": 0.0, "abs_err_model": 10.0, "who_was_closer": "espn"},
        {"player_id": "00-2", "player_name": "B", "position": "WR", "espn_pts": 4.0,
         "model_pts": 8.0, "actual_pts": 8.0, "played": True,
         "abs_err_espn": 4.0, "abs_err_model": 0.0, "who_was_closer": "model"},
        {"player_id": "00-3", "player_name": "C", "position": "WR", "espn_pts": 6.0,
         "model_pts": 10.0, "actual_pts": 0.0, "played": False,
         "abs_err_espn": 6.0, "abs_err_model": 10.0, "who_was_closer": "espn"},
    ]


def test_blend_arms_endpoints_reproduce_the_two_incumbents():
    """w=0 must equal the model alone and w=1 the ESPN column, or the arm is not measuring what it claims."""
    arms = {arm["w_espn"]: arm for arm in espn_compare._blend_arms(_blend_rows())}
    assert set(arms) == set(espn_compare.BLEND_WEIGHTS)
    assert arms[0.0]["mae_played"] == 5.0      # (10 + 0) / 2
    assert arms[1.0]["mae_played"] == 2.0      # (0 + 4) / 2
    assert arms[0.5]["mae_played"] == 3.5      # (|15-10| + |6-8|) / 2 -- strictly between
    assert arms[0.0]["mae_incl_dnp"] == pytest.approx(20.0 / 3)
    assert arms[1.0]["mae_incl_dnp"] == pytest.approx(10.0 / 3)
    assert arms[1.0]["by_position"] == {"RB": 0.0, "WR": 4.0}


def test_blend_arms_never_reach_the_published_history():
    """The arm is private research; the published allow-list must drop it."""
    aggregate = espn_compare._aggregate(_blend_rows())
    assert "blend_arms" in aggregate
    assert "blend_arms" not in espn_compare.public_aggregate(aggregate)
    history = espn_compare.new_history(2026)
    espn_compare.record_graded_week(
        history, week=3, aggregate=aggregate, graded_at="2026-09-28T12:00:00+00:00",
        projections_sha256="f" * 64)
    espn_compare.validate_history(history)
    assert "blend_arms" not in history["weeks"]["3"]


def test_blend_weight_grid_is_frozen_with_its_preregistration():
    """Changing the grid after a week is graded is a protocol violation; pin it."""
    assert espn_compare.BLEND_WEIGHTS == (0.0, 0.25, 0.5, 0.75, 1.0)
    prereg = ROOT / "docs" / "prereg" / "LEVER3_ESPN_BLEND_2026-09-22.md"
    assert prereg.is_file()
    text = prereg.read_text()
    assert "REGISTERED, NOT RUN" in text
    assert "6 graded weeks" in text
