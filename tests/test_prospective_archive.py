"""Immutable, timestamped archive of the weekly prediction payload.

The 2026 freeze (docs/PROTOCOL_FREEZE_2026.md §2) says a prediction counts
only if its decision snapshot predates kickoff. Until now the only copy of the
weekly projection snapshot was a single overwritten file and a Pages URL --
neither of which can show *when* a forecast existed relative to *which* game.
These tests pin the contract of the archive that fixes that:

* one immutable entry per run, never overwritten, hash-verified on load;
* an append-only index, and a `latest` pointer kept SEPARATE from history;
* eligibility decided per GAME against its own kickoff, with strict clocks,
  so a partial slate is labelled partial rather than the week getting one
  label;
* a run after a deadline labels that cohort missed -- it never reconstructs
  a forecast and calls it prospective;
* nothing about a league, a roster or an ESPN row is in it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue.fantasy import private_boundary
from nflvalue.fantasy import prospective_archive as archive
from nflvalue.fantasy.config import ScoringRules

KICKOFFS = {
    "2026_01_NE_SEA": "2026-09-10T00:20:00+00:00",   # Wednesday night ET
    "2026_01_SF_LA": "2026-09-11T00:35:00+00:00",    # Thursday night ET
    "2026_01_CHI_CAR": "2026-09-13T17:00:00+00:00",  # Sunday early window
}


def snapshot(generated_at: str, information_as_of: str | None = None) -> dict:
    players = [
        {"player_id": "00-0001", "player_name": "A", "position": "QB", "team": "NE",
         "opponent_team": "SEA", "game_id": "2026_01_NE_SEA", "correlation_group": "x",
         "availability_probability": 0.9, "components": {}},
        {"player_id": "00-0002", "player_name": "B", "position": "RB", "team": "SF",
         "opponent_team": "LA", "game_id": "2026_01_SF_LA", "correlation_group": "y",
         "availability_probability": 0.9, "components": {}},
        {"player_id": "00-0003", "player_name": "C", "position": "WR", "team": "CHI",
         "opponent_team": "CAR", "game_id": "2026_01_CHI_CAR", "correlation_group": "z",
         "availability_probability": 0.9, "components": {}},
        {"player_id": "00-0004", "player_name": "D", "position": "TE", "team": "DEN",
         "opponent_team": "KC", "game_id": "2026_01_DEN_KC", "correlation_group": "w",
         "availability_probability": 0.9, "components": {}},
    ]
    return {
        "schema_version": 1, "season": 2026, "week": 1,
        "generated_at": generated_at,
        "information_as_of": information_as_of or generated_at,
        "model_version": "abc123",
        "players_canonical_csv_sha256": "f" * 64,
        "simulation": {"simulations": 10000, "random_seed": 6304527},
        # shaped like the real document: the source manifest's quality report
        # is keyed by nflverse TABLE names, one of which is "rosters"
        "source_manifest": {"retrieved_at": generated_at, "seasons": [2025, 2026],
                            "quality": {"tables": {"rosters": {"rows": 10}}}},
        "players": players,
    }


def summaries() -> pd.DataFrame:
    return pd.DataFrame({
        "player_id": ["00-0001", "00-0002", "00-0003", "00-0004"],
        "mean": [18.2, 14.1, 11.9, 6.3],
        "sd": [7.0, 6.0, 6.5, 4.0],
        "p10": [9.0, 6.0, 4.0, 1.0],
        "p50": [17.5, 13.4, 11.0, 5.5],
        "p90": [28.0, 23.0, 21.0, 12.0],
        "availability_probability": [0.9, 0.9, 0.9, 0.9],
    })


def build(generated_at: str, **kwargs) -> dict:
    return archive.build_entry(
        snapshot(generated_at, kwargs.pop("information_as_of", None)),
        summaries(),
        scoring=ScoringRules.preset("ppr"), scoring_preset="ppr",
        kickoffs_utc=KICKOFFS,
        run_context=kwargs.pop("run_context", archive.run_context_from_env({})),
        archived_at=kwargs.pop("archived_at", "2026-09-08T20:00:00+00:00"),
    )


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

def test_entry_carries_identity_rules_clocks_deadlines_and_hash():
    entry = build("2026-09-08T19:00:00+00:00")
    assert entry["kind"] == archive.ARCHIVE_KIND
    assert (entry["season"], entry["week"]) == (2026, 1)
    assert entry["model_version"] == "abc123"
    assert entry["scoring"]["preset"] == "ppr"
    assert entry["scoring"]["rules"]["reception"] == 1.0
    clocks = entry["clocks"]
    assert clocks["forecast_generated_at"] == "2026-09-08T19:00:00+00:00"
    assert clocks["information_as_of"] == "2026-09-08T19:00:00+00:00"
    assert clocks["archived_at"] == "2026-09-08T20:00:00+00:00"
    # unknown provenance stays unknown: nothing was scheduled here
    assert clocks["scheduled_for"] is None
    assert entry["run_context"]["event"] is None
    for game_id, kickoff in KICKOFFS.items():
        deadline = entry["deadlines"][game_id]
        assert deadline["kickoff_utc"] == kickoff
        assert deadline["decision_deadline_utc"] == kickoff
    assert entry["projection_snapshot"]["players_canonical_csv_sha256"] == "f" * 64
    assert len(entry["payload_sha256"]) == 64
    assert {row["player_id"] for row in entry["players"]} == {
        "00-0001", "00-0002", "00-0003", "00-0004"}
    scored = next(row for row in entry["players"] if row["player_id"] == "00-0001")
    assert scored["mean"] == pytest.approx(18.2)
    assert scored["p10"] == pytest.approx(9.0)
    assert scored["p90"] == pytest.approx(28.0)


def test_entry_is_public_safe_even_though_the_snapshot_manifest_names_a_rosters_table():
    """The first end-to-end run failed this: nesting the whole snapshot
    document put `source_manifest.quality.tables.rosters` inside the entry and
    the public guard, rightly, refused the key. The document is archived as
    its own file instead and the entry carries only its hash and path."""
    entry = build("2026-09-08T19:00:00+00:00")
    private_boundary.assert_public_safe(entry, what="archive entry")
    assert "espn" not in json.dumps(entry).lower()
    assert "document" not in entry["projection_snapshot"]
    assert entry["projection_snapshot"]["document_path"].startswith("2026/snapshot_2026_wk01_")


def test_the_snapshot_document_is_archived_verbatim_beside_the_entry(tmp_path):
    document = snapshot("2026-09-08T19:00:00+00:00")
    entry = build("2026-09-08T19:00:00+00:00")
    path = archive.write_entry(entry, tmp_path, projection_snapshot=document)
    stored = archive.load_document(tmp_path, entry)
    assert stored == document
    assert (tmp_path / entry["projection_snapshot"]["document_path"]).exists()
    # never overwritten, and a different document for the same entry is refused
    with pytest.raises(FileExistsError):
        archive.write_entry(entry, tmp_path, projection_snapshot=document)
    other = build("2026-09-08T19:00:01+00:00")
    with pytest.raises(archive.ArchiveIntegrityError, match="not the one"):
        archive.write_entry(other, tmp_path, projection_snapshot=document)
    # the sibling document is not mistaken for an entry by the index guard
    assert [e["path"] for e in archive.load_index(tmp_path)["entries"]] == [
        path.relative_to(tmp_path).as_posix()]
    # and tampering with the archived document is detected
    doc_path = tmp_path / entry["projection_snapshot"]["document_path"]
    tampered = json.loads(doc_path.read_text()); tampered["players"][0]["team"] = "XX"
    doc_path.write_text(json.dumps(tampered))
    with pytest.raises(archive.ArchiveIntegrityError):
        archive.load_document(tmp_path, entry)


# --------------------------------------------------------------------------- #
# Eligibility: per game, strict, both clocks
# --------------------------------------------------------------------------- #

def test_full_slate_before_every_deadline_is_prospective():
    entry = build("2026-09-08T19:00:00+00:00")
    assert entry["eligibility"]["label"] == "prospective_full_slate"
    assert all(d["prospective"] for d in entry["deadlines"].values())
    assert entry["eligibility"]["games_prospective"] == 3
    assert entry["eligibility"]["games_missed"] == 0


def test_a_forecast_after_one_kickoff_is_partial_not_whole_week():
    # after the Wednesday game, before Thursday's
    entry = build("2026-09-10T03:00:00+00:00")
    assert entry["eligibility"]["label"] == "prospective_partial_slate"
    assert entry["deadlines"]["2026_01_NE_SEA"]["prospective"] is False
    assert entry["deadlines"]["2026_01_NE_SEA"]["reason"] == "forecast_generated_at_or_after_kickoff"
    assert entry["deadlines"]["2026_01_SF_LA"]["prospective"] is True
    assert entry["eligibility"]["games_prospective"] == 2
    assert entry["eligibility"]["games_missed"] == 1
    by_player = {row["player_id"]: row for row in entry["players"]}
    assert by_player["00-0001"]["prospective"] is False
    assert by_player["00-0002"]["prospective"] is True


def test_a_forecast_after_every_kickoff_is_not_prospective():
    entry = build("2026-09-15T12:00:00+00:00")
    assert entry["eligibility"]["label"] == "not_prospective"
    assert entry["eligibility"]["games_prospective"] == 0


def test_deadline_boundary_is_strict():
    # generated exactly AT kickoff, with source data from well before: the
    # forecast clock alone must fail it, and with the right reason
    at_kickoff = build("2026-09-10T00:20:00+00:00", information_as_of="2026-09-09T12:00:00+00:00")
    assert at_kickoff["deadlines"]["2026_01_NE_SEA"]["prospective"] is False
    assert at_kickoff["deadlines"]["2026_01_NE_SEA"]["reason"] == "forecast_generated_at_or_after_kickoff"
    one_second_before = build("2026-09-10T00:19:59+00:00", information_as_of="2026-09-09T12:00:00+00:00")
    assert one_second_before["deadlines"]["2026_01_NE_SEA"]["prospective"] is True
    # and the same strictness for the as-of clock on its own
    as_of_at_kickoff = build("2026-09-09T12:00:00+00:00", information_as_of="2026-09-10T00:20:00+00:00")
    assert as_of_at_kickoff["deadlines"]["2026_01_NE_SEA"]["prospective"] is False
    assert as_of_at_kickoff["deadlines"]["2026_01_NE_SEA"]["reason"] == "information_as_of_at_or_after_kickoff"


def test_information_as_of_must_also_precede_kickoff():
    entry = build("2026-09-09T12:00:00+00:00", information_as_of="2026-09-10T01:00:00+00:00")
    deadline = entry["deadlines"]["2026_01_NE_SEA"]
    assert deadline["prospective"] is False
    assert deadline["reason"] == "information_as_of_at_or_after_kickoff"
    assert entry["deadlines"]["2026_01_SF_LA"]["prospective"] is True


def test_timezones_are_compared_as_instants():
    # 20:20 Eastern on the 9th == 00:20 UTC on the 10th
    entry = archive.build_entry(
        snapshot("2026-09-09T20:19:00-04:00"), summaries(),
        scoring=ScoringRules.preset("ppr"), scoring_preset="ppr",
        kickoffs_utc={"2026_01_NE_SEA": "2026-09-09T20:20:00-04:00"},
        run_context=archive.run_context_from_env({}),
        archived_at="2026-09-10T00:30:00+00:00",
    )
    assert entry["deadlines"]["2026_01_NE_SEA"]["prospective"] is True
    assert entry["deadlines"]["2026_01_NE_SEA"]["kickoff_utc"] == "2026-09-10T00:20:00+00:00"


def test_naive_timestamps_are_refused_not_assumed():
    with pytest.raises(ValueError, match="timezone"):
        build("2026-09-08T19:00:00")
    with pytest.raises(ValueError, match="timezone"):
        archive.build_entry(
            snapshot("2026-09-08T19:00:00+00:00"), summaries(),
            scoring=ScoringRules.preset("ppr"), scoring_preset="ppr",
            kickoffs_utc={"2026_01_NE_SEA": "2026-09-10T00:20:00"},
            run_context=archive.run_context_from_env({}),
            archived_at="2026-09-08T20:00:00+00:00",
        )


def test_a_player_whose_game_has_no_kickoff_is_never_prospective():
    entry = build("2026-09-08T19:00:00+00:00")
    row = next(row for row in entry["players"] if row["player_id"] == "00-0004")
    assert row["prospective"] is False
    assert row["reason"] == "kickoff_unknown"
    assert entry["eligibility"]["players_no_kickoff"] == 1
    # an unknown deadline is not a met deadline: it does not count as prospective
    assert entry["eligibility"]["players_prospective"] == 3


# --------------------------------------------------------------------------- #
# Immutability, index, latest pointer
# --------------------------------------------------------------------------- #

def test_write_refuses_to_overwrite_and_verifies_on_load(tmp_path):
    entry = build("2026-09-08T19:00:00+00:00")
    path = archive.write_entry(entry, tmp_path)
    assert path.exists()
    with pytest.raises(FileExistsError):
        archive.write_entry(entry, tmp_path)
    loaded = archive.load_entry(path)
    assert loaded["payload_sha256"] == entry["payload_sha256"]
    # tampering with a stored forecast is detected
    tampered = json.loads(path.read_text())
    tampered["players"][0]["mean"] = 99.0
    path.write_text(json.dumps(tampered))
    with pytest.raises(archive.ArchiveIntegrityError):
        archive.load_entry(path)


def test_index_is_append_only_and_latest_is_a_separate_pointer(tmp_path):
    first = build("2026-09-08T19:00:00+00:00")
    first_path = archive.write_entry(first, tmp_path)
    first_bytes = first_path.read_bytes()
    second = build("2026-09-09T22:00:00+00:00", archived_at="2026-09-09T22:05:00+00:00")
    second_path = archive.write_entry(second, tmp_path)

    index = archive.load_index(tmp_path)
    assert index["kind"] == archive.INDEX_KIND
    assert [e["forecast_generated_at"] for e in index["entries"]] == [
        "2026-09-08T19:00:00+00:00", "2026-09-09T22:00:00+00:00"]
    assert first_path.read_bytes() == first_bytes  # the rerun touched nothing
    latest = json.loads((tmp_path / archive.LATEST_FILE).read_text())
    assert latest["kind"] == archive.LATEST_KIND
    assert latest["path"] == second_path.relative_to(tmp_path).as_posix()
    assert latest["payload_sha256"] == second["payload_sha256"]
    # the pointer file is not an index entry, and the index is not the pointer
    assert archive.LATEST_FILE not in {e["path"] for e in index["entries"]}


def test_an_index_that_lost_an_entry_is_refused(tmp_path):
    archive.write_entry(build("2026-09-08T19:00:00+00:00"), tmp_path)
    index_path = tmp_path / archive.INDEX_FILE
    index = json.loads(index_path.read_text())
    index["entries"] = []
    index_path.write_text(json.dumps(index))
    with pytest.raises(archive.ArchiveIntegrityError, match="append-only"):
        archive.write_entry(
            build("2026-09-09T22:00:00+00:00", archived_at="2026-09-09T22:05:00+00:00"),
            tmp_path)


def test_a_late_rerun_records_a_missed_cohort_rather_than_backfilling(tmp_path):
    late = build("2026-09-10T03:00:00+00:00", archived_at="2026-09-10T03:05:00+00:00")
    archive.write_entry(late, tmp_path)
    index = archive.load_index(tmp_path)
    assert index["entries"][-1]["label"] == "prospective_partial_slate"
    assert index["entries"][-1]["games_missed"] == 1
    # nothing in the archive claims a pre-kickoff forecast for the missed game
    assert not any(
        e["deadlines"]["2026_01_NE_SEA"]["prospective"]
        for e in (archive.load_entry(tmp_path / e["path"]) for e in index["entries"]))


def test_run_context_comes_only_from_the_environment():
    context = archive.run_context_from_env({
        "TAILSTAIL_RUN_EVENT": "schedule", "TAILSTAIL_RUN_SCHEDULE": "35 23 * 9-12,1 3",
        "TAILSTAIL_RUN_ID": "123", "TAILSTAIL_RUN_ATTEMPT": "1",
        "GITHUB_SHA": "deadbeef",
    })
    assert context == {"event": "schedule", "schedule": "35 23 * 9-12,1 3",
                       "run_id": "123", "run_attempt": "1"}
    assert archive.run_context_from_env({}) == {
        "event": None, "schedule": None, "run_id": None, "run_attempt": None}


def test_payload_hash_covers_the_forecast_not_the_location():
    entry = build("2026-09-08T19:00:00+00:00")
    body = {k: v for k, v in entry.items() if k not in ("payload_sha256", "archived_path")}
    expected = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert entry["payload_sha256"] == expected
