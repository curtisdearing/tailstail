"""Where the prospective archive lives, how it travels, and what may not change.

The archive is Tailstail's own forecast, already published in full on the
Pages site, so it is public material and travels in the checksummed public
state release with the rest of the fantasy profile. It must NOT be confused
with the ESPN captures, which stay on the private side. And the run that
produces it must stay the frozen production run: mixture off, same seed rule.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nflvalue.fantasy import private_boundary
from nflvalue.fantasy.config import SimulationConfig
from scripts import state_store

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / ".github" / "workflows" / "fantasy-weekly.yml").read_text()
WEEKLY = (ROOT / "scripts" / "fantasy_weekly.py").read_text()
GITIGNORE = (ROOT / ".gitignore").read_text()


def _build_step() -> str:
    start = WORKFLOW.index("- name: Build current projections")
    end = WORKFLOW.index("- name:", start + 10)
    return WORKFLOW[start:end]


def _verify_step() -> str:
    start = WORKFLOW.index("- name: Verify generated contracts")
    end = WORKFLOW.index("- name:", start + 10)
    return WORKFLOW[start:end]


# --------------------------------------------------------------------------- #
# Durability and privacy of the archive
# --------------------------------------------------------------------------- #

def test_the_archive_travels_in_the_public_fantasy_state_profile(tmp_path):
    root = tmp_path / "repo"
    (root / "data" / "prospective_archive" / "2026").mkdir(parents=True)
    (root / "data" / "prospective_archive" / "index.json").write_text("{}")
    (root / "data" / "prospective_archive" / "latest.json").write_text("{}")
    (root / "data" / "prospective_archive" / "2026" / "projection_2026_wk01_x.json").write_text("{}")
    (root / "data" / "player_projection_snapshot.json").write_text("{}")
    archive = tmp_path / "state.tar.gz"
    manifest = state_store.pack(archive, root, profile="fantasy")
    assert "data/prospective_archive/index.json" in manifest["files"]
    assert "data/prospective_archive/latest.json" in manifest["files"]
    assert "data/prospective_archive/2026/projection_2026_wk01_x.json" in manifest["files"]
    restored_root = tmp_path / "restored"
    restored = state_store.restore(archive, manifest["sha256"], restored_root, profile="fantasy")
    assert "data/prospective_archive/2026/projection_2026_wk01_x.json" in restored


def test_the_archive_directory_is_gitignored():
    assert re.search(r"^data/prospective_archive/$", GITIGNORE, re.M), GITIGNORE


def test_the_archive_never_shares_a_path_with_raw_espn_state():
    from nflvalue.fantasy import private_state, prospective_archive
    assert not any("prospective_archive" in p for p in private_state.ALLOWED_FILES)
    assert not any("prospective_archive" in p for p in private_state.ALLOWED_TREES)
    assert "espn" not in prospective_archive.DEFAULT_DIRECTORY


def test_the_public_payload_carries_the_capture_summary_and_the_guard_accepts_it():
    assert "prospective_capture" in private_boundary.PUBLIC_PAYLOAD_KEYS
    payload = {
        "generated_at": "2026-09-08T19:00:00+00:00", "season": 2026, "week": 1,
        "players": [], "prospective_capture": {
            "label": "prospective_full_slate", "games_total": 16, "games_prospective": 16,
            "games_missed": 0, "archived_path": "data/prospective_archive/2026/x.json",
            "payload_sha256": "a" * 64,
        },
    }
    public = private_boundary.public_weekly_payload(payload)
    assert public["prospective_capture"]["label"] == "prospective_full_slate"


def test_the_public_payload_still_strips_every_private_section():
    payload = {
        "generated_at": "2026-09-08T19:00:00+00:00", "season": 2026, "week": 1,
        "players": [], "prospective_capture": {"label": "not_prospective"},
        "my_team": {"schema_version": "my_team/1.0.0", "league": {"league_name": "x"}},
        "espn_comparison": {"status": "ok", "current_week_rows": [{"espn_pts": 1.0}]},
    }
    public = private_boundary.public_weekly_payload(payload)
    assert "my_team" not in public
    assert "current_week_rows" not in public["espn_comparison"]
    assert set(public) <= set(private_boundary.PUBLIC_PAYLOAD_KEYS) | {
        "espn_comparison", "visibility", "withheld"}


# --------------------------------------------------------------------------- #
# The workflow: run context in, archive verified out
# --------------------------------------------------------------------------- #

def test_the_build_step_passes_run_context_through_the_environment():
    step = _build_step()
    for name, expression in (
        ("TAILSTAIL_RUN_EVENT", "${{ github.event_name }}"),
        ("TAILSTAIL_RUN_SCHEDULE", "${{ github.event.schedule }}"),
        ("TAILSTAIL_RUN_ID", "${{ github.run_id }}"),
        ("TAILSTAIL_RUN_ATTEMPT", "${{ github.run_attempt }}"),
    ):
        assert f"{name}: {expression}" in step, name
    # ...and never spliced into the shell script
    run_block = step[step.index("run:"):]
    assert "${{" not in run_block


def test_the_verify_step_checks_the_archive_before_anything_is_published():
    step = _verify_step()
    assert "prospective_archive" in step
    assert "latest.json" in step and "load_index" in step
    assert "payload_sha256" in step
    assert "assert_public_safe" in step


def test_the_pages_site_does_not_receive_the_archive_history():
    start = WORKFLOW.index("- name: Prepare Tailstail Pages site")
    end = WORKFLOW.index("- uses:", start)
    assert "prospective_archive" not in WORKFLOW[start:end]


def test_the_schedule_leaves_margin_before_the_sunday_early_window():
    """GitHub delayed both September scheduled runs by 1h47m and 2h49m. A
    Sunday cron at 14:15 UTC therefore starts *after* the 17:00 UTC early
    window on a bad day. The Sunday run has to fire at least four hours ahead."""
    crons = re.findall(r'- cron: "([^"]+)"', WORKFLOW)
    assert len(crons) == 2, crons
    sunday = next(c for c in crons if c.split()[-1] == "0")
    minute, hour = (int(field) for field in sunday.split()[:2])
    assert hour * 60 + minute <= 13 * 60, sunday
    wednesday = next(c for c in crons if c.split()[-1] == "3")
    assert wednesday.split()[2] == "*" and wednesday.split()[3] == "9-12,1"


# --------------------------------------------------------------------------- #
# The frozen production run
# --------------------------------------------------------------------------- #

def test_the_role_mixture_stays_off_by_default_and_the_weekly_run_never_enables_it():
    assert SimulationConfig().role_scenario_mixture is False
    assert "role_scenario_mixture" not in WEEKLY
    assert "--shadow" not in WORKFLOW


def test_the_weekly_run_archives_after_the_snapshot_and_before_espn():
    order = [WEEKLY.index(marker) for marker in (
        "write_projection_snapshot(projection_snapshot", "prospective_archive.write_entry(",
        "espn_comparison = run_espn_comparison(")]
    assert order == sorted(order)


@pytest.mark.parametrize("name", [
    "nflvalue/fantasy/simulation.py", "nflvalue/fantasy/models.py",
    "nflvalue/fantasy/features.py", "nflvalue/projection_snapshot.py",
])
def test_the_frozen_center_files_are_not_touched_by_the_capture_work(name):
    """The capture work adds an archive beside the forecast; it does not reach
    into how the forecast is made. Enforced by a text check on the frozen files:
    none of them may mention the archive."""
    assert "prospective_archive" not in (ROOT / name).read_text()
