"""ESPN is an external challenger: scored against, never scored with.

Two separate claims are asserted here, because either one failing alone would
be enough to invalidate every ESPN-vs-Tailstail comparison ever published:

1. The comparison actually runs.  It shipped once in a merged PR that CI never
   executed, so "we have tests for it" is not evidence -- the selection lane is
   asserted directly.
2. ESPN numbers cannot reach a Tailstail projection.  Both structurally (no
   module on the projection path imports the ESPN comparison layer) and
   behaviourally (replacing every ESPN value with garbage leaves every stored
   model projection byte-identical).
"""

from __future__ import annotations

import ast
import copy
import importlib
from pathlib import Path

import pytest

from nflvalue.fantasy import espn_compare

REPO = Path(__file__).resolve().parents[1]

#: Everything that computes or carries a Tailstail projection.  If any of
#: these ever imports the ESPN layer, the comparison stops being external.
PROJECTION_PATH_MODULES = (
    "nflvalue/fantasy/models.py",
    "nflvalue/fantasy/features.py",
    "nflvalue/fantasy/scoring.py",
    "nflvalue/fantasy/simulation.py",
    "nflvalue/fantasy/hierarchy.py",
    "nflvalue/fantasy/role_state.py",
    "nflvalue/fantasy/config.py",
    "nflvalue/fantasy/data.py",
    "nflvalue/fantasy/projection_snapshot.py",
    "nflvalue/projection_snapshot.py",
)

FORBIDDEN_IMPORTS = ("espn_compare", "espn_projections", "espn_client", "espn")


# --------------------------------------------------------------------------
# 1. It runs in CI
# --------------------------------------------------------------------------

def test_espn_comparison_suite_is_in_the_offline_ci_lane():
    """The offline lane is CI's `pytest -q -m offline` job."""
    from tests.conftest import NEEDS_HISTORY, NEEDS_NETWORK

    for name in ("test_espn_compare.py", "test_espn_external_challenger.py"):
        assert (REPO / "tests" / name).exists(), f"{name} is missing"
        assert name not in NEEDS_HISTORY, f"{name} was excluded from CI as history-dependent"
        assert name not in NEEDS_NETWORK, f"{name} was excluded from CI as network-dependent"


def test_ci_runs_the_offline_marker_rather_than_a_file_list():
    workflow = (REPO / ".github/workflows/ci.yml").read_text()
    assert "pytest -q -m offline" in workflow
    assert "test_espn_compare.py" not in workflow, (
        "CI must select the ESPN suite by marker, not by naming the file"
    )


# --------------------------------------------------------------------------
# 2. ESPN cannot reach a projection -- structurally
# --------------------------------------------------------------------------

@pytest.mark.parametrize("relative", PROJECTION_PATH_MODULES)
def test_projection_path_never_imports_the_espn_layer(relative):
    path = REPO / relative
    if not path.exists():
        pytest.skip(f"{relative} is not present in this tree")
    tree = ast.parse(path.read_text(), filename=str(path))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            imported.append(base)
            imported.extend(f"{base}.{alias.name}" for alias in node.names)
    offenders = sorted({
        name for name in imported
        if any(part in FORBIDDEN_IMPORTS for part in name.split("."))
    })
    assert not offenders, f"{relative} imports the ESPN layer: {offenders}"


def test_the_comparison_module_itself_does_import_espn():
    """Guard against a vacuous check above: the scanner must be able to see it."""
    source = (REPO / "nflvalue/fantasy/espn_compare.py").read_text()
    assert "espn" in source
    module = importlib.import_module("nflvalue.fantasy.espn_compare")
    assert hasattr(module, "record_week")


# --------------------------------------------------------------------------
# 3. ESPN cannot reach a projection -- behaviourally
# --------------------------------------------------------------------------

KICKOFF = "2026-09-13T17:00:00+00:00"
BEFORE = "2026-09-13T12:00:00+00:00"

MODEL_POINTS = {"00-0000001": 18.25, "00-0000002": 9.5, "00-0000003": 4.125}
MODEL_META = {
    "00-0000001": {"team": "AAA"},
    "00-0000002": {"team": "BBB"},
    "00-0000003": {"team": "AAA"},
}
MATCHED = {101: "00-0000001", 102: "00-0000002", 103: "00-0000003"}
PLAYER_GAMES = dict.fromkeys(MODEL_POINTS, "2026_01_AAA_BBB")
KICKOFFS = {"2026_01_AAA_BBB": KICKOFF}


def _espn_players(scale: float) -> list[dict]:
    return [
        {
            "espn_id": espn_id,
            "player_name": f"Player {espn_id}",
            "position": "WR",
            "team": "AAA",
            "espn_ppr_points": value * scale,
            "points_basis": "projected",
        }
        for espn_id, value in ((101, 12.0), (102, 8.0), (103, 3.0))
    ]


def _record(scale: float, snapshot_sha: str) -> dict:
    ledger = espn_compare.new_ledger(2026)
    espn_compare.record_week(
        ledger,
        week=1,
        espn_players=_espn_players(scale),
        espn_retrieved_at=BEFORE,
        espn_snapshot_sha256=snapshot_sha,
        matched=MATCHED,
        model_points=MODEL_POINTS,
        model_meta=MODEL_META,
        model_generated_at=BEFORE,
        player_games=PLAYER_GAMES,
        kickoffs_utc=KICKOFFS,
    )
    return ledger


def test_wildly_different_espn_values_leave_every_model_row_identical():
    quiet = _record(1.0, "a" * 64)
    loud = _record(-500.0, "b" * 64)
    quiet_rows = quiet["weeks"]["1"]["rows"]
    loud_rows = loud["weeks"]["1"]["rows"]
    assert len(quiet_rows) == len(loud_rows) == len(MODEL_POINTS)
    for a, b in zip(quiet_rows, loud_rows):
        assert a["player_id"] == b["player_id"]
        assert a["model_pts"] == b["model_pts"] == MODEL_POINTS[a["player_id"]]
        assert a["espn_pts"] != b["espn_pts"], "the ESPN side must have actually moved"


def test_the_model_points_mapping_is_never_mutated_by_recording():
    before = copy.deepcopy(MODEL_POINTS)
    _record(3.0, "c" * 64)
    assert before == MODEL_POINTS


def test_grading_does_not_rewrite_the_stored_model_projection():
    ledger = _record(1.0, "d" * 64)
    stored = copy.deepcopy(ledger["weeks"]["1"]["rows"])
    espn_compare.grade_week(
        ledger,
        week=1,
        actual_points={"00-0000001": 25.0, "00-0000002": 2.0, "00-0000003": 11.0},
    )
    graded_rows = ledger["weeks"]["1"]["rows"]
    assert graded_rows == stored, "grading rewrote the prospective rows"


def test_the_payload_labels_espn_as_an_external_reference():
    ledger = _record(1.0, "e" * 64)
    payload = espn_compare.build_payload(
        ledger, current_week=1, espn_provenance=None, identity_report=None
    )
    assert payload["disclaimer"], "an ESPN comparison must ship its disclaimer"
    assert "prospective_rule" in payload
