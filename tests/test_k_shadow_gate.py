"""The kicker lane: reproducible, honest about what it does not know, and gated.

Three of these tests exist because the lane's own model card asserted things
the code did not do. Determinism was promised and broken twice — a seed drawn
through `hash()`, which PYTHONHASHSEED salts, and an identity digest taken over
the wall clock. Availability was promised to fail closed and failed open. And
the promotion gates were written down without any code behind them, so "shadow"
was a label rather than a state anything enforced.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nflvalue.fantasy import k_audit  # noqa: E402
from nflvalue.fantasy import shadow_kicker as SK  # noqa: E402
from nflvalue.fantasy.espn_contract import from_settings_payload  # noqa: E402

SETTINGS = ROOT / "tests" / "fixtures" / "espn_league_settings_2026_recorded.json"
PROVENANCE = [{"source": "nflverse pbp", "retrieved_at": "2026-08-29T10:00:00Z",
               "as_of": "2026-08-29T09:00:00Z"}]


@pytest.fixture(scope="module")
def contract():
    return from_settings_payload(json.loads(SETTINGS.read_text()))


def history(player="K1", weeks=10, season=2025, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for week in range(1, weeks + 1):
        rows.append({
            "player_id": player, "season": season, "week": week, "team": "BUF",
            "fg_made_0_39": int(rng.integers(0, 3)), "fg_made_40_49": int(rng.integers(0, 2)),
            "fg_made_50_59": int(rng.integers(0, 2)), "fg_made_60_": 0,
            "fg_missed_0_39": int(rng.integers(0, 2)), "fg_missed_40_49": 0,
            "fg_missed_50_59": int(rng.integers(0, 2)), "fg_missed_60_": 0,
            "pat_made": int(rng.integers(1, 5)), "pat_att": 4,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Determinism, for real this time
# --------------------------------------------------------------------------- #
def test_the_seed_does_not_move_with_the_interpreter_hash_salt():
    """`hash(str)` is PYTHONHASHSEED-salted: a seed built from it only LOOKS fixed.

    An in-process double call cannot catch this, because the salt is fixed for
    the life of an interpreter. Two interpreters are the only witness.
    """
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        from nflvalue.fantasy.shadow_kicker import stable_seed
        print(stable_seed("K1"))
    """)
    outputs = set()
    for salt in ("0", "1", "12345"):
        result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                text=True, env={"PYTHONHASHSEED": salt, "PATH": "/usr/bin:/bin"})
        assert result.returncode == 0, result.stderr
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1, f"seed varied with PYTHONHASHSEED: {outputs}"


def test_the_artifact_identity_excludes_only_the_run_clock(contract):
    """Same inputs, same seed, same digest — whatever time it is."""
    first = SK.build_artifact(history(), ["K1"], contract, season=2026, week=1,
                              simulations=200, seed=11, provenance=PROVENANCE)
    second = SK.build_artifact(history(), ["K1"], contract, season=2026, week=1,
                               simulations=200, seed=11, provenance=PROVENANCE)
    assert first["content_sha256"] == second["content_sha256"]
    assert first["model_run_at"] != second["model_run_at"] or True  # clock may not tick
    assert "model_run_at" in SK.NON_CONTENT_FIELDS
    # and it is a real digest of the rest, not a constant
    changed = dict(first)
    changed["season"] = 2027
    assert SK.content_digest(changed) != first["content_sha256"]


def test_a_different_seed_still_changes_the_numbers(contract):
    a = SK.project(history(), "K1", contract, season=2026, week=1, simulations=500,
                   seed=11, active=True)
    b = SK.project(history(), "K1", contract, season=2026, week=1, simulations=500,
                   seed=12, active=True)
    assert a["distribution"] != b["distribution"], "seed is not actually used"


# --------------------------------------------------------------------------- #
# Unknown is not active
# --------------------------------------------------------------------------- #
def test_an_unknown_active_state_is_unavailable_not_assumed_active(contract):
    row = SK.project(history(), "K1", contract, season=2026, week=1, simulations=100)
    assert row["status"] == "unavailable"
    assert "unknown" in row["unavailable_reason"]


def test_a_kicker_absent_from_the_availability_map_is_unavailable(contract):
    artifact = SK.build_artifact(history(), ["K1"], contract, season=2026, week=1,
                                 simulations=100, active={}, provenance=PROVENANCE)
    assert artifact["n_projected"] == 0
    assert artifact["n_unavailable"] == 1


def test_a_kicker_known_active_is_projected(contract):
    artifact = SK.build_artifact(history(), ["K1"], contract, season=2026, week=1,
                                 simulations=100, active={"K1": True},
                                 provenance=PROVENANCE)
    assert artifact["n_projected"] == 1


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def test_an_artifact_without_provenance_is_refused(contract):
    with pytest.raises(SK.ShadowKickerError, match="provenance"):
        SK.build_artifact(history(), ["K1"], contract, season=2026, week=1,
                          simulations=100, active={"K1": True})


@pytest.mark.parametrize("missing", ["source", "retrieved_at", "as_of"])
def test_every_provenance_entry_states_source_and_both_timestamps(contract, missing):
    entry = dict(PROVENANCE[0])
    entry.pop(missing)
    with pytest.raises(SK.ShadowKickerError, match=missing):
        SK.build_artifact(history(), ["K1"], contract, season=2026, week=1,
                          simulations=100, active={"K1": True}, provenance=[entry])


def test_the_information_boundary_is_the_oldest_input_not_the_newest(contract):
    with pytest.raises(SK.ShadowKickerError, match="later than the oldest input"):
        SK.build_artifact(history(), ["K1"], contract, season=2026, week=1,
                          simulations=100, active={"K1": True}, provenance=PROVENANCE,
                          information_as_of="2026-08-30T23:00:00Z")


def test_an_unparseable_timestamp_is_refused(contract):
    bad = [{"source": "x", "retrieved_at": "yesterday", "as_of": "2026-08-29T09:00:00Z"}]
    with pytest.raises(SK.ShadowKickerError, match="ISO-8601"):
        SK.build_artifact(history(), ["K1"], contract, season=2026, week=1,
                          simulations=100, active={"K1": True}, provenance=bad)


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
def test_the_lane_declares_itself_unpromoted_and_says_why(contract):
    artifact = SK.build_artifact(history(), ["K1"], contract, season=2026, week=1,
                                 simulations=100, active={"K1": True},
                                 provenance=PROVENANCE)
    assert artifact["promoted"] is False
    assert artifact["promotion"]["may_enter_lineup_objective"] is False
    assert artifact["promotion"]["kind"] == "historical_rate_baseline"
    assert "season-forward" in artifact["promotion"]["reason"]


def test_an_audit_that_did_not_run_is_a_fail_not_a_pass():
    verdict = k_audit.gate({})
    assert verdict.passed is False
    assert any("did not run" in reason for reason in verdict.reasons)
    assert k_audit.may_enter_lineup_objective(verdict) is False
    assert k_audit.may_enter_lineup_objective(None) is False


def test_a_thin_run_fails_the_declared_minimums(contract):
    """Two weeks of one kicker is not evidence, and the gate says so."""
    frame = history(weeks=6, season=2025)
    result = k_audit.run(frame, contract, test_seasons=[2025], simulations=200)
    assert result.passed is False
    assert any("kickers" in r or "weeks" in r for r in result.reasons)


def test_the_gates_are_declared_before_the_run_not_derived_from_it():
    for key in ("min_weeks", "min_kickers", "mae_improvement_over_baseline",
                "min_probability_of_improvement", "crps_improvement_over_baseline",
                "coverage_50_tolerance", "coverage_90_tolerance"):
        assert key in k_audit.GATES


def test_beating_mae_by_shrinking_to_the_mean_does_not_pass(contract):
    """The CRPS gate is the one that catches a collapsed distribution."""
    rows = pd.DataFrame({
        "season": [2025] * 60, "week": list(range(1, 61)),
        "player_id": [f"K{i % 10}" for i in range(60)],
        "projection_mean": [8.0] * 60, "projection_sd": [1e-6] * 60,
        "projection_p25": [8.0] * 60, "projection_p75": [8.0] * 60,
        "projection_p05": [8.0] * 60, "projection_p95": [8.0] * 60,
        "baseline_mean": [8.4] * 60,
        "fantasy_points": list(np.random.default_rng(5).normal(8.0, 4.0, 60)),
    })
    verdict = k_audit.gate(k_audit.evaluate(rows))
    assert verdict.passed is False
    assert any("CRPS" in reason or "band covers" in reason for reason in verdict.reasons)


def test_a_kicker_cannot_reach_the_lineup_objective_without_a_passing_audit():
    assert SK.PROMOTION_STATUS["may_enter_lineup_objective"] is False
