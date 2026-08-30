"""Accuracy harness: registry schema, gates, drift check, and honesty contract."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS = os.path.join(ROOT, "analysis", "eval_harness.py")


def run(*args):
    return subprocess.run([sys.executable, HARNESS, *args], cwd=ROOT,
                          capture_output=True, text=True)


def _registry(out: str) -> dict:
    result = run("--output", out)
    assert result.returncode == 0, result.stderr
    with open(os.path.join(ROOT, out)) as handle:
        return json.load(handle)


def test_registry_written_with_schema():
    out = "data/accuracy_registry_test.json"
    registry = _registry(out)
    for key in ["schema_version", "generated", "git_head", "environment",
                "holdout_policy", "accept_gates", "release_thresholds",
                "protocol", "inputs", "metrics"]:
        assert key in registry, key
    assert registry["schema_version"] == 3
    assert registry["accept_gates"]["ranker_log_loss"] < 0
    assert registry["accept_gates"]["sim_undercoverage_penalty_pp"] < 0
    assert registry["release_thresholds"]["sanity_top10_overlap_min"] == 0.50
    assert registry["protocol"]["schema_version"] == 1
    assert "undercoverage_penalty_pp" in registry["metrics"]["simulation_calibration"]
    assert isinstance(registry["inputs"], dict) and registry["inputs"]
    os.remove(os.path.join(ROOT, out))


def test_check_mode_detects_no_drift():
    out = "data/accuracy_registry_test2.json"
    assert run("--output", out).returncode == 0
    result = run("--check", "--output", out)
    assert result.returncode == 0 and "unchanged" in result.stdout
    os.remove(os.path.join(ROOT, out))


def test_environment_is_recorded_so_a_number_can_be_reproduced():
    """The registry this replaced recorded two package versions and no
    interpreter, which is not enough to reproduce a scikit-learn fit."""
    out = "data/accuracy_registry_test3.json"
    registry = _registry(out)
    environment = registry["environment"]
    assert environment["python"], "no interpreter version recorded"
    for package in ("numpy", "pandas", "scipy", "scikit-learn"):
        assert environment["packages"][package], f"{package} version missing"
    os.remove(os.path.join(ROOT, out))


def test_a_missing_artifact_is_reported_as_missing_not_as_a_result():
    out = "data/accuracy_registry_test4.json"
    registry = _registry(out)
    for name, value in registry["metrics"].items():
        if isinstance(value, dict) and value.get("status") == "missing":
            assert value["build_with"], f"{name} is missing without a build command"
            assert "path" in value
        # The old registry wrote the string "not built yet ..." as if it were a
        # metric value, beside real numbers. Nothing may read that way again.
        assert "not built yet" not in json.dumps(value), name
    os.remove(os.path.join(ROOT, out))


def test_the_champion_challenger_decision_is_recorded():
    out = "data/accuracy_registry_test5.json"
    registry = _registry(out)
    decision = registry["metrics"]["champion_challenger_decision"]
    assert "REJECT" in decision or "PROMOTE" in decision
    shadow = registry["metrics"]["shadow_challenge"]
    assert shadow["preregistration"]["sha256"], "the preregistration must be hashed"
    assert shadow["preregistration"]["written_before_any_challenger_was_scored"] is True
    assert shadow["prospective_2026"]["gradeable_rows"] == 0
    os.remove(os.path.join(ROOT, out))


def test_pinned_inputs_are_the_paths_the_pipeline_actually_writes():
    """The previous list pinned four parquet paths no command had written
    since the fantasy cache moved, so every hash was null by construction."""
    from analysis.eval_harness import INPUTS

    assert "historical/fantasy/feature_frame.parquet" in INPUTS
    assert "analysis/shadow_challenge_2026.json" in INPUTS
    assert not [path for path in INPUTS if path.startswith("historical/historical_")]
    assert "historical_lines.parquet" not in INPUTS
