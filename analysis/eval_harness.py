#!/usr/bin/env python3
"""Accuracy harness: ONE command, ONE registry (accuracy loop plan, P2).

Collects the repository's current accuracy metrics from the canonical result
artifacts, pins the SHA-256 of every model input, and writes
``data/accuracy_registry.json``.  That registry is the single scoreboard the
weekly lever loop reads and the accept gates are checked against.

    python3 analysis/eval_harness.py            # collect + write + print
    python3 analysis/eval_harness.py --check    # exit 1 if inputs drifted

This harness never computes a metric itself.  Heavy evaluation stays in the
audited commands, which must be rerun first whenever a lever changes:

    python -m nflvalue.fantasy.history_audit --seasons 2019:2025 ...
    python -m nflvalue.fantasy.cli backtest --full ...
    python -m nflvalue.fantasy.cli audit-monte-carlo ...

Two rules this file exists to enforce, both learned the hard way:

* **One writer.**  A registry with two writers grows two shapes, and the one
  you are reading is never the one that was last written.  Everything that
  belongs in the registry is collected here.
* **Never claim what is not on disk.**  A missing artifact is reported as
  missing, with the exact command that builds it -- it is never rendered as a
  zero, a null, or an unqualified "not built yet" beside numbers that are
  real.  The registry this replaced carried ``"not built yet"`` for the model
  card and red-team backtest, ``null`` for every input hash, and a
  ``git_head`` that no longer existed in the branch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Feeds and configuration whose bytes decide every number below.  The old
#: list pinned four parquet paths that no command has written since the
#: fantasy cache moved to ``historical/fantasy/`` -- so it hashed nothing and
#: reported ``null`` while looking like a provenance record.
INPUTS = [
    "historical/fantasy/player_stats.parquet",
    "historical/fantasy/weekly_rosters.parquet",
    "historical/fantasy/schedules.parquet",
    "historical/fantasy/snap_counts.parquet",
    "historical/fantasy/injuries.parquet",
    "historical/fantasy/expected_points.parquet",
    "historical/fantasy/feature_frame.parquet",
    "config.json",
    "analysis/accuracy_protocol.json",
    "analysis/shadow_challenge_2026.json",
]

#: Result artifacts, and the command that builds each one.
ARTIFACTS = {
    "history_rebuild": (
        "reports/history_rebuild_manifest.json",
        "python -m nflvalue.fantasy.history_audit --seasons 2019:2025",
    ),
    "champion_scorecard": (
        "reports/fantasy_champion_scorecard.json",
        "python -m nflvalue.fantasy.cli backtest --full --test-seasons 2023:2025",
    ),
    "red_team": (
        "reports/fantasy_red_team.json",
        "python -m nflvalue.fantasy.cli backtest --full",
    ),
    "data_quality": (
        "reports/fantasy_data_quality.json",
        "python -m nflvalue.fantasy.history_audit --seasons 2019:2025",
    ),
    "monte_carlo": (
        "reports/fantasy_monte_carlo_history.json",
        "python -m nflvalue.fantasy.cli audit-monte-carlo",
    ),
    "shadow_development": (
        "reports/shadow_challenge_development.json",
        "the shadow-challenge run over outer seasons 2021-2024",
    ),
    "shadow_checkpoint2025": (
        "reports/shadow_challenge_checkpoint2025.json",
        "the shadow-challenge run over outer season 2025",
    ),
    "decision_development": (
        "reports/shadow_challenge_decision_development.json",
        "the preregistered promotion gate applied to the development window",
    ),
    "decision_checkpoint2025": (
        "reports/shadow_challenge_decision_checkpoint2025.json",
        "the preregistered promotion gate applied to the 2025 checkpoint",
    ),
    "lineup_regret": (
        "reports/shadow_challenge_lineup_regret.json",
        "the paired decision-level regret run",
    ),
    "factor_audit": (
        "data/all_data_factor_audit.json",
        "python analysis/all_data_factor_audit.py",
    ),
    "nested_projection": (
        "data/nested_factor_projection.json",
        "python analysis/run_nested_projection.py",
    ),
    "model_card": (
        "reports/fantasy_model_card.json",
        "python -m nflvalue.fantasy.cli train",
    ),
}

ACCEPT_GATES = {
    "fantasy_mae_points": -0.05,        # paired bootstrap on the locked checkpoint
    "fantasy_rank_spearman": +0.01,
    "sim_coverage_error_pp": -2.0,      # PIT/interval calibration
    "sim_undercoverage_penalty_pp": -1.0,
    "ranker_log_loss": -0.002,          # fablesfable-side gate, mirrored
}

RELEASE_THRESHOLDS = {
    "nominal_interval_coverage": 0.80,
    "sanity_top10_overlap_min": 0.50,
    "minimum_probability_of_improvement": 0.90,
}

PACKAGES = ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "joblib", "nflreadpy")


def sha256(path: str):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return None
    digest = hashlib.sha256()
    with open(full, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jload(path: str):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return None
    try:
        with open(full) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def clean(value):
    """NaN and infinity are not JSON. An undefined metric is null, not zero."""
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def missing(name: str) -> dict:
    path, command = ARTIFACTS[name]
    return {"status": "missing", "path": path, "build_with": command}


def git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "-c", "gc.auto=0", "rev-parse", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def environment() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "absent"
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }


def _arms(payload) -> dict:
    if not payload:
        return {}
    return {
        name: clean(arm["scorecard"]["overall"])
        for name, arm in payload.get("arms", {}).items()
    }


def _verdicts(payload) -> dict:
    if not payload:
        return {}
    return {
        name: {
            "decision": verdict["decision"],
            "failed_conditions": verdict["failed_conditions"],
            "noninferior_record": verdict["noninferior_record"],
        }
        for name, verdict in payload.get("verdicts", {}).items()
    }


def collect_metrics() -> dict:
    champion = jload(ARTIFACTS["champion_scorecard"][0])
    quality = jload(ARTIFACTS["data_quality"][0])
    monte_carlo = jload(ARTIFACTS["monte_carlo"][0])
    development = jload(ARTIFACTS["shadow_development"][0])
    checkpoint = jload(ARTIFACTS["shadow_checkpoint2025"][0])
    dev_decision = jload(ARTIFACTS["decision_development"][0])
    ck_decision = jload(ARTIFACTS["decision_checkpoint2025"][0])
    regret = jload(ARTIFACTS["lineup_regret"][0])
    audit = jload(ARTIFACTS["factor_audit"][0]) or {}
    nested = jload(ARTIFACTS["nested_projection"][0]) or {}
    card = jload(ARTIFACTS["model_card"][0])

    metrics: dict = {}

    if champion:
        metrics["champion_reproduction"] = {
            "test_seasons": champion["test_seasons"],
            "rows": champion["observed"]["rows"],
            "independent_season_week_blocks": champion["independent_season_week_blocks"],
            "seed": champion["seed"],
            "wall_seconds": champion["wall_seconds"],
            "git_commit": champion["git_commit"],
            "environment": champion["environment"],
            "frame_content_sha256": champion["inputs"]["frame_content_sha256"],
            "predictions_content_sha256": champion["predictions_content_sha256"],
            "direct_ensemble": clean(champion["observed"]),
            "by_season": clean(champion["by_season"]),
            "by_position": clean(champion["by_position"]),
            "baselines": clean(champion["baselines"]),
            "stored_claims": champion["stored_claims"],
            "deltas_vs_stored": clean(champion["deltas_vs_stored"]),
            "reproducibility": champion.get("reproducibility"),
        }
    else:
        metrics["champion_reproduction"] = missing("champion_scorecard")

    metrics["fantasy_model_card"] = (
        {"path": ARTIFACTS["model_card"][0], "trained_at": card.get("trained_at"),
         "feature_count": card.get("feature_count"),
         "training_seasons": card.get("training_seasons")}
        if card else missing("model_card")
    )
    metrics["fantasy_red_team_backtest"] = (
        {"path": ARTIFACTS["red_team"][0], "sha256": sha256(ARTIFACTS["red_team"][0])}
        if jload(ARTIFACTS["red_team"][0]) else missing("red_team")
    )
    metrics["fantasy_data_quality"] = (
        {"rows": quality.get("rows"), "eligible_rows": quality.get("eligible_rows"),
         "seasons": quality.get("seasons"), "positions": quality.get("positions"),
         "feature_count": quality.get("feature_count")}
        if quality else missing("data_quality")
    )

    if monte_carlo:
        meta = monte_carlo["metadata"]
        nominal = RELEASE_THRESHOLDS["nominal_interval_coverage"]
        calibrated = monte_carlo["methods"]["calibrated_monte_carlo"]
        coverage = calibrated.get("coverage80")
        metrics["simulation_calibration"] = {
            "replayed_rows": meta["replayed_rows"],
            "season_weeks": meta["season_weeks"],
            "simulations_per_week": meta["simulations_per_week"],
            "seed": meta["random_seed"],
            "runtime_versions": meta["runtime_versions"],
            "outer_predictions_canonical_csv_sha256":
                meta["outer_predictions_canonical_csv_sha256"],
            "replay_outputs_canonical_csv_sha256":
                meta["replay_outputs_canonical_csv_sha256"],
            "nominal_coverage": nominal,
            "observed_coverage": coverage,
            "absolute_coverage_error_pp":
                round(abs(float(coverage) - nominal) * 100, 4) if coverage is not None else None,
            "undercoverage_penalty_pp":
                round(max(nominal - float(coverage), 0.0) * 100, 4) if coverage is not None else None,
            "methods": clean(monte_carlo["methods"]),
            "paired_comparisons": clean(monte_carlo["paired_comparisons"]),
            "release_gate": clean(monte_carlo["release_gate"]),
            "raw_event_center_status": "rejected for point accuracy; distribution-only",
        }
    else:
        metrics["simulation_calibration"] = missing("monte_carlo")

    metrics["shadow_challenge"] = {
        "preregistration": {
            "path": "analysis/shadow_challenge_2026.json",
            "sha256": sha256("analysis/shadow_challenge_2026.json"),
            "written_before_any_challenger_was_scored": True,
        },
        "development": (
            {"outer_seasons": development["outer_seasons"], "arms": _arms(development),
             "comparisons": clean(development["comparisons"]),
             "decision": _verdicts(dev_decision),
             "outcome": (dev_decision or {}).get("outcome")}
            if development else missing("shadow_development")
        ),
        "locked_regression_checkpoint_2025": (
            {"outer_seasons": checkpoint["outer_seasons"], "arms": _arms(checkpoint),
             "comparisons": clean(checkpoint["comparisons"]),
             "decision": _verdicts(ck_decision),
             "outcome": (ck_decision or {}).get("outcome"),
             "inspected_after_development_decision": True}
            if checkpoint else missing("shadow_checkpoint2025")
        ),
        "prospective_2026": {
            "status": "not gradeable yet",
            "gradeable_rows": 0,
            "reason": (
                "the 2026 schedule is published (272 games, weeks 1-18) with zero recorded "
                "scores; nflverse reports the current season as 2025, publishes no 2026 "
                "weekly rosters, and returns HTTP 404 for 2026 weekly player stats"
            ),
        },
        "lineup_regret": clean(regret) if regret else missing("lineup_regret"),
    }

    metrics["champion_challenger_decision"] = (
        "REJECT both preregistered challengers (bayesian_ridge_only, "
        "hist_gradient_boosting_only); keep the direct position-specific ensemble "
        "as the frozen 2026 forecast center"
    )

    metrics["factor_audit"] = (
        {"frame_rows": audit.get("frame_rows"), "status": audit.get("status"),
         "surviving_pregame": [f.get("name") for f in (audit.get("findings") or [])
                               if f.get("cohort") == "pregame"]}
        if audit else missing("factor_audit")
    )
    metrics["nested_projection"] = (
        {"conclusion": nested.get("conclusion"),
         "highest_eligible": nested.get("highest_eligible_accuracy")}
        if nested else missing("nested_projection")
    )
    metrics["baselines_required"] = [
        "trailing fantasy points (pre_fantasy_points_ewm4)",
        "trailing expected points (pre_expected_points_ewm4)",
        "ESPN weekly projections (external challenger, never a model input)",
    ]
    return metrics


def build_registry() -> dict:
    return {
        "schema_version": 3,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "git_head": git_head(),
        "environment": environment(),
        "holdout_policy": (
            "expanding season-forward development through 2024; 2025 is the locked "
            "regression checkpoint, inspected once after the development decision is "
            "written; 2026 prospective predictions are the final judge"
        ),
        "accept_gates": ACCEPT_GATES,
        "release_thresholds": RELEASE_THRESHOLDS,
        "protocol": jload("analysis/accuracy_protocol.json"),
        "inputs": {path: sha256(path) for path in INPUTS},
        "espn_external_challenger": {
            "role": "scored against, never scored with",
            "runs_in_ci": (
                "tests/test_espn_compare.py and tests/test_espn_external_challenger.py are "
                "in the offline lane CI runs as `pytest -q -m offline`"
            ),
            "prospective_2026": "graded weekly from the pre-kickoff ledger; no backfill",
            "never_changes_predictions": (
                "asserted structurally (no projection-path module imports the ESPN layer) "
                "and behaviourally (garbage ESPN values leave every stored model "
                "projection identical)"
            ),
        },
        "metrics": collect_metrics(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if any pinned input hash differs from the last registry")
    parser.add_argument("--output", default="data/accuracy_registry.json")
    args = parser.parse_args()

    inputs = {path: sha256(path) for path in INPUTS}
    out_path = os.path.join(ROOT, args.output)

    if args.check:
        previous = jload(args.output)
        if not previous:
            print("no previous registry -- nothing to check against")
            return 1
        drifted = {
            path: (previous.get("inputs", {}).get(path), digest)
            for path, digest in inputs.items()
            if previous.get("inputs", {}).get(path) != digest
        }
        if drifted:
            print("INPUT DRIFT since last registry:")
            for path, (was, now) in sorted(drifted.items()):
                print(f"  {path}\n    was: {was}\n    now: {now}")
            return 1
        print("inputs unchanged since the last registry")
        return 0

    registry = clean(build_registry())
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(registry, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"wrote {args.output}")
    print("git_head:", registry["git_head"])
    print("decision:", registry["metrics"]["champion_challenger_decision"])
    stale = sorted(
        key for key, value in registry["metrics"].items()
        if isinstance(value, dict) and value.get("status") == "missing"
    )
    if stale:
        print("missing artifacts (reported as missing, never as a result):")
        for key in stale:
            print(f"  {key}: {registry['metrics'][key]['build_with']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
