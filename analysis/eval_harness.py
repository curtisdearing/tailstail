#!/usr/bin/env python3
"""Accuracy harness (fantasy flavor): one command, one registry (accuracy loop plan, P2).

Collects the repo's CURRENT accuracy metrics from the canonical result
artifacts, pins the SHA-256 of every model input, and writes
data/accuracy_registry.json. The registry is the single scoreboard the
weekly lever loop reads and the accept gates are checked against.

    python3 analysis/eval_harness.py            # collect + write + print
    python3 analysis/eval_harness.py --check    # exit 1 if inputs drifted
                                                # since the last registry

This harness never computes new metrics itself: heavy evaluation stays in
the audited CLIs (python -m nflvalue.fantasy.cli backtest/train, analysis/all_data_factor_audit.py).
Rerun those first when a lever changes the model, then this collector.

Accept gates (pre-registered, accuracy_loop_plan.md): a lever is accepted
only if it moves a primary metric by at least the gate at a declared 2025
locked-regression checkpoint. Prospective 2026 predictions are the final judge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUTS = [
    "historical/historical_pbp.parquet", "historical_lines.parquet",
    "historical/rosters_weekly.parquet", "historical/injuries.parquet",
    "historical/players_meta.parquet", "historical/fantasy/feature_frame.parquet",
    "data/factor_frame.parquet", "config.json", "analysis/accuracy_protocol.json",
]

ACCEPT_GATES = {
    "fantasy_mae_points": -0.05,        # paired bootstrap p<0.10 on 2025 holdout
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


def sha256(path: str):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jload(path: str, default=None):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return default
    with open(p) as fh:
        return json.load(fh)


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def collect() -> dict:
    audit = jload("data/all_data_factor_audit.json", {}) or {}
    nested = jload("data/nested_factor_projection.json", {}) or {}
    card = jload("reports/fantasy_model_card.json", {}) or {}
    red = jload("reports/fantasy_red_team.json", {}) or {}
    quality = jload("reports/fantasy_data_quality.json", {}) or {}
    mc = jload("reports/fantasy_monte_carlo_history.json", {}) or {}
    calibrated = (mc.get("methods") or {}).get("calibrated_monte_carlo") or {}
    coverage = calibrated.get("coverage80")
    nominal = RELEASE_THRESHOLDS["nominal_interval_coverage"]

    metrics = {
        "fantasy_model_card": card or "not built yet (run: python -m nflvalue.fantasy.cli train)",
        "fantasy_red_team_backtest": red or "not built yet (run: python -m nflvalue.fantasy.cli backtest)",
        "fantasy_data_quality": {"present": bool(quality)},
        "factor_audit": {
            "frame_rows": audit.get("frame_rows"),
            "status": audit.get("status"),
            "surviving_pregame": [f.get("name") for f in (audit.get("findings") or [])
                                   if f.get("cohort") == "pregame"],
        },
        "nested_projection": {
            "conclusion": nested.get("conclusion"),
            "highest_eligible": nested.get("highest_eligible_accuracy"),
        },
        "simulation_calibration": {
            "n": calibrated.get("n"),
            "nominal_coverage": nominal,
            "observed_coverage": coverage,
            "absolute_coverage_error_pp": (
                round(abs(float(coverage) - nominal) * 100, 4)
                if coverage is not None else None
            ),
            "undercoverage_penalty_pp": (
                round(max(nominal - float(coverage), 0.0) * 100, 4)
                if coverage is not None else None
            ),
            "mean_interval_width": calibrated.get("mean_interval_width"),
            "release_gate": mc.get("release_gate"),
        },
        "baselines_required": ["trailing-mean (3-game rolling)", "public consensus"],
    }
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any pinned input hash differs from the last registry")
    ap.add_argument("--output", default="data/accuracy_registry.json")
    args = ap.parse_args()

    inputs = {p: sha256(p) for p in INPUTS}
    out_path = os.path.join(ROOT, args.output)

    if args.check:
        prev = jload(args.output)
        if not prev:
            print("no previous registry -- nothing to check against")
            return 1
        drifted = {p: (prev.get("inputs", {}).get(p), h) for p, h in inputs.items()
                   if prev.get("inputs", {}).get(p) != h}
        if drifted:
            print("INPUT DRIFT since last registry:")
            for p, (old, new) in drifted.items():
                print(f"  {p}: {str(old)[:12]} -> {str(new)[:12]}")
            return 1
        print("inputs unchanged since last registry")
        return 0

    registry = {
        "schema_version": 2,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "git_head": git_head(),
        "holdout_policy": "tune on 2020-2024 walk-forward; 2025 is a locked regression checkpoint; 2026 prospective predictions are final",
        "accept_gates": ACCEPT_GATES,
        "release_thresholds": RELEASE_THRESHOLDS,
        "protocol": jload("analysis/accuracy_protocol.json", {}),
        "inputs": inputs,
        "metrics": collect(),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(registry, fh, indent=1)

    m = registry["metrics"]
    print(f"accuracy registry @ {registry['git_head']} -> {args.output}")
    card = m.get("fantasy_model_card")
    if isinstance(card, dict) and card:
        for k in ["mae", "rmse", "positions", "scoring", "seasons"]:
            if k in card:
                print(f"  model card {k}: {card[k]}")
    else:
        print(f"  fantasy model card: {card}")
    fa = m.get("factor_audit") or {}
    print(f"  factor audit: {fa.get('frame_rows')} rows, surviving pregame: {len(fa.get('surviving_pregame') or [])}")
    print(f"  nested: {(m.get('nested_projection') or {}).get('conclusion')}")
    sc = m.get("simulation_calibration") or {}
    print(f"  simulation 80% interval: observed {sc.get('observed_coverage')} "
          f"absolute error {sc.get('absolute_coverage_error_pp')}pp "
          f"undercoverage penalty {sc.get('undercoverage_penalty_pp')}pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
