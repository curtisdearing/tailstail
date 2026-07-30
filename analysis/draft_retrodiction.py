"""Retrodiction grade for the draft board methodology.

For each target season N in 2023-2025, build a preseason board using ONLY
season N-1 information (per-game baselines, availability, age priors), then
grade the projected season ranking against realized season-N PPR totals.

Leakage honesty: the shipped ensemble was trained through 2025, so blending
its weekly predictions into a 2022-sourced baseline would leak future
knowledge into the retrodiction.  The headline grade therefore runs with
``model_weight=0`` (realized-points baseline + the same shrinkage, age and
availability machinery) — a LOWER BOUND on the board's quality.  A second,
clearly-labeled leaky variant (``model_weight=0.6`` with the 2025 model)
is reported for reference only and must never be quoted as evidence.

Baselines to beat:
* ``naive_lastyear`` — rank by last season's total points (what a casual
  drafter does);
* ``naive_pergame`` — rank by last season's per-game points, min 6 games.

Usage: PYTHONPATH=. python analysis/draft_retrodiction.py
Writes reports/draft_retrodiction.json + .md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nflvalue.fantasy.draft import (
    apply_offseason_adjustments,
    pergame_baselines,
    simulate_season,
)
from nflvalue.fantasy.models import FantasyEnsemble

FRAME = "historical/fantasy/feature_frame.parquet"
MODEL = "data/fantasy_model.joblib"
TARGETS = (2023, 2024, 2025)
TOP_K = (12, 24, 36)


def season_byes(frame: pd.DataFrame, season: int) -> dict[str, list[int]]:
    rows = frame[frame["season"] == season]
    byes: dict[str, list[int]] = {}
    for team, group in rows.groupby("team"):
        played = set(group["week"].unique())
        byes[str(team)] = sorted(set(range(1, 19)) - played)
    return byes


def actual_totals(frame: pd.DataFrame, season: int) -> pd.Series:
    rows = frame[(frame["season"] == season) & frame["fantasy_points"].notna()]
    return rows.groupby("player_id")["fantasy_points"].sum()


class _NullModel:
    """Stands in for the ensemble when model_weight=0 (no leakage)."""

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "player_id": rows["player_id"],
            "week": rows["week"],
            "projection_mean": np.nan,
        })


def grade(projected: pd.Series, actual: pd.Series) -> dict[str, float]:
    joined = pd.concat([projected.rename("proj"), actual.rename("act")], axis=1).dropna()
    rho = float(spearmanr(joined["proj"], joined["act"]).statistic)
    metrics = {"n": int(len(joined)), "spearman": round(rho, 4)}
    proj_rank = joined["proj"].rank(ascending=False)
    act_rank = joined["act"].rank(ascending=False)
    for k in TOP_K:
        hits = int(((proj_rank <= k) & (act_rank <= k)).sum())
        metrics[f"top{k}_hits"] = hits
        metrics[f"top{k}_rate"] = round(hits / k, 3)
    return metrics


def run() -> dict:
    frame = pd.read_parquet(FRAME)
    model = FantasyEnsemble.load(MODEL)
    null_model = _NullModel()
    report: dict = {"targets": {}}

    for target in TARGETS:
        source = target - 1
        byes = season_byes(frame, target)
        actual = actual_totals(frame, target)
        variants = {}

        for label, mdl, weight in (
            ("board_no_model_LOWER_BOUND", null_model, 0.0),
            ("board_leaky_model_REFERENCE_ONLY", model, 0.6),
        ):
            baselines = pergame_baselines(
                frame, mdl, source_season=source, model_weight=weight
            )
            baselines = apply_offseason_adjustments(baselines)
            outlook = simulate_season(baselines, byes, simulations=1500, random_seed=42)
            projected = outlook.board.set_index("player_id")["season_mean"]
            variants[label] = grade(projected, actual)

        # Naive baselines from season N-1 only.
        last = frame[(frame["season"] == source) & frame["fantasy_points"].notna()]
        by_player = last.groupby("player_id")["fantasy_points"].agg(["sum", "mean", "count"])
        variants["naive_lastyear_total"] = grade(by_player["sum"], actual)
        pergame = by_player.loc[by_player["count"] >= 6, "mean"]
        variants["naive_pergame_min6"] = grade(pergame, actual)

        report["targets"][str(target)] = variants

    # Aggregate spearman across seasons per variant.
    summary = {}
    for variant in next(iter(report["targets"].values())):
        rhos = [report["targets"][str(t)][variant]["spearman"] for t in TARGETS]
        top24 = [report["targets"][str(t)][variant]["top24_rate"] for t in TARGETS]
        summary[variant] = {
            "mean_spearman": round(float(np.mean(rhos)), 4),
            "mean_top24_rate": round(float(np.mean(top24)), 3),
        }
    report["summary"] = summary
    return report


def main() -> int:
    report = run()
    Path("reports").mkdir(exist_ok=True)
    Path("reports/draft_retrodiction.json").write_text(json.dumps(report, indent=1))
    lines = [
        "# Draft-board retrodiction (2023-2025)", "",
        "Preseason boards from season N-1 data graded against realized season-N",
        "PPR totals. `board_no_model_LOWER_BOUND` is the honest, leakage-free",
        "grade; the leaky variant exists only to bound what model blending adds.", "",
        "| variant | mean Spearman | mean top-24 hit rate |", "|---|---|---|",
    ]
    for variant, row in report["summary"].items():
        lines.append(f"| {variant} | {row['mean_spearman']} | {row['mean_top24_rate']} |")
    lines.append("")
    for target, variants in report["targets"].items():
        lines.append(f"## {target}")
        for variant, metrics in variants.items():
            lines.append(
                f"- {variant}: n={metrics['n']}, spearman={metrics['spearman']}, "
                f"top24 {metrics['top24_hits']}/24"
            )
        lines.append("")
    Path("reports/draft_retrodiction.md").write_text("\n".join(lines))
    print(json.dumps(report["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
