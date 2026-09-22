"""RB role-change error diagnosis (2026-09-22) — see docs/prereg/RB_ROLE_CHANGE_DIAGNOSIS_2026-09-22.md.

Reproduces every number in that note.  Read-only: it fits the production
ensemble season-forward and measures where running-back error lives.  It
changes no serving path and registers no lever.

Usage (from the repo root, with historical/fantasy/*.parquet present)::

    python analysis/rb_role_change_diagnosis.py --out-dir <dir>

Stage 1 builds the 2019-2025 roster-first feature frame, stage 2 refits the
production ``ModelConfig`` before each development test season (2021-2023) and
predicts it with strictly-prior training data, stage 3 measures the cohorts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nflvalue.fantasy.config import ModelConfig
from nflvalue.fantasy.data import HistoricalData
from nflvalue.fantasy.features import build_feature_frame
from nflvalue.fantasy.models import fit_ensemble

#: Development folds only (protocol freeze 2026 §2: expanding through 2024,
#: 2025 stays a locked checkpoint).  2021 is the first season with two prior
#: training seasons available.
DEV_TEST_SEASONS = (2021, 2022, 2023)
#: Realized opportunity-change thresholds, identical to the frozen role-state
#: audit labels in ``nflvalue.fantasy.role_state``.
DECREASE, INCREASE, MAJOR = -5.0, 5.0, 12.0
BOOTSTRAP_SEED = 20260922
BOOTSTRAP_DRAWS = 5000


def build_frame(history_dir: str, max_season: int = 2025) -> pd.DataFrame:
    data = HistoricalData.load(history_dir)

    def cut(frame: pd.DataFrame | None) -> pd.DataFrame | None:
        if frame is None:
            return None
        return frame[pd.to_numeric(frame["season"], errors="coerce").le(max_season)].copy()

    trimmed = HistoricalData(
        stats=cut(data.stats), rosters=cut(data.rosters), schedules=cut(data.schedules),
        snaps=cut(data.snaps), injuries=cut(data.injuries),
        expected_points=cut(data.expected_points),
    )
    return build_feature_frame(trimmed)


def walk_forward(frame: pd.DataFrame, seasons=DEV_TEST_SEASONS) -> pd.DataFrame:
    """Refit before every test season; training rows are strictly earlier seasons."""
    config = ModelConfig(fast=False, stack_validation_seasons=3)
    predictions = []
    for season in seasons:
        train = frame[frame["season"].astype(int) < season]
        test = frame[
            frame["season"].astype(int).eq(season) & frame["model_eligible"].fillna(False)
        ]
        if int(train["season"].max()) != season - 1:
            raise AssertionError("training frame is not strictly prior to the test season")
        predictions.append(fit_ensemble(train, config=config).predict(test))
    return pd.concat(predictions, ignore_index=True)


def add_cohorts(frame: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Attach realized regimes and strictly-pregame role-shock flags."""
    out = frame.copy()
    out["role_delta"] = out["opportunities"] - out["pre_opportunities_ewm4"]
    out["regime"] = pd.cut(
        out["role_delta"], [-np.inf, DECREASE, INCREASE, MAJOR, np.inf],
        labels=["down", "stable", "up", "major_up"],
    ).astype(str)
    out["role_change"] = out["role_delta"].abs().ge(INCREASE)

    # Serving-safe vacancy: the official injury report is published on Friday,
    # so ``injury_out`` exists at projection time.  ``status_inactive`` comes
    # from the weekly-roster file, which nflverse refreshes AFTER gameday, so a
    # backtest that reads it sees information the serving path does not have.
    seats = source[source["season"].isin(out["season"].unique())].copy()
    seats["_rank"] = seats.groupby(["season", "week", "team", "position"])[
        "pre_opportunities_ewm4"
    ].rank(method="first", ascending=False)
    for name, flag in (("report", "injury_out"), ("status", "status_inactive")):
        vacant = (
            seats["position"].eq("RB") & seats["_rank"].eq(1) & seats[flag].eq(1)
        ).astype(int)
        seats[f"rb1_out_{name}"] = vacant.groupby(
            [seats["season"], seats["week"], seats["team"]]
        ).transform("max")
    seats["role_rank"] = seats["_rank"].fillna(99.0).clip(1, 99)
    keep = ["season", "week", "player_id", "role_rank", "rb1_out_report", "rb1_out_status"]
    out = out.merge(seats[keep], on=["season", "week", "player_id"], how="left")

    prior_rank = out.sort_values(["player_id", "season", "week"]).groupby("player_id")[
        "pre_position_role_rank"
    ].shift(1)
    out["rank_moved"] = (prior_rank.notna() & prior_rank.ne(out["pre_position_role_rank"])).astype(int)
    out["trend_big"] = out["pre_opportunities_trend_2v8"].abs().ge(4).astype(int)
    out["thin_history"] = out["pre_played_games"].lt(4).astype(int)
    out["pregame_shock"] = (
        out["team_changed"].eq(1) | out["rb1_out_report"].eq(1)
        | out["rank_moved"].eq(1) | out["trend_big"].eq(1)
    ).astype(int)
    return out


def cell_metrics(group: pd.DataFrame) -> dict[str, float]:
    actual = group["fantasy_points"].to_numpy(float)
    predicted = group["projection_mean"].to_numpy(float)
    slope = float(np.polyfit(predicted, actual, 1)[0]) if len(group) > 2 else float("nan")
    covered = group["fantasy_points"].between(
        group["projection_lower80"], group["projection_upper80"]
    )
    return {
        "n": int(len(group)),
        "mae": float(np.abs(actual - predicted).mean()),
        "bias": float((predicted - actual).mean()),
        "sd_pred": float(predicted.std(ddof=1)),
        "sd_actual": float(actual.std(ddof=1)),
        "spearman": float(spearmanr(actual, predicted).statistic),
        "calibration_slope": slope,
        "coverage80": float(covered.mean()),
    }


def oracle_shift_bounds(rb: pd.DataFrame) -> dict[str, float]:
    """Upper bound on ANY role-conditional mean adjustment, granted oracle shifts.

    The MAE-optimal constant shift for a cell is its MEDIAN residual; the
    RMSE-optimal one is its mean.  Both are computed in sample, so they bound
    from above what a fitted lever could achieve out of sample.
    """
    residual = rb["projection_mean"] - rb["fantasy_points"]
    quantile = pd.qcut(rb["pre_opportunities_ewm4"], 5, duplicates="drop").astype(str)
    cell = (
        rb["role_rank"].clip(1, 4).astype(int).astype(str) + "|"
        + rb["rb1_out_report"].fillna(0).astype(int).astype(str) + "|" + quantile
    )
    base = float(np.abs(residual).mean())
    result = {"baseline_mae": base, "cells": int(cell.nunique())}
    for stat in ("mean", "median"):
        for name, key in (("per_cell", cell), ("global", pd.Series("g", index=rb.index))):
            adjusted = rb["projection_mean"] - residual.groupby(key).transform(stat)
            result[f"{stat}_{name}_mae"] = float(
                np.abs(rb["fantasy_points"] - adjusted).mean()
            )
    for factor in (1.1, 1.25, 1.5):
        centre = rb["projection_mean"].mean()
        expanded = centre + (rb["projection_mean"] - centre) * factor
        result[f"expand_x{factor}_mae"] = float(
            np.abs(rb["fantasy_points"] - expanded).mean()
        )
    return result


def week_block_bootstrap(rb: pd.DataFrame) -> dict[str, list[float]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    blocks = rb.groupby(["season", "week"]).indices
    keys = list(blocks)
    error = (rb["fantasy_points"] - rb["projection_mean"]).abs().to_numpy()
    change = rb["role_change"].to_numpy()

    def statistics(index: np.ndarray) -> tuple[float, ...]:
        chg = change[index]
        return (
            error[index].mean(), error[index][~chg].mean(),
            error[index][chg].mean(), chg.mean(),
        )

    observed = statistics(np.arange(len(rb)))
    draws = np.empty((BOOTSTRAP_DRAWS, 4))
    for draw in range(BOOTSTRAP_DRAWS):
        pick = rng.choice(len(keys), len(keys), replace=True)
        draws[draw] = statistics(np.concatenate([blocks[keys[i]] for i in pick]))
    names = ["mae", "mae_stable", "mae_role_change", "role_change_rate"]
    report = {
        name: [float(observed[i]), *np.percentile(draws[:, i], [2.5, 97.5]).tolist()]
        for i, name in enumerate(names)
    }
    gap = draws[:, 2] - draws[:, 1]
    report["mae_change_minus_stable"] = [
        float(observed[2] - observed[1]), *np.percentile(gap, [2.5, 97.5]).tolist()
    ]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir", default="historical/fantasy")
    parser.add_argument("--out-dir", default="reports/rb_role_change")
    arguments = parser.parse_args()
    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = build_frame(arguments.history_dir)
    predicted = walk_forward(frame)
    played = add_cohorts(predicted[predicted["played"].fillna(False)], frame)
    rb = played[played["position"].eq("RB")].copy()

    report: dict[str, object] = {
        "dev_test_seasons": list(DEV_TEST_SEASONS),
        "rows": {"played": int(len(played)), "rb_played": int(len(rb))},
        "by_position_regime": {},
        "rb_pregame_cohorts": {},
        "oracle_bounds": oracle_shift_bounds(rb),
        "bootstrap_week_clustered": week_block_bootstrap(rb),
    }
    for position, group in played.groupby("position"):
        report["by_position_regime"][position] = {
            "role_change_rate": float(group["role_change"].mean()),
            "regimes": {
                regime: cell_metrics(cell)
                for regime, cell in group.groupby("regime") if len(cell) >= 8
            },
        }
    for flag in ("team_changed", "rb1_out_report", "rb1_out_status", "rank_moved",
                 "trend_big", "thin_history", "pregame_shock"):
        report["rb_pregame_cohorts"][flag] = {
            str(value): cell_metrics(cell)
            for value, cell in rb.groupby(flag) if len(cell) >= 20
        }

    (out_dir / "rb_role_change_diagnosis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    rb.to_parquet(out_dir / "rb_dev_rows.parquet", index=False)
    print(json.dumps(report["bootstrap_week_clustered"], indent=2))
    print(json.dumps(report["oracle_bounds"], indent=2))


if __name__ == "__main__":
    main()
