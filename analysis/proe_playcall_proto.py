#!/usr/bin/env python3
"""PROE/playcall prototype (E-family, wave 2) -- NOT part of the 40-test battery.

Strictly-prior team PROE (mean pass_oe, shift-then-roll) + opponent PROE-allowed +
pregame spread/total/implied -> predicted team pass attempts / dropbacks / plays
via expanding season-forward ridge. Compared against:
  B0  trailing volume: pre_team_pass_attempts_ewm4 (production feature), and
  B1  current sim volume node: clip(B0 * (1 + spread/75), 15, 55)
      (nflvalue/fantasy/simulation.py pass_lambda center; QB inherits team attempts).
Then the QB-attempts translation on the frozen outer QB rows (2023-2025): replace
the volume node, keep everything else; report exact before/after bias and MAE
(production sim reference: attempts overprojected +1.63, pass yards +21.5).

Run (chunked, 45s-safe):
    PYTHONPATH=. python3 analysis/proe_playcall_proto.py --stage frames
    PYTHONPATH=. python3 analysis/proe_playcall_proto.py --stage eval
Scratch/results: /tmp/exp_families/proe_frame.parquet, /tmp/exp_families/proe_results.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.factor_families_study import (
    EXP,
    ROOT,
    SEED,
    _ewm_trailing,
    _load_pbp,
    build_defense_week,
    build_team_week,
    load_game_meta,
)

FEATURES = ["pre_team_pass_attempts_ewm4", "tr_pass_att", "tr_plays", "tr_dropbacks",
            "tr_proe", "tr_neutral_proe", "opp_tr_proe_allowed", "opp_tr_pass_att_allowed",
            "opp_tr_plays_allowed", "team_spread", "total_line", "implied_pts", "is_home"]
TEST_SEASONS = (2023, 2024, 2025)

def stage_frames():
    os.makedirs(EXP, exist_ok=True)
    pbp = _load_pbp()
    cov = pbp.groupby("season")["pass_oe"].apply(lambda s: float(s.notna().mean()))
    xcov = pbp.groupby("season")["xpass"].apply(lambda s: float(s.notna().mean()))
    tw = build_team_week(pbp)
    # neutral-situation PROE (wp 20-80%, first 3 quarters)
    neu = pbp[pbp.wp.between(0.2, 0.8) & (pbp.qtr <= 3) & pbp.pass_oe.notna()]
    np_ = neu.groupby(["game_id", "posteam"], as_index=False)["pass_oe"].mean() \
             .rename(columns={"posteam": "team", "pass_oe": "neutral_proe"})
    tw = tw.merge(np_, on=["game_id", "team"], how="left")
    tw = _ewm_trailing(tw, ["team"], ["team", "season", "week"],
                       ["pass_att", "plays", "dropbacks", "proe", "neutral_proe"])
    dw = build_defense_week(pbp, tw)
    dw = _ewm_trailing(dw, ["team"], ["team", "season", "week"],
                       ["proe_allowed", "pass_att_allowed", "plays_allowed"])
    dkey = dw[["season", "week", "team", "tr_proe_allowed", "tr_pass_att_allowed",
               "tr_plays_allowed"]].rename(columns={
                   "team": "opponent", "tr_proe_allowed": "opp_tr_proe_allowed",
                   "tr_pass_att_allowed": "opp_tr_pass_att_allowed",
                   "tr_plays_allowed": "opp_tr_plays_allowed"})
    gm = load_game_meta()[["season", "week", "game_id", "team", "opponent", "is_home",
                           "team_spread", "total_line", "implied_pts"]]
    tm = tw.merge(gm, on=["season", "week", "game_id", "team"], how="inner")
    tm = tm.merge(dkey, on=["season", "week", "opponent"], how="left")
    # production trailing feature + feature-frame actual attempts (player-sum units)
    ff = pd.read_parquet(os.path.join(ROOT, "historical", "fantasy", "feature_frame.parquet"),
                         columns=["season", "week", "team", "attempts", "played",
                                  "pre_team_pass_attempts_ewm4"])
    act = ff[ff.played == 1].groupby(["season", "week", "team"], as_index=False) \
        .agg(ff_pass_att=("attempts", "sum"))
    pre = ff.dropna(subset=["pre_team_pass_attempts_ewm4"]).drop_duplicates(
        ["season", "week", "team"])[["season", "week", "team", "pre_team_pass_attempts_ewm4"]]
    tm = tm.merge(act, on=["season", "week", "team"], how="left")
    tm = tm.merge(pre, on=["season", "week", "team"], how="left")
    tm.to_parquet(os.path.join(EXP, "proe_frame.parquet"), index=False)
    meta = dict(pass_oe_nonnull_by_season={int(k): round(v, 4) for k, v in cov.items()},
                xpass_nonnull_by_season={int(k): round(v, 4) for k, v in xcov.items()},
                n_team_games=len(tm),
                att_convention_gap=float((tm.pass_att - tm.ff_pass_att).abs().mean()))
    with open(os.path.join(EXP, "proe_frames_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    print("proe frames done")

def _season_forward_ridge(tm, target):
    """Expanding season-forward ridge; returns OOS predictions for TEST_SEASONS."""
    ok = tm.dropna(subset=FEATURES + [target])
    preds = []
    for y in TEST_SEASONS:
        tr = ok[ok.season < y]
        te = ok[ok.season == y]
        if te.empty:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED))
        model.fit(tr[FEATURES], tr[target])
        p = te[["season", "week", "game_id", "team"]].copy()
        p[f"pred_{target}"] = model.predict(te[FEATURES])
        preds.append(p)
    return pd.concat(preds, ignore_index=True)

def _metrics(pred, actual):
    m = pd.notna(pred) & pd.notna(actual)
    e = pred[m] - actual[m]
    return dict(mae=float(e.abs().mean()), bias=float(e.mean()), n=int(m.sum()))

def stage_eval():
    tm = pd.read_parquet(os.path.join(EXP, "proe_frame.parquet"))
    test = tm[tm.season.isin(TEST_SEASONS)].copy()
    test["b1_current_node"] = np.clip(
        test.pre_team_pass_attempts_ewm4 * (1 + test.team_spread / 75.0), 15, 55)
    out = dict(team_eval={}, features=FEATURES,
               train_scheme="expanding season-forward (2019..y-1 -> y), test 2023-2025")
    with open(os.path.join(EXP, "proe_frames_meta.json")) as f:
        fm = json.load(f)
    out["coverage_note"] = ("pass_oe/xpass nonnull by season " +
                            str(fm["pass_oe_nonnull_by_season"]) +
                            f"; attempts-convention gap pbp-vs-player-sum {fm['att_convention_gap']:.2f} att")
    # primary: team pass attempts in production (player-sum) units
    ridge_att = _season_forward_ridge(tm, "ff_pass_att")
    test = test.merge(ridge_att, on=["season", "week", "game_id", "team"], how="left")
    node = {}
    for name, col in [("B0_trailing_ewm4", "pre_team_pass_attempts_ewm4"),
                      ("B1_current_sim_node", "b1_current_node"),
                      ("ridge_proe", "pred_ff_pass_att")]:
        mm = _metrics(test[col], test.ff_pass_att)
        mm["per_season"] = {int(y): _metrics(test.loc[test.season == y, col],
                                             test.loc[test.season == y, "ff_pass_att"])
                            for y in TEST_SEASONS}
        node[name] = mm
    out["team_eval"]["pass_att"] = node
    out["n_team_games_test"] = int(test.ff_pass_att.notna().sum())
    # secondary: dropbacks and plays vs trailing baselines
    for target, base in [("dropbacks", "tr_dropbacks"), ("plays", "tr_plays")]:
        r = _season_forward_ridge(tm, target)
        t2 = tm[tm.season.isin(TEST_SEASONS)].merge(
            r, on=["season", "week", "game_id", "team"], how="left")
        out["team_eval"][target] = {
            "B0_trailing": _metrics(t2[base], t2[target]),
            "ridge_proe": _metrics(t2[f"pred_{target}"], t2[target])}
    # QB translation on the frozen outer predictions
    op = pd.read_parquet(os.path.join(ROOT, "reports", "fantasy_outer_predictions.parquet"))
    qbs = op[op.position == "QB"].copy()
    # the simulator hands TEAM attempts to its chosen QB = argmax projection_mean per
    # team-week; the recorded +1.63 production bias lives on that starter pathway.
    starter_idx = qbs.groupby(["season", "week", "team"]).projection_mean.idxmax()
    qbs["is_primary"] = qbs.index.isin(starter_idx)
    qb = qbs[["season", "week", "player_id", "team", "is_primary"]].copy()
    ffq = pd.read_parquet(os.path.join(ROOT, "historical", "fantasy", "feature_frame.parquet"),
                          columns=["season", "week", "player_id", "attempts", "passing_yards",
                                   "pre_attempts_ewm4", "pre_passing_yards_ewm4",
                                   "pre_team_pass_attempts_ewm4", "team_spread"])
    qb = qb.merge(ffq, on=["season", "week", "player_id"], how="left")
    qb = qb.merge(ridge_att.rename(columns={"pred_ff_pass_att": "att_ridge"})[
        ["season", "week", "team", "att_ridge"]], on=["season", "week", "team"], how="left")
    qb["att_before"] = np.clip(qb.pre_team_pass_attempts_ewm4 * (1 + qb.team_spread / 75.0), 15, 55)
    n_fallback = int(qb.att_ridge.isna().sum())
    qb["att_after"] = qb.att_ridge.fillna(qb.att_before)
    ypa = (qb.pre_passing_yards_ewm4 / qb.pre_attempts_ewm4.clip(lower=1)).clip(4, 12)
    qb["ypa_tr"] = ypa.fillna(7.0)
    qb["yds_before"] = qb.att_before * qb.ypa_tr
    qb["yds_after"] = qb.att_after * qb.ypa_tr
    pri = qb[qb.is_primary]
    out["qb_translation"] = dict(
        n=int(pri.attempts.notna().sum()), fallback_rows=n_fallback,
        cohort="primary QB per team-week (argmax projection_mean), matching the sim's starter pathway",
        attempts=dict(before=_metrics(pri.att_before, pri.attempts),
                      after=_metrics(pri.att_after, pri.attempts)),
        pass_yards=dict(before=_metrics(pri.yds_before, pri.passing_yards),
                        after=_metrics(pri.yds_after, pri.passing_yards)),
        all_qb_rows_secondary=dict(
            n=int(qb.attempts.notna().sum()),
            attempts=dict(before=_metrics(qb.att_before, qb.attempts),
                          after=_metrics(qb.att_after, qb.attempts)),
            pass_yards=dict(before=_metrics(qb.yds_before, qb.passing_yards),
                            after=_metrics(qb.yds_after, qb.passing_yards))))
    # verdict
    pa = out["team_eval"]["pass_att"]
    r_beats = (pa["ridge_proe"]["mae"] < pa["B0_trailing_ewm4"]["mae"] and
               pa["ridge_proe"]["mae"] < pa["B1_current_sim_node"]["mae"])
    per_ok = all(pa["ridge_proe"]["per_season"][y]["mae"] <
                 pa["B1_current_sim_node"]["per_season"][y]["mae"] for y in TEST_SEASONS)
    at = out["qb_translation"]["attempts"]
    bias_ok = abs(at["after"]["bias"]) < abs(at["before"]["bias"])
    mae_ok = at["after"]["mae"] <= at["before"]["mae"] + 1e-9
    if r_beats and per_ok and bias_ok and mae_ok:
        out["registry_status"] = "research_only"
        out["verdict"] = ("PROE volume node beats both baselines on team pass attempts in every "
                          "test season and shrinks the QB attempts bias without hurting MAE; "
                          "promote to a full-simulation ablation (role-shock harness) before shadow.")
    elif r_beats:
        out["registry_status"] = "research_only"
        out["verdict"] = ("PROE volume node beats baselines pooled but not uniformly; "
                          "keep research_only.")
    else:
        out["registry_status"] = "rejected"
        out["verdict"] = "PROE volume node does not beat the current node season-forward."
    with open(os.path.join(EXP, "proe_results.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["frames", "eval", "all"])
    a = ap.parse_args()
    if a.stage in ("frames", "all"):
        stage_frames()
    if a.stage in ("eval", "all"):
        stage_eval()

if __name__ == "__main__":
    main()
