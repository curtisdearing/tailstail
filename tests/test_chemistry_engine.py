"""Tests for nflvalue.fantasy.chemistry_engine (as-of safety, shrinkage sanity, determinism)."""
import numpy as np
import pandas as pd

from nflvalue.fantasy.chemistry_engine import (
    add_strictly_prior_cums,
    asof_pair_volume_features,
    build_pair_week_panel,
    capped_adjustment,
    cumulative_through_tables,
    derive_team_starters,
    fit_volume_constants,
)


def _toy_panel():
    rows = []
    # QB1-R1 heavy usage weeks 1-3 2020; QB2 takes over week 4; R2 background player
    for wk, (qb, tgt, rts) in enumerate([("QB1", 8, 30), ("QB1", 9, 32), ("QB1", 7, 31), ("QB2", 2, 30)], start=1):
        rows.append(dict(season=2020, week=wk, team="AAA", qb_id=qb, rec_id="R1",
                         rec_pos="WR", routes=float(rts), targets=float(tgt),
                         completions=float(tgt - 1), yards=10.0 * tgt, epa_sum=0.5 * tgt))
        rows.append(dict(season=2020, week=wk, team="AAA", qb_id=qb, rec_id="R2",
                         rec_pos="WR", routes=float(rts), targets=3.0,
                         completions=2.0, yards=20.0, epa_sum=0.2))
    p = pd.DataFrame(rows)
    p["t"] = p.season * 100 + p.week
    return p


def _toy_consts():
    return dict(mu_tpr={"WR": 0.17, "TE": 0.14, "RB": 0.15}, sig2_vol=0.13,
                tau2_rec_vol=1e-3, tau2_pair_vol=2e-4)


def _starters(p):
    db = p.groupby(["season", "week", "team", "qb_id"], as_index=False).agg(dropbacks=("routes", "max"))
    return derive_team_starters(p, db)


def test_strictly_prior_cums_exclude_current_week():
    p = add_strictly_prior_cums(_toy_panel(), ["routes", "targets"])
    r1 = p[(p.rec_id == "R1")].sort_values("t")
    assert r1.iloc[0].P_targets == 0.0 and r1.iloc[0].P_games == 0
    wk3 = r1[r1.week == 3].iloc[0]
    assert wk3.P_targets == 8 + 9 and wk3.P_routes == 30 + 32


def test_asof_features_use_only_past_weeks():
    p = _toy_panel()
    pair_cum, rec_cum = cumulative_through_tables(p)
    starters = _starters(p)
    q = pd.DataFrame([dict(rec_id="R1", position="WR", team="AAA", season=2020, week=3)])
    out = asof_pair_volume_features(q, pair_cum, rec_cum, starters, _toy_consts())
    # expected QB at week 3 is the week-2 starter (QB1), nothing from week >= 3
    assert out.iloc[0].exp_qb == "QB1"
    # pair cum at week 3 sees weeks 1-2 only: 17 targets on 62 routes
    assert out.iloc[0].C_targets == 17.0 and out.iloc[0].C_routes == 62.0
    # mutate week-3+ rows: features at week 3 must not move
    p2 = p.copy()
    p2.loc[p2.week >= 3, "targets"] = 99.0
    pc2, rc2 = cumulative_through_tables(p2)
    out2 = asof_pair_volume_features(q, pc2, rc2, starters, _toy_consts())
    assert out2.iloc[0].C_targets == out.iloc[0].C_targets
    assert np.isclose(out2.iloc[0].pair_vol_post, out.iloc[0].pair_vol_post)


def test_shrinkage_sanity():
    p = _toy_panel()
    pair_cum, rec_cum = cumulative_through_tables(p)
    starters = _starters(p)
    q = pd.DataFrame([dict(rec_id="R1", position="WR", team="AAA", season=2020, week=4)])
    out = asof_pair_volume_features(q, pair_cum, rec_cum, starters, _toy_consts()).iloc[0]
    raw_dev = out.C_targets / out.C_routes - out.rec_b_tpr
    # posterior strictly shrinks toward zero and keeps the sign
    assert abs(out.pair_vol_post) < abs(raw_dev)
    assert np.sign(out.pair_vol_post) == np.sign(raw_dev)
    # no history -> exactly zero
    q0 = pd.DataFrame([dict(rec_id="R9", position="WR", team="AAA", season=2020, week=4)])
    out0 = asof_pair_volume_features(q0, pair_cum, rec_cum, starters, _toy_consts()).iloc[0]
    assert out0.pair_vol_post == 0.0 and out0.chem_raw == 0.0
    # shrinkage weight increases with n
    w_small = 2e-4 / (2e-4 + 0.13 / 10)
    w_big = 2e-4 / (2e-4 + 0.13 / 1000)
    assert w_big > w_small


def test_fit_constants_respect_season_fence():
    p = _toy_panel()
    later = p.copy()
    later["season"] = 2024
    later["targets"] = 0.0  # radically different later data
    later["t"] = later.season * 100 + later.week
    both = pd.concat([p, later], ignore_index=True)
    c_fenced = fit_volume_constants(both, max_season=2021)
    c_all = fit_volume_constants(both, max_season=2025)
    c_early = fit_volume_constants(p, max_season=2021)
    assert c_fenced["fit_max_season"] == 2021
    assert c_fenced["mu_tpr"] == c_early["mu_tpr"]
    assert c_fenced["mu_tpr"] != c_all["mu_tpr"]


def test_determinism_and_cap():
    p = _toy_panel()
    a = fit_volume_constants(p, 2021)
    b = fit_volume_constants(p.sample(frac=1.0, random_state=7), 2021)  # row order must not matter
    assert a == b
    x = np.array([np.nan, -9.0, 0.3, 9.0])
    out = capped_adjustment(x, cap=1.5)
    assert out.tolist() == [0.0, -1.5, 0.3, 1.5]


def test_build_panel_target_implies_route():
    pass_plays = pd.DataFrame([dict(season=2020, week=1, team="AAA", qb_id="QB1", rec_id="R3",
                                    play_id=1, complete_pass=1.0, passing_yards=12.0, epa=0.8)])
    routes = pd.DataFrame([dict(season=2020, week=1, team="AAA", qb_id="QB1", rec_id="R1",
                                rec_pos="WR", routes=20.0)])
    r3 = routes.copy(); r3["rec_id"] = "R3"; r3["routes"] = 0.0; r3["rec_pos"] = "WR"
    panel = build_pair_week_panel(pass_plays, pd.concat([routes, r3], ignore_index=True))
    row = panel[panel.rec_id == "R3"].iloc[0]
    assert row.routes >= row.targets >= 1.0
