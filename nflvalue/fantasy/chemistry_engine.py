"""QB-receiver chemistry engine: pure, as-of-safe builders (NEW file, 2026-07-16 chemistry agent).

Contract
--------
* Every "as-of week t" feature uses ONLY plays from weeks strictly before t
  (merge_asof with allow_exact_matches=False on t = season*100 + week).
* Hierarchy: league/position mean -> receiver main effect (EB) -> pair deviation (EB).
* Volume chemistry (targets per route with a given QB) is kept SEPARATE from
  efficiency chemistry (per-target value). On 2016-2022 data the efficiency pair
  variance component fits to exactly zero (tau2_pair_epa = tau2_pair_cpoe = 0),
  so the efficiency posterior is identically 0 -- the volume channel is the only
  live chemistry signal. See reports/chemistry_condition_engine.md.
* Constants must be fit on seasons strictly before any evaluation season
  (fit_volume_constants(pair_week, max_season=Y) for test year Y).

This module is deliberately I/O-free: callers pass DataFrames in and get
DataFrames back. analysis/chemistry_study.py shows the full pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RECEIVER_POSITIONS = ("WR", "TE", "RB")
ADJUSTMENT_CAP_PTS = 1.5


def _t(season: pd.Series, week: pd.Series) -> pd.Series:
    """Sortable season-week clock."""
    return (season.astype("int64") * 100 + week.astype("int64")).astype("int64")


def build_pair_week_panel(pass_plays: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    """Pair-week panel from targeted pass plays + on-field route proxies.

    pass_plays: one row per targeted pass attempt with columns
        season, week, team, qb_id, rec_id, play_id, complete_pass, passing_yards, epa
    routes: one row per (season, week, team, qb_id, rec_id, rec_pos) with `routes`
        = count of dropbacks of qb_id with rec_id on the field (route proxy).
    """
    agg = (pass_plays.groupby(["season", "week", "team", "qb_id", "rec_id"], as_index=False)
           .agg(targets=("play_id", "size"),
                completions=("complete_pass", "sum"),
                yards=("passing_yards", "sum"),
                epa_sum=("epa", "sum")))
    panel = routes.merge(agg, on=["season", "week", "team", "qb_id", "rec_id"], how="outer")
    # a target implies at least one route even if participation missed the player
    panel["routes"] = panel[["routes", "targets"]].max(axis=1)
    for c in ("targets", "completions", "yards", "epa_sum", "routes"):
        panel[c] = panel[c].fillna(0.0)
    panel = panel[panel.rec_pos.isin(RECEIVER_POSITIONS + ("FB", "HB"))].copy()
    panel["rec_pos"] = panel["rec_pos"].replace({"FB": "RB", "HB": "RB"})
    panel["t"] = _t(panel.season, panel.week)
    return panel.sort_values(["qb_id", "rec_id", "t"]).reset_index(drop=True)


def add_strictly_prior_cums(panel: pd.DataFrame, sum_cols: list[str]) -> pd.DataFrame:
    """P_<col> = pair cumulative sum over weeks strictly before the row's week."""
    panel = panel.sort_values(["qb_id", "rec_id", "t"]).reset_index(drop=True)
    g = panel.groupby(["qb_id", "rec_id"], sort=False)
    for c in sum_cols:
        panel[f"P_{c}"] = g[c].cumsum() - panel[c]
    panel["P_games"] = g.cumcount()
    return panel


def cumulative_through_tables(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cum-through-week tables for as-of lookups at arbitrary (qb, rec, t).

    Returns (pair_cum, rec_cum). Query them with merge_asof(...,
    allow_exact_matches=False) so a lookup at week t sees data < t only.
    """
    p = panel.sort_values(["qb_id", "rec_id", "t"]).reset_index(drop=True)
    g = p.groupby(["qb_id", "rec_id"], sort=False)
    pair_cum = p[["qb_id", "rec_id", "t"]].copy()
    pair_cum["C_routes"] = g.routes.cumsum()
    pair_cum["C_targets"] = g.targets.cumsum()

    r = (p.groupby(["rec_id", "rec_pos", "t"], as_index=False)[["routes", "targets"]].sum()
         .sort_values(["rec_id", "t"]).reset_index(drop=True))
    gr = r.groupby("rec_id", sort=False)
    r["CR_routes"] = gr.routes.cumsum()
    r["CR_targets"] = gr.targets.cumsum()
    r["R_games"] = gr.cumcount() + 1
    rec_cum = r[["rec_id", "t", "CR_routes", "CR_targets", "R_games"]]
    return pair_cum, rec_cum


def fit_volume_constants(pair_week: pd.DataFrame, max_season: int) -> dict:
    """EB variance components for the volume channel, fit on seasons < max_season only."""
    sub = pair_week[pair_week.season < max_season]
    mu = {}
    for pos in RECEIVER_POSITIONS:
        s = sub[sub.rec_pos == pos]
        mu[pos] = float(s.targets.sum() / max(s.routes.sum(), 1.0))
    snap = sub.groupby(["qb_id", "rec_id", "rec_pos"], as_index=False)[["routes", "targets"]].sum()
    rsnap = (snap.groupby(["rec_id", "rec_pos"], as_index=False)[["routes", "targets"]].sum()
             .sort_values("routes", ascending=False).drop_duplicates("rec_id"))
    p_tpr = rsnap.targets / np.maximum(rsnap.routes, 1)
    sig2 = float(np.average(p_tpr.clip(.02, .5) * (1 - p_tpr.clip(.02, .5)), weights=rsnap.routes))
    dev_r = p_tpr - rsnap.rec_pos.map(mu)
    m = rsnap.routes >= 200
    if m.sum() >= 30:
        tau2_rec = float(max(0.0, np.average(dev_r[m] ** 2, weights=rsnap.routes[m])
                             - np.average(sig2 / rsnap.routes[m], weights=rsnap.routes[m])))
    else:  # not enough qualifying receivers to identify a variance component
        tau2_rec = 0.0
    w = tau2_rec / (tau2_rec + sig2 / np.maximum(rsnap.routes, 1e-9))
    rsnap["b_tpr"] = rsnap.rec_pos.map(mu) + w * dev_r.fillna(0)
    sn = snap.merge(rsnap[["rec_id", "b_tpr"]], on="rec_id")
    dev_p = sn.targets / np.maximum(sn.routes, 1) - sn.b_tpr
    mp = sn.routes >= 150
    if mp.sum() >= 30:
        tau2_pair = float(max(0.0, np.average(dev_p[mp] ** 2, weights=sn.routes[mp])
                              - np.average(sig2 / sn.routes[mp], weights=sn.routes[mp])))
    else:
        tau2_pair = 0.0
    return {"mu_tpr": mu, "sig2_vol": sig2, "tau2_rec_vol": tau2_rec,
            "tau2_pair_vol": tau2_pair, "fit_max_season": int(max_season)}


def derive_team_starters(panel: pd.DataFrame, dropbacks: pd.DataFrame) -> pd.DataFrame:
    """Realized primary QB per team-week (max dropbacks). Input dropbacks:
    (season, week, team, qb_id, dropbacks)."""
    db = dropbacks.copy()
    db["t"] = _t(db.season, db.week)
    db = db.sort_values(["team", "t", "dropbacks"], ascending=[True, True, False])
    st = db.drop_duplicates(["team", "t"])[["season", "week", "t", "team", "qb_id"]]
    return st.rename(columns={"qb_id": "starter_qb"}).sort_values(["team", "t"]).reset_index(drop=True)


def asof_pair_volume_features(queries: pd.DataFrame, pair_cum: pd.DataFrame,
                              rec_cum: pd.DataFrame, starters: pd.DataFrame,
                              consts: dict) -> pd.DataFrame:
    """As-of volume-chemistry features for (rec_id, position, team, season, week) queries.

    Expected QB (pregame-legal) = team's most recent primary starter strictly before t.
    Returns queries + exp_qb, pair_vol_post (targets/route above receiver baseline,
    EB-shrunk) and chem_raw (extra targets/game = pair_vol_post * as-of routes/game).
    """
    q = queries.copy()
    q["t"] = _t(q.season, q.week)
    q = q.sort_values("t").reset_index(drop=True)
    stq = starters[["team", "t", "starter_qb"]].sort_values("t").rename(columns={"starter_qb": "exp_qb"})
    q = pd.merge_asof(q, stq, on="t", by="team", allow_exact_matches=False)
    pcs = pair_cum.rename(columns={"qb_id": "exp_qb"}).sort_values("t")
    q = pd.merge_asof(q, pcs, on="t", by=["exp_qb", "rec_id"], allow_exact_matches=False)
    q = pd.merge_asof(q, rec_cum.sort_values("t"), on="t", by="rec_id", allow_exact_matches=False)
    for c in ("C_routes", "C_targets", "CR_routes", "CR_targets", "R_games"):
        q[c] = q[c].fillna(0.0)
    mu = q.position.map(consts["mu_tpr"])
    w_r = consts["tau2_rec_vol"] / (consts["tau2_rec_vol"] + consts["sig2_vol"] / np.maximum(q.CR_routes, 1e-9))
    b_tpr = mu + np.where(q.CR_routes > 0, w_r * (q.CR_targets / np.maximum(q.CR_routes, 1) - mu), 0.0)
    w_p = consts["tau2_pair_vol"] / (consts["tau2_pair_vol"] + consts["sig2_vol"] / np.maximum(q.C_routes, 1e-9))
    pair_dev = np.where(q.C_routes > 0, q.C_targets / np.maximum(q.C_routes, 1) - b_tpr, 0.0)
    q["rec_b_tpr"] = b_tpr
    q["pair_vol_post"] = np.where(q.C_routes > 0, w_p * pair_dev, 0.0)
    q["exp_routes_pg"] = np.where(q.R_games > 0, q.CR_routes / np.maximum(q.R_games, 1), 0.0)
    q["chem_raw"] = q.pair_vol_post * q.exp_routes_pg
    return q


def capped_adjustment(x: np.ndarray, cap: float = ADJUSTMENT_CAP_PTS) -> np.ndarray:
    """Hard cap on any post-hoc projection adjustment (points)."""
    return np.clip(np.nan_to_num(np.asarray(x, dtype=float)), -cap, cap)
