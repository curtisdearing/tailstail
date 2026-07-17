"""Chemistry/condition study driver -- reproduces every table in
reports/chemistry_condition_engine.md (chemistry agent, 2026-07-16).

Run step by step (each fits the 45s sandbox budget; state persists in /tmp/exp_chemistry):

    python3 analysis/chemistry_study.py step1 <season>     # per season 2016..2025 (~8s each)
    python3 analysis/chemistry_study.py stepDPI            # referee DPI per game
    python3 analysis/chemistry_study.py step2              # pair-week panel
    python3 analysis/chemistry_study.py step3              # strictly-prior as-of cums + starters
    python3 analysis/chemistry_study.py step4              # EB constants + A1-A3 pair variance tests
    python3 analysis/chemistry_study.py step4b             # A4-A9, B1-B4 (forward predictive)
    python3 analysis/chemistry_study.py step5              # condition panel flags
    python3 analysis/chemistry_study.py step5b             # backup QB / WR1-out / deep WR
    python3 analysis/chemistry_study.py step5c             # strict backup + pregame new-QB flags
    python3 analysis/chemistry_study.py step6 0 17         # condition battery C01..C51 (3 chunks)
    python3 analysis/chemistry_study.py step6 17 34
    python3 analysis/chemistry_study.py step6 34 50
    python3 analysis/chemistry_study.py step6b             # C51 + EB posterior + BH-FDR (all 64)
    python3 analysis/chemistry_study.py step7              # as-of chemistry features per test year
    python3 analysis/chemistry_study.py step7b             # 5-variant season-forward gate

Inputs (READ-ONLY): tailstail/historical/* plus, for 2016-2018 extended history,
nflverse release assets downloaded to /tmp/exp_chemistry/:
  https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{2016..2018}.parquet
      -> pbp_{Y}_full.parquet   (2019-2025 full pbp -> pbpfull_{Y}.parquet, same release tag)
  https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_{2016..2018}.parquet
  historical/pbp_participation_{2016..2025}.parquet (already in repo historical/)
The test battery (64 tests) was PREDECLARED in /tmp/exp_chemistry/battery.json before
any condition test ran; BH-FDR runs across the full declared family.
Baseline for variant grading: reports/fantasy_outer_predictions.parquet (READ-ONLY,
reproduced baseline MAE 5.105968; see FOUNDATION.md).

The as-of pair machinery lives in nflvalue/fantasy/chemistry_engine.py (pure, tested);
this driver keeps the study-specific batteries and the variant gate.
"""
import sys

def step1(arg1=None, arg2=None):
    import sys, os
    import pandas as pd, numpy as np

    REPO = "/sessions/dreamy-compassionate-goodall/repos/tailstail"
    OUT = "/tmp/exp_chemistry"
    season = int(arg1)

    PBP_COLS = ["season","week","game_id","season_type","posteam","defteam","epa","pass_attempt",
                "complete_pass","pass_touchdown","air_yards","yards_after_catch","passing_yards",
                "receiver_player_id","passer_player_id","play_id","down","ydstogo","yardline_100",
                "qtr","cpoe","sack","pass"]

    def load_pbp(season):
        if season <= 2023 and season >= 2019:
            df = pd.read_parquet(f"{REPO}/historical/historical_pbp.parquet", columns=PBP_COLS)
            return df[df.season == season].copy()
        elif season in (2024, 2025):
            return pd.read_parquet(f"{REPO}/historical/pbp_{season}.parquet", columns=PBP_COLS)
        else:
            # extended history downloaded to /tmp (full nflverse pbp)
            f = f"{OUT}/pbp_{season}_full.parquet"
            df = pd.read_parquet(f, columns=PBP_COLS)
            return df[df.season == season].copy()

    pbp = load_pbp(season)
    pbp = pbp[pbp["pass"] == 1].copy()          # dropbacks incl. sacks/scrambles
    pbp["play_id"] = pbp["play_id"].astype("int64")

    # participation
    part = pd.read_parquet(f"{REPO}/historical/pbp_participation_{season}.parquet",
                           columns=["nflverse_game_id","play_id","offense_players","was_pressure","route","ngs_air_yards","time_to_throw"])
    part["play_id"] = part["play_id"].astype("int64")
    part = part.rename(columns={"nflverse_game_id":"game_id"})
    pbp = pbp.merge(part, on=["game_id","play_id"], how="left")
    pbp["was_pressure"] = pbp["was_pressure"].astype("float64")

    # rosters for positions (2019+ local; earlier from /tmp if downloaded)
    rw_path = f"{REPO}/historical/rosters_weekly.parquet"
    rw = pd.read_parquet(rw_path, columns=["season","week","team","position","player_id"])
    if season < 2019:
        rw = pd.read_parquet(f"{OUT}/roster_weekly_{season}.parquet", columns=["season","week","team","position","gsis_id"]).rename(columns={"gsis_id":"player_id"})
    rw = rw[rw.season == season]
    pos_map = rw.drop_duplicates(["player_id"])[["player_id","position"]].set_index("player_id")["position"].to_dict()

    ELIG = {"WR","TE","RB","FB","HB"}
    # explode offense to find QB-on-field and eligible receivers on field
    off = pbp[["game_id","play_id","week","posteam","passer_player_id","pass_attempt","offense_players"]].copy()
    off["offense_players"] = off["offense_players"].fillna("")
    off["olist"] = off["offense_players"].str.split(";")
    ex = off.explode("olist").rename(columns={"olist":"onfield_id"})
    ex = ex[ex.onfield_id.str.len() > 3]
    ex["pos"] = ex.onfield_id.map(pos_map)

    # QB on field per play (exactly one)
    qbs = ex[ex.pos == "QB"].groupby(["game_id","play_id"]).agg(qb_id=("onfield_id","first"), nqb=("onfield_id","size")).reset_index()
    qbs = qbs[qbs.nqb == 1][["game_id","play_id","qb_id"]]
    # where passer known, prefer passer id (dropbacks with attempt)
    pbp = pbp.merge(qbs, on=["game_id","play_id"], how="left")
    pbp["qb_eff_id"] = pbp["passer_player_id"].fillna(pbp["qb_id"])

    # routes proxy: eligible receiver on field on a dropback of qb_eff_id
    elig = ex[ex.pos.isin(ELIG)][["game_id","play_id","week","posteam","onfield_id","pos"]].rename(columns={"onfield_id":"rec_id","pos":"rec_pos"})
    elig = elig.merge(pbp[["game_id","play_id","qb_eff_id"]], on=["game_id","play_id"], how="left")
    elig = elig.dropna(subset=["qb_eff_id"])
    routes = (elig.groupby(["week","posteam","qb_eff_id","rec_id","rec_pos"], as_index=False)
                  .agg(routes=("play_id","size")))
    routes["season"] = season
    routes.to_parquet(f"{OUT}/routes/routes_{season}.parquet", index=False)

    # FTN catchable 2022-2024
    if season in (2022, 2023, 2024):
        ftn = pd.read_parquet(f"{REPO}/historical/ftn_charting_{season}.parquet",
                              columns=["nflverse_game_id","nflverse_play_id","is_catchable_ball","is_drop","is_contested_ball"])
        ftn = ftn.rename(columns={"nflverse_game_id":"game_id","nflverse_play_id":"play_id"})
        ftn["play_id"] = ftn["play_id"].astype("int64")
        pbp = pbp.merge(ftn, on=["game_id","play_id"], how="left")
    else:
        pbp["is_catchable_ball"] = np.nan; pbp["is_drop"] = np.nan; pbp["is_contested_ball"] = np.nan

    # save targeted pass plays (pair attribution) + all dropbacks minimal
    tgt = pbp[(pbp.pass_attempt == 1) & pbp.receiver_player_id.notna()].copy()
    tgt["rec_pos"] = tgt.receiver_player_id.map(pos_map)
    tgt["success"] = (tgt.epa > 0).astype(float)
    tgt["explosive"] = (tgt.passing_yards.fillna(0) >= 20).astype(float)
    tgt["rz"] = (tgt.yardline_100 <= 20).astype(float)
    keep = ["season","week","game_id","season_type","posteam","defteam","play_id","passer_player_id","qb_eff_id",
            "receiver_player_id","rec_pos","complete_pass","pass_touchdown","air_yards","yards_after_catch",
            "passing_yards","epa","cpoe","success","explosive","rz","yardline_100","down","ydstogo","qtr",
            "was_pressure","is_catchable_ball","is_drop","is_contested_ball","ngs_air_yards","time_to_throw"]
    tgt[keep].to_parquet(f"{OUT}/pass/pass_{season}.parquet", index=False)
    db = pbp.groupby(["week","posteam","qb_eff_id"], as_index=False).agg(dropbacks=("play_id","size"), pressured_db=("was_pressure","sum"))
    db["season"] = season
    db.to_parquet(f"{OUT}/pass/dropbacks_{season}.parquet", index=False)
    print(season, "pass plays:", len(tgt), "routes rows:", len(routes), "qb match rate:", pbp.qb_eff_id.notna().mean().round(4))


def stepDPI(arg1=None, arg2=None):
    """DPI count per game from full pbp, 2019-2025 -- matches the panel the battery
    consumed (crew priors for season s use seasons < s inside this window)."""
    import pandas as pd
    OUT = "/tmp/exp_chemistry"
    rows = []
    for y in range(2019, 2026):
        f = f"{OUT}/pbpfull_{y}.parquet"
        df = pd.read_parquet(f, columns=["game_id", "season", "week", "penalty_type"])
        g = (df.assign(dpi=df.penalty_type.eq("Defensive Pass Interference").astype(float))
               .groupby(["game_id", "season", "week"], as_index=False).dpi.sum())
        rows.append(g)
    pd.concat(rows, ignore_index=True).to_parquet(f"{OUT}/game_dpi.parquet", index=False)

def step2(arg1=None, arg2=None):
    import pandas as pd, numpy as np
    OUT = "/tmp/exp_chemistry"
    SEASONS = list(range(2016, 2026))

    routes = pd.concat([pd.read_parquet(f"{OUT}/routes/routes_{y}.parquet") for y in SEASONS], ignore_index=True)
    routes = routes.rename(columns={"posteam":"team","qb_eff_id":"qb_id"})

    pas = pd.concat([pd.read_parquet(f"{OUT}/pass/pass_{y}.parquet") for y in SEASONS], ignore_index=True)
    pas = pas.rename(columns={"posteam":"team","receiver_player_id":"rec_id"})
    pas["qb_id"] = pas["passer_player_id"]
    pas["catch_n"] = pas["is_catchable_ball"].notna().astype(float)
    pas["catchable"] = pas["is_catchable_ball"].astype("float64")
    pas["cpoe_n"] = pas["cpoe"].notna().astype(float)
    pas["press_n"] = pas["was_pressure"].notna().astype(float)
    pas["pressured"] = pas["was_pressure"].astype("float64")
    pas["epa_press"] = np.where(pas["was_pressure"].astype("float64")==1, pas["epa"], np.nan)
    pas["epa_clean"] = np.where(pas["was_pressure"].astype("float64")==0, pas["epa"], np.nan)
    pas["yards"] = pas["passing_yards"].fillna(0.0)

    agg = (pas.groupby(["season","week","team","qb_id","rec_id"], as_index=False)
           .agg(targets=("play_id","size"), completions=("complete_pass","sum"), yards=("yards","sum"),
                air_sum=("air_yards","sum"), epa_sum=("epa","sum"), success_sum=("success","sum"),
                cpoe_sum=("cpoe","sum"), cpoe_n=("cpoe_n","sum"), explosive=("explosive","sum"),
                rz_tgt=("rz","sum"), td=("pass_touchdown","sum"),
                catchable_sum=("catchable","sum"), catchable_n=("catch_n","sum"),
                press_tgt=("pressured","sum"), press_n=("press_n","sum"),
                epa_press_sum=("epa_press","sum"), n_press_epa=("epa_press","count"),
                epa_clean_sum=("epa_clean","sum"), n_clean_epa=("epa_clean","count")))

    # team-week (per QB) totals for shares
    teamq = (pas.groupby(["season","week","team","qb_id"], as_index=False)
             .agg(tq_targets=("play_id","size"), tq_air=("air_yards","sum"), tq_rz=("rz","sum")))

    panel = routes.merge(agg, on=["season","week","team","qb_id","rec_id"], how="outer")
    panel["routes"] = panel[["routes","targets"]].max(axis=1)  # a target implies >=1 route
    for c in ["targets","completions","yards","air_sum","epa_sum","success_sum","cpoe_sum","cpoe_n",
              "explosive","rz_tgt","td","catchable_sum","catchable_n","press_tgt","press_n",
              "epa_press_sum","n_press_epa","epa_clean_sum","n_clean_epa","routes"]:
        panel[c] = panel[c].fillna(0.0)
    panel = panel.merge(teamq, on=["season","week","team","qb_id"], how="left")
    db = pd.concat([pd.read_parquet(f"{OUT}/pass/dropbacks_{y}.parquet") for y in SEASONS], ignore_index=True)
    db = db.rename(columns={"posteam":"team","qb_eff_id":"qb_id"})
    panel = panel.merge(db[["season","week","team","qb_id","dropbacks"]], on=["season","week","team","qb_id"], how="left")

    # fill rec_pos from any season roster
    rw = pd.read_parquet("/sessions/dreamy-compassionate-goodall/repos/tailstail/historical/rosters_weekly.parquet", columns=["player_id","position"])
    pm = rw.drop_duplicates("player_id").set_index("player_id")["position"].to_dict()
    panel["rec_pos"] = panel["rec_pos"].fillna(panel["rec_id"].map(pm))
    panel = panel[panel.rec_pos.isin(["WR","TE","RB","FB","HB"])].copy()
    panel["rec_pos"] = panel["rec_pos"].replace({"FB":"RB","HB":"RB"})
    panel = panel.sort_values(["season","week","team","qb_id","rec_id"]).reset_index(drop=True)
    panel.to_parquet(f"{OUT}/pair_week.parquet", index=False)
    print("panel:", panel.shape, "pairs:", panel.groupby(['qb_id','rec_id']).ngroups)
    print(panel[["routes","targets"]].sum())
    print("seasons:", panel.season.value_counts().sort_index().to_dict())


def step3(arg1=None, arg2=None):
    import pandas as pd, numpy as np
    OUT = "/tmp/exp_chemistry"
    p = pd.read_parquet(f"{OUT}/pair_week.parquet")
    p["t"] = p.season * 100 + p.week

    SUMS = ["routes","targets","completions","yards","air_sum","epa_sum","success_sum","cpoe_sum","cpoe_n",
            "explosive","rz_tgt","td","catchable_sum","catchable_n","press_tgt","press_n",
            "epa_press_sum","n_press_epa","epa_clean_sum","n_clean_epa","tq_targets","tq_air","tq_rz"]
    p[SUMS] = p[SUMS].fillna(0.0)

    p = p.sort_values(["qb_id","rec_id","t"]).reset_index(drop=True)
    g = p.groupby(["qb_id","rec_id"], sort=False)
    for c in SUMS:
        p[f"P_{c}"] = g[c].cumsum() - p[c]          # strictly prior (excludes current row)
    p["P_games"] = g.cumcount()

    # receiver-week aggregates (across QBs)
    rw = (p.groupby(["rec_id","rec_pos","t"], as_index=False)[SUMS].sum()
            .sort_values(["rec_id","t"]).reset_index(drop=True))
    gr = rw.groupby("rec_id", sort=False)
    for c in SUMS:
        rw[f"R_{c}"] = gr[c].cumsum() - rw[c]
    rw["R_games"] = gr.cumcount()
    rcols = ["rec_id","t","R_games"] + [f"R_{c}" for c in SUMS]
    p = p.merge(rw[rcols], on=["rec_id","t"], how="left")

    # QB-week aggregates (across receivers)
    qw = (p.groupby(["qb_id","t"], as_index=False)[["targets","completions","yards","air_sum","epa_sum","success_sum","cpoe_sum","cpoe_n","epa_press_sum","n_press_epa","epa_clean_sum","n_clean_epa"]].sum()
            .sort_values(["qb_id","t"]).reset_index(drop=True))
    gq = qw.groupby("qb_id", sort=False)
    for c in ["targets","completions","yards","air_sum","epa_sum","success_sum","cpoe_sum","cpoe_n","epa_press_sum","n_press_epa","epa_clean_sum","n_clean_epa"]:
        qw[f"Q_{c}"] = gq[c].cumsum() - qw[c]
    qw["Q_games"] = gq.cumcount()
    qcols = ["qb_id","t","Q_games"] + [f"Q_{c}" for c in ["targets","completions","yards","air_sum","epa_sum","success_sum","cpoe_sum","cpoe_n","epa_press_sum","n_press_epa","epa_clean_sum","n_clean_epa"]]
    p = p.merge(qw[qcols], on=["qb_id","t"], how="left")

    p.to_parquet(f"{OUT}/pair_asof.parquet", index=False)

    # team-week QB dropbacks -> prior-game starter per team
    db = p.groupby(["season","week","t","team","qb_id"], as_index=False).agg(qb_db=("dropbacks","first"))
    db = db.sort_values(["team","t","qb_db"], ascending=[True,True,False])
    starter = db.drop_duplicates(["team","t"]).rename(columns={"qb_id":"starter_qb"})[["season","week","t","team","starter_qb","qb_db"]]
    starter = starter.sort_values(["team","t"])
    starter["prev_starter_qb"] = starter.groupby("team")["starter_qb"].shift(1)
    starter.to_parquet(f"{OUT}/team_starters.parquet", index=False)
    print("asof:", p.shape, "| starter table:", starter.shape)
    print("sample P_games dist:", p.P_games.describe()[["mean","50%","max"]].round(2).to_dict())


def step4(arg1=None, arg2=None):
    import pandas as pd, numpy as np, json
    OUT = "/tmp/exp_chemistry"
    p = pd.read_parquet(f"{OUT}/pair_asof.parquet")
    INNER_T = 202300  # constants fit strictly on t < 202300 (<=2022)

    pas = pd.concat([pd.read_parquet(f"{OUT}/pass/pass_{y}.parquet") for y in range(2016,2023)], ignore_index=True)
    consts = {}
    # play-level residual variances (sigma^2) from <=2022 plays
    consts["sig2_epa"] = float(pas.epa.var())
    consts["sig2_cpoe"] = float(pas.cpoe.var())
    consts["sig2_yds"] = float(pas.passing_yards.fillna(0).var())
    mu_pos = {}
    pas["rp"] = pas.rec_pos.replace({"FB":"RB","HB":"RB"})
    for pos, gdf in pas.groupby("rp"):
        if pos not in ("WR","TE","RB"): continue
        mu_pos[pos] = dict(epa_t=float(gdf.epa.mean()), yds_t=float(gdf.passing_yards.fillna(0).mean()),
                           succ=float(gdf.success.mean()), expl=float(gdf.explosive.mean()),
                           cpoe=float(gdf.cpoe.mean()))
    # volume: league target rate per route by position (from panel <=2022)
    inner = p[p.t < INNER_T]
    for pos in ("WR","TE","RB"):
        sub = inner[inner.rec_pos == pos]
        mu_pos[pos]["tpr"] = float(sub.targets.sum() / sub.routes.sum())
        mu_pos[pos]["airshare"] = float(sub.air_sum.sum() / sub.tq_air.sum())
        mu_pos[pos]["rzshare"] = float(sub.rz_tgt.sum() / sub.tq_rz.sum())
    consts["mu_pos"] = mu_pos

    def mom_tau2(dev, n, sig2, min_n):
        m = (n >= min_n) & dev.notna()
        dev, n = dev[m], n[m]
        if len(dev) < 30: return 0.0, int(m.sum())
        v = sig2 / n
        w = n
        tau2 = float(max(0.0, (np.average(dev**2, weights=w) - np.average(v, weights=w))))
        return tau2, int(m.sum())

    # ---- end-of-2022 career snapshot for variance-component fitting ----
    snap = (p[p.t < INNER_T].groupby(["qb_id","rec_id","rec_pos"], as_index=False)
            [["routes","targets","completions","yards","air_sum","epa_sum","success_sum","cpoe_sum","cpoe_n","explosive","rz_tgt","tq_air","tq_rz","tq_targets"]].sum())
    rsnap = snap.groupby(["rec_id","rec_pos"], as_index=False)[["routes","targets","epa_sum","cpoe_sum","cpoe_n","air_sum","tq_air","rz_tgt","tq_rz","explosive","success_sum","yards"]].sum()
    qsnap = snap.groupby("qb_id", as_index=False)[["targets","epa_sum","cpoe_sum","cpoe_n","completions"]].sum()

    MU = {pos: mu_pos[pos] for pos in ("WR","TE","RB")}
    rsnap["mu_tpr"] = rsnap.rec_pos.map(lambda x: MU[x]["tpr"])
    rsnap["mu_epa"] = rsnap.rec_pos.map(lambda x: MU[x]["epa_t"])
    # receiver-level tau2 (vs position mean)
    p_tpr = rsnap.targets / rsnap.routes
    sig2_bin = (p_tpr.clip(.02,.5) * (1-p_tpr.clip(.02,.5)))
    tau2_r_vol, nfit = mom_tau2(p_tpr - rsnap.mu_tpr, rsnap.routes, float(np.average(sig2_bin, weights=rsnap.routes)), 200)
    consts["tau2_rec_vol"] = tau2_r_vol
    tau2_r_epa, _ = mom_tau2(rsnap.epa_sum/rsnap.targets - rsnap.mu_epa, rsnap.targets, consts["sig2_epa"], 50)
    consts["tau2_rec_epa"] = tau2_r_epa
    r_cp = rsnap[rsnap.cpoe_n > 0]
    tau2_r_cpoe, _ = mom_tau2(r_cp.cpoe_sum/r_cp.cpoe_n - float(pas.cpoe.mean()), r_cp.cpoe_n, consts["sig2_cpoe"], 50)
    consts["tau2_rec_cpoe"] = tau2_r_cpoe
    tau2_q_cpoe, _ = mom_tau2(qsnap.cpoe_sum/qsnap.cpoe_n.clip(lower=1) - float(pas.cpoe.mean()), qsnap.cpoe_n, consts["sig2_cpoe"], 100)
    consts["tau2_qb_cpoe"] = tau2_q_cpoe
    tau2_q_epa, _ = mom_tau2(qsnap.epa_sum/qsnap.targets - float(pas.epa.mean()), qsnap.targets, consts["sig2_epa"], 100)
    consts["tau2_qb_epa"] = tau2_q_epa

    def shrink(dev, n, tau2, sig2):
        n = np.maximum(n, 0)
        w = tau2 / (tau2 + sig2 / np.maximum(n, 1e-9))
        w = np.where(n <= 0, 0.0, w)
        return w * np.nan_to_num(dev), w

    # ---- pair-level deviations at end-2022 (for tau2_pair fitting), after removing shrunk main effects ----
    mu_epa_league = float(pas.epa.mean()); mu_cpoe_league = float(pas.cpoe.mean())
    r_eff_vol, _ = shrink(p_tpr - rsnap.mu_tpr, rsnap.routes / (1/np.maximum(sig2_bin,1e-4)).mean()**0 , tau2_r_vol, float(np.average(sig2_bin, weights=rsnap.routes)))
    rsnap["eff_vol"] = r_eff_vol
    rsnap["eff_epa"], _ = shrink(rsnap.epa_sum/np.maximum(rsnap.targets,1) - rsnap.mu_epa, rsnap.targets, tau2_r_epa, consts["sig2_epa"])
    rsnap["eff_cpoe"], _ = shrink(np.where(rsnap.cpoe_n>0, rsnap.cpoe_sum/np.maximum(rsnap.cpoe_n,1) - mu_cpoe_league, np.nan), rsnap.cpoe_n, tau2_r_cpoe, consts["sig2_cpoe"])
    qsnap["eff_cpoe"], _ = shrink(qsnap.cpoe_sum/np.maximum(qsnap.cpoe_n,1) - mu_cpoe_league, qsnap.cpoe_n, tau2_q_cpoe, consts["sig2_cpoe"])
    qsnap["eff_epa"], _ = shrink(qsnap.epa_sum/np.maximum(qsnap.targets,1) - mu_epa_league, qsnap.targets, tau2_q_epa, consts["sig2_epa"])

    sn = snap.merge(rsnap[["rec_id","eff_vol","eff_epa","eff_cpoe"]], on="rec_id").merge(
         qsnap[["qb_id","eff_cpoe","eff_epa"]], on="qb_id", suffixes=("_r","_q"))
    sn["mu_tpr"] = sn.rec_pos.map(lambda x: MU[x]["tpr"]); sn["mu_epa"] = sn.rec_pos.map(lambda x: MU[x]["epa_t"])
    tests = {}
    # volume pair dev
    dev_vol = sn.targets/sn.routes - (sn.mu_tpr + sn.eff_vol)
    sig2_vol = float(np.average((sn.targets/sn.routes).clip(.02,.5)*(1-(sn.targets/sn.routes).clip(.02,.5)), weights=sn.routes))
    consts["sig2_vol"] = sig2_vol
    tau2_pair_vol, nv = mom_tau2(dev_vol, sn.routes, sig2_vol, 150)
    consts["tau2_pair_vol"] = tau2_pair_vol
    # efficiency pair devs
    dev_epa = sn.epa_sum/np.maximum(sn.targets,1) - (sn.mu_epa + sn.eff_epa_r + sn.eff_epa_q)
    tau2_pair_epa, ne = mom_tau2(dev_epa, sn.targets, consts["sig2_epa"], 60)
    consts["tau2_pair_epa"] = tau2_pair_epa
    dev_cpoe = np.where(sn.cpoe_n>0, sn.cpoe_sum/np.maximum(sn.cpoe_n,1) - (mu_cpoe_league + sn.eff_cpoe_r + sn.eff_cpoe_q), np.nan)
    tau2_pair_cpoe, nc = mom_tau2(pd.Series(dev_cpoe), sn.cpoe_n, consts["sig2_cpoe"], 60)
    consts["tau2_pair_cpoe"] = tau2_pair_cpoe

    # chi2 variance tests (H0 tau2_pair=0)
    from scipy import stats
    def chi2_test(dev, n, sig2, min_n):
        m = (n >= min_n) & pd.Series(dev).notna()
        d, nn = pd.Series(dev)[m], n[m]
        T = float(np.sum(d**2 / (sig2/nn))); df = int(m.sum())
        return {"T": T, "df": df, "p": float(stats.chi2.sf(T, df)), "n_pairs": df}
    tests["pair_vol_var"] = chi2_test(dev_vol, sn.routes, sig2_vol, 150)
    tests["pair_epa_var"] = chi2_test(dev_epa, sn.targets, consts["sig2_epa"], 60)
    tests["pair_cpoe_var"] = chi2_test(dev_cpoe, sn.cpoe_n, consts["sig2_cpoe"], 60)

    json.dump(consts, open(f"{OUT}/ckpt/eb_consts.json","w"), indent=1)
    json.dump(tests, open(f"{OUT}/ckpt/pair_var_tests.json","w"), indent=1)
    print("consts:", {k: (round(v,6) if isinstance(v,float) else "...") for k,v in consts.items() if k!="mu_pos"})
    print("tests:", json.dumps(tests, indent=0))
    print("mu_pos:", json.dumps(mu_pos))


def step4b(arg1=None, arg2=None):
    import pandas as pd, numpy as np, json
    from scipy import stats
    OUT="/tmp/exp_chemistry"
    p = pd.read_parquet(f"{OUT}/pair_asof.parquet")
    consts = json.load(open(f"{OUT}/ckpt/eb_consts.json"))
    MU = consts["mu_pos"]
    tests = json.load(open(f"{OUT}/ckpt/pair_var_tests.json"))

    def chi2_test(dev, n, sig2, min_n):
        dev = pd.Series(np.asarray(dev, dtype=float)); n = pd.Series(np.asarray(n, dtype=float))
        m = (n >= min_n) & dev.notna()
        d, nn = dev[m], n[m]
        T = float(np.sum(d**2/(sig2/nn))); df = int(m.sum())
        return {"T": round(T,1), "df": df, "p": float(stats.chi2.sf(T, df)), "tau2_mom": float(max(0.0,(np.average(d**2,weights=nn)-np.average(sig2/nn,weights=nn)))) if df>=30 else None}

    inner = p[p.t < 202300]
    snap = (inner.groupby(["qb_id","rec_id","rec_pos"], as_index=False)
            [["routes","targets","completions","yards","air_sum","epa_sum","success_sum","cpoe_sum","cpoe_n",
              "explosive","rz_tgt","tq_air","tq_rz","tq_targets","catchable_sum","catchable_n",
              "epa_press_sum","n_press_epa","epa_clean_sum","n_clean_epa"]].sum())
    rsnap = snap.groupby(["rec_id","rec_pos"], as_index=False)[["routes","targets","yards","air_sum","epa_sum","success_sum","explosive","rz_tgt","tq_air","tq_rz","tq_targets"]].sum()
    rsnap = rsnap.sort_values("routes", ascending=False).drop_duplicates("rec_id").reset_index(drop=True)
    # receiver shrunk baselines for each metric
    def rec_base(metric_sum, nvar, mu_key, tau2, sig2):
        dev = rsnap[metric_sum]/np.maximum(rsnap[nvar],1) - rsnap.rec_pos.map(lambda x: MU[x][mu_key])
        w = tau2/(tau2 + sig2/np.maximum(rsnap[nvar],1e-9))
        return rsnap.rec_pos.map(lambda x: MU[x][mu_key]) + w*dev.fillna(0)
    # yds/tgt
    sig2_yds = consts["sig2_yds"]
    tau2_r_yds = max(0.0, float(np.average(((rsnap.yards/np.maximum(rsnap.targets,1)) - rsnap.rec_pos.map(lambda x: MU[x]["yds_t"]))[rsnap.targets>=50]**2, weights=rsnap.targets[rsnap.targets>=50]) - np.average(sig2_yds/rsnap.targets[rsnap.targets>=50], weights=rsnap.targets[rsnap.targets>=50])))
    rsnap["b_yds"] = rec_base("yards","targets","yds_t",tau2_r_yds,sig2_yds)
    # success (binomial)
    sig2_succ = 0.25
    tau2_r_succ = max(0.0, float(np.average(((rsnap.success_sum/np.maximum(rsnap.targets,1)) - rsnap.rec_pos.map(lambda x: MU[x]["succ"]))[rsnap.targets>=50]**2, weights=rsnap.targets[rsnap.targets>=50]) - np.average(sig2_succ/rsnap.targets[rsnap.targets>=50], weights=rsnap.targets[rsnap.targets>=50])))
    rsnap["b_succ"] = rec_base("success_sum","targets","succ",tau2_r_succ,sig2_succ)
    # explosive
    sig2_expl = 0.09
    tau2_r_expl = max(0.0, float(np.average(((rsnap.explosive/np.maximum(rsnap.targets,1)) - rsnap.rec_pos.map(lambda x: MU[x]["expl"]))[rsnap.targets>=50]**2, weights=rsnap.targets[rsnap.targets>=50]) - np.average(sig2_expl/rsnap.targets[rsnap.targets>=50], weights=rsnap.targets[rsnap.targets>=50])))
    rsnap["b_expl"] = rec_base("explosive","targets","expl",tau2_r_expl,sig2_expl)
    # air share & rz share (denominators = team totals in receiver games)
    for nm, num, den, mu_key, s2 in [("airshr","air_sum","tq_air","airshare",None),("rzshr","rz_tgt","tq_rz","rzshare",None)]:
        rate = rsnap[num]/np.maximum(rsnap[den],1)
        mu = rsnap.rec_pos.map(lambda x: MU[x][mu_key])
        sig2 = float((rate*(1-rate)).clip(0.01,0.25).mean()) if nm=="rzshr" else None
        rsnap[f"b_{nm}"] = rate  # store raw; shrink at pair stage via tq denominators
    sn = snap.merge(rsnap[["rec_id","b_yds","b_succ","b_expl"]], on="rec_id")
    # QB main effects for yds (reuse epa infra: QB yds/att dev)
    qsnap = snap.groupby("qb_id", as_index=False)[["targets","yards","success_sum","explosive"]].sum()
    mu_yds_l = float((inner.yards.sum())/max(inner.targets.sum(),1))
    qdev_yds = qsnap.yards/np.maximum(qsnap.targets,1) - mu_yds_l
    w = 0.5*qdev_yds.notna()
    tau2_q_yds = max(0.0, float(np.average(qdev_yds[qsnap.targets>=100]**2, weights=qsnap.targets[qsnap.targets>=100]) - np.average(sig2_yds/qsnap.targets[qsnap.targets>=100], weights=qsnap.targets[qsnap.targets>=100])))
    qsnap["q_yds"] = (tau2_q_yds/(tau2_q_yds+sig2_yds/np.maximum(qsnap.targets,1e-9)))*qdev_yds
    sn = sn.merge(qsnap[["qb_id","q_yds"]], on="qb_id")

    tests["pair_yds_var"] = chi2_test(sn.yards/np.maximum(sn.targets,1) - (sn.b_yds + sn.q_yds), sn.targets, sig2_yds, 60)
    tests["pair_succ_var"] = chi2_test(sn.success_sum/np.maximum(sn.targets,1) - sn.b_succ, sn.targets, sig2_succ, 60)
    tests["pair_expl_var"] = chi2_test(sn.explosive/np.maximum(sn.targets,1) - sn.b_expl, sn.targets, sig2_expl, 60)
    # air / rz share: pair share vs receiver overall share
    tests["pair_airshare_var"] = chi2_test(sn.air_sum/np.maximum(sn.tq_air,1) - sn.rec_id.map(rsnap.set_index("rec_id").b_airshr), sn.tq_targets, 0.02, 150)
    tests["pair_rzshare_var"] = chi2_test(sn.rz_tgt/np.maximum(sn.tq_rz,1) - sn.rec_id.map(rsnap.set_index("rec_id").b_rzshr), sn.tq_rz, 0.11, 30)
    # catchable (inner=2022 only)
    cb = sn[sn.catchable_n >= 25]
    mu_c = float(sn.catchable_sum.sum()/max(sn.catchable_n.sum(),1))
    tests["pair_catchable_var"] = chi2_test(cb.catchable_sum/np.maximum(cb.catchable_n,1) - mu_c, cb.catchable_n, mu_c*(1-mu_c), 25)
    # pressure gap: pair clean-vs-pressure epa gap vs qb-league gap
    pg = sn[(sn.n_press_epa >= 15) & (sn.n_clean_epa >= 30)].copy()
    gap = (pg.epa_clean_sum/pg.n_clean_epa) - (pg.epa_press_sum/pg.n_press_epa)
    mu_gap = float((sn.epa_clean_sum.sum()/max(sn.n_clean_epa.sum(),1)) - (sn.epa_press_sum.sum()/max(sn.n_press_epa.sum(),1)))
    nn = 1/(1/pg.n_press_epa + 1/pg.n_clean_epa)
    tests["pair_pressure_gap_var"] = chi2_test(gap - mu_gap, nn, consts["sig2_epa"], 10)

    # ---- B1-B3 season-forward within inner: prior posterior -> next-season realized dev ----
    def forward_test(metric):
        rows=[]
        for Y in (2019,2020,2021,2022):
            pre = p[p.t < Y*100].groupby(["qb_id","rec_id","rec_pos"], as_index=False)[["routes","targets","epa_sum","cpoe_sum","cpoe_n"]].sum()
            cur = p[(p.season==Y)].groupby(["qb_id","rec_id","rec_pos"], as_index=False)[["routes","targets","epa_sum","cpoe_sum","cpoe_n"]].sum()
            rpre = pre.groupby(["rec_id","rec_pos"], as_index=False)[["routes","targets","epa_sum","cpoe_sum","cpoe_n"]].sum()
            rpre["tpr"] = rpre.targets/np.maximum(rpre.routes,1)
            rpre["mu"] = rpre.rec_pos.map(lambda x: MU[x]["tpr"])
            w = consts["tau2_rec_vol"]/(consts["tau2_rec_vol"]+consts["sig2_vol"]/np.maximum(rpre.routes,1e-9))
            rpre["b_tpr"] = rpre.mu + w*(rpre.tpr-rpre.mu).fillna(0)
            rpre["epa_t"] = rpre.epa_sum/np.maximum(rpre.targets,1)
            rpre["mu_e"] = rpre.rec_pos.map(lambda x: MU[x]["epa_t"])
            we = consts["tau2_rec_epa"]/(consts["tau2_rec_epa"]+consts["sig2_epa"]/np.maximum(rpre.targets,1e-9))
            rpre["b_epa"] = rpre.mu_e + we*(rpre.epa_t-rpre.mu_e).fillna(0)
            rpre["cp"] = rpre.cpoe_sum/np.maximum(rpre.cpoe_n,1)
            wc = consts["tau2_rec_cpoe"]/(consts["tau2_rec_cpoe"]+consts["sig2_cpoe"]/np.maximum(rpre.cpoe_n,1e-9))
            rpre["b_cp"] = wc*rpre.cp.fillna(0)
            m = pre.merge(rpre[["rec_id","b_tpr","b_epa","b_cp"]], on="rec_id").merge(
                cur, on=["qb_id","rec_id"], suffixes=("_pre","_cur"))
            if metric=="vol":
                dev_pre = m.targets_pre/np.maximum(m.routes_pre,1) - m.b_tpr
                w_pre = consts["tau2_pair_vol"]/(consts["tau2_pair_vol"]+consts["sig2_vol"]/np.maximum(m.routes_pre,1e-9))
                x = w_pre*dev_pre
                ydev = m.targets_cur/np.maximum(m.routes_cur,1) - m.b_tpr
                wt = np.minimum(m.routes_pre, m.routes_cur); keep = (m.routes_pre>=80)&(m.routes_cur>=80)
            elif metric=="epa":
                dev_pre = m.epa_sum_pre/np.maximum(m.targets_pre,1) - m.b_epa
                tau2 = max(consts["tau2_pair_epa"], 1e-6)
                x = (tau2/(tau2+consts["sig2_epa"]/np.maximum(m.targets_pre,1e-9)))*dev_pre
                x = dev_pre  # tau2=0 -> use raw dev for predictive check
                ydev = m.epa_sum_cur/np.maximum(m.targets_cur,1) - m.b_epa
                wt = np.minimum(m.targets_pre, m.targets_cur); keep = (m.targets_pre>=30)&(m.targets_cur>=30)
            else:
                dev_pre = m.cpoe_sum_pre/np.maximum(m.cpoe_n_pre,1) - m.b_cp
                x = dev_pre
                ydev = m.cpoe_sum_cur/np.maximum(m.cpoe_n_cur,1) - m.b_cp
                wt = np.minimum(m.cpoe_n_pre, m.cpoe_n_cur); keep = (m.cpoe_n_pre>=30)&(m.cpoe_n_cur>=30)
            sub = pd.DataFrame({"x":x,"y":ydev,"w":wt})[keep.values]
            rows.append(sub)
        allr = pd.concat(rows, ignore_index=True).dropna()
        def wcorr(df):
            xm = np.average(df.x, weights=df.w); ym = np.average(df.y, weights=df.w)
            cov = np.average((df.x-xm)*(df.y-ym), weights=df.w)
            return cov/np.sqrt(np.average((df.x-xm)**2,weights=df.w)*np.average((df.y-ym)**2,weights=df.w))
        r = wcorr(allr)
        rng = np.random.default_rng(6102026)
        bs = [wcorr(allr.sample(frac=1, replace=True, random_state=int(rng.integers(1e9)))) for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5,97.5])
        pv = 2*min((np.array(bs)<=0).mean(), (np.array(bs)>=0).mean()); pv = max(pv, 1/2000)
        return {"n_pairs": int(len(allr)), "weighted_r": round(float(r),4), "ci": [round(float(lo),4), round(float(hi),4)], "p": float(pv)}

    tests["fwd_pair_vol"] = forward_test("vol")
    tests["fwd_pair_epa"] = forward_test("epa")
    tests["fwd_pair_cpoe"] = forward_test("cpoe")
    json.dump(tests, open(f"{OUT}/ckpt/pair_var_tests.json","w"), indent=1)
    print(json.dumps(tests, indent=1))


def step5(arg1=None, arg2=None):
    import pandas as pd, numpy as np
    REPO = "/sessions/dreamy-compassionate-goodall/repos/tailstail"
    OUT = "/tmp/exp_chemistry"

    ff = pd.read_parquet(f"{REPO}/historical/fantasy/feature_frame.parquet",
        columns=["season","week","player_id","player_name","position","team","opponent_team","game_id",
                 "fantasy_points","pre_fantasy_points_ewm4","pre_fantasy_points_ewm8","targets","carries","attempts","receptions"])
    ff = ff[ff.position.isin(["QB","RB","WR","TE"])].copy()
    ff["played"] = (ff[["targets","carries","attempts","receptions"]].fillna(0).sum(axis=1) > 0) | ff.fantasy_points.fillna(0).ne(0)
    sc = pd.read_parquet(f"{REPO}/historical/fantasy/schedules.parquet")
    sc = sc[["game_id","game_type","gameday","weekday","gametime","home_team","away_team","spread_line",
             "roof","surface","temp","wind","stadium","stadium_id","away_rest","home_rest","location","referee"]]
    d = ff.merge(sc, on="game_id", how="left")
    print("game_type dist:", d.game_type.value_counts().to_dict())
    d = d[d.game_type == "REG"].copy()
    d["is_home"] = (d.team == d.home_team)
    d["team_spread"] = np.where(d.is_home, d.spread_line, -d.spread_line)   # + = favored
    d["rest"] = np.where(d.is_home, d.home_rest, d.away_rest)
    TURF = {"fieldturf","matrixturf","sportturf","astroturf","a_turf","turf","astroplay","fieldturf "}
    d["surface_l"] = d.surface.str.lower().str.strip()
    d["is_turf"] = d.surface_l.isin(TURF)
    d["is_grass"] = d.surface_l.str.contains("grass", na=False) | d.surface_l.eq("dessograss")
    # stadium type: fixed dome = stadium whose roof is always 'dome'/'closed'? use roof value: dome=fixed; open/closed=retractable
    d["roof_l"] = d.roof.str.lower()
    d["fixed_dome"] = d.roof_l.eq("dome")
    d["outdoors"] = d.roof_l.eq("outdoors")
    d["retract"] = d.roof_l.isin(["open","closed"])
    d["retract_open"] = d.roof_l.eq("open")
    d["indoor_env"] = d.roof_l.isin(["dome","closed"])
    d["hour"] = pd.to_numeric(d.gametime.str.split(":").str[0], errors="coerce")
    d["primetime"] = (d.hour >= 19) & d.weekday.isin(["Thursday","Sunday","Monday"])
    d["short_week"] = d.rest <= 5
    d["post_bye"] = d.rest >= 13
    d["big_fav"] = d.team_spread >= 7
    d["big_dog"] = d.team_spread <= -7
    d["alt_den"] = (d.home_team == "DEN") | d.stadium.str.contains("Azteca", na=False)
    d["cold"] = d.temp <= 32
    d["windy"] = d.wind >= 15
    # birthdays
    pm = pd.read_parquet(f"{REPO}/historical/players_meta.parquet")
    d = d.merge(pm.rename(columns={"birth_date":"bd"}), on="player_id", how="left")
    gd = pd.to_datetime(d.gameday); bd = pd.to_datetime(d.bd, errors="coerce")
    bdays = bd.dt.dayofyear; gdays = gd.dt.dayofyear
    diff = (gdays - bdays).abs()
    d["birthday_week"] = (np.minimum(diff, 365 - diff) <= 3) & bd.notna()
    # revenge: opponent is a franchise the player was rostered by in a PRIOR season (2016+)
    rw = pd.read_parquet(f"{REPO}/historical/rosters_weekly.parquet", columns=["season","team","player_id"]).drop_duplicates()
    old = pd.concat([pd.read_parquet(f"{OUT}/roster_weekly_{y}.parquet", columns=["season","team","gsis_id"]).rename(columns={"gsis_id":"player_id"}).drop_duplicates() for y in (2016,2017,2018)])
    hist = pd.concat([rw, old], ignore_index=True).dropna().drop_duplicates()
    hist = hist.rename(columns={"team":"opponent_team","season":"prior_season"})
    mm = d[["player_id","opponent_team","season"]].reset_index().merge(hist, on=["player_id","opponent_team"], how="left")
    rev = mm[mm.prior_season < mm.season].groupby("index").size()
    d["revenge"] = False; d.loc[rev.index, "revenge"] = True
    # referee DPI crew (prior-season DPI/game rate of this referee, strictly prior seasons)
    off = pd.read_parquet(f"{REPO}/historical/officials.parquet")
    refs = off[off.position == "Referee"][["game_id","official_name"]].drop_duplicates("game_id")
    dpi = pd.read_parquet(f"{OUT}/game_dpi.parquet").merge(refs, on="game_id", how="left")
    ref_season = dpi.groupby(["official_name","season"], as_index=False).agg(dpi_pg=("dpi","mean"), g=("dpi","size"))
    rows = []
    for s in range(2019, 2026):
        prior = ref_season[ref_season.season < s].groupby("official_name").agg(dpi_prior=("dpi_pg","mean"), g=("g","sum")).reset_index()
        prior = prior[prior.g >= 10]
        prior["season"] = s
        prior["dpi_hi"] = prior.dpi_prior >= prior.dpi_prior.quantile(2/3)
        rows.append(prior)
    refp = pd.concat(rows)
    d = d.merge(sc[["game_id"]].assign(referee2=sc.referee), on="game_id", how="left")
    d = d.merge(refp[["official_name","season","dpi_hi","dpi_prior"]].rename(columns={"official_name":"referee2"}), on=["referee2","season"], how="left")
    d["ref_dpi_hi"] = d.dpi_hi.fillna(False)
    d.drop(columns=["dpi_hi"], inplace=True)
    # outcome residual
    d["exp_pts"] = d.pre_fantasy_points_ewm8.fillna(d.pre_fantasy_points_ewm4)
    posmean = d.groupby("position").fantasy_points.transform("mean")
    d["exp_pts"] = d.exp_pts.fillna(posmean)
    d["resid"] = d.fantasy_points - d.exp_pts
    d = d[d.played & d.fantasy_points.notna()].copy()
    d.to_parquet(f"{OUT}/cond_panel.parquet", index=False)
    print("cond panel:", d.shape, "| primetime rate:", d.primetime.mean().round(3), "| turf:", d.is_turf.mean().round(3),
          "| revenge:", d.revenge.mean().round(4), "| birthday:", d.birthday_week.mean().round(4), "| refhi:", d.ref_dpi_hi.mean().round(3))
    print("resid mean/sd:", d.resid.mean().round(3), d.resid.std().round(3))


def step5b(arg1=None, arg2=None):
    import pandas as pd, numpy as np
    OUT="/tmp/exp_chemistry"
    d = pd.read_parquet(f"{OUT}/cond_panel.parquet")
    d = d.sort_values(["season","week"]).reset_index(drop=True)
    # team-game sequence
    tg = d[["team","season","week"]].drop_duplicates().sort_values(["team","season","week"]).reset_index(drop=True)
    tg["gnum"] = tg.groupby("team").cumcount()
    d = d.merge(tg, on=["team","season","week"], how="left")

    # realized primary QB per team-game (max pass attempts)
    qb = d[d.position=="QB"].sort_values(["team","gnum","attempts"], ascending=[True,True,False])
    prim = qb.drop_duplicates(["team","gnum"])[["team","gnum","player_id","player_name"]].rename(columns={"player_id":"primary_qb","player_name":"primary_qb_name"})
    prim = prim.sort_values(["team","gnum"])
    # modal starter over prior 8 team games (strictly prior)
    def modal_prior(s):
        out=[]
        hist=[]
        for v in s:
            if hist:
                vc = pd.Series(hist[-8:]).mode()
                out.append(vc.iloc[0])
            else:
                out.append(None)
            hist.append(v)
        return out
    prim["modal_prior_qb"] = prim.groupby("team")["primary_qb"].transform(modal_prior)
    prim["prev_game_qb"] = prim.groupby("team")["primary_qb"].shift(1)
    prim["backup_game"] = prim.primary_qb.ne(prim.modal_prior_qb) & prim.modal_prior_qb.notna()
    d = d.merge(prim[["team","gnum","primary_qb","modal_prior_qb","prev_game_qb","backup_game"]], on=["team","gnum"], how="left")
    d["backup_game"] = d.backup_game.fillna(False)

    # WR1-as-of (trailing 6 team-games targets) and absence
    wr = d[d.position=="WR"][["team","gnum","player_id","targets"]].copy()
    wr["targets"] = wr.targets.fillna(0)
    mat = wr.pivot_table(index=["team","gnum"], columns="player_id", values="targets", aggfunc="sum").fillna(0)
    mat = mat.groupby(level="team").apply(lambda x: x.rolling(6, min_periods=1).sum().shift(1)).fillna(0)
    mat.index = mat.index.droplevel(0)
    best = mat.idxmax(axis=1); bestval = mat.max(axis=1)
    wr1 = pd.DataFrame({"wr1_id": best, "wr1_trail": bestval}).reset_index()
    wr1 = wr1[wr1.wr1_trail >= 12]
    d = d.merge(wr1, on=["team","gnum"], how="left")
    played_ids = d.groupby(["team","gnum"]).player_id.apply(set).rename("played_set")
    d = d.join(played_ids, on=["team","gnum"])
    d["wr1_absent"] = d.apply(lambda r: (r.wr1_id is not None and isinstance(r.played_set,set) and r.wr1_id not in r.played_set) if pd.notna(r.wr1_id) else False, axis=1)
    d["wr1_out_flag"] = d.wr1_absent & d.player_id.ne(d.wr1_id)
    d = d.drop(columns=["played_set"])

    # deep WR: career aDOT strictly prior seasons
    pas = pd.concat([pd.read_parquet(f"{OUT}/pass/pass_{y}.parquet", columns=["season","receiver_player_id","air_yards"]) for y in range(2016,2026)], ignore_index=True)
    ag = pas.groupby(["receiver_player_id","season"], as_index=False).agg(air=("air_yards","sum"), n=("air_yards","count"))
    rows=[]
    for s in range(2019,2026):
        pr = ag[ag.season < s].groupby("receiver_player_id").agg(air=("air","sum"), n=("n","sum")).reset_index()
        pr = pr[pr.n >= 30]
        pr["adot_prior"] = pr.air/pr.n
        pr["season"] = s
        rows.append(pr[["receiver_player_id","season","adot_prior"]])
    adot = pd.concat(rows).rename(columns={"receiver_player_id":"player_id"})
    d = d.merge(adot, on=["player_id","season"], how="left")
    d["deep_wr"] = (d.position=="WR") & (d.adot_prior >= 12.5)

    # demeaned residual (position x season)
    d["r_adj"] = d.resid - d.groupby(["position","season"]).resid.transform("mean")
    d.to_parquet(f"{OUT}/cond_panel.parquet", index=False)
    print("backup_game rate:", d.backup_game.mean().round(3), "| wr1_out rows:", int(d.wr1_out_flag.sum()),
          "| deep_wr rate among WR:", d.loc[d.position=='WR','deep_wr'].mean().round(3))
    print("r_adj sd:", d.r_adj.std().round(2), "| rows:", len(d))


def step5c(arg1=None, arg2=None):
    import pandas as pd, numpy as np
    OUT="/tmp/exp_chemistry"
    d = pd.read_parquet(f"{OUT}/cond_panel.parquet")
    prim = (d[["team","gnum","primary_qb"]].drop_duplicates(["team","gnum"])
            .sort_values(["team","gnum"]).reset_index(drop=True))
    # starts of current primary in team's prior 8 games (strictly prior)
    def starts_prior8(g):
        out=[]; hist=[]
        for v in g:
            h = hist[-8:]
            out.append(sum(1 for x in h if x == v))
            hist.append(v)
        return out
    prim["n_starts_prior8"] = prim.groupby("team")["primary_qb"].transform(starts_prior8)
    prim["prev_qb"] = prim.groupby("team")["primary_qb"].shift(1)
    def prev_starts_prior8(g):
        # for the EXPECTED starter (prev game's primary), how many of prior 8 did he start?
        out=[]; hist=[]
        for v in g:
            h = hist[-8:]
            prev = hist[-1] if hist else None
            out.append(sum(1 for x in h if x == prev) if prev is not None else np.nan)
            hist.append(v)
        return out
    prim["prev_qb_starts_prior8"] = prim.groupby("team")["primary_qb"].transform(prev_starts_prior8)
    d = d.merge(prim[["team","gnum","n_starts_prior8","prev_qb","prev_qb_starts_prior8"]], on=["team","gnum"], how="left")
    d["backup_strict"] = (d.gnum >= 4) & (d.n_starts_prior8 <= 2)
    d["pregame_new_qb"] = (d.gnum >= 4) & ((d.prev_qb != d.modal_prior_qb) | (d.prev_qb_starts_prior8 <= 2))
    d.to_parquet(f"{OUT}/cond_panel.parquet", index=False)
    print("backup_strict rate:", round(d.backup_strict.mean(),4), "| pregame_new_qb rate:", round(d.pregame_new_qb.mean(),4))
    print("by season strict:", d.groupby('season').backup_strict.mean().round(3).to_dict())
    print("overlap strict&loose:", round((d.backup_strict & d.backup_game).mean(),4), "loose:", round(d.backup_game.mean(),4))


def step6(arg1=None, arg2=None):
    import pandas as pd, numpy as np, json, sys, os
    OUT="/tmp/exp_chemistry"
    d = pd.read_parquet(f"{OUT}/cond_panel.parquet")
    d["block"] = d.season*100 + d.week
    d["late"] = d.season >= 2023

    # name -> id for the flagged players
    def pid(name_sub):
        m = d[d.player_name.str.contains(name_sub, case=False, na=False)]
        return m.player_id.mode().iloc[0] if len(m) else None
    GODWIN, AIYUK, DSMITH = pid("Godwin"), pid("Aiyuk"), pid("DeVonta Smith")

    surf_known = (d.is_turf | d.is_grass)
    SPEC = [
     ("C01_surface_turf_QB","QB","is_turf",surf_known,False),
     ("C02_surface_turf_RB","RB","is_turf",surf_known,False),
     ("C03_surface_turf_WR","WR","is_turf",surf_known,False),
     ("C04_surface_turf_TE","TE","is_turf",surf_known,False),
     ("C05_dome_QB","QB","indoor_env",None,False),
     ("C06_dome_RB","RB","indoor_env",None,False),
     ("C07_dome_WR","WR","indoor_env",None,False),
     ("C08_dome_TE","TE","indoor_env",None,False),
     ("C09_retractable_open_vs_closed_pooled_RESEARCHONLY","ALL","retract_open",d.retract,True),
     ("C10_cold_le32_pooled_RESEARCHONLY","ALL","cold",d.outdoors & d.temp.notna(),True),
     ("C11_cold_le32_QB_RESEARCHONLY","QB","cold",d.outdoors & d.temp.notna(),True),
     ("C12_wind_ge15_QB_RESEARCHONLY","QB","windy",d.outdoors & d.wind.notna(),True),
     ("C13_wind_ge15_WRTE_RESEARCHONLY","WRTE","windy",d.outdoors & d.wind.notna(),True),
     ("C14_altitude_DEN_QB","QB","alt_den",None,False),
     ("C15_altitude_DEN_RB","RB","alt_den",None,False),
     ("C16_altitude_DEN_WR","WR","alt_den",None,False),
     ("C17_altitude_DEN_TE","TE","alt_den",None,False),
     ("C18_primetime_QB","QB","primetime",None,False),
     ("C19_primetime_RB","RB","primetime",None,False),
     ("C20_primetime_WR","WR","primetime",None,False),
     ("C21_primetime_TE","TE","primetime",None,False),
     ("C22_shortweek_QB","QB","short_week",None,False),
     ("C23_shortweek_RB","RB","short_week",None,False),
     ("C24_shortweek_WR","WR","short_week",None,False),
     ("C25_shortweek_TE","TE","short_week",None,False),
     ("C26_postbye_QB","QB","post_bye",None,False),
     ("C27_postbye_RB","RB","post_bye",None,False),
     ("C28_postbye_WR","WR","post_bye",None,False),
     ("C29_postbye_TE","TE","post_bye",None,False),
     ("C30_bigfav_QB","QB","big_fav",d.team_spread.notna(),False),
     ("C31_bigfav_RB","RB","big_fav",d.team_spread.notna(),False),
     ("C32_bigfav_WR","WR","big_fav",d.team_spread.notna(),False),
     ("C33_bigfav_TE","TE","big_fav",d.team_spread.notna(),False),
     ("C34_bigdog_QB","QB","big_dog",d.team_spread.notna(),False),
     ("C35_bigdog_RB","RB","big_dog",d.team_spread.notna(),False),
     ("C36_bigdog_WR","WR","big_dog",d.team_spread.notna(),False),
     ("C37_bigdog_TE","TE","big_dog",d.team_spread.notna(),False),
     ("C38_backupQB_RB","RB","backup_strict",d.gnum>=4,False),
     ("C39_backupQB_WR","WR","backup_strict",d.gnum>=4,False),
     ("C40_backupQB_TE","TE","backup_strict",d.gnum>=4,False),
     ("C41_WR1out_WR","WR","wr1_out_flag",d.wr1_id.notna(),False),
     ("C42_WR1out_TE","TE","wr1_out_flag",d.wr1_id.notna(),False),
     ("C43_WR1out_RB","RB","wr1_out_flag",d.wr1_id.notna(),False),
     ("C44_refDPIhigh_x_deepWR","WR","ref_dpi_hi",d.deep_wr & d.dpi_prior.notna(),False),
     ("C45_refDPIhigh_x_allWR","WR","ref_dpi_hi",d.dpi_prior.notna(),False),
     ("C46_birthday_week_pooled","ALL","birthday_week",d.bd.notna(),False),
     ("C47_revenge_game_pooled","ALL","revenge",None,False),
     ("C48_primetime_Godwin","PLAYER:"+str(GODWIN),"primetime",None,False),
     ("C49_primetime_Aiyuk","PLAYER:"+str(AIYUK),"primetime",None,False),
     ("C50_primetime_DSmith","PLAYER:"+str(DSMITH),"primetime",None,False),
    ]

    B = 4000
    rng = np.random.default_rng(6102026)
    blocks = np.sort(d.block.unique())
    nb = len(blocks)
    IDX = rng.integers(0, nb, size=(B, nb))   # shared block resample matrix
    bpos = {b:i for i,b in enumerate(blocks)}

    def run_test(tid, pos, flag, mask, research):
        if pos.startswith("PLAYER:"):
            sub = d[d.player_id == pos.split(":")[1]]
        elif pos == "ALL": sub = d
        elif pos == "WRTE": sub = d[d.position.isin(["WR","TE"])]
        else: sub = d[d.position == pos]
        if mask is not None: sub = sub[mask.reindex(sub.index, fill_value=False)] if len(mask)!=len(sub) else sub[mask]
        f = sub[flag].fillna(False).astype(bool)
        y = sub.r_adj.values
        n1, n0 = int(f.sum()), int((~f).sum())
        if n1 < 8 or n0 < 8:
            return {"test": tid, "n_raw": n1, "note": "insufficient n", "research_only": research}
        m1, m0 = float(y[f].mean()), float(y[~f].mean())
        eff = m1 - m0
        s1, s0 = float(y[f].std(ddof=1)), float(y[~f].std(ddof=1))
        se_iid = float(np.sqrt(s1*s1/n1 + s0*s0/n0))
        if pos.startswith("PLAYER:"):
            # iid bootstrap over this player's games
            rg = np.random.default_rng(6102026 + hash(tid)%10000)
            y1, y0 = y[f.values], y[~f.values]
            bs = np.array([rg.choice(y1,len(y1)).mean() - rg.choice(y0,len(y0)).mean() for _ in range(B)])
        else:
            bi = sub.block.map(bpos).values
            s1a = np.bincount(bi, weights=np.where(f, y, 0.0), minlength=nb)
            c1a = np.bincount(bi, weights=f.astype(float), minlength=nb)
            s0a = np.bincount(bi, weights=np.where(~f, y, 0.0), minlength=nb)
            c0a = np.bincount(bi, weights=(~f).astype(float), minlength=nb)
            S1, C1 = s1a[IDX].sum(1), c1a[IDX].sum(1)
            S0, C0 = s0a[IDX].sum(1), c0a[IDX].sum(1)
            ok = (C1 > 0) & (C0 > 0)
            bs = (S1[ok]/C1[ok]) - (S0[ok]/C0[ok])
        se_b = float(bs.std(ddof=1))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = 2*min(float((bs <= 0).mean()), float((bs >= 0).mean())); p = max(p, 1.0/B)
        n_eff = int(round(n1 * min(1.0, (se_iid/se_b)**2))) if se_b > 0 else n1
        # per-season stability
        sea = []
        for s, g in sub.groupby("season"):
            fg = g[flag].fillna(False).astype(bool)
            if fg.sum() >= 5 and (~fg).sum() >= 5:
                sea.append((int(s), float(g.r_adj[fg].mean() - g.r_adj[~fg].mean()), int(fg.sum())))
        same = sum(1 for _,e,_ in sea if np.sign(e) == np.sign(eff))
        # early/late
        def sub_eff(m):
            g = sub[m]; fg = g[flag].fillna(False).astype(bool)
            if fg.sum() >= 5 and (~fg).sum() >= 5:
                return round(float(g.r_adj[fg].mean() - g.r_adj[~fg].mean()), 3)
            return None
        return {"test": tid, "n_raw": n1, "n_ctrl": n0, "n_eff": n_eff,
                "effect_raw": round(eff,3), "se_boot": round(se_b,3),
                "ci": [round(float(lo),3), round(float(hi),3)], "p": round(p,5),
                "baseline_pts_cond": round(float(sub.fantasy_points[f].mean()),2),
                "baseline_pts_ctrl": round(float(sub.fantasy_points[~f].mean()),2),
                "seasons_same_sign": f"{same}/{len(sea)}",
                "eff_2019_22": sub_eff(~sub.late), "eff_2023_25": sub_eff(sub.late),
                "research_only": research}

    ck = f"{OUT}/ckpt/cond_results.json"
    res = json.load(open(ck)) if os.path.exists(ck) else {}
    start, end = int(arg1), int(arg2)
    for spec in SPEC[start:end]:
        r = run_test(*spec)
        res[spec[0]] = r
        print(spec[0], "eff", r.get("effect_raw"), "p", r.get("p"), "n", r.get("n_raw"))
    json.dump(res, open(ck, "w"), indent=1)
    print(f"done {start}:{end}, total {len(res)}")


def step6b(arg1=None, arg2=None):
    import pandas as pd, numpy as np, json
    from scipy import stats
    OUT="/tmp/exp_chemistry"
    d = pd.read_parquet(f"{OUT}/cond_panel.parquet")
    res = json.load(open(f"{OUT}/ckpt/cond_results.json"))

    # ---- C51: per-player home-vs-away residual deviation variance (beyond league home effect)
    lg_home = d.r_adj[d.is_home].mean() - d.r_adj[~d.is_home].mean()
    rows=[]
    for pidv, g in d.groupby("player_id"):
        h, a = g.r_adj[g.is_home], g.r_adj[~g.is_home]
        if len(h) >= 15 and len(a) >= 15:
            dev = (h.mean()-a.mean()) - lg_home
            se2 = h.var(ddof=1)/len(h) + a.var(ddof=1)/len(a)
            rows.append((dev, se2, len(h)))
    dev = np.array([r[0] for r in rows]); se2 = np.array([r[1] for r in rows])
    T = float(np.sum(dev**2/se2)); df = len(rows)
    p51 = float(stats.chi2.sf(T, df))
    tau2_stad = float(max(0.0, np.mean(dev**2 - se2)))
    res["C51_own_stadium_quirk_variance"] = {"test":"C51_own_stadium_quirk_variance","n_players":df,
        "league_home_edge_pts": round(float(lg_home),3), "T": round(T,1), "df": df, "p": round(p51,5),
        "tau_player_stadium_pts": round(float(np.sqrt(tau2_stad)),3), "research_only": False}
    print("C51:", res["C51_own_stadium_quirk_variance"])

    # ---- EB posterior across C01-C50 standard split tests
    eff, se, ids = [], [], []
    for k,v in res.items():
        if k.startswith("C") and "effect_raw" in v and v.get("se_boot"):
            eff.append(v["effect_raw"]); se.append(v["se_boot"]); ids.append(k)
    eff = np.array(eff); se = np.array(se)
    tau2 = float(max(0.0, np.mean(eff**2 - se**2)))  # MoM across battery
    for i,k in enumerate(ids):
        w = tau2/(tau2 + se[i]**2)
        pm = w*eff[i]
        psd = float(np.sqrt(tau2*se[i]**2/(tau2+se[i]**2)))
        res[k]["posterior_mean"] = round(float(pm),3)
        res[k]["posterior_ci"] = [round(float(pm-1.96*psd),3), round(float(pm+1.96*psd),3)]
        res[k]["shrink_w"] = round(float(w),3)
    print("battery tau (pts):", round(np.sqrt(tau2),3))

    # ---- assemble all 64 p-values, BH-FDR
    pair = json.load(open(f"{OUT}/ckpt/pair_var_tests.json"))
    AMAP = {"A1_pair_var_vol_tpr":"pair_vol_var","A2_pair_var_epa_per_tgt":"pair_epa_var","A3_pair_var_cpoe":"pair_cpoe_var",
     "A4_pair_var_yds_per_tgt":"pair_yds_var","A5_pair_var_success":"pair_succ_var","A6_pair_var_explosive":"pair_expl_var",
     "A7_pair_var_airshare":"pair_airshare_var","A8_pair_var_rzshare":"pair_rzshare_var","A9_pair_var_catchable2022":"pair_catchable_var",
     "B1_fwd_pair_vol":"fwd_pair_vol","B2_fwd_pair_epa":"fwd_pair_epa","B3_fwd_pair_cpoe":"fwd_pair_cpoe","B4_pair_var_pressure_gap":"pair_pressure_gap_var"}
    battery = json.load(open(f"{OUT}/battery.json"))
    allp = []
    for t in battery["tests"]:
        if t in AMAP: allp.append((t, float(pair[AMAP[t]]["p"])))
        elif t in res: allp.append((t, float(res[t]["p"])))
        else: print("MISSING:", t)
    assert len(allp) == 64, len(allp)
    m = len(allp)
    srt = sorted(allp, key=lambda x: x[1])
    qs = {}
    prev = 1.0
    for rank in range(m, 0, -1):
        t, p = srt[rank-1]
        q = min(prev, p*m/rank)
        qs[t] = q; prev = q
    fdr = {t: {"p": p, "q": round(qs[t],5)} for t,p in allp}
    json.dump({"tau_battery_pts": round(np.sqrt(tau2),4), "fdr": fdr}, open(f"{OUT}/ckpt/fdr.json","w"), indent=1)
    json.dump(res, open(f"{OUT}/ckpt/cond_results.json","w"), indent=1)
    sig10 = [t for t in fdr if fdr[t]["q"] <= 0.10]
    sig05 = [t for t in fdr if fdr[t]["q"] <= 0.05]
    print(f"q<=0.10: {len(sig10)}"); [print("  ", t, fdr[t]) for t in sorted(sig10, key=lambda x: fdr[x]['q'])]
    print(f"q<=0.05: {len(sig05)}")


def step7(arg1=None, arg2=None):
    import pandas as pd, numpy as np, json
    OUT="/tmp/exp_chemistry"
    pw = pd.read_parquet(f"{OUT}/pair_week.parquet")
    pw["t"] = (pw.season*100 + pw.week).astype("int64")

    # cumulative-through tables (value AS OF just after week t) -> query with strict < t via merge_asof allow_exact_matches=False
    pw = pw.sort_values(["qb_id","rec_id","t"])
    g = pw.groupby(["qb_id","rec_id"], sort=False)
    pc = pw[["qb_id","rec_id","t"]].copy()
    pc["C_routes"] = g.routes.cumsum(); pc["C_targets"] = g.targets.cumsum()
    rw = pw.groupby(["rec_id","rec_pos","t"], as_index=False)[["routes","targets"]].sum().sort_values(["rec_id","t"])
    gr = rw.groupby("rec_id", sort=False)
    rw["CR_routes"] = gr.routes.cumsum(); rw["CR_targets"] = gr.targets.cumsum(); rw["R_games"] = gr.cumcount()+1

    st = pd.read_parquet(f"{OUT}/team_starters.parquet").sort_values(["team","t"])
    st["t"] = st.t.astype("int64")  # starter_qb per team-week

    # query set: receiver rows of cond_panel + outer preds
    cdp = pd.read_parquet(f"{OUT}/cond_panel.parquet", columns=["season","week","player_id","position","team","r_adj","targets","fantasy_points"])
    cdp = cdp[cdp.position.isin(["WR","TE","RB"])].copy(); cdp["src"]="train"
    op = pd.read_parquet("/sessions/dreamy-compassionate-goodall/repos/tailstail/reports/fantasy_outer_predictions.parquet",
                         columns=["season","week","player_id","position","team"])
    op = op[op.position.isin(["WR","TE","RB"])].copy(); op["src"]="outer"
    q = pd.concat([cdp[["season","week","player_id","position","team","src"]], op], ignore_index=True).drop_duplicates(["season","week","player_id","team"])
    q["t"] = (q.season*100 + q.week).astype("int64")

    # expected QB: last starter strictly before t
    q = q.sort_values("t").reset_index(drop=True)
    stq = st[["team","t","starter_qb"]].sort_values("t")
    q = pd.merge_asof(q, stq.rename(columns={"starter_qb":"exp_qb"}), on="t", by="team", allow_exact_matches=False)

    # pair cums strictly before t
    q = q.rename(columns={"player_id":"rec_id"})
    q = q.sort_values("t")
    pcs = pc.rename(columns={"qb_id":"exp_qb"}).sort_values("t")
    q = pd.merge_asof(q, pcs, on="t", by=["exp_qb","rec_id"], allow_exact_matches=False)
    rws = rw[["rec_id","t","CR_routes","CR_targets","R_games"]].sort_values("t")
    q = pd.merge_asof(q, rws, on="t", by="rec_id", allow_exact_matches=False)
    for c in ["C_routes","C_targets","CR_routes","CR_targets","R_games"]: q[c] = q[c].fillna(0.0)

    def fit_consts(maxseason):
        sub = pw[pw.season < maxseason]
        consts = {}
        mu = {}
        for pos in ("WR","TE","RB"):
            s = sub[sub.rec_pos==pos]; mu[pos] = float(s.targets.sum()/max(s.routes.sum(),1))
        consts["mu_tpr"] = mu
        snap = sub.groupby(["qb_id","rec_id","rec_pos"], as_index=False)[["routes","targets"]].sum()
        rsnap = snap.groupby(["rec_id","rec_pos"], as_index=False)[["routes","targets"]].sum()
        rsnap = rsnap.sort_values("routes",ascending=False).drop_duplicates("rec_id")
        p_tpr = rsnap.targets/np.maximum(rsnap.routes,1)
        sig2 = float(np.average((p_tpr.clip(.02,.5)*(1-p_tpr.clip(.02,.5))), weights=rsnap.routes))
        consts["sig2_vol"] = sig2
        dev_r = p_tpr - rsnap.rec_pos.map(mu)
        m = rsnap.routes>=200
        consts["tau2_rec_vol"] = float(max(0.0, np.average(dev_r[m]**2,weights=rsnap.routes[m]) - np.average(sig2/rsnap.routes[m],weights=rsnap.routes[m])))
        # pair tau2: pair dev after receiver EB baseline
        rsnap["w"] = consts["tau2_rec_vol"]/(consts["tau2_rec_vol"]+sig2/np.maximum(rsnap.routes,1e-9))
        rsnap["b_tpr"] = rsnap.rec_pos.map(mu) + rsnap.w*dev_r.fillna(0)
        sn = snap.merge(rsnap[["rec_id","b_tpr"]], on="rec_id")
        dev_p = sn.targets/np.maximum(sn.routes,1) - sn.b_tpr
        mp = sn.routes>=150
        consts["tau2_pair_vol"] = float(max(0.0, np.average(dev_p[mp]**2,weights=sn.routes[mp]) - np.average(sig2/sn.routes[mp],weights=sn.routes[mp])))
        return consts

    CONSTS = {Y: fit_consts(Y) for Y in (2023,2024,2025)}
    json.dump(CONSTS, open(f"{OUT}/ckpt/consts_by_year.json","w"), indent=1)

    for Y, cn in CONSTS.items():
        mu = q.position.map(cn["mu_tpr"])
        wr_ = cn["tau2_rec_vol"]/(cn["tau2_rec_vol"]+cn["sig2_vol"]/np.maximum(q.CR_routes,1e-9))
        b_tpr = mu + np.where(q.CR_routes>0, wr_*(q.CR_targets/np.maximum(q.CR_routes,1)-mu), 0.0)
        wp = cn["tau2_pair_vol"]/(cn["tau2_pair_vol"]+cn["sig2_vol"]/np.maximum(q.C_routes,1e-9))
        pair_dev = np.where(q.C_routes>0, q.C_targets/np.maximum(q.C_routes,1)-b_tpr, 0.0)
        q[f"pair_vol_post_{Y}"] = np.where(q.C_routes>0, wp*pair_dev, 0.0)
        q[f"exp_routes_{Y}"] = np.where(q.R_games>0, q.CR_routes/np.maximum(q.R_games,1), 0.0)
        q[f"chem_raw_{Y}"] = q[f"pair_vol_post_{Y}"] * q[f"exp_routes_{Y}"]   # extra targets/game vs receiver baseline with this QB
    q.to_parquet(f"{OUT}/chem_features.parquet", index=False)
    print("rows:", len(q), "| exp_qb found:", q.exp_qb.notna().mean().round(4), "| pair history >0:", (q.C_routes>0).mean().round(4))
    print("chem_raw_2025 describe:", q.chem_raw_2025.describe()[["mean","std","min","max"]].round(3).to_dict())
    print(json.dumps({str(k): {kk: (round(vv,6) if isinstance(vv,float) else vv) for kk,vv in v.items() if kk!='mu_tpr'} for k,v in CONSTS.items()}, indent=1))


def step7b(arg1=None, arg2=None):
    import pandas as pd, numpy as np, json
    OUT="/tmp/exp_chemistry"
    REPO="/sessions/dreamy-compassionate-goodall/repos/tailstail"
    CAP = 1.5

    op = pd.read_parquet(f"{REPO}/reports/fantasy_outer_predictions.parquet")
    cdp = pd.read_parquet(f"{OUT}/cond_panel.parquet")
    chem = pd.read_parquet(f"{OUT}/chem_features.parquet").rename(columns={"rec_id":"player_id"})

    # game-level flags per (season, week, team) from schedules (pregame-legal)
    sc = pd.read_parquet(f"{REPO}/historical/fantasy/schedules.parquet")
    sc = sc[sc.game_type=="REG"]
    rows=[]
    for side in ("home","away"):
        t = sc[["season","week","game_id","home_team","away_team","spread_line","roof","surface","gametime","weekday","home_rest","away_rest"]].copy()
        t["team"] = t[f"{side}_team"]
        t["team_spread"] = np.where(side=="home", t.spread_line, -t.spread_line)
        t["rest"] = t[f"{side}_rest"]
        rows.append(t)
    fl = pd.concat(rows, ignore_index=True)
    TURF={"fieldturf","matrixturf","sportturf","astroturf","a_turf","turf","astroplay","fieldturf "}
    fl["is_turf"]=fl.surface.str.lower().str.strip().isin(TURF).astype(float)
    fl["indoor_env"]=fl.roof.str.lower().isin(["dome","closed"]).astype(float)
    fl["hour"]=pd.to_numeric(fl.gametime.str.split(":").str[0],errors="coerce")
    fl["primetime"]=((fl.hour>=19)&fl.weekday.isin(["Thursday","Sunday","Monday"])).astype(float)
    fl["short_week"]=(fl.rest<=5).astype(float); fl["post_bye"]=(fl.rest>=13).astype(float)
    fl["big_fav"]=(fl.team_spread>=7).astype(float); fl["big_dog"]=(fl.team_spread<=-7).astype(float)
    fl["alt_den"]=(fl.home_team=="DEN").astype(float)
    FLAGS=["indoor_env","is_turf","primetime","short_week","post_bye","big_fav","big_dog","alt_den"]
    fl = fl[["season","week","team"]+FLAGS].drop_duplicates(["season","week","team"])

    # training table: cond_panel rows + chem features + flags(already in cond_panel as booleans)
    tr = cdp[["season","week","player_id","position","team","r_adj"]].copy()
    for f in FLAGS: tr[f] = cdp[f].astype(float).values
    key=["season","week","player_id","team"]
    tr = tr.merge(chem[key+["chem_raw_2023","chem_raw_2024","chem_raw_2025"]], on=key, how="left")
    for c in ["chem_raw_2023","chem_raw_2024","chem_raw_2025"]: tr[c]=tr[c].fillna(0.0)

    # outer eval table
    ev = op.merge(fl, on=["season","week","team"], how="left")
    ev = ev.merge(chem[key+["chem_raw_2023","chem_raw_2024","chem_raw_2025"]], on=key, how="left")
    for c in FLAGS: ev[c]=ev[c].fillna(0.0)
    for c in ["chem_raw_2023","chem_raw_2024","chem_raw_2025"]: ev[c]=ev[c].fillna(0.0)
    print("flag coverage on outer:", (op.merge(fl,on=['season','week','team'],how='left')[FLAGS].notna().all(axis=1)).mean().round(4))

    def ols(X, y):
        X1 = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
        return beta  # [intercept, coefs...]

    adj = {v: np.zeros(len(ev)) for v in ("b","c","d","e")}
    diag = {}
    for Y in (2023,2024,2025):
        ch = f"chem_raw_{Y}"
        trY = tr[(tr.season>=2019)&(tr.season<Y)]
        mY = ev.season==Y
        for pos in ("QB","RB","WR","TE"):
            tp = trY[trY.position==pos]
            mp = mY & (ev.position==pos)
            if not mp.any(): continue
            # centering means from TRAIN
            mu_f = tp[FLAGS].mean()
            mu_c = tp[ch].mean()
            Xf_tr = (tp[FLAGS]-mu_f).values; y = tp.r_adj.values
            Xf_ev = (ev.loc[mp,FLAGS]-mu_f).values
            c_tr = (tp[ch]-mu_c).values.reshape(-1,1); c_ev = (ev.loc[mp,ch]-mu_c).values.reshape(-1,1)
            # (b) chemistry only (receivers; QB=0)
            if pos!="QB":
                bb = ols(c_tr, y)
                adj["b"][mp.values] = (c_ev @ bb[1:]).ravel()
                diag[f"{Y}_{pos}_chem_slope"] = round(float(bb[1]),3)
            # (c) conditions only
            bc = ols(Xf_tr, y)
            adj["c"][mp.values] = Xf_ev @ bc[1:]
            # (d) both
            Xd_tr = np.column_stack([Xf_tr, c_tr]) if pos!="QB" else Xf_tr
            Xd_ev = np.column_stack([Xf_ev, c_ev]) if pos!="QB" else Xf_ev
            bd = ols(Xd_tr, y)
            adj["d"][mp.values] = Xd_ev @ bd[1:]
            # (e) interactions challenger
            def inter(Xf, c):
                pt = Xf[:,FLAGS.index("primetime")]; pb = Xf[:,FLAGS.index("post_bye")]
                ind = Xf[:,FLAGS.index("indoor_env")]; tf = Xf[:,FLAGS.index("is_turf")]
                bd_ = Xf[:,FLAGS.index("big_dog")]
                cc = c.ravel()
                return np.column_stack([Xf, cc, cc*pt, cc*bd_, pt*pb, ind*tf])
            Xe_tr = inter(Xf_tr, c_tr); Xe_ev = inter(Xf_ev, c_ev)
            be = ols(Xe_tr, y)
            adj["e"][mp.values] = Xe_ev @ be[1:]

    res = {}
    err0 = np.abs(op.fantasy_points - op.projection_mean).values
    blocks = (op.season*100+op.week).values
    ub = np.unique(blocks)
    rng = np.random.default_rng(6102026)
    BIDX = rng.integers(0, len(ub), size=(10000, len(ub)))
    bmap = {b:i for i,b in enumerate(ub)}
    bi = np.array([bmap[b] for b in blocks])
    from scipy import stats as st_
    def grade(a):
        a = np.clip(a, -CAP, CAP)
        pm = op.projection_mean.values + a
        err = np.abs(op.fantasy_points.values - pm)
        d = err - err0
        ds = np.bincount(bi, weights=d, minlength=len(ub)); dc = np.bincount(bi, minlength=len(ub))
        bs = ds[BIDX].sum(1)/dc[BIDX].sum(1)
        lo, hi = np.percentile(bs,[2.5,97.5])
        cov = float(((op.fantasy_points.values>=op.projection_lower80.values+a)&(op.fantasy_points.values<=op.projection_upper80.values+a)).mean())
        rho = float(st_.spearmanr(op.fantasy_points.values, pm).statistic)
        gate = "shadow" if (d.mean()<=-0.05 and hi<0) else ("research_only" if (lo<0<hi or (d.mean()<0 and hi<0)) else "rejected")
        if d.mean()<=-0.05 and hi<0: gate="shadow"
        elif lo<0 and hi<0: gate="research_only_negative_but_small" if d.mean()>-0.05 else "shadow"
        elif lo<=0<=hi: gate="research_only" if d.mean()<0 else "rejected"
        else: gate="rejected"
        return {"mae": round(float(err.mean()),6), "delta_mae": round(float(d.mean()),6),
                "ci": [round(float(lo),6), round(float(hi),6)], "cov80": round(cov,4),
                "spearman": round(rho,4), "adj_sd": round(float(a.std()),4), "adj_maxabs": round(float(np.abs(a).max()),3),
                "pct_capped": round(float((np.abs(a)>=CAP-1e-9).mean()),4), "gate": gate}
    base_cov = float(((op.fantasy_points>=op.projection_lower80)&(op.fantasy_points<=op.projection_upper80)).mean())
    res["a_none"] = {"mae": round(float(err0.mean()),6), "cov80": round(base_cov,4),
                     "spearman": round(float(st_.spearmanr(op.fantasy_points, op.projection_mean).statistic),4)}
    for v in ("b","c","d","e"):
        res[{"b":"b_chemistry","c":"c_conditions","d":"d_both","e":"e_interactions"}[v]] = grade(adj[v])
    json.dump({"variants": res, "diag": diag}, open(f"{OUT}/ckpt/variants.json","w"), indent=1)
    print(json.dumps(res, indent=1))


STEPS = {k: v for k, v in list(globals().items()) if k.startswith("step") and callable(v)}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STEPS:
        print("usage: chemistry_study.py <" + "|".join(sorted(STEPS)) + "> [args]")
        raise SystemExit(1)
    STEPS[sys.argv[1]](*sys.argv[2:])
