#!/usr/bin/env python3
"""Factor-families study (wave 2): opponent/scheme, environment, game context, market.

Predeclared confirmatory battery of exactly 40 tests (see BATTERY below and the
frozen header of reports/factor_families_audit.md). Strictly pregame features:
shift-then-roll trailing stats, schedule-derivable flags, official pregame injury
designations. Observed-game weather is NEVER an exposure here (leakage_proxy).

Reproducible end-to-end, chunked for the 45s cap:
    python3 analysis/factor_families_study.py --stage frames
    python3 analysis/factor_families_study.py --stage battery --family A|B|C|D
    python3 analysis/factor_families_study.py --stage fdr
    python3 analysis/factor_families_study.py --stage gates
    python3 analysis/factor_families_study.py --stage registry
    python3 analysis/factor_families_study.py --stage report
All scratch goes to /tmp/exp_families/. Repo data/ caches are read-only inputs;
the only repo outputs are data/factor_registry_families.json and
reports/factor_families_audit.md (results appended below the frozen header).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = "/tmp/exp_families"
SEED = 6102026
N_BOOT = 1500
SEASONS = range(2019, 2026)
POSITIONS = ("QB", "RB", "WR", "TE")
EWM_HALFLIFE = 5
EWM_MIN = 3

# ---------------------------------------------------------------------------
# Predeclared battery (frozen with the report header before any computation).
# outcome: d_fp / d_pass_yards / d_rush_yards / d_rec_yards (player) or
#          d_plays / d_pass_att (team).
# exposure values: 1 exposed, 0 control, NaN excluded.
# ---------------------------------------------------------------------------
def _t(tid, family, level, positions, exposure, outcome, prior_sd, definition, years="2019-2025"):
    return dict(id=tid, family=family, level=level, positions=positions,
                exposure=exposure, outcome=outcome, prior_sd=prior_sd,
                definition=definition, years=years)

BATTERY = [
    _t("A01_opp_qb_points_soft", "A", "player", ("QB",), "x_opp_fp_soft_QB", "d_fp", 1.0,
       "Opponent trailing QB fantasy points allowed, within-week top vs bottom tercile"),
    _t("A02_opp_rb_points_soft", "A", "player", ("RB",), "x_opp_fp_soft_RB", "d_fp", 1.0,
       "Opponent trailing RB fantasy points allowed, top vs bottom tercile"),
    _t("A03_opp_wr_points_soft", "A", "player", ("WR",), "x_opp_fp_soft_WR", "d_fp", 1.0,
       "Opponent trailing WR fantasy points allowed, top vs bottom tercile"),
    _t("A04_opp_te_points_soft", "A", "player", ("TE",), "x_opp_fp_soft_TE", "d_fp", 1.0,
       "Opponent trailing TE fantasy points allowed, top vs bottom tercile"),
    _t("A05_opp_pass_epa_soft_qb", "A", "player", ("QB",), "x_opp_pass_epa_soft", "d_fp", 1.0,
       "Opponent trailing EPA/dropback allowed, top (softest) vs bottom tercile"),
    _t("A06_opp_pass_epa_soft_wr", "A", "player", ("WR",), "x_opp_pass_epa_soft", "d_fp", 1.0,
       "Opponent trailing EPA/dropback allowed, top vs bottom tercile"),
    _t("A07_opp_rush_epa_soft_rb", "A", "player", ("RB",), "x_opp_rush_epa_soft", "d_rush_yards", 5.0,
       "Opponent trailing EPA/rush allowed, top vs bottom tercile"),
    _t("A08_opp_pressure_high_qb", "A", "player", ("QB",), "x_opp_pressure_high", "d_fp", 1.0,
       "Opponent trailing (sack+qb_hit)/dropback, top vs bottom tercile"),
    _t("A09_opp_blitz_high_qb", "A", "player", ("QB",), "x_opp_blitz_high", "d_fp", 1.0,
       "Opponent trailing blitz share (FTN n_blitzers>0 on dropbacks), top vs bottom tercile",
       years="2022-2025"),
    _t("A10_opp_heavy_box_rb", "A", "player", ("RB",), "x_opp_heavy_box", "d_rush_yards", 5.0,
       "Opponent trailing share of rushes faced with 8+ box (FTN), top vs bottom tercile",
       years="2022-2025"),
    _t("A11_opp_pace_fast_all", "A", "player", POSITIONS, "x_opp_pace_fast", "d_fp", 1.0,
       "Opponent trailing offensive plays/game, top vs bottom tercile (pace-up spillover)"),
    _t("A12_opp_proe_allowed_qb", "A", "player", ("QB",), "x_opp_proe_allowed_high", "d_fp", 1.0,
       "Opponent trailing PROE allowed (mean pass_oe faced), top vs bottom tercile"),
    _t("A13_opp_explosive_pass_wr", "A", "player", ("WR",), "x_opp_explosive_high", "d_fp", 1.0,
       "Opponent trailing 20+yd completions per dropback allowed, top vs bottom tercile"),
    _t("A14_opp_db2_out_qb", "A", "player", ("QB",), "x_opp_db2_out", "d_fp", 1.0,
       "2+ opponent DBs officially Out/Doubtful pregame (factor_frame flag)"),
    _t("A15_opp_db2_out_wr", "A", "player", ("WR",), "x_opp_db2_out", "d_fp", 1.0,
       "2+ opponent DBs officially Out/Doubtful pregame"),
    _t("A16_opp_front2_out_rb", "A", "player", ("RB",), "x_opp_front2_out", "d_fp", 1.0,
       "2+ opponent front-7 defenders officially Out/Doubtful pregame"),
    _t("A17_opp_man_rate_wr", "A", "player", ("WR",), "x_opp_man_high", "d_fp", 1.0,
       "Opponent trailing man-coverage share (participation MAN/(MAN+ZONE)), top vs bottom tercile",
       years="2019-2025 (tag coverage varies)"),
    _t("B01_turf_all", "B", "player", POSITIONS, "x_turf", "d_fp", 1.0,
       "Game on artificial turf vs natural grass"),
    _t("B02_closed_roof_all", "B", "player", POSITIONS, "x_closed_roof", "d_fp", 1.0,
       "Roof dome/closed vs outdoors/open"),
    _t("B03_closed_roof_qb_passyds", "B", "player", ("QB",), "x_closed_roof", "d_pass_yards", 8.0,
       "Roof dome/closed vs outdoors/open, QB passing yards vs trailing"),
    _t("B04_altitude_road_all", "B", "player", POSITIONS, "x_altitude", "d_fp", 1.0,
       "Road team at Denver / any team at Azteca (5,280ft+); Denver home rows excluded"),
    _t("B05_travel_1500_all", "B", "player", POSITIONS, "x_travel_1500", "d_fp", 1.0,
       "Road great-circle travel >=1500mi vs shorter road trips; international excluded"),
    _t("B06_tz2_all", "B", "player", POSITIONS, "x_tz2", "d_fp", 1.0,
       "Road 2+ time zones crossed vs 0-1; international excluded"),
    _t("B07_body_clock_all", "B", "player", POSITIONS, "x_body_clock", "d_fp", 1.0,
       "Pacific-home team on road at 13:xx ET kickoff vs other road games"),
    _t("B08_international_all", "B", "player", POSITIONS, "x_international", "d_fp", 1.0,
       "International site (London/Mexico City/Munich/Frankfurt/Sao Paulo/Dublin/Madrid/Berlin)"),
    _t("B09_short_rest_all", "B", "player", POSITIONS, "x_short_rest", "d_fp", 1.0,
       "Rest <=5 days vs regular 6-11 day rest (post-bye excluded from control)"),
    _t("B10_post_bye_all", "B", "player", POSITIONS, "x_post_bye", "d_fp", 1.0,
       "Rest >=12 days (post-bye) vs regular 6-11 day rest (short rest excluded)"),
    _t("B11_post_bye_wr_recyds", "B", "player", ("WR",), "x_post_bye", "d_rec_yards", 5.0,
       "Post-bye WR receiving yards vs trailing (fantasy-space re-derivation of prop survivor)"),
    _t("B12_after_ot_all", "B", "player", POSITIONS, "x_after_ot", "d_fp", 1.0,
       "Team played overtime last week (fatigue carryover)"),
    _t("C01_big_fav_all", "C", "player", POSITIONS, "x_big_fav", "d_fp", 1.0,
       "Team spread <= -7 (big favorite) vs all other lined games"),
    _t("C02_big_dog_all", "C", "player", POSITIONS, "x_big_dog", "d_fp", 1.0,
       "Team spread >= +7 (big underdog) vs all other lined games"),
    _t("C03_high_total_all", "C", "player", POSITIONS, "x_high_total", "d_fp", 1.0,
       "Game total >= 47.5 vs all other lined games"),
    _t("C04_low_total_all", "C", "player", POSITIONS, "x_low_total", "d_fp", 1.0,
       "Game total <= 41.5 vs all other lined games"),
    _t("C05_implied_high_all", "C", "player", POSITIONS, "x_implied_high", "d_fp", 1.0,
       "Implied team total, within-season-week top vs bottom tercile"),
    _t("C06_primetime_all", "C", "player", POSITIONS, "x_primetime", "d_fp", 1.0,
       "Primetime kickoff (feature-frame definition) vs day games"),
    _t("C07_div_rematch_all", "C", "player", POSITIONS, "x_div_rematch", "d_fp", 1.0,
       "Second divisional meeting vs first meeting (population: division games)"),
    _t("C08_ref_pace_team", "C", "team", (), "x_ref_pace_high", "d_plays", 2.5,
       "Referee (crew chief) trailing combined plays/game, top vs bottom tercile -> team plays vs trailing"),
    _t("C09_fourth_aggr_team", "C", "team", (), "x_fourth_aggr_high", "d_plays", 2.5,
       "Team trailing 4th-and-<=2 go rate (neutral situations), top vs bottom tercile -> team plays vs trailing"),
    _t("C10_new_hc_all", "C", "player", POSITIONS, "x_new_hc", "d_fp", 1.0,
       "Head coach differs from team's final game of prior season (2020+)", years="2020-2025"),
    _t("D01_implied_high_passatt_team", "D", "team", (), "x_implied_high", "d_pass_att", 2.0,
       "Implied team total within-week top vs bottom tercile -> team pass attempts vs trailing"),
]
assert len(BATTERY) == 40, f"predeclared battery must be exactly 40, got {len(BATTERY)}"

# ---------------------------------------------------------------------------
# Geography: home site per (team, season), standard-time UTC offset, coords.
# Approximate city coordinates; used only for distance bands / tz counts.
# ---------------------------------------------------------------------------
TEAM_SITE = {
    "ARI": (33.53, -112.26, -7), "ATL": (33.75, -84.40, -5), "BAL": (39.28, -76.62, -5),
    "BUF": (42.77, -78.79, -5), "CAR": (35.23, -80.84, -5), "CHI": (41.86, -87.62, -6),
    "CIN": (39.10, -84.51, -5), "CLE": (41.51, -81.70, -5), "DAL": (32.75, -97.09, -6),
    "DEN": (39.74, -105.02, -7), "DET": (42.34, -83.05, -5), "GB": (44.50, -88.06, -6),
    "HOU": (29.68, -95.41, -6), "IND": (39.76, -86.16, -5), "JAX": (30.32, -81.64, -5),
    "KC": (39.05, -94.48, -6), "MIA": (25.96, -80.24, -5), "MIN": (44.97, -93.26, -6),
    "NE": (42.09, -71.26, -5), "NO": (29.95, -90.08, -6), "NYG": (40.81, -74.07, -5),
    "NYJ": (40.81, -74.07, -5), "PHI": (39.90, -75.17, -5), "PIT": (40.45, -80.02, -5),
    "SEA": (47.60, -122.33, -8), "SF": (37.40, -121.97, -8), "TB": (27.98, -82.50, -5),
    "TEN": (36.17, -86.77, -6), "WAS": (38.91, -76.86, -5),
    "OAK": (37.75, -122.20, -8), "LV": (36.09, -115.18, -8),
    "LA": (33.95, -118.34, -8), "LAC": (33.95, -118.34, -8),
}
# 2019 exceptions handled by same metro coords (LA Coliseum/Dignity Health ~ SoFi).
INTL_SITES = {
    "wembley": (51.556, -0.280, 0), "tottenham": (51.604, -0.066, 0),
    "azteca": (19.303, -99.151, -6), "allianz": (48.219, 11.625, 1),
    "deutsche bank": (50.069, 8.645, 1), "frankfurt": (50.069, 8.645, 1),
    "neo quimica": (-23.545, -46.474, -3), "corinthians": (-23.545, -46.474, -3),
    "arena corinthians": (-23.545, -46.474, -3), "croke": (53.361, -6.251, 0),
    "bernabeu": (40.453, -3.688, 1), "santiago bernab": (40.453, -3.688, 1),
    "olympiastadion": (52.515, 13.239, 1), "olympic stadium (berlin)": (52.515, 13.239, 1),
}
ALTITUDE_SITES = {"DEN"}  # + azteca via intl map

def haversine_mi(lat1, lon1, lat2, lon2):
    r = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))

def _ewm_trailing(df, entity_cols, order_cols, value_cols, halflife=EWM_HALFLIFE, min_periods=EWM_MIN):
    """Strict shift-then-roll: at game g, use only games strictly before g."""
    out = df.sort_values(order_cols).copy()
    g = out.groupby(list(entity_cols), sort=False)
    for c in value_cols:
        out[f"tr_{c}"] = g[c].transform(
            lambda s: s.shift(1).ewm(halflife=halflife, min_periods=min_periods).mean())
    return out

def _load_pbp():
    frames = [pd.read_parquet(os.path.join(ROOT, "historical", "historical_pbp.parquet"))]
    for y in (2024, 2025):
        frames.append(pd.read_parquet(os.path.join(ROOT, "historical", f"pbp_{y}.parquet")))
    pbp = pd.concat(frames, ignore_index=True)
    pbp = pbp[(pbp.season_type == "REG") & pbp.posteam.notna() & (pbp.posteam != "")]
    return pbp

# ---------------------------------------------------------------------------
# Stage: frames
# ---------------------------------------------------------------------------
def build_team_week(pbp):
    grp = pbp.groupby(["season", "week", "game_id", "posteam"], as_index=False)
    tw = grp.agg(
        plays=("play_id", lambda s: 0),  # replaced below
        dropbacks=("pass", "sum"),
        rush_att=("rush", "sum"),
        pass_attempt_raw=("pass_attempt", "sum"),
        sacks=("sack", "sum"),
        qb_hits=("qb_hit", "sum"),
    )
    pr = pbp[pbp["pass"] == 1]
    proe = pr[pr.pass_oe.notna()].groupby(["game_id", "posteam"])["pass_oe"].mean().rename("proe")
    epa_p = pr.groupby(["game_id", "posteam"])["epa"].mean().rename("epa_pass")
    epa_r = pbp[pbp["rush"] == 1].groupby(["game_id", "posteam"])["epa"].mean().rename("epa_rush")
    expl = (pr.assign(expl=((pr.passing_yards >= 20) & (pr.complete_pass == 1)).astype(float))
              .groupby(["game_id", "posteam"])["expl"].sum().rename("explosive_pass"))
    plays = pbp[(pbp["pass"] == 1) | (pbp["rush"] == 1)].groupby(["game_id", "posteam"]).size().rename("n_plays")
    # 4th-and-short go attempts (neutral-ish situations)
    f4 = pbp[(pbp.down == 4) & (pbp.ydstogo <= 2) & (pbp.qtr <= 3) & pbp.wp.between(0.05, 0.95)]
    f4g = f4.groupby(["game_id", "posteam"]).agg(
        fourth_short_n=("play_id", "size"),
        fourth_short_go=("pass", lambda s: 0)).reset_index()
    go = f4.assign(go=((f4["pass"] == 1) | (f4["rush"] == 1)).astype(float)).groupby(
        ["game_id", "posteam"])["go"].sum().rename("fourth_short_go")
    f4g = f4g.drop(columns=["fourth_short_go"]).merge(go.reset_index(), on=["game_id", "posteam"], how="left")
    for extra in (proe, epa_p, epa_r, expl, plays):
        tw = tw.merge(extra.reset_index(), on=["game_id", "posteam"], how="left")
    tw = tw.merge(f4g, on=["game_id", "posteam"], how="left")
    tw["plays"] = tw.pop("n_plays").fillna(0)
    tw["pass_att"] = tw.pass_attempt_raw - tw.sacks          # official attempts
    tw["pressure_rate"] = (tw.sacks + tw.qb_hits) / tw.dropbacks.clip(lower=1)
    tw["explosive_rate"] = tw.explosive_pass / tw.dropbacks.clip(lower=1)
    tw = tw.rename(columns={"posteam": "team"})
    return tw

def build_defense_week(pbp, tw):
    """Allowed view keyed by the defense, mirroring offense aggregates."""
    d = tw.rename(columns={"team": "off_team"}).copy()
    key = pbp[["game_id", "posteam", "defteam"]].drop_duplicates()
    d = d.merge(key.rename(columns={"posteam": "off_team", "defteam": "team"}),
                on=["game_id", "off_team"], how="left")
    keep = ["season", "week", "game_id", "team", "plays", "dropbacks", "pass_att",
            "proe", "epa_pass", "epa_rush", "explosive_rate", "pressure_rate_off"]
    d["pressure_rate_off"] = d["pressure_rate"]  # pressure the DEFENSE generated
    return d[keep].rename(columns={
        "plays": "plays_allowed", "dropbacks": "dropbacks_faced", "pass_att": "pass_att_allowed",
        "proe": "proe_allowed", "epa_pass": "epa_pass_allowed", "epa_rush": "epa_rush_allowed",
        "explosive_rate": "explosive_allowed", "pressure_rate_off": "pressure_generated"})

def build_ftn_defense(pbp):
    frames = []
    for y, fn in [(2022, "ftn_charting_2022.parquet"), (2023, "ftn_charting_2023.parquet"),
                  (2024, "ftn_charting_2024.parquet"), (2025, "ftn_2025.parquet")]:
        f = pd.read_parquet(os.path.join(ROOT, "historical", fn),
                            columns=["nflverse_game_id", "nflverse_play_id", "n_blitzers", "n_defense_box"])
        f["season"] = y
        frames.append(f)
    ftn = pd.concat(frames, ignore_index=True)
    ftn = ftn.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    base = pbp[["game_id", "play_id", "season", "week", "defteam", "pass", "rush"]]
    m = base.merge(ftn.drop(columns=["season"]), on=["game_id", "play_id"], how="inner")
    dp = m[m["pass"] == 1].groupby(["season", "week", "game_id", "defteam"], as_index=False).agg(
        blitz_share=("n_blitzers", lambda s: (s.fillna(0) > 0).mean()))
    dr = m[m["rush"] == 1].groupby(["season", "week", "game_id", "defteam"], as_index=False).agg(
        heavy_box_share=("n_defense_box", lambda s: (s >= 8).mean()))
    out = dp.merge(dr, on=["season", "week", "game_id", "defteam"], how="outer")
    return out.rename(columns={"defteam": "team"})

def build_manzone_defense(pbp):
    frames = []
    for y in SEASONS:
        p = os.path.join(ROOT, "historical", f"pbp_participation_{y}.parquet")
        if not os.path.exists(p):
            continue
        f = pd.read_parquet(p, columns=["nflverse_game_id", "play_id", "defense_man_zone_type"])
        frames.append(f)
    part = pd.concat(frames, ignore_index=True).rename(columns={"nflverse_game_id": "game_id"})
    part = part[part.defense_man_zone_type.isin(["MAN_COVERAGE", "ZONE_COVERAGE"])]
    base = pbp[["game_id", "play_id", "season", "week", "defteam"]]
    m = base.merge(part, on=["game_id", "play_id"], how="inner")
    g = m.groupby(["season", "week", "game_id", "defteam"], as_index=False).agg(
        man_n=("defense_man_zone_type", lambda s: (s == "MAN_COVERAGE").sum()),
        tag_n=("defense_man_zone_type", "size"))
    g["man_share"] = g.man_n / g.tag_n.clip(lower=1)
    g.loc[g.tag_n < 5, "man_share"] = np.nan
    return g.rename(columns={"defteam": "team"})[["season", "week", "game_id", "team", "man_share"]]

def build_pos_points_allowed():
    ff = pd.read_parquet(os.path.join(ROOT, "historical", "fantasy", "feature_frame.parquet"),
                         columns=["season", "week", "position", "played", "fantasy_points",
                                  "opponent_team", "game_id"])
    ff = ff[(ff.played == 1) & ff.position.isin(POSITIONS) & ff.opponent_team.notna()]
    g = ff.groupby(["season", "week", "game_id", "opponent_team", "position"], as_index=False)[
        "fantasy_points"].sum()
    wide = g.pivot_table(index=["season", "week", "game_id", "opponent_team"],
                         columns="position", values="fantasy_points", aggfunc="sum").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"opponent_team": "team", **{p: f"fp_allowed_{p}" for p in POSITIONS}})
    for p in POSITIONS:
        c = f"fp_allowed_{p}"
        if c in wide:
            wide[c] = wide[c].fillna(0.0)
    return wide

def _tercile_within_week(df, col, min_valid=20):
    pct = df.groupby(["season", "week"])[col].rank(pct=True)
    x = pd.Series(np.nan, index=df.index)
    x[pct >= 2 / 3] = 1.0
    x[pct <= 1 / 3] = 0.0
    nweek = df.groupby(["season", "week"])[col].transform(lambda s: s.notna().sum())
    x[nweek < min_valid] = np.nan
    return x

def load_game_meta():
    sch = pd.read_parquet(os.path.join(ROOT, "historical", "fantasy", "schedules.parquet"))
    sch = sch[(sch.game_type == "REG") & sch.season.isin(list(SEASONS))].copy()
    sanity = sch.dropna(subset=["spread_line", "home_moneyline"])
    corr = np.corrcoef(sanity.spread_line, -sanity.home_moneyline)[0, 1]
    assert corr > 0.5, f"spread_line sign convention unexpected (corr={corr:.2f})"
    rows = []
    for _, g in sch.iterrows():
        stad = str(g.stadium or "").lower()
        intl = any(k in stad for k in INTL_SITES) if g.location == "Neutral" or "azteca" in stad else False
        if intl:
            site = next(v for k, v in INTL_SITES.items() if k in stad)
        else:
            site = TEAM_SITE.get(g.home_team, (np.nan, np.nan, np.nan))
        altitude_site = (g.home_team in ALTITUDE_SITES and not intl) or ("azteca" in stad)
        hour = int(str(g.gametime)[:2]) if pd.notna(g.gametime) else np.nan
        for team, opp, home, rest, coach, _opp_coach in (
                (g.home_team, g.away_team, 1, g.home_rest, g.home_coach, g.away_coach),
                (g.away_team, g.home_team, 0, g.away_rest, g.away_coach, g.home_coach)):
            tlat, tlon, ttz = TEAM_SITE.get(team, (np.nan, np.nan, np.nan))
            travel = haversine_mi(tlat, tlon, site[0], site[1])
            spread = -g.spread_line if home == 1 else g.spread_line
            implied = (g.total_line / 2.0) - (spread / 2.0) if pd.notna(g.total_line) and pd.notna(spread) else np.nan
            rows.append(dict(
                season=g.season, week=g.week, game_id=g.game_id, team=team, opponent=opp,
                is_home=home, gameday=g.gameday, rest=rest, coach=coach,
                referee=g.referee, overtime=g.overtime, div_game=g.div_game,
                stadium=g.stadium, roof=g.roof, surface=g.surface,
                intl=int(intl), altitude=int(bool(altitude_site) and not (home == 1 and team in ALTITUDE_SITES)),
                travel_mi=travel, tz_diff=abs(ttz - site[2]) if pd.notna(ttz) and pd.notna(site[2]) else np.nan,
                kick_hour=hour, west_home=int(ttz == -8),
                team_spread=spread, total_line=g.total_line, implied_pts=implied,
            ))
    gm = pd.DataFrame(rows)
    gm = gm.sort_values(["team", "gameday", "week"])
    gm["after_ot"] = gm.groupby(["team", "season"])["overtime"].shift(1).fillna(0.0)
    prev_coach = gm.groupby("team", sort=False)["coach"].shift(1)
    prev_season = gm.groupby("team", sort=False)["season"].shift(1)
    first = gm.groupby(["team", "season"]).cumcount() == 0
    gm["new_hc"] = np.where(first & prev_season.notna() & (prev_season == gm.season - 1),
                            (gm.coach != prev_coach).astype(float), np.nan)
    gm["new_hc"] = gm.groupby(["team", "season"])["new_hc"].transform("max")
    pair = gm.apply(lambda r: "_".join(sorted([r.team, r.opponent])) + f"_{r.season}", axis=1)
    meet = gm.assign(pair=pair).sort_values("gameday").groupby("pair").cumcount()
    gm["div_rematch"] = np.where(gm.div_game == 1, (meet.reindex(gm.index) >= 2).astype(float), np.nan)
    gm["body_clock"] = ((gm.west_home == 1) & (gm.is_home == 0) & (gm.kick_hour == 13)).astype(float)
    return gm

def stage_frames():
    os.makedirs(EXP, exist_ok=True)
    pbp = _load_pbp()
    tw = build_team_week(pbp)
    dw = build_defense_week(pbp, tw)
    ftn = build_ftn_defense(pbp)
    mz = build_manzone_defense(pbp)
    fpa = build_pos_points_allowed()

    # --- defense trailing (allowed) features, strictly shift-then-roll ------
    dfe = dw.merge(ftn, on=["season", "week", "game_id", "team"], how="left")
    dfe = dfe.merge(mz, on=["season", "week", "game_id", "team"], how="left")
    dfe = dfe.merge(fpa, on=["season", "week", "game_id", "team"], how="left")
    dcols = ["plays_allowed", "pass_att_allowed", "proe_allowed", "epa_pass_allowed",
             "epa_rush_allowed", "explosive_allowed", "pressure_generated",
             "blitz_share", "heavy_box_share", "man_share"] + [f"fp_allowed_{p}" for p in POSITIONS]
    dfe = _ewm_trailing(dfe, ["team"], ["team", "season", "week"], dcols)

    # --- offense trailing -----------------------------------------------
    tw = tw.sort_values(["team", "season", "week"])
    g = tw.groupby("team", sort=False)
    for c in ("plays", "pass_att", "dropbacks", "proe"):
        tw[f"tr_{c}"] = g[c].transform(lambda s: s.shift(1).ewm(halflife=EWM_HALFLIFE, min_periods=EWM_MIN).mean())
    tw["f4n"] = tw.fourth_short_n.fillna(0.0)
    tw["f4go"] = tw.fourth_short_go.fillna(0.0)
    g = tw.groupby("team", sort=False)
    tw["tr_f4n"] = g["f4n"].transform(lambda s: s.shift(1).ewm(halflife=12, min_periods=8).mean())
    tw["tr_f4go"] = g["f4go"].transform(lambda s: s.shift(1).ewm(halflife=12, min_periods=8).mean())
    tw["tr_fourth_go_rate"] = np.where(tw.tr_f4n > 0.2, tw.tr_f4go / tw.tr_f4n, np.nan)

    gm = load_game_meta()
    # referee trailing pace (combined plays per game officiated)
    game_plays = tw.groupby(["season", "week", "game_id"], as_index=False)["plays"].sum() \
                   .rename(columns={"plays": "game_plays"})
    ref = gm[gm.is_home == 1][["season", "week", "game_id", "gameday", "referee"]] \
        .merge(game_plays, on=["season", "week", "game_id"], how="left")
    ref = ref.sort_values(["referee", "gameday"])
    ref["tr_ref_plays"] = ref.groupby("referee", sort=False)["game_plays"].transform(
        lambda s: s.shift(1).ewm(halflife=10, min_periods=5).mean())
    gm = gm.merge(ref[["game_id", "tr_ref_plays"]], on="game_id", how="left")

    # --- team-level analysis frame ---------------------------------------
    tm = tw.merge(gm, on=["season", "week", "game_id", "team"], how="left", suffixes=("", "_gm"))
    tm["d_plays"] = tm.plays - tm.tr_plays
    tm["d_pass_att"] = tm.pass_att - tm.tr_pass_att
    tm["x_ref_pace_high"] = _tercile_within_week(tm, "tr_ref_plays", min_valid=16)
    tm["x_fourth_aggr_high"] = _tercile_within_week(tm, "tr_fourth_go_rate", min_valid=16)
    tm["x_implied_high"] = _tercile_within_week(tm, "implied_pts", min_valid=16)
    tm["vol_q"] = pd.qcut(tm.tr_plays, 5, labels=False, duplicates="drop")
    tm["volp_q"] = pd.qcut(tm.tr_pass_att, 5, labels=False, duplicates="drop")
    tm.to_parquet(os.path.join(EXP, "team_frame.parquet"), index=False)

    # --- player frame ------------------------------------------------------
    cols = ["season", "week", "player_id", "full_name", "position", "played",
            "pre_played_games", "team", "opponent_team", "game_id", "is_home", "rest",
            "fantasy_points", "pre_fantasy_points_ewm4", "passing_yards",
            "pre_passing_yards_ewm4", "rushing_yards", "pre_rushing_yards_ewm4",
            "receiving_yards", "pre_receiving_yards_ewm4", "attempts", "pre_attempts_ewm4",
            "pre_team_pass_attempts_ewm4", "pre_position_role_rank",
            "team_spread", "total_line", "implied_team_points", "primetime",
            "roof", "surface", "line_missing"]
    ff = pd.read_parquet(os.path.join(ROOT, "historical", "fantasy", "feature_frame.parquet"), columns=cols)
    pf = ff[(ff.played == 1) & (ff.pre_played_games >= 3) & ff.position.isin(POSITIONS)].copy()
    pf["d_fp"] = pf.fantasy_points - pf.pre_fantasy_points_ewm4
    pf["d_pass_yards"] = pf.passing_yards - pf.pre_passing_yards_ewm4
    pf["d_rush_yards"] = pf.rushing_yards - pf.pre_rushing_yards_ewm4
    pf["d_rec_yards"] = pf.receiving_yards - pf.pre_receiving_yards_ewm4
    pf["role_bin"] = pf.pre_position_role_rank.clip(upper=3).fillna(3).astype(int)
    pf["base_q"] = pf.groupby("position")["pre_fantasy_points_ewm4"].transform(
        lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))

    # opponent trailing exposures (keyed on the defense = opponent_team)
    dkey = dfe[["season", "week", "team"] + [f"tr_{c}" for c in dcols]].rename(
        columns={"team": "opponent_team"})
    pf = pf.merge(dkey, on=["season", "week", "opponent_team"], how="left")
    tkey = tw[["season", "week", "team", "tr_plays"]].rename(
        columns={"team": "opponent_team", "tr_plays": "tr_opp_plays"})
    pf = pf.merge(tkey, on=["season", "week", "opponent_team"], how="left")

    # tercile exposures computed once per (season, week, opponent) then merged
    opp_tw = pf[["season", "week", "opponent_team"]].drop_duplicates().merge(
        dkey, on=["season", "week", "opponent_team"], how="left").merge(
        tkey, on=["season", "week", "opponent_team"], how="left")
    ter_map = {}
    ter_specs = {
        "x_opp_pass_epa_soft": ("tr_epa_pass_allowed", False),
        "x_opp_rush_epa_soft": ("tr_epa_rush_allowed", False),
        "x_opp_pressure_high": ("tr_pressure_generated", False),
        "x_opp_blitz_high": ("tr_blitz_share", False),
        "x_opp_heavy_box": ("tr_heavy_box_share", False),
        "x_opp_pace_fast": ("tr_opp_plays", False),
        "x_opp_proe_allowed_high": ("tr_proe_allowed", False),
        "x_opp_explosive_high": ("tr_explosive_allowed", False),
        "x_opp_man_high": ("tr_man_share", False),
        **{f"x_opp_fp_soft_{p}": (f"tr_fp_allowed_{p}", False) for p in POSITIONS},
    }
    for xcol, (src, _) in ter_specs.items():
        opp_tw[xcol] = _tercile_within_week(opp_tw, src, min_valid=16)
    pf = pf.merge(opp_tw[["season", "week", "opponent_team"] + list(ter_specs)],
                  on=["season", "week", "opponent_team"], how="left")

    # factor-frame pregame injury-cluster flags (game-level, per opponent)
    fac = pd.read_parquet(os.path.join(ROOT, "data", "factor_frame.parquet"),
                          columns=["season", "week", "team", "opponent_db_2_plus",
                                   "opponent_front_2_plus"]).drop_duplicates(
                                       ["season", "week", "team"])
    pf = pf.merge(fac.rename(columns={"opponent_db_2_plus": "x_opp_db2_out",
                                      "opponent_front_2_plus": "x_opp_front2_out"}),
                  on=["season", "week", "team"], how="left")
    for c in ("x_opp_db2_out", "x_opp_front2_out"):
        pf[c] = pf[c].astype(float)

    # game-meta exposures
    gmk = gm[["season", "week", "game_id", "team", "intl", "altitude", "travel_mi",
              "tz_diff", "body_clock", "after_ot", "new_hc", "div_rematch", "div_game"]]
    pf = pf.merge(gmk, on=["season", "week", "game_id", "team"], how="left")

    surf = pf.surface.fillna("").str.strip().str.lower()
    pf["x_turf"] = np.where(surf.str.contains("turf"), 1.0, np.where(surf == "grass", 0.0, np.nan))
    roof = pf.roof.fillna("")
    pf["x_closed_roof"] = np.where(roof.isin(["dome", "closed"]), 1.0,
                                   np.where(roof.isin(["outdoors", "open"]), 0.0, np.nan))
    pf["x_altitude"] = pf.altitude.astype(float)
    pf.loc[(pf.team == "DEN") & (pf.is_home == 1), "x_altitude"] = np.nan
    road = pf.is_home == 0
    ok_dom = (pf.intl == 0)
    pf["x_travel_1500"] = np.where(road & ok_dom & pf.travel_mi.notna(),
                                   (pf.travel_mi >= 1500).astype(float), np.nan)
    pf["x_tz2"] = np.where(road & ok_dom & pf.tz_diff.notna(), (pf.tz_diff >= 2).astype(float), np.nan)
    pf["x_body_clock"] = np.where(road, pf.body_clock, np.nan)
    pf["x_international"] = pf.intl.astype(float)
    wk1 = pf.week == 1
    rest = pf.rest
    pf["x_short_rest"] = np.where(wk1 | rest.isna(), np.nan,
                                  np.where(rest <= 5, 1.0, np.where(rest.between(6, 11), 0.0, np.nan)))
    pf["x_post_bye"] = np.where(wk1 | rest.isna(), np.nan,
                                np.where(rest >= 12, 1.0, np.where(rest.between(6, 11), 0.0, np.nan)))
    pf["x_after_ot"] = np.where(wk1, np.nan, pf.after_ot.astype(float))
    lined = pf.line_missing != 1
    pf["x_big_fav"] = np.where(lined & pf.team_spread.notna(), (pf.team_spread <= -7).astype(float), np.nan)
    pf["x_big_dog"] = np.where(lined & pf.team_spread.notna(), (pf.team_spread >= 7).astype(float), np.nan)
    pf["x_high_total"] = np.where(lined & pf.total_line.notna(), (pf.total_line >= 47.5).astype(float), np.nan)
    pf["x_low_total"] = np.where(lined & pf.total_line.notna(), (pf.total_line <= 41.5).astype(float), np.nan)
    imp = pf[["season", "week", "game_id", "team", "implied_team_points"]].drop_duplicates(
        ["season", "week", "team"]).copy()
    imp["x_implied_high"] = _tercile_within_week(imp, "implied_team_points", min_valid=16)
    pf = pf.merge(imp[["season", "week", "team", "x_implied_high"]], on=["season", "week", "team"], how="left")
    pf["x_primetime"] = pf.primetime.astype(float)
    pf["x_div_rematch"] = pf.div_rematch
    pf["x_new_hc"] = pf.new_hc

    pf.to_parquet(os.path.join(EXP, "player_frame.parquet"), index=False)
    dfe.to_parquet(os.path.join(EXP, "defense_features.parquet"), index=False)
    meta = dict(n_player=len(pf), n_team=len(tm),
                n_by_pos=pf.groupby("position").size().to_dict(),
                exposure_counts={c: dict(exposed=int((pf[c] == 1).sum()), control=int((pf[c] == 0).sum()))
                                 for c in pf.columns if c.startswith("x_")})
    with open(os.path.join(EXP, "frames_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print(json.dumps({k: meta[k] for k in ("n_player", "n_team")}, indent=1))
    print("frames done")

# ---------------------------------------------------------------------------
# Matched-effect estimator with (season, week)-block cluster bootstrap.
# ---------------------------------------------------------------------------
def matched_effect(df, xcol, ycol, strata_cols, prior_sd, n_boot=N_BOOT, seed=SEED):
    d = df[df[xcol].isin([0.0, 1.0]) & df[ycol].notna()].copy()
    # NaN strata keys would make ngroup() return -1 and np.add.at would silently
    # wrap to the last stratum -- drop them (they cannot be matched anyway).
    d = d.dropna(subset=list(strata_cols))
    n_missing = int(len(df) - len(d))
    if d.empty or d[xcol].nunique() < 2:
        return dict(n_exposed=int((d[xcol] == 1).sum()), n_control=int((d[xcol] == 0).sum()),
                    n_missing=n_missing, empty=True)
    d["_arm"] = d[xcol].astype(int)
    d["_stratum"] = d.groupby(strata_cols, sort=False).ngroup()
    d["_block"] = d.groupby(["season", "week"], sort=False).ngroup()
    nb, ns = d._block.nunique(), d._stratum.nunique()
    sums = np.zeros((nb, ns, 2))
    cnts = np.zeros((nb, ns, 2))
    np.add.at(sums, (d._block.values, d._stratum.values, d._arm.values), d[ycol].values)
    np.add.at(cnts, (d._block.values, d._stratum.values, d._arm.values), 1.0)

    def eff(mult):
        s = np.tensordot(mult, sums, axes=1)
        c = np.tensordot(mult, cnts, axes=1)
        ok = (c[:, 0] > 0) & (c[:, 1] > 0)
        if not ok.any():
            return np.nan
        w = c[ok, 1] / c[ok, 1].sum()
        return float(np.sum(w * (s[ok, 1] / c[ok, 1] - s[ok, 0] / c[ok, 0])))

    point = eff(np.ones(nb))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        mult = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)
        boots[i] = eff(mult)
    boots = boots[~np.isnan(boots)]
    se = float(boots.std(ddof=1)) if len(boots) > 10 else np.nan
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if len(boots) > 10 else (np.nan, np.nan)
    p_lo = (boots <= 0).mean() if len(boots) else np.nan
    p_hi = (boots >= 0).mean() if len(boots) else np.nan
    pval = float(min(1.0, 2 * min(p_lo, p_hi) + 1.0 / (len(boots) + 1))) if len(boots) else np.nan
    if se and np.isfinite(se) and se > 0:
        post_var = 1.0 / (1.0 / se ** 2 + 1.0 / prior_sd ** 2)
        post_mean = post_var * (point / se ** 2)
        post_lo, post_hi = post_mean - 1.96 * post_var ** 0.5, post_mean + 1.96 * post_var ** 0.5
    else:
        post_mean = post_lo = post_hi = np.nan
    # strata actually used at full sample
    c_full = cnts.sum(axis=0)
    used = (c_full[:, 0] > 0) & (c_full[:, 1] > 0)
    n_used = int(c_full[used].sum())
    exp_games = d[d._arm == 1].drop_duplicates(["season", "week", "team"]) if "team" in d.columns else d[d._arm == 1]
    per_season = {}
    for s_, ds in d.groupby("season"):
        b = ds.groupby(["season", "week"], sort=False).ngroup()
        st = ds.groupby(strata_cols, sort=False).ngroup()
        sm = np.zeros((b.nunique(), st.nunique(), 2)); cn = np.zeros_like(sm)
        np.add.at(sm, (b.values, st.values, ds._arm.values), ds[ycol].values)
        np.add.at(cn, (b.values, st.values, ds._arm.values), 1.0)
        s2, c2 = sm.sum(axis=0), cn.sum(axis=0)
        ok = (c2[:, 0] > 0) & (c2[:, 1] > 0)
        if ok.any():
            w = c2[ok, 1] / c2[ok, 1].sum()
            per_season[int(s_)] = dict(
                effect=float(np.sum(w * (s2[ok, 1] / c2[ok, 1] - s2[ok, 0] / c2[ok, 0]))),
                n_exposed=int(c2[ok, 1].sum()), n_control=int(c2[ok, 0].sum()))
        else:
            per_season[int(s_)] = dict(effect=np.nan, n_exposed=int(c2[:, 1].sum()),
                                       n_control=int(c2[:, 0].sum()))
    return dict(
        effect=point, se=se, ci_lo=lo, ci_hi=hi, p=pval,
        posterior_mean=post_mean, posterior_lo=post_lo, posterior_hi=post_hi,
        n_exposed=int((d._arm == 1).sum()), n_control=int((d._arm == 0).sum()),
        n_effective=int(len(exp_games)), n_in_matched_strata=n_used, n_missing=n_missing,
        n_strata_used=int(used.sum()), per_season=per_season, empty=False)

def _train_test_split_effects(res_full, df, xcol, ycol, strata_cols, prior_sd):
    """Season-forward: point estimate on <=2022, per-year effects 2023-2025."""
    out = {}
    tr = df[df.season <= 2022]
    r = matched_effect(tr, xcol, ycol, strata_cols, prior_sd, n_boot=400, seed=SEED + 1)
    out["train_le2022"] = {k: r.get(k) for k in ("effect", "n_exposed", "n_control")}
    for y in (2023, 2024, 2025):
        ry = res_full["per_season"].get(y, {})
        out[f"y{y}"] = ry
    tr_eff = out["train_le2022"]["effect"]
    signs = [np.sign(out[f"y{y}"].get("effect") or np.nan) for y in (2023, 2024, 2025)]
    signs = [s for s in signs if np.isfinite(s)]
    if tr_eff is not None and np.isfinite(tr_eff) and signs:
        agree = sum(1 for s in signs if s == np.sign(tr_eff))
        out["sign_agreement_2023_25"] = f"{agree}/{len(signs)}"
        post = [out[f"y{y}"].get("effect") for y in (2023, 2024, 2025)]
        post = [p for p in post if p is not None and np.isfinite(p)]
        mag_ok = bool(post) and abs(tr_eff) > 0 and 0.25 <= abs(np.mean(post)) / max(abs(tr_eff), 1e-9) <= 4.0
        out["magnitude_order_holds"] = bool(mag_ok)
    else:
        out["sign_agreement_2023_25"] = "n/a"
        out["magnitude_order_holds"] = None
    return out

def stage_battery(family):
    pf = pd.read_parquet(os.path.join(EXP, "player_frame.parquet"))
    tm = pd.read_parquet(os.path.join(EXP, "team_frame.parquet"))
    results = {}
    for t in [t for t in BATTERY if t["family"] == family]:
        if t["level"] == "player":
            d = pf[pf.position.isin(t["positions"])]
            if t["id"].startswith("C07"):
                d = d[d.div_game == 1]
            strata = ["position", "role_bin", "base_q"]
        else:
            d = tm.copy()
            strata = ["volp_q" if t["outcome"] == "d_pass_att" else "vol_q"]
        res = matched_effect(d, t["exposure"], t["outcome"], strata, t["prior_sd"])
        if not res.get("empty"):
            res["season_forward"] = _train_test_split_effects(res, d, t["exposure"], t["outcome"],
                                                              strata, t["prior_sd"])
        res.update(id=t["id"], family=family, definition=t["definition"], outcome=t["outcome"],
                   exposure=t["exposure"], years=t["years"], prior_sd=t["prior_sd"],
                   positions=list(t["positions"]))
        results[t["id"]] = res
        print(f"{t['id']}: eff={res.get('effect', float('nan')):+.3f} "
              f"p={res.get('p', float('nan')):.4f} nE={res.get('n_exposed')} nC={res.get('n_control')}")
    with open(os.path.join(EXP, f"results_{family}.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)
    print(f"family {family}: {len(results)} tests done")

FP_PER_UNIT = {"d_fp": 1.0, "d_pass_yards": 0.04, "d_rush_yards": 0.1, "d_rec_yards": 0.1,
               "d_plays": np.nan, "d_pass_att": np.nan}
ADJ_CAP = 1.0  # fantasy points

def _bh(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    q = np.empty(m)
    prev = 1.0
    for rank_idx in range(m - 1, -1, -1):
        i = order[rank_idx]
        val = pvals[i] * m / (rank_idx + 1)
        prev = min(prev, val)
        q[i] = prev
    return q

def stage_fdr():
    results = {}
    for fam in "ABCD":
        with open(os.path.join(EXP, f"results_{fam}.json")) as f:
            results.update(json.load(f))
    assert len(results) == len(BATTERY) == 40, f"expected 40 results, got {len(results)}"
    ids = [t["id"] for t in BATTERY]
    pvals = np.array([results[i].get("p") if np.isfinite(results[i].get("p") or np.nan) else 1.0
                      for i in ids])
    qvals = _bh(pvals)
    for i, tid in enumerate(ids):
        r = results[tid]
        r["q"] = float(qvals[i])
        sf = r.get("season_forward", {})
        agree = sf.get("sign_agreement_2023_25", "n/a")
        agree_ok = agree not in ("n/a",) and int(agree.split("/")[0]) * 3 >= int(agree.split("/")[1]) * 2
        r["forward_holds"] = bool(agree_ok and sf.get("magnitude_order_holds"))
        if r.get("empty"):
            r["battery_status"] = "research_only_no_cohort"
        elif r["q"] < 0.05 and r["forward_holds"]:
            r["battery_status"] = "survivor_candidate"
        elif r["q"] < 0.05:
            r["battery_status"] = "research_only_unstable_forward"
        else:
            r["battery_status"] = "rejected_null"
    with open(os.path.join(EXP, "battery_results.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)
    surv = [i for i in ids if results[i]["battery_status"] == "survivor_candidate"]
    print("survivor candidates:", surv)
    print("q<0.05 unstable:", [i for i in ids if results[i]["battery_status"] == "research_only_unstable_forward"])

def stage_gates():
    with open(os.path.join(EXP, "battery_results.json")) as f:
        results = json.load(f)
    pf = pd.read_parquet(os.path.join(EXP, "player_frame.parquet"))
    op = pd.read_parquet(os.path.join(ROOT, "reports", "fantasy_outer_predictions.parquet"))
    gates = {}
    for tid, r in results.items():
        if r["battery_status"] != "survivor_candidate":
            continue
        if tid.split("_")[0] in ("C08", "C09", "D01") or r["outcome"] in ("d_plays", "d_pass_att"):
            gates[tid] = dict(gate="n/a_team_volume_node",
                              note="team-volume pathway; see PROE prototype for the volume-node ablation")
            continue
        tr_eff = (r.get("season_forward", {}).get("train_le2022", {}) or {}).get("effect")
        if tr_eff is None or not np.isfinite(tr_eff):
            gates[tid] = dict(gate="no_train_estimate")
            continue
        mult = FP_PER_UNIT.get(r["outcome"], 1.0)
        adj = float(np.clip(tr_eff * mult, -ADJ_CAP, ADJ_CAP))
        expo = pf[["season", "week", "player_id", r["exposure"]]].rename(columns={r["exposure"]: "x"})
        m = op.merge(expo, on=["season", "week", "player_id"], how="left")
        if r["positions"]:
            m["x"] = np.where(m.position.isin(r["positions"]), m.x, np.nan)
        m["proj_adj"] = m.projection_mean + np.where(m.x == 1.0, adj, 0.0)
        m["ae_base"] = (m.fantasy_points - m.projection_mean).abs()
        m["ae_adj"] = (m.fantasy_points - m.proj_adj).abs()
        blocks = m.groupby(["season", "week"])
        base = blocks.ae_base.mean().values
        adj_ = blocks.ae_adj.mean().values
        deltas = adj_ - base
        rng = np.random.default_rng(SEED)
        nb = len(deltas)
        boot = np.array([deltas[rng.integers(0, nb, nb)].mean() for _ in range(4000)])
        mean_d = float((m.ae_adj.mean() - m.ae_base.mean()))
        lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        n_touched = int((m.x == 1.0).sum())
        resid = m.loc[m.x == 1.0, "fantasy_points"] - m.loc[m.x == 1.0, "projection_mean"]
        if mean_d <= -0.05 and hi < 0:
            verdict = "shadow"
        elif lo < 0 < hi or (mean_d <= 0 and hi >= 0):
            verdict = "research_only"
        else:
            verdict = "rejected"
        gates[tid] = dict(gate=verdict, adjustment_fp=adj, mae_delta_mean=mean_d,
                          mae_delta_ci=[lo, hi], n_rows_touched=n_touched,
                          exposed_model_residual_mean=float(resid.mean()) if len(resid) else np.nan,
                          note="delta over frozen outer MAE 5.105968; paired week-block bootstrap B=4000")
        print(tid, gates[tid])
    with open(os.path.join(EXP, "gates.json"), "w") as f:
        json.dump(gates, f, indent=1, default=float)
    print("gates done:", len(gates))

def _market_inventory():
    inv = {}
    for name, path in [("historical_lines(base)", os.path.join(ROOT, "historical_lines.parquet")),
                       ("lines_extra", os.path.join(ROOT, "historical", "lines_extra.parquet")),
                       ("fantasy/schedules", os.path.join(ROOT, "historical", "fantasy", "schedules.parquet"))]:
        df = pd.read_parquet(path)
        cols = set(df.columns)
        inv[name] = dict(
            rows=len(df), seasons=f"{int(df.season.min())}-{int(df.season.max())}",
            has_open_close=bool({"opening_spread", "open_spread", "spread_open"} & cols),
            has_multibook=bool({"book", "sportsbook", "provider"} & cols),
            line_cols=sorted(c for c in cols if any(k in c for k in
                             ("spread", "total", "moneyline", "odds"))),
            spread_nonnull=float(df.spread_line.notna().mean()) if "spread_line" in cols else None,
            total_nonnull=float(df.total_line.notna().mean()) if "total_line" in cols else None)
    return inv

def _research_only_weather(pf):
    ff = pd.read_parquet(os.path.join(ROOT, "historical", "fantasy", "feature_frame.parquet"),
                         columns=["season", "week", "player_id", "temp", "wind"])
    d = pf.merge(ff, on=["season", "week", "player_id"], how="left")
    outdoor = d[d.x_closed_roof == 0].copy()
    outdoor["x_wind15"] = np.where(outdoor.wind.notna(), (outdoor.wind >= 15).astype(float), np.nan)
    outdoor["x_cold32"] = np.where(outdoor.temp.notna(), (outdoor.temp <= 32).astype(float), np.nan)
    out = {}
    r = matched_effect(outdoor[outdoor.position == "QB"], "x_wind15", "d_fp",
                       ["role_bin", "base_q"], 1.0, n_boot=800)
    out["wind15_qb_dfp"] = {k: r.get(k) for k in ("effect", "ci_lo", "ci_hi", "n_exposed", "n_control")}
    r = matched_effect(outdoor, "x_cold32", "d_fp", ["position", "role_bin", "base_q"], 1.0, n_boot=800)
    out["cold32_pooled_dfp"] = {k: r.get(k) for k in ("effect", "ci_lo", "ci_hi", "n_exposed", "n_control")}
    return out

# ---------------------------------------------------------------------------
# Stage: extras -- market inventory + research-only observed-weather probes.
# Explicitly OUTSIDE the 40-test battery (declared in the frozen header).
# ---------------------------------------------------------------------------
def stage_extras():
    pf = pd.read_parquet(os.path.join(EXP, "player_frame.parquet"))
    out = dict(market_inventory=_market_inventory(),
               weather_research_only=_research_only_weather(pf))
    with open(os.path.join(EXP, "extras.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:2500])
    print("extras done")

# ---------------------------------------------------------------------------
# Stage: registry -- data/factor_registry_families.json
# ---------------------------------------------------------------------------
FAMILY_META = {
    "A": dict(source="pbp caches 2019-2025 + FTN charting 2022-25 + pbp_participation man/zone + fantasy feature_frame + data/factor_frame.parquet injury flags",
              as_of_clock="opponent trailing stats strictly shift-then-roll (ewm hl=5, min 3 prior games, cross-season carry); DB/front-7 flags from official pregame Out/Doubtful designations",
              mechanism="opponent quality/scheme shifts player opportunity and efficiency"),
    "B": dict(source="nflverse schedules (roof/surface/site/rest/stadium) + static team geodesy table",
              as_of_clock="fully schedule-derivable before kickoff",
              mechanism="surface/roof/altitude/travel/recovery affects execution"),
    "C": dict(source="nflverse schedules (closing lines, kickoff, coaches, referee) + pbp trailing rates",
              as_of_clock="closing line stamped pregame (leakage-adjacent for LIVE use: acceptable as projection feature, flag before intra-week deployment); schedule flags; trailing rates shift-then-roll",
              mechanism="expected game state drives team volume and script"),
    "D": dict(source="historical_lines.parquet + historical/lines_extra.parquet + fantasy/schedules.parquet (single snapshot, no book id)",
              as_of_clock="closing line stamped pregame (see C)",
              mechanism="market prices encode volume/scoring expectations"),
}

def _status_for(r, gate):
    if r.get("empty"):
        return "research_only", "empty_cohort"
    bs = r.get("battery_status")
    if bs == "survivor_candidate":
        v = (gate or {}).get("gate")
        if v == "shadow":
            return "shadow", "passed battery q<0.05 + forward + adjustment gate"
        if v == "research_only":
            return "research_only", "battery survivor; adjustment-gate CI straddles 0"
        if v == "rejected":
            return "rejected", "battery survivor but adjustment gate failed"
        return "research_only", "battery survivor; team-volume node (no player adjustment gate; see PROE prototype)"
    if bs == "research_only_unstable_forward":
        return "research_only", "q<0.05 but 2023-25 sign/order unstable"
    return "rejected", "null at BH-FDR q>=0.05"

def stage_registry():
    with open(os.path.join(EXP, "battery_results.json")) as f:
        results = json.load(f)
    gates = {}
    gp = os.path.join(EXP, "gates.json")
    if os.path.exists(gp):
        with open(gp) as f:
            gates = json.load(f)
    extras = {}
    ep = os.path.join(EXP, "extras.json")
    if os.path.exists(ep):
        with open(ep) as f:
            extras = json.load(f)
    proe = None
    pp = os.path.join(EXP, "proe_results.json")
    if os.path.exists(pp):
        with open(pp) as f:
            proe = json.load(f)
    recs = []
    for t in BATTERY:
        r = results[t["id"]]
        g = gates.get(t["id"])
        sf = r.get("season_forward", {}) or {}
        status, reason = _status_for(r, g)
        fm = FAMILY_META[t["family"]]
        recs.append(dict(
            id=f"fam_{t['id']}", family=t["family"], definition=t["definition"],
            source=fm["source"], years=t["years"], as_of_clock=fm["as_of_clock"],
            positions=list(t["positions"]) or ["team"], component_node=t["outcome"],
            mechanism=fm["mechanism"],
            missingness=f"{r.get('n_missing', 0)} frame rows outside exposure/outcome/strata",
            n_raw=dict(exposed=r.get("n_exposed", 0), control=r.get("n_control", 0)),
            n_effective=r.get("n_effective"),
            cohort="played player-weeks 2019-2025 REG, >=3 prior played games" if t["level"] == "player"
                   else "team-games 2019-2025 REG with trailing volume",
            control="matched strata position x role bin x trailing-baseline quintile, (season,week) cluster bootstrap"
                    if t["level"] == "player" else "matched trailing-volume quintile strata, (season,week) cluster bootstrap",
            prior=f"normal(0, {t['prior_sd']}) on the {t['outcome']} effect",
            effect=dict(point=r.get("effect"), ci95=[r.get("ci_lo"), r.get("ci_hi")], p=r.get("p")),
            posterior=dict(mean=r.get("posterior_mean"), ci95=[r.get("posterior_lo"), r.get("posterior_hi")]),
            multiplicity_q=r.get("q"),
            season_forward=dict(train_le2022=sf.get("train_le2022"),
                                sign_agreement_2023_25=sf.get("sign_agreement_2023_25"),
                                magnitude_order_holds=sf.get("magnitude_order_holds")),
            ablation=g, calibration_effect=(g or {}).get("mae_delta_mean"),
            status=status, status_reason=reason))
    wx = (extras.get("weather_research_only") or {})
    inv = extras.get("market_inventory") or {}
    single_book = not any(v.get("has_multibook") for v in inv.values())
    no_openclose = not any(v.get("has_open_close") for v in inv.values())
    recs += [
        dict(id="fam_D90_line_movement_openclose", family="D",
             definition="Open->close spread/total movement as a factor",
             source="ABSENT locally", years="n/a",
             as_of_clock="would be pregame if an open+close archive existed",
             positions=["team"], component_node="d_pass_att",
             mechanism="steam/News captured by line movement",
             missingness=("no open/close columns in any local lines source (verified: "
                          + ", ".join(sorted(inv)) + "); single snapshot per game" if no_openclose else "unexpected"),
             n_raw=dict(exposed=0, control=0), n_effective=0, cohort="n/a", control="n/a",
             prior="n/a", effect=None, posterior=None, multiplicity_q=None,
             season_forward=None, ablation=None, calibration_effect=None,
             status="proposed", status_reason="acquisition needed: historical open/close odds archive"),
        dict(id="fam_D91_crossbook_disagreement", family="D",
             definition="Cross-book spread/total disagreement as uncertainty factor",
             source="ABSENT locally", years="n/a", as_of_clock="pregame if multi-book archive existed",
             positions=["team"], component_node="d_fp",
             mechanism="book disagreement flags uncertain game states",
             missingness="no book/sportsbook column anywhere; single-book snapshot" if single_book else "unexpected",
             n_raw=dict(exposed=0, control=0), n_effective=0, cohort="n/a", control="n/a",
             prior="n/a", effect=None, posterior=None, multiplicity_q=None,
             season_forward=None, ablation=None, calibration_effect=None,
             status="proposed", status_reason="acquisition needed: multi-book odds archive"),
        dict(id="fam_D92_prop_implied_player_volume", family="D",
             definition="Player-prop implied volume (attempts/receptions lines) as workload prior",
             source="ABSENT locally", years="n/a", as_of_clock="pregame props close",
             positions=["QB", "RB", "WR", "TE"], component_node="opportunities",
             mechanism="prop market aggregates sharp workload info",
             missingness="no player-prop lines in either repo snapshot (price/edge/CLV stay out of shared snapshot by policy; volume-implied lines would be admissible)",
             n_raw=dict(exposed=0, control=0), n_effective=0, cohort="n/a", control="n/a",
             prior="n/a", effect=None, posterior=None, multiplicity_q=None,
             season_forward=None, ablation=None, calibration_effect=None,
             status="proposed", status_reason="acquisition needed: historical player-prop archive (volume markets only)"),
        dict(id="fam_C90_ref_dpi_pass_volume", family="C",
             definition="Referee-crew trailing DPI rate -> team pass volume",
             source="officials.parquet ready; pbp cache LACKS penalty columns", years="n/a",
             as_of_clock="crew announced pregame; trailing DPI shift-then-roll",
             positions=["team"], component_node="d_pass_att",
             mechanism="DPI-friendly crews extend drives / encourage passing",
             missingness="40-col pbp subset has no penalty/penalty_type; needs penalty-level pbp re-pull",
             n_raw=dict(exposed=0, control=0), n_effective=0, cohort="n/a", control="n/a",
             prior="n/a", effect=None, posterior=None, multiplicity_q=None,
             season_forward=None, ablation=None, calibration_effect=None,
             status="proposed", status_reason="data acquisition: full pbp penalty columns"),
        dict(id="fam_C91_playoffs_cohort", family="C",
             definition="Playoff-game context effects",
             source="frames are REG-only by method contract", years="n/a",
             as_of_clock="schedule-derivable", positions=["QB", "RB", "WR", "TE"],
             component_node="d_fp", mechanism="win-or-go-home usage shifts",
             missingness="analysis population excludes POST by construction",
             n_raw=dict(exposed=0, control=0), n_effective=0, cohort="n/a", control="n/a",
             prior="n/a", effect=None, posterior=None, multiplicity_q=None,
             season_forward=None, ablation=None, calibration_effect=None,
             status="proposed", status_reason="separate POST-season study needed"),
        dict(id="fam_B90_wind15_observed", family="B",
             definition="Observed game wind >=15mph, outdoor QB d_fp",
             source="feature_frame temp/wind (stadium observation)", years="2019-2025",
             as_of_clock="LEAKAGE_PROXY: observed during game, not a pregame forecast",
             positions=["QB"], component_node="d_fp",
             mechanism="wind suppresses passing",
             missingness="outdoor games with wind recorded only",
             n_raw=dict(exposed=(wx.get("wind15_qb_dfp") or {}).get("n_exposed", 0),
                        control=(wx.get("wind15_qb_dfp") or {}).get("n_control", 0)),
             n_effective=None, cohort="outdoor QB player-weeks", control="matched role/baseline strata",
             prior="normal(0,1)", effect=wx.get("wind15_qb_dfp"), posterior=None,
             multiplicity_q=None, season_forward=None, ablation=None, calibration_effect=None,
             status="research_only", status_reason="leakage_proxy (observed weather); pregame forecast feed required for production"),
        dict(id="fam_B91_cold32_observed", family="B",
             definition="Observed game temp <=32F, pooled d_fp",
             source="feature_frame temp/wind (stadium observation)", years="2019-2025",
             as_of_clock="LEAKAGE_PROXY: observed during game, not a pregame forecast",
             positions=["QB", "RB", "WR", "TE"], component_node="d_fp",
             mechanism="cold suppresses offense",
             missingness="outdoor games with temp recorded only",
             n_raw=dict(exposed=(wx.get("cold32_pooled_dfp") or {}).get("n_exposed", 0),
                        control=(wx.get("cold32_pooled_dfp") or {}).get("n_control", 0)),
             n_effective=None, cohort="outdoor player-weeks", control="matched position/role/baseline strata",
             prior="normal(0,1)", effect=wx.get("cold32_pooled_dfp"), posterior=None,
             multiplicity_q=None, season_forward=None, ablation=None, calibration_effect=None,
             status="research_only", status_reason="leakage_proxy (observed weather)"),
        dict(id="fam_A90_matchup_interactions", family="A",
             definition="Matchup interactions (e.g., deep-ball QB x explosive-allowed D; man-beater WR x man-heavy D)",
             source="deferred", years="n/a", as_of_clock="pregame (both sides trailing)",
             positions=["QB", "RB", "WR", "TE"], component_node="d_fp",
             mechanism="style-on-style interaction", missingness="not fit in this wave by predeclaration",
             n_raw=dict(exposed=0, control=0), n_effective=0, cohort="n/a", control="n/a",
             prior="n/a", effect=None, posterior=None, multiplicity_q=None,
             season_forward=None, ablation=None, calibration_effect=None,
             status="proposed", status_reason="logged as hypotheses for the interaction wave; single-factor mains estimated here"),
        dict(id="fam_A91_manzone_matchup_interaction", family="A",
             definition="Receiver-type x man/zone-share interaction",
             source="pbp_participation coverage tags", years="2019-2025",
             as_of_clock="pregame (trailing man share)", positions=["WR", "TE"],
             component_node="d_fp", mechanism="separator-vs-man folklore",
             missingness="tag coverage 38-50% of plays by season",
             n_raw=dict(exposed=0, control=0), n_effective=0, cohort="n/a", control="n/a",
             prior="n/a", effect=None, posterior=None, multiplicity_q=None,
             season_forward=None, ablation=None, calibration_effect=None,
             status="proposed", status_reason="interaction wave; industry evidence negative (methodology scan) -- low prior"),
    ]
    if proe is not None:
        recs.append(dict(
            id="fam_E01_proe_playcall_v1", family="E",
            definition="Strictly-prior team PROE + opponent PROE-allowed + spread/total -> predicted team dropbacks/plays/pass attempts (volume node)",
            source="pbp xpass/pass_oe + schedules lines; analysis/proe_playcall_proto.py",
            years="2019-2025", as_of_clock="all features shift-then-roll or pregame closing line",
            positions=["team", "QB"], component_node="team_pass_attempts -> QB attempts/pass yards",
            mechanism="pass-rate-over-expected persists and prices team volume better than raw trailing attempts",
            missingness=proe.get("coverage_note", ""),
            n_raw=dict(exposed=proe.get("n_team_games_test", 0), control=0),
            n_effective=proe.get("n_team_games_test", 0),
            cohort="team-games 2023-2025 season-forward; QB translation on outer QB rows",
            control="baselines: trailing ewm4 attempts; current sim volume node clip(ewm4*(1+spread/75),15,55)",
            prior="n/a (MAE evaluation, not NHST)", effect=None, posterior=None,
            multiplicity_q=None, season_forward=proe.get("team_eval"),
            ablation=proe.get("qb_translation"), calibration_effect=None,
            status=proe.get("registry_status", "research_only"),
            status_reason=proe.get("verdict", "")))
    outp = os.path.join(ROOT, "data", "factor_registry_families.json")
    with open(outp, "w") as f:
        json.dump(recs, f, indent=1, default=float)
    print(f"registry written: {outp} ({len(recs)} records)")

# ---------------------------------------------------------------------------
# Stage: report -- append results below the frozen header of
# reports/factor_families_audit.md (idempotent: truncate after the marker).
# ---------------------------------------------------------------------------
MARKER = "Results below were computed only after this header was frozen."

def _f(x, nd=2):
    try:
        if x is None or not np.isfinite(float(x)):
            return "--"
        return f"{float(x):+.{nd}f}"
    except (TypeError, ValueError):
        return "--"

def _sig3(r):
    sf = r.get("season_forward", {}) or {}
    ps = r.get("per_season", {}) or {}
    marks = []
    for y in (2023, 2024, 2025):
        e = (ps.get(str(y)) or ps.get(y) or {}).get("effect")
        try:
            marks.append("+" if float(e) > 0 else "-" if float(e) < 0 else "0")
        except (TypeError, ValueError):
            marks.append(".")
    return "".join(marks), (sf.get("train_le2022") or {}).get("effect")

def stage_report():
    with open(os.path.join(EXP, "battery_results.json")) as f:
        results = json.load(f)
    gates = {}
    if os.path.exists(os.path.join(EXP, "gates.json")):
        with open(os.path.join(EXP, "gates.json")) as f:
            gates = json.load(f)
    extras = {}
    if os.path.exists(os.path.join(EXP, "extras.json")):
        with open(os.path.join(EXP, "extras.json")) as f:
            extras = json.load(f)
    proe = None
    if os.path.exists(os.path.join(EXP, "proe_results.json")):
        with open(os.path.join(EXP, "proe_results.json")) as f:
            proe = json.load(f)
    with open(os.path.join(EXP, "frames_meta.json")) as f:
        fmeta = json.load(f)

    rp = os.path.join(ROOT, "reports", "factor_families_audit.md")
    head = open(rp).read()
    cut = head.find(MARKER)
    assert cut > 0, "frozen header marker missing"
    head = head[:cut + len(MARKER)] + "\n"

    L = [""]
    L.append("## Battery results (all 40 tests, BH-FDR across the battery)")
    L.append("")
    L.append(f"Population as built: {fmeta['n_player']:,} player-weeks "
             f"(QB {fmeta['n_by_pos'].get('QB', 0):,} / RB {fmeta['n_by_pos'].get('RB', 0):,} / "
             f"WR {fmeta['n_by_pos'].get('WR', 0):,} / TE {fmeta['n_by_pos'].get('TE', 0):,}); "
             f"{fmeta['n_team']:,} team-games. Effect units: fantasy points for d_fp, yards for "
             "yardage outcomes, plays/attempts for team outcomes; all vs trailing baseline, "
             "exposed minus matched control.")
    L.append("")
    L.append("| id | pos | outcome | effect [95% CI] | post. mean [95% CI] | p | q | nE/nC (eff) | tr<=22 | 23/24/25 | status |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    order = {t["id"]: i for i, t in enumerate(BATTERY)}
    for tid in sorted(results, key=lambda k: order.get(k, 99)):
        r = results[tid]
        if r.get("empty"):
            L.append(f"| {tid} | {'/'.join(r.get('positions', []) or ['team'])} | {r.get('outcome','')} "
                     f"| EMPTY COHORT | -- | -- | -- | {r.get('n_exposed',0)}/{r.get('n_control',0)} (0) "
                     f"| -- | -- | research_only |")
            continue
        signs, tr = _sig3(r)
        pos = "/".join(r.get("positions") or []) or "team"
        st = r.get("battery_status", "")
        gate = gates.get(tid, {})
        if st == "survivor_candidate" and gate.get("gate"):
            st = f"survivor->{gate['gate']}"
        L.append(
            f"| {tid} | {pos} | {r['outcome']} | {_f(r['effect'])} [{_f(r['ci_lo'])},{_f(r['ci_hi'])}] "
            f"| {_f(r['posterior_mean'])} [{_f(r['posterior_lo'])},{_f(r['posterior_hi'])}] "
            f"| {r['p']:.4f} | {r['q']:.3f} | {r['n_exposed']:,}/{r['n_control']:,} ({r.get('n_effective', 0):,}) "
            f"| {_f(tr)} | {signs} | {st} |")
    L.append("")

    surv = [tid for tid, r in results.items() if r.get("battery_status") == "survivor_candidate"]
    unst = [tid for tid, r in results.items() if r.get("battery_status") == "research_only_unstable_forward"]
    L.append("## Survivors and adjustment gates")
    L.append("")
    if not surv:
        L.append("No test passed q<0.05 with a stable 2023-25 season-forward check.")
    for tid in sorted(surv, key=lambda k: order.get(k, 99)):
        r = results[tid]
        sf = r.get("season_forward", {}) or {}
        g = gates.get(tid, {})
        L.append(f"- **{tid}** ({r['definition']}): effect {_f(r['effect'])} "
                 f"[{_f(r['ci_lo'])},{_f(r['ci_hi'])}], q={r['q']:.3f}, "
                 f"nE={r['n_exposed']:,} (eff {r.get('n_effective', 0):,}), "
                 f"sign agreement {sf.get('sign_agreement_2023_25')}, "
                 f"train<=2022 {_f((sf.get('train_le2022') or {}).get('effect'))}.")
        if g.get("gate") in ("shadow", "research_only", "rejected"):
            ci = g.get("mae_delta_ci", [None, None])
            L.append(f"  - Gate on frozen outer predictions (adj {_f(g.get('adjustment_fp'))} fp capped, "
                     f"n_touched={g.get('n_rows_touched'):,}): MAE delta {g.get('mae_delta_mean'):+.4f} "
                     f"[{_f(ci[0], 4)},{_f(ci[1], 4)}] -> **{g['gate']}** "
                     f"(exposed-row model residual {_f(g.get('exposed_model_residual_mean'))}).")
        elif g:
            L.append(f"  - Gate: {g.get('gate')} -- {g.get('note', '')}")
    if unst:
        L.append("")
        L.append("q<0.05 but season-forward unstable (research_only): " + ", ".join(sorted(unst)))
    L.append("")

    L.append("## Folklore graveyard (nulls at q>=0.05, with n)")
    L.append("")
    for tid in sorted(results, key=lambda k: order.get(k, 99)):
        r = results[tid]
        if r.get("battery_status") == "rejected_null":
            L.append(f"- {tid}: {r['definition']} -- effect {_f(r.get('effect'))} "
                     f"[{_f(r.get('ci_lo'))},{_f(r.get('ci_hi'))}], q={r.get('q', 1):.2f}, "
                     f"nE={r.get('n_exposed', 0):,}.")
    L.append("")

    inv = extras.get("market_inventory") or {}
    L.append("## Market data inventory (verified, family D ground truth)")
    L.append("")
    for name, v in inv.items():
        L.append(f"- **{name}**: {v['rows']:,} rows, seasons {v['seasons']}; "
                 f"open/close: {'YES' if v['has_open_close'] else 'NO'}; multi-book: "
                 f"{'YES' if v['has_multibook'] else 'NO'}; spread nonnull "
                 f"{(v['spread_nonnull'] or 0) * 100:.1f}%, total nonnull {(v['total_nonnull'] or 0) * 100:.1f}%; "
                 f"line cols: {', '.join(v['line_cols'])}.")
    L.append("- Consequence: line-movement and cross-book-disagreement factors are NOT fittable "
             "from local data (single closing snapshot, one implicit book). Registered as "
             "`proposed` with acquisition notes. Closing spread/total/implied ARE stamped "
             "pregame and usable as projection features (leakage-adjacent for live use).")
    L.append("")

    wx = extras.get("weather_research_only") or {}
    L.append("## Observed-weather probes (research_only, leakage_proxy -- NOT in battery)")
    L.append("")
    for k, v in wx.items():
        if v:
            L.append(f"- {k}: effect {_f(v.get('effect'))} [{_f(v.get('ci_lo'))},{_f(v.get('ci_hi'))}], "
                     f"nE={v.get('n_exposed', 0):,}/nC={v.get('n_control', 0):,}. Observed in-game weather, "
                     "not a pregame forecast: research_only(leakage_proxy).")
    L.append("")

    L.append("## PROE / playcall prototype (E-family, MAE evaluation -- NOT in battery)")
    L.append("")
    if proe is None:
        L.append("PENDING: run analysis/proe_playcall_proto.py.")
    else:
        L.append(f"- Column coverage: {proe.get('coverage_note', '')}")
        te = proe.get("team_eval", {})
        for node, models in te.items():
            for mname, mv in models.items():
                per = mv.get("per_season", {})
                per_s = ", ".join(f"{y}: {per[y]['mae']:.3f}" for y in sorted(per))
                L.append(f"- team {node} / {mname}: MAE {mv['mae']:.4f} (bias {mv['bias']:+.3f}) [{per_s}]")
        qt = proe.get("qb_translation", {})
        if qt:
            a, y = qt.get("attempts", {}), qt.get("pass_yards", {})
            L.append(f"- QB translation (n={qt.get('n'):,} outer QB rows; fallback rows {qt.get('fallback_rows', 0)}):")
            L.append(f"  - attempts: before bias {a['before']['bias']:+.3f} MAE {a['before']['mae']:.3f} -> "
                     f"after bias {a['after']['bias']:+.3f} MAE {a['after']['mae']:.3f} "
                     f"(production sim reference bias +1.63)")
            L.append(f"  - pass yards: before bias {y['before']['bias']:+.3f} MAE {y['before']['mae']:.3f} -> "
                     f"after bias {y['after']['bias']:+.3f} MAE {y['after']['mae']:.3f} "
                     f"(production sim reference bias +21.5)")
        L.append(f"- Verdict: {proe.get('verdict', '')} Registry status: {proe.get('registry_status', '')}.")
    L.append("")

    L.append("## Caveats and missing cohorts")
    L.append("")
    L.append("- FTN-based tests (A09/A10) train on 2022 only for the season-forward split; treat "
             "their forward stability with extra caution.")
    L.append("- Man/zone participation tags cover 38-50% of plays depending on season; A17 trailing "
             "man-share uses tagged plays only (games with <5 tagged plays dropped).")
    L.append("- C10 new-HC is 2020-2025 (no 2018 tape for 2019); B04 altitude excludes DEN home rows "
             "from both arms; B05/B06 exclude international rows; B07 body-clock is the folklore "
             "1pm-ET-west-coast cohort.")
    L.append("- Referee crews tested at TEAM-VOLUME level only (C08 plays/game). DPI-rate pathway "
             "needs penalty-level pbp (registered proposed). Officials.parquet confirms crew "
             "assignments but adds no rate columns here.")
    L.append("- Matchup interactions (deep-ball QB x explosive-allowed, man-beater WR x man-share, "
             "pressure-sensitive QB x pressure D) are LOGGED AS HYPOTHESES for the interaction "
             "wave; mains estimated here, interactions deliberately not fit.")
    L.append("- Gate rejections do NOT retract battery findings: effects are measured vs the "
             "player's TRAILING baseline, while the gate asks for improvement over the trained "
             "ensemble, which already consumes spread/total/implied/roof/surface as features and "
             "evidently prices most of these (exposed-row model residuals are near zero where the "
             "trailing-space effect is large). 'rejected' at the gate = no post-hoc adjustment "
             "on top of the current model.")
    L.append("- Sub-threshold certain improvements: C04 (-0.0273), C05 (-0.0123), C01 (-0.0092), "
             "A10 (-0.0031), B04 (-0.0022) all had MAE-delta CIs entirely below zero but means "
             "above the -0.05 materiality bar, so they are rejected per the predeclared gate. A "
             "JOINT capped game-script adjustment (C01+C04+C05 together, non-overlapping rows) is "
             "logged as a hypothesis for the interaction/integration wave -- deliberately NOT fit "
             "here to respect predeclaration.")
    L.append("")
    L.append(f"STATUS: battery complete; {len(surv)} survivor(s); "
             f"{len(unst)} unstable-forward; {sum(1 for r in results.values() if r.get('battery_status') == 'rejected_null')} nulls; "
             f"{sum(1 for r in results.values() if r.get('empty'))} empty cohorts.")
    with open(rp, "w") as f:
        f.write(head + "\n".join(L) + "\n")
    print(f"report written: {rp}")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=["frames", "battery", "fdr", "gates", "extras", "registry", "report"])
    ap.add_argument("--family", default=None, choices=list("ABCD"))
    a = ap.parse_args()
    os.makedirs(EXP, exist_ok=True)
    if a.stage == "frames":
        stage_frames()
    elif a.stage == "battery":
        for fam in ([a.family] if a.family else list("ABCD")):
            stage_battery(fam)
    elif a.stage == "fdr":
        stage_fdr()
    elif a.stage == "gates":
        stage_gates()
    elif a.stage == "extras":
        stage_extras()
    elif a.stage == "registry":
        stage_registry()
    elif a.stage == "report":
        stage_report()

if __name__ == "__main__":
    main()
