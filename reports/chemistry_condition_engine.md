# Chemistry & Condition Engine — full study report

2026-07-16, chemistry/condition agent (wave 2). Reproduce with `analysis/chemistry_study.py`
(step list in its docstring); pure as-of builders in `nflvalue/fantasy/chemistry_engine.py`
(tests `tests/test_chemistry_engine.py`, 6/6 green). Scratch + checkpoints: `/tmp/exp_chemistry/`.

## TL;DR

- **QB-receiver chemistry exists only as a VOLUME phenomenon** (targets/route with a given
  QB): existence chi-2 p~1e-29, season-forward predictive r=0.168 [0.092, 0.242].
  **Efficiency chemistry (EPA/CPOE/yds/success/explosive/RZ-share per target) is null** —
  EB variance components fit to exactly zero on 2016-2022 and forward tests fail.
- **Conditions surviving BH-FDR (q<=0.05, predeclared 64-test family)**: WR1-out -> WR/TE
  bumps; dome QB/WR; turf WR; primetime QB fade; big-dog QB bump; big-fav QB fade;
  D.Smith primetime (raw; posterior straddles 0); wind (research-only). q<=0.10 adds
  primetime WR fade and post-bye WR/TE fades.
- **But NONE of it helps the production model**: all four post-hoc adjustment variants
  (chemistry / conditions / both / interactions; +-1.5 pt cap; constants strictly-prior per
  test year) **DEGRADE outer MAE** (+0.011 to +0.027, all 95% CIs entirely above 0) ->
  **all rejected; variant (a) none wins**. Recency-weighted baselines already price these
  effects; the training slope on chemistry is *negative* (mean reversion), so post-hoc
  layering double-counts.
- **Folklore**: birthday NULL, revenge NULL, DPI-crew x deep-target NULL, own-stadium quirk
  NULL (p=0.128). Primetime fades are real at position level (QB -0.98, WR -0.37 posterior
  pts vs own baseline) but likely slate-strength confounded.

## Method

- **Pair panel**: pbp + pbp_participation 2016-2025 (65,109 pair-weeks; route proxy =
  dropbacks of QB q with receiver r on field). Strictly-prior cums (cumsum minus current
  row); arbitrary (qb, rec, t) lookups via merge_asof `allow_exact_matches=False` (< t
  only). FTN catchability 2022-2024; pressure states from participation `was_pressure`.
- **Hierarchy** league -> position -> (QB main, receiver main) -> pair, all EB
  method-of-moments; existence constants fit <=2022; variant-gate constants refit on
  < test-year only. Volume chemistry separated from efficiency chemistry everywhere;
  both are reported per pair (efficiency posterior is identically 0 because its variance
  component fits to 0 — that IS the per-pair efficiency report).
- **Condition panel**: 36,380 player-weeks 2019-2025 REG, outcome `r_adj` = fantasy_points
  minus ewm8 pregame expectation, demeaned position x season (sd 6.81).
- **Battery predeclared**: 64 tests in `/tmp/exp_chemistry/battery.json`, declared BEFORE
  any condition test ran. Every split: raw n, effective n (block-bootstrap design effect),
  raw effect, 95% CI (season-week block bootstrap B=4000 seed 6102026), EB posterior
  (battery-wide tau=0.94 pts), per-season sign stability, 2019-22 vs 2023-25 check, BH-FDR q.
- **Spread** = schedules closing line (pregame-known; verified stamped at schedule level,
  not in-game). **Temp/wind = observed-game values -> pregame-ILLEGAL; all weather splits
  RESEARCH-ONLY.** Retractable roof state decided ~90min pregame; also research-only.
- **Starter/backup**: `backup_strict` = realized primary QB started <=2 of team's prior 8
  team-games (18.5% of rows); pregame-legal `pregame_new_qb` 20.3%. First-pass loose modal
  definition (25.1%) retired as season-boundary-inflated.

## 1. Pair chemistry: existence + forward prediction (A1-A9, B1-B4)

| test | what | statistic | p | BH-q | verdict |
|---|---|---|---|---|---|
| A1_pair_var_vol_tpr | targets/route given QB (volume) | chi2 T=1939.1 df=1284 | 1.50e-29 | 0.0000 | SIGNAL |
| A2_pair_var_epa_per_tgt | EPA/target over QB+receiver expectation | chi2 T=400.2 df=543 | 1.00e+00 | 1.0000 | null |
| A3_pair_var_cpoe | CPOE catchability over QB+receiver expectation | chi2 T=425.8 df=543 | 1.00e+00 | 1.0000 | null |
| A4_pair_var_yds_per_tgt | yards/target over expectation | chi2 T=432.1 df=539, tau2=0.00e+00 | 1.00e+00 | 1.0000 | null |
| A5_pair_var_success | success rate over receiver base | chi2 T=493.5 df=539, tau2=0.00e+00 | 9.20e-01 | 1.0000 | null |
| A6_pair_var_explosive | explosive (20+) rate over receiver base | chi2 T=483.6 df=539, tau2=0.00e+00 | 9.58e-01 | 1.0000 | null |
| A7_pair_var_airshare | share of QB air yards vs receiver norm | chi2 T=58116.0 df=2145, tau2=9.57e-04 | 0.00e+00 | 0.0000 | SIGNAL |
| A8_pair_var_rzshare | red-zone target share vs receiver norm | chi2 T=1388.6 df=1643, tau2=0.00e+00 | 1.00e+00 | 1.0000 | null |
| A9_pair_var_catchable2022 | FTN catchable-ball rate (2022 inner) | chi2 T=452.4 df=222, tau2=3.00e-03 | 7.02e-18 | 0.0000 | SIGNAL |
| B1_fwd_pair_vol | prior pair volume posterior -> next-season dev | fwd r=0.1683 [0.0921, 0.2423], n=566 pairs | 5.00e-04 | 0.0040 | SIGNAL |
| B2_fwd_pair_epa | prior pair EPA dev -> next-season dev | fwd r=0.0631 [-0.0478, 0.1704], n=326 pairs | 2.65e-01 | 0.4584 | null |
| B3_fwd_pair_cpoe | prior pair CPOE dev -> next-season dev | fwd r=0.0165 [-0.0956, 0.1253], n=325 pairs | 7.95e-01 | 0.8926 | null |
| B4_pair_var_pressure_gap | pair clean-vs-pressure EPA gap variance | chi2 T=604.5 df=553, tau2=7.89e-03 | 6.37e-02 | 0.1728 | null |

Reading: volume chemistry (A1) and air-share allocation (A7) show pair variance far beyond
noise, and the volume posterior carries forward across seasons (B1). A7's chi-2 uses an
assumed share-variance scale — treat its magnitude qualitatively (tau~0.031 share-sd is
the meaningful number). Catchability chemistry (A9) exists within 2022 FTN but has no
forward test (single inner season) — research-only. Every per-target EFFICIENCY channel
(A2-A6, A8) is null with EB tau2 = 0; B2/B3 confirm no forward signal. Pressure-gap
variance (B4, p=0.064) marginal -> not admitted.

Pair volume posterior scale: sd ~0.009 targets/route among pairs with >=300 shared routes
(~ +-0.5 targets/game at the extremes). Top end-2025 posteriors (face-valid):
Dalton->A.J. Green +0.035, Darnold->Smith-Njigba +0.035, Tua->Hill +0.034,
Goff->Swift +0.030; bottom: Hurts->Zaccheaus -0.030, Murray->Green -0.029.

Team-season/coordinator continuity limitation: pair cums cross team-season boundaries by
design (roster-of-record identity); coordinator-change regimes are NOT modeled — a pair's
history under a fired OC still counts. Noted, not fixed (no coordinator table in snapshot).

## 2. Condition battery (C01-C51) — every split, nulls first-class

| test | n | n_eff | effect | 95% CI | posterior [95%] | p | q | seasons same sign | 2019-22 / 2023-25 | base pts cond/ctrl |
|---|---|---|---|---|---|---|---|---|---|---|
| C01_surface_turf_QB | 1976 | 1915 | 0.41 | [-0.082, 0.922] | 0.382 [-0.098, 0.863] | 0.1075 | 0.247 | 5/7 | 0.319 / 0.533 | 13.98 / 13.7 |
| C02_surface_turf_RB | 4057 | 3647 | 0.288 | [-0.01, 0.588] | 0.281 [-0.013, 0.575] | 0.062 | 0.173 | 4/7 | 0.446 / 0.072 | 8.96 / 8.7 |
| C03_surface_turf_WR | 6381 | 5596 | 0.373 | [0.133, 0.615] | 0.367 [0.126, 0.608] | 0.0015 | 0.010 | 6/7 | 0.344 / 0.416 | 8.73 / 8.14 |
| C04_surface_turf_TE | 3171 | 3079 | 0.057 | [-0.188, 0.315] | 0.056 [-0.195, 0.306] | 0.6615 | 0.814 | 4/7 | 0.069 / 0.039 | 6.14 / 6.27 |
| C05_dome_QB | 1431 | 1387 | 0.763 | [0.254, 1.294] | 0.705 [0.198, 1.212] | 0.004 | 0.018 | 7/7 | 0.722 / 0.814 | 14.36 / 13.57 |
| C06_dome_RB | 2938 | 2938 | 0.14 | [-0.122, 0.411] | 0.137 [-0.133, 0.407] | 0.321 | 0.503 | 4/7 | 0.098 / 0.194 | 9.03 / 8.7 |
| C07_dome_WR | 4622 | 4622 | 0.609 | [0.391, 0.838] | 0.6 [0.372, 0.827] | 0.00025 | 0.002 | 7/7 | 0.612 / 0.607 | 9.02 / 8.13 |
| C08_dome_TE | 2269 | 2195 | 0.136 | [-0.122, 0.419] | 0.133 [-0.136, 0.403] | 0.305 | 0.501 | 5/7 | 0.242 / 0.006 | 6.38 / 6.13 |
| C09_retractable_open_vs_closed_pooled_RESEARCHONLY | 713 | 442 | 0.235 | [-0.453, 0.92] | 0.206 [-0.442, 0.854] | 0.5275 | 0.689 | 3/4 | 0.111 / 0.513 | 9.18 / 9.13 |
| C10_cold_le32_pooled_RESEARCHONLY | 1279 | 768 | -0.235 | [-0.661, 0.264] | -0.221 [-0.671, 0.229] | 0.35 | 0.533 | 5/7 | -0.541 / 0.093 | 8.22 / 8.57 |
| C11_cold_le32_QB_RESEARCHONLY | 156 | 156 | -0.775 | [-1.858, 0.432] | -0.562 [-1.53, 0.406] | 0.1995 | 0.376 | 5/7 | -0.881 / -0.665 | 13.54 / 13.55 |
| C12_wind_ge15_QB_RESEARCHONLY | 341 | 251 | -1.644 | [-2.7, -0.561] | -1.228 [-2.156, -0.3] | 0.0035 | 0.017 | 7/7 | -1.428 / -2.066 | 12.27 / 13.73 |
| C13_wind_ge15_WRTE_RESEARCHONLY | 1604 | 1048 | -0.829 | [-1.216, -0.42] | -0.793 [-1.18, -0.405] | 0.00025 | 0.002 | 7/7 | -0.824 / -0.844 | 6.94 / 7.54 |
| C14_altitude_DEN_QB | 144 | 122 | -0.874 | [-2.28, 0.559] | -0.55 [-1.674, 0.574] | 0.23 | 0.409 | 4/7 | -1.313 / -0.261 | 12.91 / 13.84 |
| C15_altitude_DEN_RB | 327 | 307 | -0.216 | [-0.918, 0.498] | -0.189 [-0.842, 0.464] | 0.5275 | 0.689 | 5/7 | -0.403 / 0.026 | 8.24 / 8.83 |
| C16_altitude_DEN_WR | 495 | 381 | -0.329 | [-0.969, 0.3] | -0.293 [-0.902, 0.316] | 0.322 | 0.503 | 5/7 | -0.532 / -0.079 | 7.56 / 8.44 |
| C17_altitude_DEN_TE | 258 | 258 | -0.084 | [-0.692, 0.541] | -0.075 [-0.663, 0.512] | 0.794 | 0.893 | 4/7 | -0.339 / 0.274 | 5.54 / 6.23 |
| C18_primetime_QB | 854 | 667 | -1.122 | [-1.831, -0.435] | -0.982 [-1.633, -0.331] | 0.001 | 0.007 | 6/7 | -1.105 / -1.143 | 14.35 / 13.69 |
| C19_primetime_RB | 1814 | 1814 | -0.282 | [-0.586, 0.019] | -0.275 [-0.573, 0.023] | 0.0675 | 0.173 | 6/7 | -0.452 / -0.067 | 8.93 / 8.78 |
| C20_primetime_WR | 2919 | 2204 | -0.384 | [-0.693, -0.067] | -0.373 [-0.688, -0.058] | 0.018 | 0.068 | 7/7 | -0.392 / -0.373 | 8.34 / 8.42 |
| C21_primetime_TE | 1398 | 1398 | 0.253 | [-0.066, 0.562] | 0.246 [-0.065, 0.557] | 0.1175 | 0.259 | 5/7 | 0.155 / 0.371 | 6.57 / 6.12 |
| C22_shortweek_QB | 284 | 222 | 0.298 | [-0.837, 1.435] | 0.216 [-0.75, 1.183] | 0.5945 | 0.746 | 4/7 | -0.014 / 0.655 | 15.17 / 13.72 |
| C23_shortweek_RB | 593 | 593 | -0.008 | [-0.547, 0.539] | -0.007 [-0.532, 0.517] | 0.9695 | 1.000 | 5/7 | -0.044 / 0.033 | 9.1 / 8.79 |
| C24_shortweek_WR | 945 | 655 | 0.412 | [-0.122, 0.959] | 0.378 [-0.149, 0.906] | 0.143 | 0.295 | 5/7 | 0.46 / 0.36 | 8.94 / 8.37 |
| C25_shortweek_TE | 473 | 473 | 0.452 | [-0.059, 0.982] | 0.42 [-0.073, 0.913] | 0.081 | 0.199 | 6/7 | 0.242 / 0.69 | 6.7 / 6.17 |
| C26_postbye_QB | 269 | 269 | -0.737 | [-1.641, 0.168] | -0.595 [-1.406, 0.217] | 0.108 | 0.247 | 6/7 | -1.195 / -0.144 | 13.53 / 13.83 |
| C27_postbye_RB | 587 | 587 | -0.375 | [-0.87, 0.143] | -0.349 [-0.837, 0.139] | 0.1605 | 0.311 | 5/7 | -0.36 / -0.395 | 8.59 / 8.82 |
| C28_postbye_WR | 917 | 912 | -0.5 | [-0.931, -0.059] | -0.473 [-0.902, -0.044] | 0.026 | 0.092 | 6/7 | -0.763 / -0.143 | 7.87 / 8.44 |
| C29_postbye_TE | 452 | 439 | -0.634 | [-1.116, -0.112] | -0.59 [-1.077, -0.102] | 0.018 | 0.068 | 5/7 | -0.7 / -0.549 | 5.72 / 6.24 |
| C30_bigfav_QB | 708 | 708 | -0.783 | [-1.386, -0.189] | -0.709 [-1.277, -0.14] | 0.01 | 0.043 | 6/7 | -0.697 / -0.919 | 15.41 / 13.52 |
| C31_bigfav_RB | 1451 | 1451 | 0.37 | [0.028, 0.726] | 0.358 [0.018, 0.697] | 0.032 | 0.108 | 6/7 | 0.373 / 0.368 | 9.66 / 8.65 |
| C32_bigfav_WR | 2155 | 2155 | -0.054 | [-0.372, 0.245] | -0.052 [-0.362, 0.257] | 0.721 | 0.869 | 3/7 | -0.105 / 0.024 | 9.12 / 8.28 |
| C33_bigfav_TE | 1069 | 1069 | 0.325 | [-0.008, 0.668] | 0.315 [-0.017, 0.646] | 0.054 | 0.165 | 6/7 | 0.32 / 0.334 | 7.18 / 6.04 |
| C34_bigdog_QB | 661 | 661 | 1.305 | [0.629, 1.956] | 1.159 [0.542, 1.776] | 0.00025 | 0.002 | 7/7 | 1.23 / 1.43 | 11.13 / 14.27 |
| C35_bigdog_RB | 1364 | 1364 | -0.111 | [-0.444, 0.217] | -0.108 [-0.432, 0.217] | 0.5265 | 0.689 | 5/7 | 0.055 / -0.371 | 7.79 / 8.97 |
| C36_bigdog_WR | 2154 | 2154 | -0.096 | [-0.378, 0.193] | -0.094 [-0.378, 0.191] | 0.5115 | 0.689 | 5/7 | -0.086 / -0.112 | 7.37 / 8.58 |
| C37_bigdog_TE | 1062 | 927 | -0.186 | [-0.523, 0.142] | -0.18 [-0.513, 0.154] | 0.2795 | 0.471 | 5/7 | -0.102 / -0.312 | 5.51 / 6.32 |
| C38_backupQB_RB | 1758 | 1758 | 0.187 | [-0.11, 0.485] | 0.182 [-0.11, 0.475] | 0.2095 | 0.383 | 4/7 | 0.255 / 0.107 | 8.27 / 8.9 |
| C39_backupQB_WR | 2719 | 2450 | 0.13 | [-0.171, 0.405] | 0.127 [-0.154, 0.408] | 0.384 | 0.572 | 4/7 | 0.026 / 0.246 | 7.69 / 8.54 |
| C40_backupQB_TE | 1361 | 1361 | -0.093 | [-0.382, 0.216] | -0.091 [-0.388, 0.207] | 0.5565 | 0.712 | 3/7 | 0.056 / -0.252 | 5.63 / 6.33 |
| C41_WR1out_WR | 1719 | 1719 | 1.949 | [1.634, 2.259] | 1.892 [1.578, 2.207] | 0.00025 | 0.002 | 7/7 | 1.973 / 1.92 | 7.56 / 8.5 |
| C42_WR1out_TE | 882 | 882 | 0.6 | [0.237, 0.964] | 0.577 [0.214, 0.94] | 0.0025 | 0.013 | 6/7 | 0.474 / 0.78 | 6.72 / 6.13 |
| C43_WR1out_RB | 1133 | 1133 | 0.473 | [0.034, 0.907] | 0.448 [0.021, 0.875] | 0.034 | 0.109 | 5/7 | 0.863 / -0.059 | 8.84 / 8.79 |
| C44_refDPIhigh_x_deepWR | 925 | 925 | 0.234 | [-0.323, 0.781] | 0.214 [-0.318, 0.747] | 0.4145 | 0.603 | 4/7 | 0.21 / 0.354 | 8.98 / 8.57 |
| C45_refDPIhigh_x_allWR | 4627 | 4627 | 0.082 | [-0.125, 0.302] | 0.081 [-0.131, 0.293] | 0.4405 | 0.626 | 5/7 | 0.172 / -0.031 | 8.51 / 8.33 |
| C46_birthday_week_pooled | 746 | 684 | -0.074 | [-0.576, 0.455] | -0.069 [-0.562, 0.424] | 0.787 | 0.893 | 4/7 | 0.097 / -0.288 | 8.47 / 8.75 |
| C47_revenge_game_pooled | 845 | 845 | 0.066 | [-0.281, 0.414] | 0.064 [-0.285, 0.412] | 0.7335 | 0.869 | 3/7 | 0.015 / 0.124 | 6.98 / 8.78 |
| C48_primetime_Godwin | 19 | 19 | -2.445 | [-5.895, 0.976] | -0.545 [-2.172, 1.083] | 0.16 | 0.311 | 2/2 | -2.527 / -2.445 | 13.28 / 16.09 |
| C49_primetime_Aiyuk | 18 | 18 | -3.214 | [-6.629, 0.265] | -0.717 [-2.344, 0.91] | 0.067 | 0.173 | 1/1 | -2.339 / -5.0 | 10.75 / 14.01 |
| C50_primetime_DSmith | 20 | 20 | -4.953 | [-8.505, -1.423] | -1.038 [-2.679, 0.603] | 0.0025 | 0.013 | 3/3 | -3.622 / -5.871 | 10.54 / 14.3 |
| C51_own_stadium_quirk_variance | 440 players | — | home-edge 0.358 | — | stadium-quirk tau=0.387 | 0.12801 | 0.273 | — | — | — |

Notes: effect = mean `r_adj` (points vs own ewm8 baseline) in condition minus out of
condition, same position subset. `_RESEARCHONLY` = pregame-illegal information (observed
weather / gameday roof state). Effective n from block-bootstrap design effect. Posterior =
EB shrink toward 0 with battery-wide tau = 0.94 pts. C44/C45 (DPI-heavy crews x deep /
all WRs) were the predeclared referee-pathway tests: both null, as predicted. C38-C40
(backup-QB splits vs receivers' own recent baseline): null — by the time a backup starts,
the ewm8 baseline has largely absorbed the regime; this does not contradict the pair-level
volume finding.

## 3. Folklore adjudication

| claim | verdict | evidence |
|---|---|---|
| Birthday-week boost (C46) | **NULL** | -0.07 pts, p=0.79, n=746 |
| Revenge game (C47) | **NULL** | +0.07 pts, p=0.73, n=845 |
| Ref DPI crew helps deep WRs (C44/45) | **NULL (as predicted)** | +0.23 p=0.41 deep; +0.08 p=0.44 all WR |
| Own-stadium quirk (C51) | **NULL** | player home-vs-away deviation variance p=0.128 (440 players); league home edge +0.36 pts is common, not player-specific |
| "Lights too bright" primetime fades | **REAL at position level** | QB -0.98 [-1.63, -0.33] q=0.007; WR -0.37 [-0.69, -0.06] q=0.068; 6-7/7 seasons. Confound: primetime slates feature stronger opponents and the baseline is opponent-blind — schedule-context effect, not psychology |
| Condition-book: D.Smith primetime fade | **monitor (research_only)** | raw -4.95 p=0.0025 q=0.013 survives FDR, but n=20 and EB posterior -1.04 [-2.68, +0.60] straddles 0; 3/3 seasons negative — keep flag, do not act |
| Condition-book: Godwin primetime fade | **not supported** | -2.45 p=0.16, n=19; posterior -0.55 [-2.17, +1.08] |
| Condition-book: Aiyuk primetime fade | **not supported** | -3.21 p=0.067, n=18; posterior -0.72 [-2.34, +0.91] |

## 4. Five-variant season-forward gate (decision table)

Baseline: READ-ONLY `reports/fantasy_outer_predictions.parquet` (reproduced ensemble, MAE 5.105968, n=11,481; outer years 2023/2024/2025, per-year base MAE 5.021/5.104/5.193). Cap +-1.5 pts (max |posterior| condition effect ~1.2 pts; cap binds on <4% of rows). Mappings = per-position OLS on `r_adj`; all constants refit strictly before each test year. Paired week-block bootstrap (54 blocks, B=10,000, seed 6102026).

| variant | MAE | dMAE vs (a) | 95% CI | cov80 (nom .80) | Spearman | adj sd | % capped | GATE |
|---|---|---|---|---|---|---|---|---|
| a_none (baseline) | 5.105968 | — | — | 0.8183 | 0.6232 | — | — | **winner** |
| b_chemistry | 5.116649 | +0.010682 | [0.0054, 0.0164] | 0.8178 | 0.6217 | 0.307 | 0.63% | **rejected** |
| c_conditions | 5.121343 | +0.015375 | [0.0047, 0.0269] | 0.8180 | 0.6209 | 0.445 | 1.72% | **rejected** |
| d_both | 5.132556 | +0.026589 | [0.0157, 0.0384] | 0.8169 | 0.6193 | 0.536 | 2.76% | **rejected** |
| e_interactions | 5.129338 | +0.023370 | [0.0117, 0.0357] | 0.8172 | 0.6190 | 0.563 | 3.61% | **rejected** |

Gate rule (predeclared): mean dMAE <= -0.05 AND 95% CI upper < 0 -> shadow; CI straddles
0 -> research_only; else rejected. All four variants have CIs entirely ABOVE zero ->
**rejected**. Spearman never improves; coverage moves <=0.14pp.

Diagnosis (why chemistry fails as an adjustment while being real): the per-position
training slope of `r_adj` on the as-of chemistry feature is **negative** every
year/position (WR -2.9..-1.6, TE -2.8..-2.3, RB -1.2..-0.9): a high as-of pair-volume
posterior means recent production is already elevated, the recency-weighted baseline
over-projects, and the residual mean-reverts. The ensemble baseline is strictly better
than ewm8, so a post-hoc layer fitted on ewm8 residuals double-counts reversion and adds
noise. Chemistry information is already priced in; there is no residual edge to harvest
post-hoc. If chemistry is revisited it must be IN-MODEL (a feature the ensemble trains
on, owned by the role/feature agents), not a post-hoc adjustment.

## 5. Limitations (honest list)

- Routes are an on-field-on-dropback proxy from participation, not charted routes; FTN
  charted routes exist 2023+ only (train-only per methodology scan) and were not used.
- Adjustment mappings were fit on ewm8 residuals (the only pregame expectation available
  across all train years), not ensemble residuals; exactly why the gate exists — gate said no.
- Coordinator/scheme continuity unmodeled; pair history crosses team-season boundaries.
- Weather splits use observed-game weather (pregame-illegal): research-only unless an
  archived pregame-forecast source is added.
- Primetime effects confounded with slate strength (opponent quality not in the baseline).
- 2016-2018 participation lacks `was_pressure`; FTN catchability 2022-2024 only (A9 has
  no forward test).
- QB rows get no chemistry adjustment in variant (b) (receiver channel only); (d)/(e)
  give QBs the condition channel.

## 6. Registry + artifacts

Registry records -> `data/factor_registry_chemistry.json` (everything at most
research_only; adjustment variants rejected). Scratch: `/tmp/exp_chemistry/` (pair_week /
pair_asof / cond_panel / chem_features parquets, ckpt/*.json). Battery predeclaration:
`/tmp/exp_chemistry/battery.json`. Baseline artifact READ-ONLY; no production file
touched; no commits.
