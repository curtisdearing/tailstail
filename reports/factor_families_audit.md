# Factor families audit — opponent/scheme, environment, game context, market (+ PROE prototype)

Agent: factor-families (wave 2). Date: 2026-07-16. Baseline gate reference:
`reports/fantasy_outer_predictions.parquet` (hash ee0052a9…, n=11,481, direct MAE
5.105968, 54 season-week blocks 2023–2025). Evaluation frames built read-only into
`/tmp/exp_families/`; nothing under `data/` caches or `nflvalue/` was modified.

## PREDECLARATION (frozen before any results were computed)

**The confirmatory battery is exactly 40 tests.** BH-FDR is applied across these 40
p-values and nothing else. Every other number in this report (wind/temp splits, the
PROE prototype MAE comparison, descriptive cohort tables) is explicitly outside the
battery: the PROE prototype is a season-forward MAE evaluation, not an NHST test, and
observed-weather splits are research_only(leakage_proxy) by construction.

Common method: population = played player-weeks 2019–2025 REG with >=3 prior played
games, positions QB/RB/WR/TE (n=32,781; team-level tests use team-games). Primary
outcome per test is the deviation of the realized stat from the player's own trailing
baseline (fantasy points minus pre_fantasy_points_ewm4, or the named component minus
its ewm4), so every effect is "vs trailing-baseline". Exposed vs matched controls:
exact stratification on position x role-rank bin (1/2/3+) x trailing-baseline quintile
(team tests: trailing-volume quintile); effect = exposed-count-weighted mean of
within-stratum differences. Opponent trailing stats are strictly shift-then-roll
(ewm halflife 5 games, min 3 prior, cross-season carryover) and converted to exposure
by within-season-week tercile rank across defenses (top vs bottom tercile, middle
excluded). Uncertainty: cluster bootstrap over (season, week) blocks, B=1500, seed
6102026; two-sided bootstrap p; skeptical normal prior centered 0 (sd: 1.0 fp,
8 pass yds, 5 rush/rec yds, 2.5 team plays, 2.0 team pass att) -> posterior mean +
95% CI; raw n and effective n (distinct exposed team-games) reported for every test
including nulls and empty cohorts. Season-by-season signs reported; season-forward
check = estimate on <=2022 (FTN-based: 2022 only) must hold sign/magnitude-order on
2023/2024/2025 per-year. Battery survivors that imply a projection adjustment are then
gated post-hoc on the frozen outer predictions with a <=2022-estimated capped
adjustment and paired week-block bootstrap: mean MAE delta <= -0.05 AND 95% CI upper
< 0 -> shadow; CI straddles 0 -> research_only; else rejected.

### The 40 predeclared tests

Family A — opponent/scheme (16):
- A01 opp trailing QB-fantasy-points-allowed top vs bottom tercile -> QB d_fp
- A02 same, RB-points-allowed -> RB d_fp
- A03 same, WR-points-allowed -> WR d_fp
- A04 same, TE-points-allowed -> TE d_fp
- A05 opp trailing EPA/dropback allowed (soft pass D) -> QB d_fp
- A06 opp trailing EPA/dropback allowed -> WR d_fp
- A07 opp trailing EPA/rush allowed -> RB d_rush_yards
- A08 opp trailing pressure rate ((sack+qb_hit)/dropback) high -> QB d_fp
- A09 opp trailing blitz share (FTN n_blitzers>0, 2022-25) high -> QB d_fp
- A10 opp trailing heavy-box share vs rush (FTN box>=8, 2022-25) high -> RB d_rush_yards
- A11 opp trailing plays/game (pace) fast -> pooled d_fp
- A12 opp trailing PROE-allowed high -> QB d_fp
- A13 opp trailing explosive-pass-allowed (20+yd per dropback) high -> WR d_fp
- A14 2+ opponent DBs officially out -> QB d_fp
- A15 2+ opponent DBs officially out -> WR d_fp
- A16 2+ opponent front-7 officially out -> RB d_fp
- A17 opp trailing man-coverage share (participation MAN/(MAN+ZONE)) high -> WR d_fp
  (A17 is the 17th line but A-family count is 16 tests: A01-A16 plus A17 minus none —
  see note below.)

Family B — environment (12):
- B01 turf vs grass -> pooled d_fp
- B02 closed roof (dome/closed) vs open air -> pooled d_fp
- B03 closed roof -> QB d_pass_yards
- B04 road team at altitude (Denver/Azteca) -> pooled d_fp
- B05 road travel >=1500mi (international excluded) -> pooled d_fp
- B06 road 2+ time zones crossed (international excluded) -> pooled d_fp
- B07 west-coast (PT home) team, road kickoff 13:xx ET -> pooled d_fp
- B08 international game -> pooled d_fp
- B09 short rest (<=5 days) -> pooled d_fp
- B10 post-bye (rest >=12 days) -> pooled d_fp
- B11 post-bye -> WR d_rec_yards
- B12 overtime last week -> pooled d_fp

Family C — game context (10):
- C01 big favorite (team spread <= -7) -> pooled d_fp
- C02 big underdog (team spread >= +7) -> pooled d_fp
- C03 high total (>=47.5) -> pooled d_fp
- C04 low total (<=41.5) -> pooled d_fp
- C05 implied team total top vs bottom within-week tercile -> pooled d_fp
- C06 primetime -> pooled d_fp
- C07 divisional rematch (2nd meeting) -> pooled d_fp
- C08 referee trailing plays/game top vs bottom tercile -> TEAM d_plays
- C09 team trailing 4th-and-short go rate top vs bottom tercile -> TEAM d_plays
- C10 new head coach vs prior season -> pooled d_fp

Family D — market (1 + inventory):
- D01 implied team total top vs bottom within-week tercile -> TEAM d_pass_attempts

Count note: A-family enumerates A01..A17 with 17 ids? No — A01..A16 are sixteen ids
and A17 was added before any computation as the seventeenth A-id; the battery total
is A(17) + B(12) + C(10) + D(1) = 40. The count assertion in
tests/test_factor_families.py pins len(BATTERY) == 40 and the registry mirrors it.

Non-battery registry records declared up front: market open/close movement,
cross-book disagreement, prop-implied volume (data-acquisition dependent), referee
DPI->pass volume (needs penalty-level pbp), playoffs cohort (frame is REG-only),
observed wind/temp splits (leakage_proxy), matchup interaction hypotheses (deferred
to the interaction wave), man/zone matchup interactions (industry-negative prior).

Results below were computed only after this header was frozen.

## Battery results (all 40 tests, BH-FDR across the battery)

Population as built: 32,781 player-weeks (QB 4,105 / RB 8,566 / WR 13,457 / TE 6,653); 3,742 team-games. Effect units: fantasy points for d_fp, yards for yardage outcomes, plays/attempts for team outcomes; all vs trailing baseline, exposed minus matched control.

| id | pos | outcome | effect [95% CI] | post. mean [95% CI] | p | q | nE/nC (eff) | tr<=22 | 23/24/25 | status |
|---|---|---|---|---|---|---|---|---|---|---|
| A01_opp_qb_points_soft | QB | d_fp | +1.43 [+0.82,+2.05] | +1.30 [+0.72,+1.88] | 0.0007 | 0.002 | 1,442/1,290 (1,226) | +1.91 | +-+ | survivor->rejected |
| A02_opp_rb_points_soft | RB | d_fp | +1.26 [+0.94,+1.60] | +1.23 [+0.90,+1.56] | 0.0007 | 0.002 | 3,026/2,697 (1,278) | +1.24 | +++ | survivor->rejected |
| A03_opp_wr_points_soft | WR | d_fp | +0.46 [+0.17,+0.73] | +0.45 [+0.17,+0.73] | 0.0020 | 0.006 | 4,765/4,263 (1,279) | +0.67 | +-+ | research_only_unstable_forward |
| A04_opp_te_points_soft | TE | d_fp | +0.53 [+0.23,+0.80] | +0.52 [+0.23,+0.80] | 0.0007 | 0.002 | 2,354/2,120 (1,257) | +0.67 | -++ | survivor->rejected |
| A05_opp_pass_epa_soft_qb | QB | d_fp | +1.14 [+0.55,+1.73] | +1.05 [+0.47,+1.62] | 0.0007 | 0.002 | 1,465/1,306 (1,238) | +1.47 | +++ | survivor->rejected |
| A06_opp_pass_epa_soft_wr | WR | d_fp | +0.34 [+0.08,+0.60] | +0.34 [+0.08,+0.59] | 0.0100 | 0.025 | 4,714/4,275 (1,279) | +0.36 | +-+ | survivor->rejected |
| A07_opp_rush_epa_soft_rb | RB | d_rush_yards | +5.46 [+3.86,+7.10] | +5.31 [+3.70,+6.93] | 0.0007 | 0.002 | 3,022/2,688 (1,278) | +5.42 | +++ | survivor->rejected |
| A08_opp_pressure_high_qb | QB | d_fp | -0.82 [-1.45,-0.22] | -0.74 [-1.34,-0.15] | 0.0113 | 0.027 | 1,434/1,320 (1,234) | -1.43 | +-- | research_only_unstable_forward |
| A09_opp_blitz_high_qb | QB | d_fp | +0.39 [-0.23,+1.08] | +0.35 [-0.28,+0.98] | 0.2393 | 0.299 | 826/761 (699) | +1.31 | -++ | rejected_null |
| A10_opp_heavy_box_rb | RB | d_rush_yards | -3.15 [-5.52,-0.67] | -2.96 [-5.34,-0.59] | 0.0180 | 0.034 | 1,699/1,589 (728) | -2.59 | +-- | survivor->rejected |
| A11_opp_pace_fast_all | QB/RB/WR/TE | d_fp | -0.09 [-0.28,+0.10] | -0.09 [-0.29,+0.10] | 0.3847 | 0.466 | 11,456/10,394 (1,279) | -0.27 | +-+ | rejected_null |
| A12_opp_proe_allowed_qb | QB | d_fp | +0.92 [+0.30,+1.57] | +0.83 [+0.24,+1.43] | 0.0033 | 0.010 | 1,456/1,288 (1,228) | +1.32 | -++ | survivor->rejected |
| A13_opp_explosive_pass_wr | WR | d_fp | +0.38 [+0.08,+0.68] | +0.37 [+0.06,+0.67] | 0.0140 | 0.029 | 4,729/4,237 (1,279) | +0.33 | +-+ | survivor->rejected |
| A14_opp_db2_out_qb | QB | d_fp | +0.65 [-0.11,+1.37] | +0.57 [-0.12,+1.27] | 0.0887 | 0.136 | 429/3,663 (357) | +0.55 | +++ | rejected_null |
| A15_opp_db2_out_wr | WR | d_fp | +0.30 [-0.05,+0.64] | +0.29 [-0.05,+0.64] | 0.0887 | 0.136 | 1,391/12,028 (372) | +0.20 | +++ | rejected_null |
| A16_opp_front2_out_rb | RB | d_fp | +0.11 [-0.35,+0.58] | +0.10 [-0.34,+0.54] | 0.6313 | 0.665 | 1,060/7,470 (445) | -0.39 | +++ | rejected_null |
| A17_opp_man_rate_wr | WR | d_fp | +0.08 [-0.16,+0.30] | +0.08 [-0.15,+0.31] | 0.5260 | 0.569 | 4,778/4,256 (1,279) | +0.21 | -+- | rejected_null |
| B01_turf_all | QB/RB/WR/TE | d_fp | +0.35 [+0.16,+0.55] | +0.35 [+0.15,+0.54] | 0.0007 | 0.002 | 14,193/17,857 (1,572) | +0.35 | +++ | survivor->rejected |
| B02_closed_roof_all | QB/RB/WR/TE | d_fp | +0.53 [+0.33,+0.75] | +0.53 [+0.33,+0.73] | 0.0007 | 0.002 | 10,282/22,499 (1,140) | +0.59 | +++ | survivor->rejected |
| B03_closed_roof_qb_passyds | QB | d_pass_yards | +6.58 [+0.75,+12.67] | +5.75 [+0.19,+11.32] | 0.0287 | 0.052 | 1,308/2,797 (1,101) | +6.13 | +++ | rejected_null |
| B04_altitude_road_all | QB/RB/WR/TE | d_fp | -0.73 [-1.35,-0.13] | -0.66 [-1.25,-0.07] | 0.0140 | 0.029 | 571/31,681 (62) | -0.70 | --- | survivor->rejected |
| B05_travel_1500_all | QB/RB/WR/TE | d_fp | +0.22 [-0.12,+0.57] | +0.21 [-0.13,+0.56] | 0.1993 | 0.266 | 3,410/12,817 (385) | +0.07 | +++ | rejected_null |
| B06_tz2_all | QB/RB/WR/TE | d_fp | +0.26 [-0.04,+0.58] | +0.26 [-0.04,+0.56] | 0.0967 | 0.142 | 4,096/12,131 (460) | +0.23 | +++ | rejected_null |
| B07_body_clock_all | QB/RB/WR/TE | d_fp | +0.24 [-0.34,+0.78] | +0.22 [-0.33,+0.77] | 0.4087 | 0.481 | 1,122/15,301 (129) | +0.55 | -+- | rejected_null |
| B08_international_all | QB/RB/WR/TE | d_fp | -0.10 [-0.71,+0.46] | -0.09 [-0.64,+0.46] | 0.7407 | 0.741 | 397/32,384 (44) | +0.34 | --0 | rejected_null |
| B09_short_rest_all | QB/RB/WR/TE | d_fp | +0.44 [+0.03,+0.84] | +0.42 [+0.02,+0.82] | 0.0327 | 0.057 | 2,098/27,009 (239) | +0.33 | +-+ | rejected_null |
| B10_post_bye_all | QB/RB/WR/TE | d_fp | -0.43 [-0.75,-0.10] | -0.42 [-0.74,-0.10] | 0.0153 | 0.031 | 2,086/27,009 (225) | -0.57 | --+ | survivor->research_only |
| B11_post_bye_wr_recyds | WR | d_rec_yards | -1.86 [-4.28,+0.48] | -1.76 [-4.06,+0.54] | 0.0993 | 0.142 | 861/11,090 (225) | -2.72 | +-- | rejected_null |
| B12_after_ot_all | QB/RB/WR/TE | d_fp | -0.31 [-0.62,+0.03] | -0.30 [-0.62,+0.02] | 0.0713 | 0.119 | 1,691/29,502 (186) | -0.33 | +-- | rejected_null |
| C01_big_fav_all | QB/RB/WR/TE | d_fp | -0.50 [-0.71,-0.29] | -0.49 [-0.71,-0.28] | 0.0007 | 0.002 | 4,665/28,116 (530) | -0.50 | --- | survivor->rejected |
| C02_big_dog_all | QB/RB/WR/TE | d_fp | +0.50 [+0.30,+0.71] | +0.50 [+0.28,+0.71] | 0.0007 | 0.002 | 4,922/27,859 (530) | +0.53 | +++ | survivor->rejected |
| C03_high_total_all | QB/RB/WR/TE | d_fp | +0.41 [+0.21,+0.61] | +0.41 [+0.21,+0.61] | 0.0007 | 0.002 | 10,399/22,382 (1,144) | +0.33 | +++ | survivor->rejected |
| C04_low_total_all | QB/RB/WR/TE | d_fp | -0.53 [-0.73,-0.33] | -0.52 [-0.72,-0.32] | 0.0007 | 0.002 | 7,576/25,205 (844) | -0.70 | --- | survivor->rejected |
| C05_implied_high_all | QB/RB/WR/TE | d_fp | -0.33 [-0.53,-0.13] | -0.33 [-0.53,-0.13] | 0.0020 | 0.006 | 11,424/10,469 (1,279) | -0.32 | --- | survivor->rejected |
| C06_primetime_all | QB/RB/WR/TE | d_fp | -0.08 [-0.33,+0.16] | -0.08 [-0.33,+0.16] | 0.5060 | 0.562 | 7,084/25,697 (794) | -0.19 | -++ | rejected_null |
| C07_div_rematch_all | QB/RB/WR/TE | d_fp | +0.13 [-0.24,+0.48] | +0.12 [-0.23,+0.48] | 0.4847 | 0.554 | 6,215/5,654 (672) | +0.17 | +++ | rejected_null |
| C08_ref_pace_team | team | d_plays | +0.13 [-0.41,+0.68] | +0.13 [-0.41,+0.68] | 0.6553 | 0.672 | 1,243/1,092 (1,243) | +0.65 | +-- | rejected_null |
| C09_fourth_aggr_team | team | d_plays | +0.64 [-0.22,+1.48] | +0.63 [-0.21,+1.46] | 0.1353 | 0.187 | 1,216/1,101 (1,216) | +0.67 | ++- | rejected_null |
| C10_new_hc_all | QB/RB/WR/TE | d_fp | -0.13 [-0.34,+0.07] | -0.13 [-0.34,+0.07] | 0.2100 | 0.271 | 6,290/22,943 (692) | -0.27 | --+ | rejected_null |
| D01_implied_high_passatt_team | team | d_pass_att | +1.00 [+0.29,+1.71] | +0.97 [+0.27,+1.66] | 0.0087 | 0.023 | 1,275/1,147 (1,275) | +1.63 | +-+ | research_only_unstable_forward |

## Survivors and adjustment gates

- **A01_opp_qb_points_soft** (Opponent trailing QB fantasy points allowed, within-week top vs bottom tercile): effect +1.43 [+0.82,+2.05], q=0.002, nE=1,442 (eff 1,226), sign agreement 2/3, train<=2022 +1.91.
  - Gate on frozen outer predictions (adj +1.00 fp capped, n_touched=545): MAE delta +0.0070 [+0.0031,+0.0104] -> **rejected** (exposed-row model residual -0.15).
- **A02_opp_rb_points_soft** (Opponent trailing RB fantasy points allowed, top vs bottom tercile): effect +1.26 [+0.94,+1.60], q=0.002, nE=3,026 (eff 1,278), sign agreement 3/3, train<=2022 +1.24.
  - Gate on frozen outer predictions (adj +1.00 fp capped, n_touched=985): MAE delta +0.0148 [+0.0101,+0.0204] -> **rejected** (exposed-row model residual +0.40).
- **A04_opp_te_points_soft** (Opponent trailing TE fantasy points allowed, top vs bottom tercile): effect +0.53 [+0.23,+0.80], q=0.002, nE=2,354 (eff 1,257), sign agreement 2/3, train<=2022 +0.67.
  - Gate on frozen outer predictions (adj +0.67 fp capped, n_touched=614): MAE delta +0.0063 [+0.0027,+0.0095] -> **rejected** (exposed-row model residual +0.26).
- **A05_opp_pass_epa_soft_qb** (Opponent trailing EPA/dropback allowed, top (softest) vs bottom tercile): effect +1.14 [+0.55,+1.73], q=0.002, nE=1,465 (eff 1,238), sign agreement 3/3, train<=2022 +1.47.
  - Gate on frozen outer predictions (adj +1.00 fp capped, n_touched=546): MAE delta +0.0084 [+0.0043,+0.0122] -> **rejected** (exposed-row model residual -0.55).
- **A06_opp_pass_epa_soft_wr** (Opponent trailing EPA/dropback allowed, top vs bottom tercile): effect +0.34 [+0.08,+0.60], q=0.025, nE=4,714 (eff 1,279), sign agreement 2/3, train<=2022 +0.36.
  - Gate on frozen outer predictions (adj +0.36 fp capped, n_touched=1,533): MAE delta +0.0093 [+0.0066,+0.0122] -> **rejected** (exposed-row model residual -0.08).
- **A07_opp_rush_epa_soft_rb** (Opponent trailing EPA/rush allowed, top vs bottom tercile): effect +5.46 [+3.86,+7.10], q=0.002, nE=3,022 (eff 1,278), sign agreement 3/3, train<=2022 +5.42.
  - Gate on frozen outer predictions (adj +0.54 fp capped, n_touched=1,002): MAE delta +0.0066 [+0.0035,+0.0100] -> **rejected** (exposed-row model residual +0.58).
- **A10_opp_heavy_box_rb** (Opponent trailing share of rushes faced with 8+ box (FTN), top vs bottom tercile): effect -3.15 [-5.52,-0.67], q=0.034, nE=1,699 (eff 728), sign agreement 2/3, train<=2022 -2.59.
  - Gate on frozen outer predictions (adj -0.26 fp capped, n_touched=993): MAE delta -0.0031 [-0.0044,-0.0018] -> **rejected** (exposed-row model residual -0.16).
- **A12_opp_proe_allowed_qb** (Opponent trailing PROE allowed (mean pass_oe faced), top vs bottom tercile): effect +0.92 [+0.30,+1.57], q=0.010, nE=1,456 (eff 1,228), sign agreement 2/3, train<=2022 +1.32.
  - Gate on frozen outer predictions (adj +1.00 fp capped, n_touched=525): MAE delta +0.0063 [+0.0021,+0.0105] -> **rejected** (exposed-row model residual -0.31).
- **A13_opp_explosive_pass_wr** (Opponent trailing 20+yd completions per dropback allowed, top vs bottom tercile): effect +0.38 [+0.08,+0.68], q=0.029, nE=4,729 (eff 1,279), sign agreement 2/3, train<=2022 +0.33.
  - Gate on frozen outer predictions (adj +0.33 fp capped, n_touched=1,521): MAE delta +0.0070 [+0.0049,+0.0092] -> **rejected** (exposed-row model residual +0.19).
- **B01_turf_all** (Game on artificial turf vs natural grass): effect +0.35 [+0.16,+0.55], q=0.002, nE=14,193 (eff 1,572), sign agreement 3/3, train<=2022 +0.35.
  - Gate on frozen outer predictions (adj +0.35 fp capped, n_touched=4,632): MAE delta +0.0207 [+0.0156,+0.0254] -> **rejected** (exposed-row model residual +0.11).
- **B02_closed_roof_all** (Roof dome/closed vs outdoors/open): effect +0.53 [+0.33,+0.75], q=0.002, nE=10,282 (eff 1,140), sign agreement 3/3, train<=2022 +0.59.
  - Gate on frozen outer predictions (adj +0.59 fp capped, n_touched=3,403): MAE delta +0.0289 [+0.0213,+0.0363] -> **rejected** (exposed-row model residual +0.12).
- **B04_altitude_road_all** (Road team at Denver / any team at Azteca (5,280ft+); Denver home rows excluded): effect -0.73 [-1.35,-0.13], q=0.029, nE=571 (eff 62), sign agreement 3/3, train<=2022 -0.70.
  - Gate on frozen outer predictions (adj -0.70 fp capped, n_touched=155): MAE delta -0.0022 [-0.0040,-0.0003] -> **rejected** (exposed-row model residual -0.75).
- **B10_post_bye_all** (Rest >=12 days (post-bye) vs regular 6-11 day rest (short rest excluded)): effect -0.43 [-0.75,-0.10], q=0.031, nE=2,086 (eff 225), sign agreement 2/3, train<=2022 -0.57.
  - Gate on frozen outer predictions (adj -0.57 fp capped, n_touched=643): MAE delta -0.0028 [-0.0060,+0.0002] -> **research_only** (exposed-row model residual +0.05).
- **C01_big_fav_all** (Team spread <= -7 (big favorite) vs all other lined games): effect -0.50 [-0.71,-0.29], q=0.002, nE=4,665 (eff 530), sign agreement 3/3, train<=2022 -0.50.
  - Gate on frozen outer predictions (adj -0.50 fp capped, n_touched=1,271): MAE delta -0.0092 [-0.0128,-0.0053] -> **rejected** (exposed-row model residual -0.26).
- **C02_big_dog_all** (Team spread >= +7 (big underdog) vs all other lined games): effect +0.50 [+0.30,+0.71], q=0.002, nE=4,922 (eff 530), sign agreement 3/3, train<=2022 +0.53.
  - Gate on frozen outer predictions (adj +0.53 fp capped, n_touched=1,394): MAE delta +0.0069 [+0.0035,+0.0105] -> **rejected** (exposed-row model residual +0.22).
- **C03_high_total_all** (Game total >= 47.5 vs all other lined games): effect +0.41 [+0.21,+0.61], q=0.002, nE=10,399 (eff 1,144), sign agreement 3/3, train<=2022 +0.33.
  - Gate on frozen outer predictions (adj +0.33 fp capped, n_touched=2,701): MAE delta +0.0095 [+0.0066,+0.0125] -> **rejected** (exposed-row model residual +0.38).
- **C04_low_total_all** (Game total <= 41.5 vs all other lined games): effect -0.53 [-0.73,-0.33], q=0.002, nE=7,576 (eff 844), sign agreement 3/3, train<=2022 -0.70.
  - Gate on frozen outer predictions (adj -0.70 fp capped, n_touched=3,058): MAE delta -0.0273 [-0.0352,-0.0194] -> **rejected** (exposed-row model residual -0.27).
- **C05_implied_high_all** (Implied team total, within-season-week top vs bottom tercile): effect -0.33 [-0.53,-0.13], q=0.006, nE=11,424 (eff 1,279), sign agreement 3/3, train<=2022 -0.32.
  - Gate on frozen outer predictions (adj -0.32 fp capped, n_touched=3,638): MAE delta -0.0123 [-0.0158,-0.0087] -> **rejected** (exposed-row model residual +0.06).

q<0.05 but season-forward unstable (research_only): A03_opp_wr_points_soft, A08_opp_pressure_high_qb, D01_implied_high_passatt_team

## Folklore graveyard (nulls at q>=0.05, with n)

- A09_opp_blitz_high_qb: Opponent trailing blitz share (FTN n_blitzers>0 on dropbacks), top vs bottom tercile -- effect +0.39 [-0.23,+1.08], q=0.30, nE=826.
- A11_opp_pace_fast_all: Opponent trailing offensive plays/game, top vs bottom tercile (pace-up spillover) -- effect -0.09 [-0.28,+0.10], q=0.47, nE=11,456.
- A14_opp_db2_out_qb: 2+ opponent DBs officially Out/Doubtful pregame (factor_frame flag) -- effect +0.65 [-0.11,+1.37], q=0.14, nE=429.
- A15_opp_db2_out_wr: 2+ opponent DBs officially Out/Doubtful pregame -- effect +0.30 [-0.05,+0.64], q=0.14, nE=1,391.
- A16_opp_front2_out_rb: 2+ opponent front-7 defenders officially Out/Doubtful pregame -- effect +0.11 [-0.35,+0.58], q=0.66, nE=1,060.
- A17_opp_man_rate_wr: Opponent trailing man-coverage share (participation MAN/(MAN+ZONE)), top vs bottom tercile -- effect +0.08 [-0.16,+0.30], q=0.57, nE=4,778.
- B03_closed_roof_qb_passyds: Roof dome/closed vs outdoors/open, QB passing yards vs trailing -- effect +6.58 [+0.75,+12.67], q=0.05, nE=1,308.
- B05_travel_1500_all: Road great-circle travel >=1500mi vs shorter road trips; international excluded -- effect +0.22 [-0.12,+0.57], q=0.27, nE=3,410.
- B06_tz2_all: Road 2+ time zones crossed vs 0-1; international excluded -- effect +0.26 [-0.04,+0.58], q=0.14, nE=4,096.
- B07_body_clock_all: Pacific-home team on road at 13:xx ET kickoff vs other road games -- effect +0.24 [-0.34,+0.78], q=0.48, nE=1,122.
- B08_international_all: International site (London/Mexico City/Munich/Frankfurt/Sao Paulo/Dublin/Madrid/Berlin) -- effect -0.10 [-0.71,+0.46], q=0.74, nE=397.
- B09_short_rest_all: Rest <=5 days vs regular 6-11 day rest (post-bye excluded from control) -- effect +0.44 [+0.03,+0.84], q=0.06, nE=2,098.
- B11_post_bye_wr_recyds: Post-bye WR receiving yards vs trailing (fantasy-space re-derivation of prop survivor) -- effect -1.86 [-4.28,+0.48], q=0.14, nE=861.
- B12_after_ot_all: Team played overtime last week (fatigue carryover) -- effect -0.31 [-0.62,+0.03], q=0.12, nE=1,691.
- C06_primetime_all: Primetime kickoff (feature-frame definition) vs day games -- effect -0.08 [-0.33,+0.16], q=0.56, nE=7,084.
- C07_div_rematch_all: Second divisional meeting vs first meeting (population: division games) -- effect +0.13 [-0.24,+0.48], q=0.55, nE=6,215.
- C08_ref_pace_team: Referee (crew chief) trailing combined plays/game, top vs bottom tercile -> team plays vs trailing -- effect +0.13 [-0.41,+0.68], q=0.67, nE=1,243.
- C09_fourth_aggr_team: Team trailing 4th-and-<=2 go rate (neutral situations), top vs bottom tercile -> team plays vs trailing -- effect +0.64 [-0.22,+1.48], q=0.19, nE=1,216.
- C10_new_hc_all: Head coach differs from team's final game of prior season (2020+) -- effect -0.13 [-0.34,+0.07], q=0.27, nE=6,290.

## Market data inventory (verified, family D ground truth)

- **historical_lines(base)**: 1,390 rows, seasons 2019-2023; open/close: NO; multi-book: NO; spread nonnull 100.0%, total nonnull 100.0%; line cols: away_moneyline, away_spread_odds, home_moneyline, home_spread_odds, over_odds, spread_line, total, total_line, under_odds.
- **lines_extra**: 842 rows, seasons 2024-2026; open/close: NO; multi-book: NO; spread nonnull 76.7%, total nonnull 76.7%; line cols: away_moneyline, home_moneyline, spread_line, total, total_line.
- **fantasy/schedules**: 1,960 rows, seasons 2019-2025; open/close: NO; multi-book: NO; spread nonnull 100.0%, total nonnull 100.0%; line cols: away_moneyline, away_spread_odds, home_moneyline, home_spread_odds, over_odds, spread_line, total, total_line, under_odds.
- Consequence: line-movement and cross-book-disagreement factors are NOT fittable from local data (single closing snapshot, one implicit book). Registered as `proposed` with acquisition notes. Closing spread/total/implied ARE stamped pregame and usable as projection features (leakage-adjacent for live use).

## Observed-weather probes (research_only, leakage_proxy -- NOT in battery)

- wind15_qb_dfp: effect -1.49 [-2.60,-0.35], nE=315/nC=2,482. Observed in-game weather, not a pregame forecast: research_only(leakage_proxy).
- cold32_pooled_dfp: effect -0.28 [-0.64,+0.13], nE=1,229/nC=18,517. Observed in-game weather, not a pregame forecast: research_only(leakage_proxy).

## PROE / playcall prototype (E-family, MAE evaluation -- NOT in battery)

- Column coverage: pass_oe/xpass nonnull by season {'2019': 0.783, '2020': 0.785, '2021': 0.7876, '2022': 0.7862, '2023': 0.7847, '2024': 0.7819, '2025': 0.7818}; attempts-convention gap pbp-vs-player-sum 0.19 att
- team pass_att / B0_trailing_ewm4: MAE 6.4667 (bias +0.073) [2023: 6.307, 2024: 6.579, 2025: 6.514]
- team pass_att / B1_current_sim_node: MAE 6.7613 (bias +0.091) [2023: 6.725, 2024: 6.765, 2025: 6.793]
- team pass_att / ridge_proe: MAE 6.0076 (bias +0.045) [2023: 5.928, 2024: 6.033, 2025: 6.062]
- team dropbacks / B0_trailing: MAE 6.9083 (bias +0.134) []
- team dropbacks / ridge_proe: MAE 6.7636 (bias -0.154) []
- team plays / B0_trailing: MAE 7.1813 (bias +0.247) []
- team plays / ridge_proe: MAE 7.0370 (bias +0.494) []
- QB translation (n=1,560 outer QB rows; fallback rows 0):
  - attempts: before bias +2.303 MAE 8.261 -> after bias +2.215 MAE 7.454 (production sim reference bias +1.63)
  - pass yards: before bias +16.690 MAE 70.826 -> after bias +16.459 MAE 68.463 (production sim reference bias +21.5)
- Verdict: PROE volume node beats both baselines on team pass attempts in every test season and shrinks the QB attempts bias without hurting MAE; promote to a full-simulation ablation (role-shock harness) before shadow. Registry status: research_only.

## Caveats and missing cohorts

- FTN-based tests (A09/A10) train on 2022 only for the season-forward split; treat their forward stability with extra caution.
- Man/zone participation tags cover 38-50% of plays depending on season; A17 trailing man-share uses tagged plays only (games with <5 tagged plays dropped).
- C10 new-HC is 2020-2025 (no 2018 tape for 2019); B04 altitude excludes DEN home rows from both arms; B05/B06 exclude international rows; B07 body-clock is the folklore 1pm-ET-west-coast cohort.
- Referee crews tested at TEAM-VOLUME level only (C08 plays/game). DPI-rate pathway needs penalty-level pbp (registered proposed). Officials.parquet confirms crew assignments but adds no rate columns here.
- Matchup interactions (deep-ball QB x explosive-allowed, man-beater WR x man-share, pressure-sensitive QB x pressure D) are LOGGED AS HYPOTHESES for the interaction wave; mains estimated here, interactions deliberately not fit.
- Gate rejections do NOT retract battery findings: effects are measured vs the player's TRAILING baseline, while the gate asks for improvement over the trained ensemble, which already consumes spread/total/implied/roof/surface as features and evidently prices most of these (exposed-row model residuals are near zero where the trailing-space effect is large). 'rejected' at the gate = no post-hoc adjustment on top of the current model.
- Sub-threshold certain improvements: C04 (-0.0273), C05 (-0.0123), C01 (-0.0092), A10 (-0.0031), B04 (-0.0022) all had MAE-delta CIs entirely below zero but means above the -0.05 materiality bar, so they are rejected per the predeclared gate. A JOINT capped game-script adjustment (C01+C04+C05 together, non-overlapping rows) is logged as a hypothesis for the interaction/integration wave -- deliberately NOT fit here to respect predeclaration.

STATUS: battery complete; 18 survivor(s); 3 unstable-forward; 19 nulls; 0 empty cohorts.
