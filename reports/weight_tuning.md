# Composite-weight tuning — walk-forward, 2019–2025

**Leans, not locks.** walk_forward rows are OUT-OF-SAMPLE (config chosen on prior seasons only); pooled_top10 is in-sample across 2019-2025 and shown for transparency, with ship_for_2026 = pooled argmax. Directional hit rate at synthetic lines; breakeven proxy 52.38%. 1-800-GAMBLER.

## Out-of-sample: each season scored with weights chosen ONLY from prior seasons

| Season | chosen on train | train hit | OOS hit (n) | old-default OOS |
|---|---|---|---|---|
| 2020 | conf 0.8, z_cap 1.5, lcm 0.6, core4 | 59.5% | **57.5%** (1280) | 58.4% |
| 2021 | conf 0.8, z_cap 2.0, lcm 0.8, all | 58.6% | **58.9%** (1360) | 58.9% |
| 2022 | conf 0.8, z_cap 2.0, lcm 0.8, all | 58.7% | **59.0%** (1355) | 58.8% |
| 2023 | conf 0.8, z_cap 1.5, lcm 0.8, all | 58.8% | **59.5%** (1360) | 58.8% |
| 2024 | conf 0.8, z_cap 1.5, lcm 0.8, all | 59.0% | **59.3%** (1360) | 59.2% |
| 2025 | conf 0.8, z_cap 1.5, lcm 0.8, all | 59.0% | **57.1%** (1360) | 56.5% |

## Pooled 2019–2025 (in-sample, for transparency)

| conf_share | z_cap | low_conf_mult | markets | n | hit |
|---|---|---|---|---|---|
| 0.8 | 1.5 | 0.6 | core4 | 9115 | 58.8% |
| 0.8 | 1.5 | 0.8 | core4 | 9115 | 58.8% |
| 0.8 | 1.5 | 1.0 | core4 | 9115 | 58.8% |
| 0.8 | 1.5 | 0.8 | all | 9115 | 58.7% |
| 0.8 | 1.5 | 0.6 | all | 9115 | 58.7% |
| 0.8 | 1.5 | 1.0 | all | 9115 | 58.5% |
| 0.8 | 2.0 | 0.6 | core4 | 9115 | 58.5% |
| 0.8 | 2.0 | 0.8 | core4 | 9115 | 58.5% |
| 0.8 | 2.0 | 1.0 | core4 | 9115 | 58.5% |
| 0.8 | 2.0 | 0.8 | all | 9115 | 58.5% |

**Shipped for 2026:** conf_share 0.8, z_cap 1.5, low_conf_mult 0.6, markets=core4 (pooled 58.8% over n=9115). Live weights stay edge-dominant (0.5) whenever a real price exists; this tunes the other half and the no-market ordering.
