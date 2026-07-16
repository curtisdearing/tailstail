# Historical fantasy Monte Carlo audit

Replayed **11,481** untouched player-weeks across **54** season-week blocks with **1,000** correlated draws per week (11,481,000 player-draws).

The player-draw count is computation, not statistical n. Confidence intervals use season-week blocks.
Residual fallback was required for **263** player-weeks whose sparse event shapes could not support stable calibrated tails.

## Point forecasts and intervals

| Method | n | MAE | RMSE | Bias | Spearman | 80% coverage | Width |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct Ensemble | 11,481 | 5.091 | 6.718 | -0.033 | 0.625 | 82.0% | 17.20 |
| Raw Event Simulator | 11,481 | 5.451 | 7.229 | 0.409 | 0.545 | — | — |
| Calibrated Monte Carlo | 11,481 | 5.091 | 6.718 | -0.033 | 0.625 | 82.2% | 17.24 |

## Paired week-block comparisons

- **raw event vs direct:** candidate-minus-direct MAE +0.359, 95% CI [+0.297, +0.423], candidate-better probability 0.0%, tie probability 0.0%.
- **calibrated mc vs direct:** candidate-minus-direct MAE +0.000, 95% CI [+0.000, +0.000], candidate-better probability 0.0%, tie probability 100.0%.

## Position results

| Position | n | Direct MAE | Raw-event MAE | Direct coverage | MC coverage |
|---|---:|---:|---:|---:|---:|
| QB | 1,759 | 5.562 | 6.068 | 80.9% | 81.0% |
| RB | 3,012 | 5.179 | 5.417 | 82.5% | 82.5% |
| TE | 1,921 | 4.261 | 4.659 | 81.5% | 80.7% |
| WR | 4,789 | 5.197 | 5.563 | 82.4% | 83.1% |

## Error regimes

Regimes are evaluation labels defined from realized outcomes; they diagnose failure modes and are not pregame features.

| Regime | Exact n | Direct MAE | Raw-event MAE | Direct coverage | MC coverage |
|---|---:|---:|---:|---:|---:|
| role_increase_5_plus | 1,676 | 7.306 | 8.255 | 71.4% | 72.0% |
| role_decrease_5_plus | 1,258 | 5.836 | 6.889 | 70.5% | 71.6% |
| stable_role_abs_lt_3 | 6,247 | 4.231 | 4.218 | 88.6% | 88.3% |
| scored_touchdown | 3,742 | 7.009 | 7.715 | 72.0% | 72.4% |
| no_touchdown | 7,739 | 4.164 | 4.356 | 86.9% | 87.0% |
| team_changed | 168 | 3.928 | 4.717 | 87.5% | 89.9% |
| qb_changed | 1,475 | 4.580 | 5.368 | 83.8% | 84.6% |
| injury_questionable | 470 | 5.182 | 5.727 | 85.5% | 90.9% |
| practice_dnp | 167 | 5.187 | 5.408 | 85.0% | 92.2% |

## Raw component diagnostics

Bias is actual minus simulated; negative values mean the raw simulator overpredicts.

| Component | Positions | Exact n | MAE | Bias | Spearman |
|---|---|---:|---:|---:|---:|
| completions | QB | 1,759 | 5.221 | -1.482 | 0.427 |
| attempts | QB | 1,759 | 7.520 | -1.629 | 0.379 |
| passing_yards | QB | 1,759 | 64.063 | -21.541 | 0.442 |
| passing_tds | QB | 1,759 | 0.864 | -0.104 | 0.252 |
| passing_interceptions | QB | 1,759 | 0.643 | -0.024 | 0.193 |
| carries | QB, RB, WR | 9,560 | 1.874 | +0.235 | 0.843 |
| rushing_yards | QB, RB, WR | 9,560 | 10.897 | +1.044 | 0.768 |
| rushing_tds | QB, RB, WR | 9,560 | 0.180 | +0.049 | 0.373 |
| targets | RB, WR, TE | 9,722 | 2.106 | +0.089 | 0.594 |
| receptions | RB, WR, TE | 9,722 | 1.616 | +0.092 | 0.523 |
| receiving_yards | RB, WR, TE | 9,722 | 21.134 | +0.781 | 0.543 |
| receiving_tds | RB, WR, TE | 9,722 | 0.290 | +0.024 | 0.237 |
| fumbles_lost | QB, RB, WR, TE | 11,481 | 0.095 | +0.008 | 0.081 |

## Release gate

**PASS**

- Warning: raw event simulator is less accurate than the direct ensemble and is distribution-only
- Warning: 263 player-weeks required residual fallback because event tails were unstable
- Warning: role_decrease_5_plus coverage is heterogeneous at 71.6%
- Warning: stable_role_abs_lt_3 coverage is heterogeneous at 88.3%
- Warning: team_changed coverage is heterogeneous at 89.9%
- Warning: injury_questionable coverage is heterogeneous at 90.9%
- Warning: practice_dnp coverage is heterogeneous at 92.2%

## Interpretation

The calibrated simulator deliberately preserves the direct ensemble center. Its historical test is whether the resulting distribution is calibrated and useful for lineup/trade risk, not whether repeated draws manufacture a lower MAE.

All simulation inputs are allow-listed pregame fields. Current-week outcomes are joined only after simulation for scoring the replay.

The raw event center is not approved for blending: it loses 0.359 MAE versus the ensemble, and its 95% week-block interval is wholly worse. Carries/rushing volume are its strongest components; passing volume and touchdown allocation remain priority weaknesses.

Role shocks dominate error: actual opportunity increases of 5+ have 7.306 MAE and decreases of 5+ have 5.836 MAE, versus 4.231 for stable roles. This is the highest-value next modeling target.

The long-term-absence cohort was added after this frozen feature frame was built, so this report does not claim historical performance for it. Rebuild the frame and rerun the same command before promotion.

## Reproducibility

- Outer predictions canonical CSV SHA-256: `1f47873954a3ef6dd1e4b89e8d4f4d59afb6eaa09b59279bf6cc0062bdc59310`
- Simulation inputs canonical CSV SHA-256: `1009b20a3c78e8f1ff39941e1b33f1aabd7df3f6a1a090008751055791997d84`
- Replay outputs canonical CSV SHA-256: `0de4c52191077df0ba48c24dfa74707e5b2c0d1a875e563636a32c96bbd0617d`
- Canonical CSV format version: `1`
