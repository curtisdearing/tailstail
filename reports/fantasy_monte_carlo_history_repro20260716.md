# Historical fantasy Monte Carlo audit

Replayed **11,481** untouched player-weeks across **54** season-week blocks with **1,000** correlated draws per week (11,481,000 player-draws).

The player-draw count is computation, not statistical n. Confidence intervals use season-week blocks.
Residual fallback was required for **264** player-weeks whose sparse event shapes could not support stable calibrated tails.

## Point forecasts and intervals

| Method | n | MAE | RMSE | Bias | Spearman | 80% coverage | Width |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct Ensemble | 11,481 | 5.106 | 6.726 | -0.067 | 0.623 | 81.8% | 17.05 |
| Raw Event Simulator | 11,481 | 5.451 | 7.229 | 0.409 | 0.545 | — | — |
| Calibrated Monte Carlo | 11,481 | 5.106 | 6.726 | -0.067 | 0.623 | 81.8% | 17.09 |

## Paired week-block comparisons

- **raw event vs direct:** candidate-minus-direct MAE +0.345, 95% CI [+0.286, +0.404], candidate-better probability 0.0%, tie probability 0.0%.
- **calibrated mc vs direct:** candidate-minus-direct MAE +0.000, 95% CI [+0.000, +0.000], candidate-better probability 0.0%, tie probability 100.0%.

## Position results

| Position | n | Direct MAE | Raw-event MAE | Direct coverage | MC coverage |
|---|---:|---:|---:|---:|---:|
| QB | 1,759 | 5.553 | 6.068 | 80.8% | 80.2% |
| RB | 3,012 | 5.189 | 5.417 | 82.6% | 82.4% |
| TE | 1,921 | 4.259 | 4.659 | 81.4% | 80.5% |
| WR | 4,789 | 5.229 | 5.563 | 81.9% | 82.5% |

## Error regimes

Regimes are evaluation labels defined from realized outcomes; they diagnose failure modes and are not pregame features.

| Regime | Exact n | Direct MAE | Raw-event MAE | Direct coverage | MC coverage |
|---|---:|---:|---:|---:|---:|
| role_increase_5_plus | 1,676 | 7.301 | 8.255 | 71.7% | 72.1% |
| role_decrease_5_plus | 1,258 | 5.886 | 6.889 | 70.3% | 71.0% |
| stable_role_abs_lt_3 | 6,247 | 4.253 | 4.218 | 88.1% | 87.7% |
| scored_touchdown | 3,742 | 6.991 | 7.715 | 71.9% | 72.0% |
| no_touchdown | 7,739 | 4.195 | 4.356 | 86.6% | 86.6% |
| team_changed | 168 | 4.085 | 4.717 | 86.3% | 86.9% |
| qb_changed | 1,475 | 4.623 | 5.368 | 83.5% | 84.5% |
| injury_questionable | 470 | 5.181 | 5.727 | 84.3% | 90.6% |
| practice_dnp | 167 | 5.158 | 5.408 | 83.2% | 92.8% |

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
- Warning: 264 player-weeks required residual fallback because event tails were unstable
- Warning: role_decrease_5_plus coverage is heterogeneous at 71.0%
- Warning: scored_touchdown coverage is heterogeneous at 72.0%
- Warning: injury_questionable coverage is heterogeneous at 90.6%
- Warning: practice_dnp coverage is heterogeneous at 92.8%

## Interpretation

The calibrated simulator deliberately preserves the direct ensemble center. Its historical test is whether the resulting distribution is calibrated and useful for lineup/trade risk, not whether repeated draws manufacture a lower MAE.

All simulation inputs are allow-listed pregame fields. Current-week outcomes are joined only after simulation for scoring the replay.

The raw event center is not approved for blending: it loses 0.359 MAE versus the ensemble, and its 95% week-block interval is wholly worse. Carries/rushing volume are its strongest components; passing volume and touchdown allocation remain priority weaknesses.

Role shocks dominate error: actual opportunity increases of 5+ have 7.306 MAE and decreases of 5+ have 5.836 MAE, versus 4.231 for stable roles. This is the highest-value next modeling target.

The long-term-absence cohort was added after this frozen feature frame was built, so this report does not claim historical performance for it. Rebuild the frame and rerun the same command before promotion.

## Reproducibility

- Outer predictions canonical CSV SHA-256: `ee0052a95e20f35b6477e6b3c014a1696fe9924593ae75a3b6d29a79f7a9499e`
- Simulation inputs canonical CSV SHA-256: `43ce071074b545b4fa59de75a02817fe270294a699459d3c9c8df5e92c6b1ffe`
- Replay outputs canonical CSV SHA-256: `dff2793107fef3ada228f2a819d64c6b98c1dc770c851466f528fdb68f152bdc`
- Canonical CSV format version: `1`
