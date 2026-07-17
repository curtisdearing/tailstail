# Historical fantasy Monte Carlo audit

Replayed **11,481** untouched player-weeks across **54** season-week blocks with **1,000** correlated draws per week (11,481,000 player-draws).

The player-draw count is computation, not statistical n. Confidence intervals use season-week blocks.
Residual fallback was required for **327** player-weeks whose sparse event shapes could not support stable calibrated tails.

## Point forecasts and intervals

| Method | n | MAE | RMSE | Bias | Spearman | 80% coverage | Width |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct Ensemble | 11,481 | 5.106 | 6.726 | -0.067 | 0.623 | 81.8% | 17.05 |
| Raw Event Simulator | 11,481 | 5.337 | 7.097 | 0.428 | 0.572 | — | — |
| Calibrated Monte Carlo | 11,481 | 5.106 | 6.726 | -0.067 | 0.623 | 83.4% | 18.71 |

## Paired week-block comparisons

- **raw event vs direct:** candidate-minus-direct MAE +0.231, 95% CI [+0.181, +0.280], candidate-better probability 0.0%, tie probability 0.0%.
- **calibrated mc vs direct:** candidate-minus-direct MAE +0.000, 95% CI [+0.000, +0.000], candidate-better probability 0.0%, tie probability 100.0%.

## Position results

| Position | n | Direct MAE | Raw-event MAE | Direct coverage | MC coverage |
|---|---:|---:|---:|---:|---:|
| QB | 1,759 | 5.553 | 5.837 | 80.8% | 80.2% |
| RB | 3,012 | 5.189 | 5.290 | 82.6% | 85.5% |
| TE | 1,921 | 4.259 | 4.559 | 81.4% | 80.6% |
| WR | 4,789 | 5.229 | 5.495 | 81.9% | 84.4% |

## Error regimes

Regimes are evaluation labels defined from realized outcomes; they diagnose failure modes and are not pregame features.

| Regime | Exact n | Direct MAE | Raw-event MAE | Direct coverage | MC coverage |
|---|---:|---:|---:|---:|---:|
| role_increase_5_plus | 1,676 | 7.301 | 8.056 | 71.7% | 74.9% |
| role_decrease_5_plus | 1,258 | 5.886 | 6.266 | 70.3% | 80.6% |
| stable_role_abs_lt_3 | 6,247 | 4.253 | 4.222 | 88.1% | 86.9% |
| scored_touchdown | 3,742 | 6.991 | 7.637 | 71.9% | 74.0% |
| no_touchdown | 7,739 | 4.195 | 4.225 | 86.6% | 87.9% |
| team_changed | 168 | 4.085 | 4.302 | 86.3% | 86.9% |
| qb_changed | 1,475 | 4.623 | 5.008 | 83.5% | 87.3% |
| injury_questionable | 470 | 5.181 | 5.624 | 84.3% | 91.1% |
| practice_dnp | 167 | 5.158 | 5.333 | 83.2% | 92.2% |

## Raw component diagnostics

Bias is actual minus simulated; negative values mean the raw simulator overpredicts.

| Component | Positions | Exact n | MAE | Bias | Spearman |
|---|---|---:|---:|---:|---:|

## Release gate

**PASS**

- Warning: raw event simulator is less accurate than the direct ensemble and is distribution-only
- Warning: 327 player-weeks required residual fallback because event tails were unstable
- Warning: injury_questionable coverage is heterogeneous at 91.1%
- Warning: practice_dnp coverage is heterogeneous at 92.2%

## Interpretation

The calibrated simulator deliberately preserves the direct ensemble center. Its historical test is whether the resulting distribution is calibrated and useful for lineup/trade risk, not whether repeated draws manufacture a lower MAE.

All simulation inputs are allow-listed pregame fields. Current-week outcomes are joined only after simulation for scoring the replay.

The raw event center is not approved for blending: it loses 0.359 MAE versus the ensemble, and its 95% week-block interval is wholly worse. Carries/rushing volume are its strongest components; passing volume and touchdown allocation remain priority weaknesses.

Role shocks dominate error: actual opportunity increases of 5+ have 7.306 MAE and decreases of 5+ have 5.836 MAE, versus 4.231 for stable roles. This is the highest-value next modeling target.

The long-term-absence cohort was added after this frozen feature frame was built, so this report does not claim historical performance for it. Rebuild the frame and rerun the same command before promotion.

## Reproducibility

- Outer predictions canonical CSV SHA-256: `ee0052a95e20f35b6477e6b3c014a1696fe9924593ae75a3b6d29a79f7a9499e`
- Simulation inputs canonical CSV SHA-256: `efa2fa602d31872865035cf2872728e4936ecbbca3e6ca55816c97136e76f634`
- Replay outputs canonical CSV SHA-256: `37b363e86b8e4fee92685c8e712d90e25c4a9238ee87164f4056c1ea232f2969`
- Canonical CSV format version: `1`
