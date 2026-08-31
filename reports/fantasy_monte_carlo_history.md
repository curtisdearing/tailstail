# Historical fantasy Monte Carlo audit

Replayed **11,482** untouched player-weeks across **54** season-week blocks with **1,000** correlated draws per week (11,482,000 player-draws).

The player-draw count is computation, not statistical n. Confidence intervals use season-week blocks.
Residual fallback was required for **254** player-weeks whose sparse event shapes could not support stable calibrated tails.

## Point forecasts and intervals

| Method | n | MAE | RMSE | Bias | Spearman | 80% coverage | Width |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct Ensemble | 11,482 | 5.094 | 6.721 | -0.017 | 0.624 | 82.3% | 17.23 |
| Raw Event Simulator | 11,482 | 5.455 | 7.232 | 0.408 | 0.545 | — | — |
| Calibrated Monte Carlo | 11,482 | 5.094 | 6.721 | -0.017 | 0.624 | 82.4% | 17.27 |

## Paired week-block comparisons

- **raw event vs direct:** candidate-minus-direct MAE +0.361, 95% CI [+0.297, +0.425], candidate-better probability 0.0%, tie probability 0.0%.
- **calibrated mc vs direct:** candidate-minus-direct MAE +0.000, 95% CI [+0.000, +0.000], candidate-better probability 0.0%, tie probability 100.0%.

## Position results

| Position | n | Direct MAE | Raw-event MAE | Direct coverage | MC coverage |
|---|---:|---:|---:|---:|---:|
| QB | 1,759 | 5.557 | 6.067 | 81.0% | 80.7% |
| RB | 3,012 | 5.178 | 5.417 | 82.6% | 82.7% |
| TE | 1,921 | 4.268 | 4.659 | 82.2% | 81.1% |
| WR | 4,790 | 5.203 | 5.573 | 82.7% | 83.4% |

## Error regimes

Regimes are evaluation labels defined from realized outcomes; they diagnose failure modes and are not pregame features.

| Regime | Exact n | Direct MAE | Raw-event MAE | Direct coverage | MC coverage |
|---|---:|---:|---:|---:|---:|
| role_increase_5_plus | 1,677 | 7.331 | 8.256 | 71.5% | 72.2% |
| role_decrease_5_plus | 1,259 | 5.808 | 6.896 | 70.8% | 71.6% |
| stable_role_abs_lt_3 | 6,244 | 4.234 | 4.221 | 88.8% | 88.3% |
| scored_touchdown | 3,743 | 7.022 | 7.718 | 72.2% | 72.4% |
| no_touchdown | 7,739 | 4.162 | 4.360 | 87.3% | 87.2% |
| team_changed | 168 | 3.967 | 4.735 | 87.5% | 89.9% |
| qb_changed | 1,475 | 4.582 | 5.377 | 84.1% | 84.9% |
| injury_questionable | 470 | 5.186 | 5.731 | 85.7% | 90.9% |
| practice_dnp | 167 | 5.172 | 5.396 | 85.6% | 92.8% |

## Raw component diagnostics

Bias is actual minus simulated; negative values mean the raw simulator overpredicts.

| Component | Positions | Exact n | MAE | Bias | Spearman |
|---|---|---:|---:|---:|---:|
| completions | QB | 1,759 | 5.219 | -1.482 | 0.427 |
| attempts | QB | 1,759 | 7.519 | -1.632 | 0.379 |
| passing_yards | QB | 1,759 | 64.024 | -21.556 | 0.443 |
| passing_tds | QB | 1,759 | 0.865 | -0.105 | 0.250 |
| passing_interceptions | QB | 1,759 | 0.643 | -0.024 | 0.193 |
| carries | QB, RB, WR | 9,561 | 1.873 | +0.235 | 0.842 |
| rushing_yards | QB, RB, WR | 9,561 | 10.901 | +1.046 | 0.768 |
| rushing_tds | QB, RB, WR | 9,561 | 0.180 | +0.049 | 0.372 |
| targets | RB, WR, TE | 9,723 | 2.108 | +0.088 | 0.593 |
| receptions | RB, WR, TE | 9,723 | 1.618 | +0.092 | 0.522 |
| receiving_yards | RB, WR, TE | 9,723 | 21.158 | +0.777 | 0.542 |
| receiving_tds | RB, WR, TE | 9,723 | 0.290 | +0.024 | 0.236 |
| fumbles_lost | QB, RB, WR, TE | 11,482 | 0.095 | +0.008 | 0.080 |

## Release gate

**PASS**

- Warning: raw event simulator is less accurate than the direct ensemble and is distribution-only
- Warning: 254 player-weeks required residual fallback because event tails were unstable
- Warning: role_decrease_5_plus coverage is heterogeneous at 71.6%
- Warning: stable_role_abs_lt_3 coverage is heterogeneous at 88.3%
- Warning: team_changed coverage is heterogeneous at 89.9%
- Warning: injury_questionable coverage is heterogeneous at 90.9%
- Warning: practice_dnp coverage is heterogeneous at 92.8%

## Interpretation

The calibrated simulator deliberately preserves the direct ensemble center. Its historical test is whether the resulting distribution is calibrated and useful for lineup/trade risk, not whether repeated draws manufacture a lower MAE.

All simulation inputs are allow-listed pregame fields. Current-week outcomes are joined only after simulation for scoring the replay.

The raw event center is not approved for blending: it loses 0.359 MAE versus the ensemble, and its 95% week-block interval is wholly worse. Carries/rushing volume are its strongest components; passing volume and touchdown allocation remain priority weaknesses.

Role shocks dominate error: actual opportunity increases of 5+ have 7.306 MAE and decreases of 5+ have 5.836 MAE, versus 4.231 for stable roles. This is the highest-value next modeling target.

The long-term-absence cohort was added after this frozen feature frame was built, so this report does not claim historical performance for it. Rebuild the frame and rerun the same command before promotion.

## Reproducibility

- Outer predictions canonical CSV SHA-256: `9cbbf5cbc8ed27936f7c28655b4ad67c0967a39e9e78d43cde89f69d0671999d`
- Simulation inputs canonical CSV SHA-256: `513f10f3d96af0da8f51d38697b001922915abcf5495d61940237f574510aa69`
- Replay outputs canonical CSV SHA-256: `c63cfea705b4c3306fb575ff096e7d85ebf7a928c5e3a2d69b9fcbb06ab63399`
- Canonical CSV format version: `1`
