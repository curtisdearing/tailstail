# ML improvement test — RF & gradient boosting vs tuned composite

**Leans, not locks.** Walk-forward by season; identical candidate pools and
top-5 selection protocol; graded at synthetic trailing-mean lines (NOT real
prices; breakeven proxy 52.38% at -110). Log-loss is the gradient-
descent objective the GBDT minimizes. 1-800-GAMBLER.

| Season | tuned composite | GBDT hit (units) | RF hit (units) | GBDT log-loss | GBDT AUC |
|---|---|---|---|---|---|
| 2021 | 58.8% (+167.3u) | **64.8%** (+323.8u) | 69.0% (+432.6u) | 0.63928 | 0.6253 |
| 2022 | 59.4% (+181.8u) | **63.5%** (+286.8u) | 65.5% (+338.4u) | 0.63458 | 0.6221 |
| 2023 | 59.5% (+184.4u) | **65.0%** (+327.6u) | 66.1% (+356.3u) | 0.62563 | 0.6353 |
| 2024 | 59.3% (+178.7u) | **63.2%** (+279.9u) | 63.8% (+297.1u) | 0.63712 | 0.6335 |
| 2025 | 57.1% (+121.5u) | **67.1%** (+383.0u) | 68.2% (+411.6u) | 0.62942 | 0.6291 |

Weekly-retrain GBDT, 2025: **66.5%** (+365.8u, top-1 69.5%, avg log-loss 0.62643)
