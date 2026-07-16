# Fantasy model search — season-forward result

Scope: full-PPR QB/RB/WR/TE decision-pool rows, 2019–2025 source history,
untouched 2023–2025 outer seasons, all features known before the evaluated game.
The current evaluation contains 11,481 player-weeks. Stack selection and
conformal residuals are learned only inside each outer cutoff.

| Result | Value |
|---|---:|
| Ensemble MAE | 5.091 |
| Ensemble RMSE | 6.718 |
| Ensemble Spearman rank correlation | 0.625 |
| Nominal 80% interval coverage | 82.0% |
| Trailing PPR MAE | 5.692 |
| Trailing expected-points MAE | 5.595 |
| MAE gain vs trailing PPR | 0.596 (95% week-block CI 0.515–0.679) |
| MAE gain vs expected points | 0.498 (95% week-block CI 0.424–0.575) |

| Position | MAE | RMSE | Rank correlation |
|---|---:|---:|---:|
| QB | 5.562 | 7.218 | 0.543 |
| RB | 5.179 | 6.800 | 0.619 |
| WR | 5.197 | 6.850 | 0.568 |
| TE | 4.261 | 5.706 | 0.541 |

The ensemble beat trailing PPR in every one of the 54 evaluated weeks and
expected points in all 54 weeks. Actual-on-predicted calibration slope was
1.035 and the mean bias was -0.033 points.

Against the previous production ensemble on the identical rows, the admitted
regularized model improved MAE from 5.1010 to 5.0926, RMSE from 6.7320 to
6.7194, rank correlation from 0.6222 to 0.6243, and interval coverage from
81.79% to 82.04% on the 11,477 rows common to both models. It improved MAE in
30 of 54 weeks. A paired week-block bootstrap gives a two-sided 95% interval
of -0.0024 to 0.0197 for the MAE gain (6.4% probability of no gain), so this
is a small directional improvement rather than evidence of a decisive
breakthrough. Four newly eligible rows explain the difference between the
common comparison pool and current evaluation pool. See
`docs/FANTASY_PROJECTION_METHODS.md` for admitted and rejected iterations.

The pooled result still conceals the same hard failures:

| Regime | n | MAE | Bias (actual − prediction) | Coverage |
|---|---:|---:|---:|---:|
| Stable role | 6,247 | 4.231 | -0.564 | 88.6% |
| Opportunity +5 or more | 1,676 | 7.306 | +5.272 | 71.4% |
| Opportunity -5 or more | 1,258 | 5.836 | -4.416 | 70.5% |
| Scored a TD | 3,742 | 7.009 | +5.682 | 72.0% |
| No TD | 7,739 | 4.164 | -2.797 | 86.9% |

Conclusion: the combined model is the best tested center and ranking system in
this repository, but discrete role and touchdown scenarios must remain in the
Monte Carlo layer. These numbers measure this public-data decision pool; they
are not a claim of universal or 100% accuracy.
