# Fantasy engine premortem and release gates

Assume the engine fails in the coming season. These are the likely causes and
the controls already built into the base.

| Failure | Detection | Prevention / response |
|---|---|---|
| Same-week result enters a feature | Outcome perturbation and feature-name contract | Shift before every rolling calculation; cutoff refits |
| Bye/unmatched roster week becomes a fake zero | Schedule-match gate and perturbation audit | Exclude from eligibility and mask from all rolling history |
| Bench zeroes make MAE look excellent | Report roster denominator and prior-only eligibility separately | Evaluate the decision pool; never filter on current participation |
| Closing lines/weather presented as Wednesday data | Snapshot metadata review | Label backtest as near-kickoff; archive real Wednesday snapshots going forward |
| New team/rookie has no history | Missingness and cold-start cohorts | Position/draft priors, team/role change points, partial pooling |
| Starter scratch redistributes role | Large role-shock error and coverage failure | Availability hurdle and correlated teammate share redistribution |
| TD variance dominates misses | TD/no-TD regime report | Simulate discrete team TDs and widen outcome distributions |
| Ensemble chases one season | Stack-weight instability | Multi-season OOF weights, L2 shrinkage, 65% family cap |
| Neural model adds complexity but no value | Family OOF metrics/weight approaches zero | Keep as optional weak learner; do not force equal weight |
| Rare quirky split becomes a headline | Raw/effective `n`, posterior interval, OOS ablation | Hierarchical shrinkage and multiple-testing ledger |
| Interval says 80% but covers 65% | Position/regime coverage report | Conformal residuals plus role/injury adaptive width; block release |
| Current roster cache silently stale | Manifest season/hash and carried-roster flag | Refresh mutable season; lower confidence until official snapshot |
| Trade calculator overvalues bench depth | Compare package sums vs simulated lineups | Optimize starters/flex in every sample and report probability of improvement |
| Same random seed creates fake ROS correlation | Cross-week sample audit | Deterministically permute weekly samples during aggregation |
| Partial slate owns all team production | Team component reconciliation | Explicit “other roster” target/carry/TD bucket |
| Model artifact cannot reproduce output | Model card/version/hash and deterministic seeds | Persist all learners/config; CI executes data-independent contracts |

## Automated release gate

The adversarial report fails when:

- nominal 80% coverage is outside 70–88%;
- paired week-block bootstrap improvement over a declared baseline has a
  nonpositive 95% lower bound; or
- the actual-on-predicted calibration slope leaves 0.80–1.20.

Passing those gates means the build is internally eligible for use. It does not
mean “perfect,” “guaranteed,” or better than every commercial model.

## Operational checklist

1. Fetch and validate current-season source coverage.
2. Materialize exactly one target-week roster snapshot.
3. Exclude the target week from all fitting.
4. Fit and save the model card and out-of-fold diagnostics.
5. Project all relevant roster players; hard-zero official inactive/out.
6. Simulate game/team/player events and retain raw component reconciliation.
7. Verify output count, finite points, interval order, and deterministic rerun.
8. Publish JSON/dashboard/model as one versioned run artifact.
9. Append actuals after games; rerun regime and release-gate audit.
