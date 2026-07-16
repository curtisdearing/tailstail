# Fantasy projection engine — contrarian engineering and data review

## Verdict

The direct fantasy ensemble is suitable as the current production center; the
raw event simulator is not. On every available untouched outer-test prediction
from 2023–2025, the ensemble produced 5.091 MAE across 11,481 player-weeks. The
raw football-event center produced 5.451 MAE. A paired season-week bootstrap
puts raw-minus-direct MAE at +0.359, 95% CI [+0.297, +0.423].

Monte Carlo remains useful for correlation, alternate scoring, tails, and
lineup/trade risk only after calibration to the validated ensemble. Its center
is intentionally identical to the ensemble and is not advertised as a second
accuracy gain. See `reports/fantasy_monte_carlo_history.md` for complete counts.

## Findings from the iterative reviewer loop

| Severity | Finding | Evidence | Resolution |
|---|---|---|---|
| Critical | Historical outcomes could have reached the simulator if it later began reading an unshifted frame column | Replay initially passed the full evaluation frame | Simulation receives an explicit pregame-only column allow-list |
| Critical | Mean calibration drifted for low projections after floor clipping | Maximum drift was 0.813 points | Replaced repeated correction with an exact monotonic clipped-mean solve; final max drift is 2.7e-12 |
| High | Skewed event draws silently narrowed conformal intervals | MC width 16.01 vs direct 17.20; role-decrease coverage 68.4% | Match p10–p90 width directly; final MC width 17.24 and role-decrease coverage 71.6% |
| High | Sparse backup-player events could be stretched into absurd tails | Fixture produced 44-point SD while appearing width-calibrated | Detect unstable scale, use explicit residual fallback, and expose the fallback flag; 263/11,481 rows use it |
| High | Component columns collided with existing expected-stat features | Completions and receptions disappeared from the first component report | Namespace all replay outputs `mc_expected_*`; all 13 components are now graded |
| High | Fablesfable could treat hash-verified draws as accuracy-verified probabilities | Contract had integrity metadata but no validation state | Added `approved/research_only/unvalidated` provenance; betting probability refuses unapproved markets by default |
| Medium | Parquet bytes were treated as reproducibility identity | Equal tables can differ by PyArrow version | Canonical typed CSV hashes identify content; Parquet SHA-256 is integrity-only |
| Medium | JSON Schema did not constrain component fields | Python validation was stronger than the published schema | Schema now requires every component and every summary statistic |

## What matters most in the historical data

The strongest error divider is role realization, not an obscure narrative tag.

| Realized regime | Exact n | Ensemble MAE | MC 80% coverage |
|---|---:|---:|---:|
| Stable role: opportunity change under 3 | 6,247 | 4.231 | 88.3% |
| Opportunity increase of 5+ | 1,676 | 7.306 | 72.0% |
| Opportunity decrease of 5+ | 1,258 | 5.836 | 71.6% |
| Scored at least one touchdown | 3,742 | 7.009 | 72.4% |
| No touchdown | 7,739 | 4.164 | 87.0% |

These are outcome-defined audit cohorts, not legal pregame features. The next
model must predict role-shock probability from pregame roster, injury, depth,
transaction, practice, and teammate-vacancy evidence. It must not use the
realized opportunity change itself.

The raw simulator's best components are carries (Spearman 0.843) and rushing
yards (0.768). Targets are moderate (0.594). Passing attempts are weak (0.379)
and overprojected by 1.63 attempts on average; passing yards are overprojected
by 21.54 yards. Touchdown component ranks are weak (0.237 receiving, 0.373
rushing, 0.252 passing), confirming that TD allocation needs a separate
hierarchical hazard model.

The scored-touchdown cohort is 3,742/11,481 player-weeks (32.6% of this
model-eligible evaluation pool), not 2%. A roughly 2–4% quantity can be a
per-opportunity touchdown rate; it must never be described as the probability
that a skill player scores in a week.

## Hardened implementation plan

### 1. Rebuild the historical clock and roster state

- Rebuild the feature frame with the new long-term incumbent-vacancy cohort;
  keep it separate from short-notice Out/Doubtful.
- Materialize weekly team rosters, transactions, reserve/IR, practice status,
  official inactives, and depth rank as immutable information-as-of snapshots.
- Model roster spells rather than assuming the current roster explains prior
  depth. A player on another team cannot remain a replacement candidate.
- Exit: every feature has source time, availability time, missingness, and a
  canonical content hash; leakage tests fail on any current-week outcome field.

### 2. Split participation from conditional workload

- Bayesian availability model: probability active and probability entering
  the game.
- Conditional role model: team pass/rush volume, route participation, target
  share, carry share, red-zone share, and starter probability given active.
- Hierarchical shrinkage by player, position, coordinator, team, and season;
  new players inherit priors and update as weekly evidence arrives.
- Treat teammate absences as vacated-role allocation, with different early
  replacement and established-replacement phases.
- Exit: season-forward role-increase/decrease probability calibration and lower
  lineup regret, not merely better in-sample fit.

### 3. Repair event components before blending them

- Passing volume: calibrate starter probability and team attempts; current raw
  attempts and passing yards are materially high.
- Receiving: jointly model routes, targets per route, catch rate, and yards per
  target rather than one target-share multiplier.
- Rushing: preserve the strong carry signal but add goal-line share, scramble
  vs designed-QB carries, offensive line, box count, and game-script state.
- Touchdowns: team scoring opportunity -> play type -> player allocation, with
  red-zone/inside-10 usage and heavy partial pooling.
- Exit: each component clears position-specific bias, rank, calibration, and
  season-forward ablation gates. The raw event center is blended only if a
  paired week-block confidence interval beats zero.

### 4. Admit contextual and obscure factors safely

- High priority: depth change, teammate vacancy, expected routes/snaps,
  offensive-line injuries, QB change, coordinator tendency, opponent coverage
  and front, implied team scoring, pace, spread/game script, rest/travel.
- Conditional context: wind/precipitation/temperature, roof, surface, stadium,
  altitude, primetime, referee pace/penalty tendencies.
- Chemistry: estimate target rate over expectation, first-read share, catch
  rate over expectation, yards/route, EPA/target, and red-zone share for a
  QB-receiver pair, all strictly prior and shrunk toward player/team priors.
- Research-only quirks: birthday, revenge, contract incentives, individual
  stadium history, and referee-player interactions. They remain labels until a
  predeclared hierarchical posterior and untouched-season ablation survive
  multiplicity control.
- Search interactions with tree models, then estimate the discovered effect in
  a separate Bayesian model and untouched season. Never search and claim on the
  same player-weeks.

### 5. Optimize the ensemble for decisions and distributions

- Keep Bayesian ridge, gradient boosting, random forest, and gradient-descent
  learners as diverse candidates; learn weights only from inner
  season-forward folds.
- Add proper distribution objectives (CRPS/log score and interval calibration)
  beside MAE/RMSE/Spearman.
- Report global, position, season, role-shock, TD, injury, and cold-start
  metrics with exact n. Bootstrap full season-week blocks.
- Keep champion/challenger artifacts. A challenger promotes only on frozen
  rules; no manual cherry-picking of the best subgroup.

### 6. Season, lineup, waiver, and trade simulation

- Weekly samples preserve game and teammate correlation and can be rescored for
  any league.
- Rest-of-season samples include schedule, byes, injury/role state transitions,
  replacement value, lineup constraints, and playoff-week weighting.
- Trade value is expected lineup-point delta plus risk/tail effects, not the sum
  of isolated player means.
- Grade start/sit regret, waiver value captured, trade delta, and calibration
  prospectively as rosters change.

### 7. Keep the two products independent

- Tailstail owns fantasy scoring, rosters, decisions, state, schedule, and
  dashboard.
- Fablesfable owns sportsbook lines, prices, de-vigging, CLV, kill checks, and
  betting delivery.
- The shared contract contains football components plus validation provenance.
  Fablesfable refuses a probability unless that exact market is `approved`.
- Extract `nflvalue-core` only after this contract survives prospective use;
  consumers pin immutable releases rather than tracking another repo's `main`.

## Plan-optimization record

Rubric: statistical validity 25, leakage/reproducibility 20, architecture 15,
operational safety 15, observability 10, actionable sequencing 10, claim
discipline 5.

Score trajectory: **72 -> 84 -> 93 -> 95 -> 95**. The largest improvements
were replacing pooled/player-draw pseudo-n with week-block inference, separating
validated centers from simulation distributions, and adding an enforceable
cross-repository accuracy gate. The final plateau leaves shared-package
extraction deferred until prospective evidence exists; extracting it now would
increase coordination cost without improving projections.
