# Fantasy projection and simulation engine

This package is a separate fantasy-football consumer of the repository's NFL
data. It does not rank bets and it does not use fantasy consensus as a target.
It produces player distributions that can be rescored for PPR, half-PPR,
standard, or custom league rules and translated into weekly lineups,
rest-of-season value, waivers, and trades.

## End-to-end contract

```text
official weekly stats + rosters + schedules + snaps + injuries + expected opportunity
        ↓
roster-first player-week table (including zero-participation rows)
        ↓
strictly shifted player/team/opponent/condition history
        ↓
position-normalized, train-only-imputed model inputs
        ↓
Bayesian ridge + gradient boosting + random forest + gradient-descent MLP
        ↓
multi-season out-of-fold, constrained position-specific stack
        ↓
shared game pace → team pass/rush volume → target/carry allocation → yards/TDs
        ↓
configurable scoring → weekly/season samples → lineup and trade value
```

The direct model supplies the best-tested center. The event simulator supplies
the shape and cross-player dependence. A hurdle calibration keeps the model's
mean and uncertainty while preserving a true zero when a player is unavailable.
The simulator reports its raw event mean and the residual calibration instead
of hiding the disagreement.

## Data

The downloader uses `nflreadpy` for:

- weekly player statistics;
- weekly rosters and roster status;
- schedules, lines, stadium, referee, roof/surface, rest, starting quarterback;
- offensive snap counts;
- official injury/practice reports; and
- ffopportunity expected fantasy points.

`historical/fantasy/manifest.json` records retrieval time, row counts, file
hashes, source-library version, and cache use. Stats, rosters, and schedules are
required. The other feeds degrade through explicit missingness flags instead of
silently becoming zero. Current-season tables refresh on every scheduled run;
immutable past seasons can use the cache.

The feature frame is roster-first. A player who was on the roster but logged no
box-score event remains a zero-outcome row. A player who appears in stats but is
missing from the roster is retained and tagged. If the target-week roster has
not yet published, only the latest roster identity is carried forward and
`roster_snapshot_carried=1`; no stats or injury outcome is copied.

## Leakage rules

- Every performance feature is shifted before rolling or exponentially
  weighting.
- The current result, current expected points, current snaps, and current
  efficiency never enter `model_features()`.
- Model imputation, scaling, PCA, fitting, stack weighting, and conformal
  residuals are learned inside historical cutoffs.
- Backtests refit before each outer season.
- Eligibility uses only prior opportunity, prior expected points, draft capital,
  a matched game schedule, and known availability. It never checks whether the
  player participated in the week being predicted.
- Roster rows without a matched game are masked out of player, team, and coach
  rolling history so bye/feed-artifact zeroes cannot decay a player's role.
- Same-week outcome perturbation tests verify that same-week features do not
  change.

Historical closing lines and observed game weather represent a near-kickoff
projection. A Wednesday backtest would require archived Wednesday snapshots;
the system must not relabel this evaluation as one.

## Features

The production manifest currently exposes 120+ numeric inputs across:

- prior PPR and expected-opportunity history at 2/4/8-game decays;
- pass attempts, carries, targets, receptions, touches, shares, WOPR and snaps;
- yards, catch rate, touchdowns, EPA and CPOE efficiency;
- team pass/rush volume and expected points;
- expected TD, yard, reception, completion, interception, first-down, and
  phase-specific fantasy-point opportunity;
- short-versus-long role momentum;
- prior-only coach pace, pass rate, positional target allocation, and QB rush
  allocation;
- active-roster reconciled attempt/target/carry shares and top-down volume;
- opponent points allowed by position;
- spread, total, implied team points, rest, home/division/primetime context;
- weather, roof, surface and stadium;
- official roster/injury/practice state;
- vacated target/carry share and RB1/WR1/TE1 absence;
- age, experience, draft capital, trades/team changes, quarterback changes;
- shrunk individual stadium, referee, surface, primetime and QB-pair history; and
- within-week/position normalized comparisons.

Rare individual-condition effects use prior counts and shrink toward the
player's ordinary baseline. A one-game stadium coincidence cannot receive the
same weight as a repeated pattern.

## Models

Four compact, regularized model families are independently fit by position:

1. Bayesian ridge on scaled, variance-filtered latent components;
2. regularized histogram gradient boosting;
3. minimum-leaf random forest; and
4. a small, early-stopped gradient-descent neural network.

Stack weights are learned on up to three complete out-of-fold seasons, constrained
to be nonnegative, sum to one, and give no family more than 65%. A shrinkage
penalty prevents one lucky validation season from producing a permanent 100%
weight. The final learners then refit on all allowed history.

The compact production capacity is intentional. Expanded forests, boosting
runs, and neural networks lowered some squared errors but worsened season-
forward MAE after the richer feature set was introduced. The tested compact
capacity improved all headline metrics. The public-method audit and iteration
ledger are in [`FANTASY_PROJECTION_METHODS.md`](FANTASY_PROJECTION_METHODS.md).

Intervals use out-of-fold residual quantiles and widen for role instability and
uncertain availability. Official inactive/out status is a hard zero gate.

## Correlated simulation

Each simulation samples shared game pace and team scoring. Team pass attempts
and carries are allocated with availability-gated Dirichlet shares. Targets lead
to receptions and receiving yards; carries lead to rushing yards. Passing
touchdowns are assigned to a receiver and the active quarterback, so teammate
outcomes are not independent. An unmodeled-roster bucket prevents a partial
fantasy slate from assigning 100% of a team's offense to listed players.

Outputs include mean, median, standard deviation, P10/P25/P75/P90, availability,
and probabilities of reaching 10/15/20/25 points. Raw football components are
retained, allowing the same samples to be rescored for another league.

## Season and trade translation

Rest-of-season aggregation intentionally permutes simulation rows between weeks
when a caller reused a seed; otherwise a player's week-one lucky draw would be
artificially correlated with every later week. Lineups are optimized inside
each simulation according to `LineupRules`. Trade packages are evaluated by the
change in startable lineup points, including replacement and flex effects—not
by adding player means.

Trade output includes mean/median/P10/P90 lineup delta and the probability the
trade improves the roster. `value_over_replacement` derives position-specific
replacement levels from league size and lineup settings.

## Commands

```bash
# Official data and reproducible cache manifest
python -m nflvalue.fantasy.cli fetch --seasons 2019:2026

# Feature frame and data-quality report
python -m nflvalue.fantasy.cli build

# Untouched outer seasons and adversarial report
python -m nflvalue.fantasy.cli backtest --test-seasons 2023:2025

# Production ensemble
python -m nflvalue.fantasy.cli train

# One weekly snapshot and correlated simulation
python -m nflvalue.fantasy.cli project --season 2026 --week 1
python -m nflvalue.fantasy.cli simulate --season 2026 --week 1 --simulations 10000

# Automatic next scheduled week
python scripts/fantasy_weekly.py
```

The scheduled GitHub workflow refreshes after Wednesday practice reports and
Sunday morning, verifies nonempty output, and uploads the JSON, dashboard,
model card, and versioned model artifact.

## Honest limitations

There is no perfect weekly projection. Surprise role changes and touchdowns
remain the largest errors. Weekly roster, injury, starter, line, and weather
snapshots are only as timely as their upstream feeds. Player-condition history
is predictive correlation, not proof of causation. The release gate validates
projection accuracy and interval coverage; it does not claim that fantasy teams
will win, because league rosters, waivers, opponents, and playoff rules are a
separate decision layer.
