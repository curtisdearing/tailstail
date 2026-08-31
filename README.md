# tailstail — fantasy football projections and simulation

Tailstail turns scoring-independent NFL player/game distributions into fantasy
football decisions: weekly projections, configurable scoring, correlated
simulations, start/sit choices, lineups, waivers, rest-of-season value, and
trades.

It does not rank bets, consume sportsbook prices, calculate edge/CLV, or send
betting recommendations. Those responsibilities belong to the sister project
[fablesfable](https://github.com/curtisdearing/fablesfable).

## Quickstart

```bash
pip install -r requirements.txt
python -m nflvalue.fantasy.cli fetch --seasons 2019:2026
python -m nflvalue.fantasy.cli build
python -m nflvalue.fantasy.cli backtest --test-seasons 2023:2025
python -m nflvalue.fantasy.cli audit-monte-carlo
python -m nflvalue.fantasy.cli train
python -m nflvalue.fantasy.cli project --season 2026 --week 1
python -m nflvalue.fantasy.cli simulate --season 2026 --week 1 --simulations 10000
```

Run the complete weekly pipeline with:

```bash
python scripts/fantasy_weekly.py --season 2026 --week 1
```

It writes:

- `data/fantasy_latest.json` — scoring-specific fantasy summaries;
- `data/player_projection_snapshot.json` — shared, scoring-independent player
  distribution contract;
- `data/player_projection_samples.parquet` — correlated football-event samples;
- `data/fantasy_model.joblib` and `reports/fantasy_model_card.json` — versioned
  model evidence; and
- `fantasy.html` — the Tailstail dashboard.

## Product boundary

Tailstail and fablesfable share public NFL inputs, stable identities,
strictly-prior feature discipline, factor evidence, and a versioned football
projection contract. They do not share consumer state or objectives.

| Shared football evidence | Tailstail only | fablesfable only |
|---|---|---|
| NFL data/provenance | League scoring | Sportsbook lines/prices |
| Pregame feature clocks | Fantasy point distributions | De-vigged probability/edge |
| Event-component simulations | Lineups/waivers/trades | CLV and kill checks |
| Factor research ledger | Fantasy dashboard/state | Betting dashboard/state |

The repositories deploy independently. Tailstail owns
`.github/workflows/fantasy-weekly.yml`, the `fantasy-model-state` release tag,
and its own GitHub Pages site. It requires no Odds API or betting Discord
secret. Legacy prop modules remain in the tree temporarily while the shared
core contract is extracted; no Tailstail production workflow invokes them.

### Public and private output

A weekly run writes two artifacts. `fantasy.html` and `data/fantasy_public.json`
are public: Tailstail's own projections and the aggregate model grading, built
from an allow-list and checked before they are written. The league this project
is pointed at is private, so its snapshot, rosters, team names and league id —
and the personalised `decision-card/1` page built from them — are written to the
gitignored `private/` directory and are never published, never uploaded as a
workflow artifact, and never packed into a release asset.

The ESPN external-challenger comparison is split the same way. The public season
grading series is a durable aggregate history
(`data/espn_comparison_history.json`) that carries counts, MAEs and hashes and
nothing that could be a player; the per-player rows it was once derived from
live in a separate private repository reached only by a trusted production run.
They are deliberately **not** in an Actions cache: GitHub documents that a
pull-request workflow can restore default-branch caches, so a cache is not a
place for restricted material. See
[docs/DECISION_CARD.md](docs/DECISION_CARD.md).

## Validation

The fantasy engine uses position-specific Bayesian ridge, gradient boosting,
random forest, and gradient-descent learners with nested season-forward
stacking. The event simulator samples shared pace, team volume, player
target/carry shares, efficiency, availability, and touchdowns before applying
league scoring.

Current 2023–2025 full-PPR evaluation, rebuilt from source on 2026-08-30 and
reproduced end to end: **11,482 player-weeks, 5.0941 MAE, 6.7215 RMSE, 0.6240
Spearman, 82.34% coverage for nominal 80% intervals** (82.41% after the
calibrated Monte Carlo replay, mean CRPS 3.693). Role shocks and touchdowns
remain the largest errors. These numbers describe the public-data decision
pool; they are not a claim of universal accuracy.

The previously published figures — 11,481 rows, 5.0914 MAE, 6.7184 RMSE,
0.6246 Spearman, 82.24% replay coverage — came from an earlier nflverse pull
under numpy 2.0.2 / pandas 2.3.3. The fresh pull carries one additional
eligible player-week and vendor restatements of snap, injury and
expected-opportunity rows, so the two runs differ by ~0.003 MAE. Both are
recorded in `data/accuracy_registry.json`; neither is reconciled away.

`data/accuracy_registry.json` is the single scoreboard, written only by
`analysis/eval_harness.py`, and every number in it carries the hash of the
input that produced it.

Read:

- [Fantasy engine](docs/FANTASY_ENGINE.md)
- [Professional-method audit](docs/FANTASY_PROJECTION_METHODS.md)
- [Factor catalog](docs/FANTASY_FACTOR_CATALOG.md)
- [Fantasy premortem](docs/FANTASY_PREMORTEM.md)
- [Season-forward model search](reports/fantasy_model_search.md)
- [Historical Monte Carlo audit](reports/fantasy_monte_carlo_history.md)
- [Contrarian engineering review and execution plan](reports/fantasy_engineering_review.md)
- [Reproducible factor audit](reports/all_data_factor_audit.md)
- [Preregistered 2026 accuracy protocol](docs/ACCURACY_PROTOCOL.md)

## Honesty invariants

- Every outcome-derived feature is shifted before use.
- Training, imputation, stack selection, and intervals are learned inside
  historical cutoffs.
- Current-week participation never determines backtest eligibility.
- Parquet hashes are transport-integrity checks; canonical tabular hashes are
  reproducibility fingerprints.
- Small player-condition samples are shrunk and remain research-only until they
  survive a predeclared season-forward test.
- Surprise availability, role movement, and discrete touchdowns remain
  uncertainty—not hidden certainty.
