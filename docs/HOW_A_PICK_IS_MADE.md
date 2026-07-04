# How a pick is made — the complete, no-black-box walkthrough

Every number on a published lean is traceable through these nine steps. File
references point at the exact code.

## 1. Data comes in (`nflvalue/ingest.py`, runs before every live pipeline)

Play-by-play, schedules (with pre-game spread/total and projected starting
QBs), weekly rosters, official injury reports, NGS tracking, FTN charting,
contracts, player DOBs — all free nflverse/ESPN/Open-Meteo feeds, cached as
parquet under `historical/`. Coverage and trust grades: `DATA_SOURCES.md`.

## 2. Walk-forward features (`features.py`)

For each (player, week): rolling usage (targets, target share, carries,
attempts), rolling efficiency (yards/target, catch rate, yards/carry, YPA)
shrunk toward position means on small samples; for each (defense, position):
rolling yards- and EPA-allowed factors (WR and TE tracked separately); for
each team: rolling pass/rush volume. **Everything is `shift(1)`-then-roll: a
row at week W aggregates only weeks < W.** `tests/test_leakage.py` poisons
future weeks and asserts nothing moves.

## 3. The deterministic projection (`projection.py`)

```
expected volume    = team rolling volume × player usage share × game-script tilt
expected efficiency = player rolling efficiency × opponent-vs-position factor
mean               = volume × efficiency
```

Game script comes from the pre-game spread (favorites tilt run, dogs tilt
pass, capped ±12%). Each market gets a distribution family (gamma for yards,
negative binomial for counts, Poisson for TDs); the per-market SD is the
standard deviation of the engine's own **past** errors (walk-forward
residuals), so P(over line) is read off a distribution whose width reflects
how wrong the model has actually been. LLMs never touch any number
(`synthesis.py` enforces this with a contract violation exception).

## 4. Candidates and gates (`candidates.py`)

Every (player, market) pairing per game — then gates: ≥3 trailing games
(cold-start), minimum trailing usage (no scrubs), anytime-TD is YES-only.
~40 candidates per game survive. With no sportsbook line, the reference line
is the player's own trailing mean (floor + .5), rendered with a † and **never
allowed to mint an edge**.

## 5. Measured situational adjustments (`candidates.py`, constants from data, not vibes)

| Trigger | Adjustment | Evidence |
|---|---|---|
| Teammate in same usage family OUT | volume boost from that player's own historical with/without splits (cap ×1.35; halved if no absent-week sample) × efficiency dampening `1 − .29·(boost−1)` | n=297 absent player-weeks: beneficiaries gained volume, lost ~31% per-touch efficiency |
| Projected QB threw <50% of trailing attempts | pass-family means ×0.92 | n=162 backup weeks: volume flat, efficiency −8.4% |
| Team's WR1 / TE1 / RB1 OUT | QB passing markets ×0.921 / ×0.947 / ×0.971 (multiplicative, floor .85) | full absence matrix, n=1,146–1,514 per cause: `data/absence_matrix.json` |

## 6. Ranking (`composite.py` + `ml_ranker.py`)

Two rankers, both always computed:

- **Composite (auditable):** `100·(w_e·edge + w_c·confidence + w_m·matchup)`,
  weights tuned by walk-forward grid over 2019–2025 (each season scored with
  weights chosen only from prior seasons). Confidence = capped |z| from the
  line; matchup = opponent yards/EPA-allowed + game-script + pace, directional
  for the chosen side; edge (below) dominates when a real price exists.
- **ML ranker (orders the list, flag-gated):** a gradient-boosted classifier
  predicting P(actual > line), stacked on the projection's own belief plus
  ~50 features (weather, PROE/pace, NGS separation/air-yards share, red-zone
  roles, defensive/O-line outs, QB chemistry, shotgun tilts, blitz/box rates,
  age, contract year, birthdays, revenge…). It weights features by outcomes —
  in practice it pruned birthdays/revenge to ~zero and leaned on red-zone
  share, NGS, and usage. Walk-forward out-of-sample 2021–25 it beat the
  composite 63–69% vs 57–59% at reference lines. Structural guard: the model
  **refuses to score any week it trained on** (`WalkForwardViolation`).

## 7. The line, and the edge (`sources/oddsapi_props.py`, `composite.py`)

Lines are pulled (budget-capped, rotating) from the configured books —
DraftKings, BetMGM, Hard Rock. Per prop: consensus point → every book de-vigged
→ sharp-weighted **consensus fair probability**; best price per side kept
with the book named. **Edge = model P(side) − consensus fair P(side)**;
`ev_best_price` shows expected value at the best quote. No two-sided price →
`no_market`, and the pick ranks on model conviction alone, labeled as such.

## 8. Selection honesty (`shortlist.py`)

Top 5 per game, max 2 per player (correlated markets), deterministic
tie-breaks, and the screen count ("5 of N") always printed — a top-5 from 40
candidates contains luck, and the denominator keeps that visible. The context
panel (injuries, birthdays, revenge, incentives, weather, mismatches) is
assembled **after** ranking from a scorer that cannot see it; tests prove
score equality with and without context.

## 9. Feedback (`prop_learning.py`, `clv.py`, `killcheck.py` — Tuesdays, automatic)

Every pick is graded; every miss attributed (volume miss / efficiency miss /
availability surprise / script flip / tail variance — queryable via the RAG
CLI). Per-market bias and reliability adjust next week's run (bounded,
walk-forward, from the full candidate pool to avoid selection bias). The ML
retrains weekly, and its training labels migrate from synthetic reference
lines to **real lines** as they accumulate. Closing-line value is logged per
pick (entry snapshot vs pre-kick snapshot, de-vigged); after 150 resolved
picks the kill-check declares GO or NO-GO — and NO-GO means the pre-committed
answer is *stop treating this as bettable*, in writing, in the report.

## Known limitations (deliberately not hidden)

Backtest hit rates are measured at synthetic (trailing-mean) reference lines,
which structurally favor unders — real books price tighter; forward CLV is
the only edge proof accepted. Exact formations/personnel data is unavailable
free post-2023 (see `DATA_SOURCES.md`); the model uses formation-adjacent
signals instead. Candidate sets for live weeks carry forward from the prior
week, so debuting/just-traded players are invisible until they play. Player
props limit fast; the value here is the research, not scalable income.
