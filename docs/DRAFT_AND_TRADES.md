# Draft board + weekly trade planner

The season-level decision layer on top of the weekly engine: who to draft,
then — every week — who to trade, for whom, and when. Built for a shallow
(6-team) full-PPR league but parameterized for any size.

## Method in one paragraph

Per-game player baselines come from the trained ensemble's weekly predictions
blended with realized points (shrunk by games played); season totals come from
Monte Carlo over the real schedule with byes, per-week availability, and a
season-ending hazard. The board ranks by ceiling-weighted value over
replacement: `score = 0.45·VOR_mean + 0.55·VOR_p90`, with replacement levels
that respect shallow-league reality (QB/TE are streamed, RB/WR benches run
deep). Age curves, team-change widening, and rookie ADP pricing are labeled
priors (`basis` column) — never silent. The trade planner evaluates packages
by *optimal-lineup* simulation deltas (never sum-of-projections) and only
proposes trades that don't hurt the counterparty's lineup — one-sided robbery
never gets accepted, so it's not worth proposing.

## Evidence

`analysis/draft_retrodiction.py` grades the methodology leakage-free:
preseason boards built from season N-1 data only, scored against realized
season-N totals (2023–2025). Result: the board beats naive last-year-points
drafting on top-24 hit rate (0.417 vs 0.389) — the decision zone — and the
model-free lower bound matches the model-blended variant, so the grade is not
riding on leaked model knowledge. Full numbers: `reports/draft_retrodiction.md`.

## Draft day

```bash
# refresh ADP + schedule first if stale (see scripts/draft_board.py for paths)
PYTHONPATH=. python scripts/draft_board.py --teams 6 --ceiling 0.55
```

Outputs `data/draft_board_2026.{csv,json}`, season sample matrix, and
baselines. Columns worth reading at the table: `tier` (gap-clustered),
`value_gap` (ADP minus our rank — positive = market lets him fall),
`basis` (what's evidence vs prior), `season_p90` (the ceiling being bought).

Snake helper: `draft.snake_picks(slot, league_teams=6)` gives overall pick
numbers; `draft.availability_probability(board, pick)` estimates who survives
to each pick from ADP spread.

## Weekly loop (in season)

1. **Tuesday** (after MNF): rerun the weekly pipeline, then rebuild baselines
   so the board reflects the latest four weeks of role evidence.
2. **Trade scan**:

```bash
PYTHONPATH=. python scripts/trade_scan.py \
    --espn-league <LEAGUE_ID> --season 2026 --my-team "<TEAM NAME>" \
    --week <W> [--espn-s2 <cookie> --swid <cookie>]   # cookies only if private
```

   Falls back to `--rosters data/rosters.json` (see module docstring for the
   format). Each proposal reports `my_gain_per_sim`, `their_gain_per_sim`
   (two-sided gate), `bye_relief` (next three weeks), and `playoff_tilt`
   (weeks 15–17 delta) — trade *timing* is the point: byes create windows,
   playoff schedules decide championships.
3. **Sell-high / buy-low**: `trade_planner.market_temperature` compares recent
   realized points against model expectation; hot players are sold into the
   market's recency bias, cold ones bought.

## Honesty notes

- The season simulator treats weekly points as exchangeable draws around a
  per-game center — it does not re-run the weekly feature model for future
  weeks (those features don't exist yet). Week-level start/sit decisions
  should use the weekly pipeline, not this board.
- Availability and season-ending hazards are calibrated heuristics; the
  retrodiction grade prices in their error honestly.
- Rookie rows are market priors by construction. The model has no evidence on
  them; the board says so instead of pretending.
