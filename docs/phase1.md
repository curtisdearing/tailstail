# Phase 1 — player-prop projection layer: how to run it

Phase 1 Sub-phase A (deterministic core) + Phase 1B Part 1 (hardening) are
complete. See `PHASE1_BUILD_PROMPT.md` / the Phase 1B build prompt for the
full spec and `PHASE1_HANDSOFF_DESIGN.md` for the guardrails this honors.
Phase 1B Part 2 (availability resolver, freshness gate, Sleeper divergence,
LLM synthesis layer) has not been built yet — do not assume it exists.

## Setup

```bash
pip install -r requirements.txt   # adds scipy on top of pandas/pyarrow/numpy
pip install pytest                # only needed to run tests/
pip install nflreadpy             # needed for real positions (nflvalue/sources/rosters.py)
```

Data already on disk: `historical/historical_pbp.parquet` (2019–2023 regular
season). `nflvalue/sources/rosters.py` additionally pulls real weekly rosters
via `nflreadpy` and caches them to `historical/rosters_weekly.parquet` (already
populated for 2019–2023 in this environment) — no live network call is needed
once that cache exists.

## Files added

| File | What it does |
|---|---|
| `nflvalue/db.py` | SQLite helpers (`connect`, `upsert`, `upsert_df`, `query_df`) + schema for `player_week`, `opp_pos_def`, `projections`, `prop_backtest`. DB lives at `data/nfl_props.db` (named `nfl_props.db`, not `nfl.db`, purely to dodge a stale-journal-file issue hit once in this dev environment — no design reason behind the literal name). |
| `nflvalue/sources/rosters.py` | **Phase 1B.** Real weekly positions (QB/RB/WR/TE) via `nflreadpy.load_rosters_weekly`, cached to parquet, with a small recorded fixture for offline tests. |
| `nflvalue/features.py` | Builds `player_week` (usage/efficiency, walk-forward), `opp_pos_def` (defense-vs-position factors, walk-forward, **WR and TE tracked separately** since 1B), `build_team_week` (team pass/rush pace, walk-forward). Position comes from `rosters.py`; the old play-by-play participation heuristic is kept only as a fallback for the rare row missing a roster match (~0.6% of rows), tagged `position_source="inferred_fallback"`. |
| `nflvalue/projection.py` | Pure math: `project(player_row, market, ...)` returns `{player_id, name, pos, market, mean, sd, dist, line, p_over, p_under, components, low_confidence, eligible_for_shortlist, roll_games}`. Deterministic, no I/O. **Phase 1B:** adds the `MIN_GAMES_ELIGIBLE` cold-start gate (see below). |
| `prop_backtest.py` | Walk-forward accuracy backtest, 2019–2023. Writes `data/prop_backtest.json` + upserts `nflvalue/db.py`'s `prop_backtest` table. Reports overall / eligible-only / by-sample-size / by-position accuracy and a calibration table. |
| `tests/` | `test_leakage.py`, `test_reproducibility.py`, `test_projection.py`, `test_positions.py`, `test_backtest_smoke.py` — 23 tests total. |

## Run the backtest

```bash
python3 prop_backtest.py                    # all 5 seasons
python3 prop_backtest.py --seasons 2019 2020  # a subset, faster
```

Per market: MAE, RMSE, correlation — **overall**, **eligible-for-shortlist**
(the honest number after the cold-start gate), by trailing-sample-size
bucket, and (for receiving_yards/receptions/anytime_td, which span more than
one real position) **by position** — plus a 10-bucket calibration table
(predicted P(over) vs. actual over-rate against each player's own trailing
rolling **mean** as a synthetic line — not a real sportsbook price).

**This measures projection accuracy only, not price-beating** — there is no
free historical player-prop *line* data, so "does the model beat the market"
can only be tested forward, live, once real prop lines are pulled (Phase 3).

## Run the tests

```bash
python3 -m pytest tests/ -q
```

23 tests, ~55–60s total (the leakage/reproducibility tests rebuild
`player_week` a handful of times, which is the slow step). Split across a
couple of invocations if your shell has a tight timeout:

```bash
python3 -m pytest tests/test_leakage.py tests/test_reproducibility.py -q
python3 -m pytest tests/test_projection.py tests/test_positions.py -q
python3 -m pytest tests/test_backtest_smoke.py -q
```

`test_backtest_smoke.py` runs `prop_backtest.run(seasons=[2019])` and
**overwrites `data/prop_backtest.json`** with that smaller slice — re-run the
full `python3 prop_backtest.py` afterward if you want the complete
2019–2023 report back on disk.

## Existing game-line app

Untouched. `build_ratings.py`, `backtest.py`, `run.py`, `weekly.py`,
`dashboard.py` etc. all still run exactly as before — the new prop layer only
adds files under `nflvalue/` and `prop_backtest.py`/`tests/` at the root.

## Phase 1B changes, in brief

1. **Real positions, not inferred roles.** `nflvalue/sources/rosters.py` pulls
   real weekly QB/RB/WR/TE from `nflreadpy`. This replaced Phase 1A's
   participation-based heuristic (which could only bucket a coarse QB/RB/REC).
   `opp_pos_def` now computes **separate** WR-defense and TE-defense factors
   (previously pooled into one "REC" bucket) using each play's real targeted-
   receiver position.
2. **Cold-start gating.** `MIN_GAMES_ELIGIBLE = 3` in `projection.py`: below
   that many trailing games, a projection is still computed and returned (for
   backtest visibility) but marked `eligible_for_shortlist=False` and forced
   `low_confidence=True`. A future ranker (Phase 2) should filter on this flag.
3. **Calibration methodology fix.** The Checkpoint-1 calibration table used
   each player's trailing rolling **median** as the synthetic benchmark line.
   For right-skewed markets (receiving/rushing yards), mean > median
   structurally, so "P(actual > median)" runs meaningfully above 50%
   regardless of the model's own probability — that's a flaw in the
   **benchmark**, not necessarily the model. Switching to a rolling **mean**
   (an apples-to-apples comparison with the model's own projected mean)
   revealed the model was calibrated considerably better than Checkpoint 1's
   number suggested — see the accuracy-deltas summary for the actual curves.

## Known limitations / honest caveats (read before trusting a number)

- **Position fallback for ~0.6% of rows.** A player missing from that week's
  roster snapshot (e.g. a same-day practice-squad elevation) falls back to
  the old participation heuristic and defaults any "REC" guess to WR (more
  common than TE); tagged `position_source="inferred_fallback"` so it stays
  visible rather than silently blending in with real data.
- **Game script is a pluggable no-op here.** `project()` accepts a
  `game_script` multiplier (trailing-team-passes-more / leading-team-runs-more),
  but `prop_backtest.py` runs it neutral (1.0/1.0) rather than wiring in a
  live spread/total via `nflvalue/montecarlo.simulate` — that hookup is
  straightforward at weekly-report time (Part 2) and was left out of the
  accuracy backtest to keep scope contained, per the build prompts.
- **The WR/TE split didn't improve AGGREGATE receiving_yards accuracy** (it's
  roughly flat vs. Checkpoint 1's combined-REC number) even though it's a
  strictly more accurate input. Two honest reasons: (a) real positions
  correctly EXCLUDE some pass-catching RBs that the old heuristic
  misclassified into the combined bucket, changing the evaluation set; (b) a
  position-specific opponent factor is computed from roughly half the plays
  of the old pooled factor, so it's individually noisier. The **by-position**
  breakdown is where the real signal shows: TE receiving yards are
  meaningfully more predictable than WR (MAE ~17 vs. ~25, 2019–2023) — a
  finding the old combined bucket couldn't see at all.
- **RB markets (rushing_yards, rush_attempts) improved cleanly** with real
  positions — both MAE and correlation got better, and the eligible sample
  size went UP (real positions correctly keep a pass-catching-back's
  low-carry game classified as RB, where the old heuristic would have
  misclassified that week as receiving).
- **Calibration is now good but not perfect**, and tail buckets (P(over)
  > 80%) have small samples (as few as n=6–40) — read those with extra
  caution. anytime_td and receptions calibrate especially well; receiving/
  rushing/passing yards are directionally right but noisier at the extremes.
- **Cold-start rows (fewer than `MIN_GAMES_ELIGIBLE` trailing games)** are
  gated out of "eligible_for_shortlist" but still reported for transparency;
  accuracy in this bucket is honestly worse (occasionally negatively
  correlated) — this is exactly why the gate exists.
