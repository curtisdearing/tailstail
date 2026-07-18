# Performance record (Phase B)

Rule: optimize only what was measured, and never trade determinism or clarity
for speed. Every change below is accompanied by an equivalence test in
`tests/test_phase_b_hardening.py` and by byte-identical canonical content
hashes from `scripts/neutrality_hashes.py`.

## Method

- Data: nflverse play-by-play 2021-2023 (142,207 regular-season plays, 81.3 MB
  in memory) plus weekly rosters, composed through `ingest.load_all_pbp`.
- Timings: 5 consecutive runs per stage in one process, median reported.
- Hardware: containerized Linux, Python 3.10, pandas 2.3.3. Absolute numbers
  are machine-specific; the ratio is the claim.

## Profile before

`cProfile` over three feature builds, sorted by cumulative time:

| Frame | Cumulative | Share |
|---|---:|---:|
| `build_player_week` | 9.54 s / 3 runs | 94% of total |
| `groupby.transform` (`_transform_general`) | 5.85 s | 61% |
| `_rolling_shifted` (46,302 calls) | 3.92 s | 41% |
| `Series.__init__` (96,717 calls) | 2.23 s | 23% |

Diagnosis: `groupby(...)[col].transform(_rolling_shifted)` passes a Python
callable, so pandas takes the `_transform_general` path and constructs a fresh
Series per group per column. With ~5,800 players x 14 rolling columns that is
~46k Series constructions doing work that the Cython group paths already do.

## Change

One change, in `nflvalue/features.py`: `_rolling_shifted_grouped` replaces the
per-group Python callable with `groupby(...).shift(1)` followed by
`groupby(...).rolling(...)` / `.ewm(...)`. Same shift, same window, same
`min_periods`, same aggregation — only the dispatch differs.

## After

| Stage | Before (median) | After (median) | Speedup |
|---|---:|---:|---:|
| `build_player_week` | 1.613 s | 0.884 s | **1.82x** |
| `build_opp_pos_def` | 0.106 s | 0.082 s | 1.29x |
| `build_team_week` | 0.013 s | 0.010 s | 1.30x |
| `load_all_pbp` | 0.067 s | 0.045 s | (I/O noise) |
| `project` x600 | 0.014 s | 0.014 s | 1.00x (untouched) |

Canonical content hashes for `player_week`, `opp_pos_def`, `team_week`,
`projections`, and `game_script` are **unchanged** — see the Phase B checkpoint.

## Memory

No unbounded cross-join exists in the feature path. Measured on the same data:

- `pbp` frame: 81.3 MB; `player_week` output: 10.5 MB.
- Peak additional allocation during `build_player_week`: 36.1 MB.
- Process max RSS for a full build: 304 MB.

The feature build is linear in plays and does not need chunking at present
scale (three seasons). The place that grows super-linearly is the
QB x receiver **pair** panel in the chemistry work
(`nflvalue/fantasy/chemistry_engine.py`, `analysis/chemistry_study.py`), which
is quadratic in receivers per team rather than in plays. It was not a measured
hot path in the weekly run and was left alone; if the pair panel is ever
extended past a single season at a time, chunk it by season-week there.

## Not optimized, deliberately

- `project()` is already closed-form and 23 microseconds per call.
- `RandomForest(n_jobs=-1)` in `ml_ranker` and `fantasy/models` is faster with
  threads but is not byte-reproducible across thread counts. Reproducibility
  runs should pin `n_jobs=1`; this is a known, documented trade and was not
  changed silently for speed.
