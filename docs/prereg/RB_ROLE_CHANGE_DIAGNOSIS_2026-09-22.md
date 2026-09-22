# Frozen protocol — 2026-09-22 / running-back role-change regime

Copy of `docs/FROZEN_PROTOCOL_TEMPLATE.md`. **Status: DIAGNOSIS ONLY — NO LEVER
REGISTERED, NO LEVER RUN.** The 2026 wk1-2 retro's qualitative read ("the model
keeps projecting a stale volume prior for backs whose role has changed") was
treated as a hypothesis and tested on history. It is refuted in the form stated.
Four candidate levers were each disqualified BEFORE any of them was tuned, so
this note registers nothing and the incumbent model is retained unchanged. Do
not edit a frozen copy; supersede it with a new dated copy.

## Identity

- Product/track: tailstail (fantasy).
- Decision owner/reviewer: Curtis (owner). Diagnosis by Claude Opus 5.
- Git commit and tag: branch `research/tt-rb-role-change-2026-09-22` off `main`
  `f78ae38`; season freeze `freeze-2026-wk1` at `a9e9765`.
- `analysis/accuracy_protocol.json` SHA-256: untouched. This work changes no
  serving path, no feature, no model, no gate.
- Input canonical hashes and registry path: `historical/fantasy/manifest.json`,
  nflverse pull `2026-09-22T10:11:30Z`, nflreadpy 0.1.5, seasons 2019-2026
  (`player_stats` 3b054866…, `weekly_rosters` 2dd9189b…, `expected_points`
  71227c08…, `snap_counts` b79ffd6b…, `injuries` 11bff7d6…, `schedules`
  5eaaec2d…).
- Runtime/package versions: Python 3.11 venv (pandas 2.x, scikit-learn, scipy);
  `requirements.txt` still has no lock file (known gap, recorded 2026-08-10).

## Registered experiment

- Lever and feature flag: **none**. See "Decision" below for the four candidate
  directions and the number that disqualified each.
- Causal/mechanical hypothesis under test: RB error concentrates in weeks where
  the player's role changed, because the `pre_*` rolling volume features
  describe a role the player no longer holds (a *stale* prior). Competing
  explanation tested at the same time: those players simply have *thin* prior
  history.
- Primary metric: RB played-only MAE, walk-forward, production `ModelConfig`.
- Development folds: walk-forward test seasons **2021, 2022, 2023**; before each
  test season the ensemble is refit on strictly earlier seasons only
  (`analysis/rb_role_change_diagnosis.py:walk_forward` asserts
  `max(train season) == test season - 1`). n = 10,849 played rows, 2,938 RB.
- Locked checkpoint opened: **none**. 2024 was not spent and 2025 stays locked.
- Population, exclusions, n: `model_eligible` rows that were played. Every cell
  below reports its own n.
- Matching keys, strata, cluster unit: position; prior-opportunity quintile;
  prior within-team positional role rank. Cluster unit is the week (54 blocks).
- Multiplicity family: the four candidate lever directions × four positions.
  Nothing here is a promotion claim, so no q-value is spent.
- Random seed, simulations, wall time: `ModelConfig.random_seed = 6102026`;
  bootstrap seed 20260922, 5,000 week-block draws; no Monte-Carlo simulation;
  ~2 min total (frame 46 s, three refits 7/11/18 s).

## Results

### 1 · Where RB error lives (walk-forward 2021-23, played rows)

Regime = realized `opportunities − pre_opportunities_ewm4`, the same ±5 / +12
thresholds as the frozen role-state audit labels.

| regime | n | % rows | % abs error | MAE | bias (pred−act) | sd pred | sd act | slope | cov80 |
|---|---|---|---|---|---|---|---|---|---|
| down ≤ −5 | 447 | 15.3 | 19.1 | 6.811 | +6.001 | 3.90 | 5.57 | 0.771 | 0.613 |
| stable \|Δ\| < 5 | 1785 | 61.0 | 49.5 | 4.415 | +0.456 | 4.42 | 7.29 | 1.040 | 0.891 |
| up +5..12 | 533 | 18.2 | 21.0 | 6.276 | −5.087 | 4.06 | 7.88 | 1.014 | 0.797 |
| major up > +12 | 163 | 5.6 | 10.1 | 9.836 | −9.455 | 3.88 | 8.36 | 0.822 | 0.601 |

39% of RB rows carry 51% of RB absolute error. Week-clustered bootstrap
(B = 5,000, seed 20260922, 54 blocks): RB MAE 5.420 [5.265, 5.582]; stable 4.420
[4.267, 4.592]; role-change 6.988 [6.716, 7.275]; **change − stable +2.568
[2.267, 2.894]**; role-change rate 0.389 [0.372, 0.408]. Stable across seasons
(stable MAE 4.50/4.48/4.27; rate 0.404/0.397/0.367).

### 2 · It is a MIX effect, not an RB-specific modelling failure

| position | n | role-change rate | MAE all | MAE stable | MAE change |
|---|---|---|---|---|---|
| RB | 2938 | 0.389 | 5.420 | 4.420 | 6.988 |
| WR | 4577 | 0.104 | 5.455 | 5.009 | 9.313 |
| TE | 1765 | 0.060 | 4.544 | 4.175 | 10.310 |
| QB | 1569 | 0.627 | 5.919 | 5.412 | 6.221 |

The model is **better at RB than at WR in both regimes**; RB looks worse overall
only because RB roles change 3.7× as often as WR roles. Re-weighted to WR's
role-change rate, RB MAE would be **4.687** vs WR's actual 5.455. Within-week
Spearman over the same 54 weeks: **RB 0.527, WR 0.491, QB 0.421, TE 0.407** —
RB ranks best, not worst. The retro's "RB ranks worst" reading is a two-week
artifact of the 2026 sample, not a property of the model.

### 3 · The stale-prior mechanism is refuted

The down and up regimes carry biases of **+6.00 and −5.09** that cancel to a
pooled RB bias of **−0.25**, and the within-regime calibration slope is ≈ 1.0.
That is the signature of a conditional mean behaving correctly under a two-sided
shock it cannot see, not of a prior stuck at a stale level. There is no
pregame-identifiable pocket to shift: bias across prior-opportunity quintiles is
−0.33 / −0.23 / −0.61 / −0.22 / +0.11; across prior role rank 1/2/3 it is −0.20 /
−0.19 / −0.69; across all 18 weeks \|bias\| < 1.3.

### 4 · Confound check — stale vs thin (they are different, and thin is not the problem)

| prior history | regime | n | MAE | bias | slope |
|---|---|---|---|---|---|
| thin (<4 g) | stable | 78 | 3.955 | +0.705 | 0.969 |
| thin (<4 g) | role-change | 47 | 5.949 | −3.399 | 0.606 |
| medium (4-11) | stable | 192 | 3.478 | +0.463 | 0.990 |
| medium (4-11) | role-change | 126 | 6.699 | −2.316 | 0.735 |
| long (12+) | stable | 1524 | 4.563 | +0.461 | 1.049 |
| long (12+) | role-change | 971 | 7.076 | −1.153 | 0.680 |

Role change costs +2.0 to +2.5 MAE inside *every* history bucket, so it is not a
proxy for thin history. And thin history on its own is **better**, not worse
(MAE 4.705 vs 5.452, coverage 0.904 vs 0.811): the prior-only eligibility gate
keeps thin-history rows near replacement level where the model is accurate. A
"thin prior" fix would be aimed at a cell that is not broken.

### 5 · Pregame-observable role-shock flags do not identify the bad cell

| flag | n=1 | MAE 1 | MAE 0 | bias 1 | rho 1 | rho 0 |
|---|---|---|---|---|---|---|
| any pregame shock | 1172 | 5.441 | 5.406 | −0.666 | 0.513 | 0.541 |
| `rank_moved` | 626 | 5.028 | 5.526 | −0.538 | 0.480 | 0.532 |
| `trend_big` (\|trend_2v8\| ≥ 4) | 746 | 5.744 | 5.310 | −0.900 | 0.494 | 0.533 |
| `team_changed` | 38 | 5.044 | 5.425 | +1.545 | 0.054 | 0.530 |
| `rb1_out` (injury report) | 53 | 6.355 | 5.403 | −1.818 | 0.228 | 0.531 |
| `thin_history` | 125 | 4.705 | 5.452 | −0.838 | 0.424 | 0.524 |

The union of every pregame role-shock flag splits RB MAE by **0.035** (5.441 vs
5.406) and interval coverage by 0.016. The information that separates the 6.99
cell from the 4.42 cell is realized, not pregame.

### 6 · Oracle bound — no mean adjustment of any kind can pay

Granting oracle knowledge of the correct shift for each of 22 pregame cells
(role rank × RB1-out × prior-opportunity quintile), fitted and scored **in
sample** so it strictly bounds what a real lever could achieve:

| adjustment | RB MAE 5.4202 → | RMSE 7.0604 → |
|---|---|---|
| RMSE-optimal (cell **mean** residual), 22 cells | 5.4357 (**worse**) | 7.0374 |
| RMSE-optimal, single global shift | 5.4525 (**worse**) | 7.0561 |
| MAE-optimal (cell **median** residual), 22 cells | 5.3366 (−0.084) | 7.1415 |
| MAE-optimal, single global shift | 5.3618 (−0.058) | 7.1568 |
| monotone expansion ×1.10 / ×1.25 / ×1.50 | 5.4187 / 5.4645 / 5.6718 | — |

Two readings. (a) The whole MAE-optimal oracle gain is 0.084, of which 0.058 is
a **single global constant shift** — the repo has already been burned by a lever
that turned out to be a constant shift. The genuinely role-conditional part is
**0.026 MAE**, ~6% of the 0.44 gap to ESPN, and it costs +0.08 RMSE. (b)
Expanding the projections to close the sd gap (sd pred 4.3 vs sd actual 8.2)
*increases* MAE monotonically and, being monotone, cannot change within-week
ranking at all. Compression with a calibration slope of 0.95 is a correctly
scaled conditional mean under large irreducible variance, not shrinkage.

### 7 · Leakage finding — `status_inactive` is not serving-safe, and `rb1_out` inherits it

`features._add_teammate_state` builds `rb1_out`/`wr1_out`/`te1_out`,
`vacated_*_share` and `unavailable_skill_count` from
`unavailable = status_inactive | injury_out`. `injury_out` is the official
injury report (published Friday) and is genuinely pregame. **`status_inactive`
is the nflverse weekly-roster `status`, and that file is refreshed after
gameday.** Over 2021-23 it supplies **141 of the 141** RB1-out team-weeks in a
backtest, while the injury report supplies only **47** — so two thirds of the
"RB1 is out" signal available in replay does not exist at T−60 min. The 2026
retro saw the same thing from the other side: 39 ledger players production had
projected as available are INA/RES in today's files. Any lever keyed on
`rb1_out` would therefore measure materially better in backtest than it could
serve. This is the same defect class as the shipped `expected_points_missing`
bug; it is recorded here, not fixed, because fixing it is a serving-path change
and this note changes nothing.

Everything used in §§1-6 was verified strictly prior: training rows are asserted
to be from strictly earlier seasons; every `pre_*` feature is a
`groupby(...).shift(1)` statistic; `assert_feature_contract` refuses raw
outcome columns; realized regimes and `role_delta` are **labels only** and never
enter a prediction.

### 8 · Out-of-sample sanity check — 2026 wk1-2 vs ESPN (exploratory, n tiny)

Corrected retro projections (fix branch `4589088`) vs ESPN pre-kickoff, played
rows, labelled by the same realized regime:

| cell | n | model MAE | ESPN MAE | model bias | ESPN bias |
|---|---|---|---|---|---|
| RB stable | 57 | **3.866** | 4.204 | +1.04 | +1.50 |
| RB role-change | 53 | 7.180 | **5.824** | — | — |
| RB down | 24 | 7.387 | 4.885 | +6.79 | +4.22 |
| RB up | 29 | 7.008 | 6.601 | −6.77 | −5.06 |

The model **beats ESPN on stable RB rows** and loses the whole 2026 RB gap in
role-change rows — which corroborates §1 and refutes §3's mechanism at the same
time, because ESPN is badly wrong there too (bias +4.2 / −5.1); it is only less
wrong. ESPN's edge is beat-reporter depth-chart and touch-share information that
does not exist anywhere in the nflverse pregame corpus. RB backups whose RB1 was
ruled out on the official injury report in 2026 wk1-2: **n = 1**. Two weeks, one
cluster pair — no interval is quoted and none is computable.

## Decision

**No lever registered. Incumbent retained.** Each candidate direction from the
2026 retro was disqualified by a number fixed before it could be tuned:

1. **Recency/role-aware reweighting of the volume prior** — refuted by §3 and
   §6. There is no pregame-identifiable bias pocket to reweight toward, and the
   oracle, in-sample, role-conditional ceiling is 0.026 MAE.
2. **Explicit team-change reset of the rolling features** — 74 eligible played
   RB rows over three seasons (71 of 87 in week 1). Exposed n ≈ 25/season fails
   freeze §3's exposed n ≥ 100 by 4×, and those rows are already *better* than
   average (MAE 5.044 vs 5.425).
3. **Depth-chart-conditioned volume prior** — the only serving-safe depth signal
   in the corpus is the official injury report, exposure n = 53 over three
   seasons and n = 1 in 2026 wk1-2. Fails §3. The larger backtest exposure is
   the non-serving-safe `status_inactive` of §7. `role_state._depth_chart_asof`
   wants `historical/depth_charts_*.parquet`, which this corpus does not carry.
4. **Widening the RB predictive interval** — coverage is broken exactly where
   the regime is (down 0.613, major-up 0.601 vs nominal 0.80) but flat across
   every pregame flag (0.808 vs 0.825), and the existing `role_volatility`
   width scaling in `models.predict` already moves width only 17.9 → 18.9. A
   widening keyed on pregame evidence would widen the wrong rows. It also does
   not address MAE or ranking.

Two protocol facts reinforce the same conclusion. Freeze §4 allows the fantasy
track **one** lever at a time, and lever 3 (market/ESPN shrinkage blend) is
already REGISTERED and OPEN with its confirmatory sample accruing from 2026 wk3
— and §8 shows lever 3 is aimed at precisely the information gap found here.
Freeze §4 also stops the track after three consecutive rejected levers, so
registering a lever whose ceiling is already known to be 0.026 MAE would spend
that budget on a foregone conclusion.

Standing of the two rules that produced this decision, stated plainly because the
distinction matters more than the outcome. Freeze §3's exposed n >= 100 / matched
unexposed n >= 100 was **fixed in advance** by the signed freeze and fails on its own
(best serving-safe exposure n = 53). The second rule -- "register a lever only if a
strictly-pregame cohort isolates a bias pocket that an oracle, in-sample,
role-conditional mean adjustment could close by materially more than a single global
constant shift" -- is a **screen I applied once to all four candidates after the cohort
tables existed**, not a preregistered gate, and it is recorded as such. Nothing here may
be cited as a passed test.

- Baseline metrics: RB played MAE 5.420 [5.265, 5.582]; within-week Spearman
  0.527. Unchanged — nothing was run against them.
- Candidate metrics and paired interval: n/a, no candidate.
- Calibration: RB 80% coverage 0.815 pooled, 0.891 stable, 0.613 down, 0.601
  major-up.
- Forward CLV: n/a (fantasy track).
- Top-10 sanity diff: n/a — no candidate projection set exists.
- **Accepted, rejected, or research-only: RESEARCH-ONLY. Incumbent retained.**
- Reviewer sign-off and date: pending Curtis.

## Operations

- Data/provider and schema checks: single nflverse pull, hashes in §Identity.
- Roster/identity freshness: see §7 — the weekly-roster refresh boundary is now
  a recorded, unfixed hazard for every feature built from `status_inactive`.
- Rollback commit/flag: nothing to roll back; no serving path was touched.
- Week 8 bye-week checkpoint owner: Curtis.
- Known caveats shown to users: none. Nothing here reaches a published surface.
- Open follow-ups, none of them authorized here: (a) audit every consumer of
  `status_inactive` for the §7 serving boundary and decide whether the feature
  should be rebuilt from the injury report alone; (b) if RB role-change weeks
  are to be attacked directly, the missing input is a pregame depth-chart /
  touch-share feed, not a re-weighting of what the corpus already has.
