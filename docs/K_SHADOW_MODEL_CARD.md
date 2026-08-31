# K (Kicker) — Shadow / Research-Only Model Card

**Status:** `shadow` — research only. Not a lineup recommendation, not promoted,
not wired into optimal-lineup decisions.
**What is implemented:** a **historical-rate baseline**, not the model this card
originally described. See §0a.
**Written:** 2026-08-30, BEFORE any implementation code, as the pre-coding gate.
**Revised:** 2026-08-30, after implementation, to say what was actually built.

---

## §0a. What the code is, as opposed to what this card designed

This section was added because the rest of the card described a model, the
repository contained a baseline, and nothing in between said so. A card that
over-describes its implementation is worse than no card: it lends the output a
provenance it has not got.

`nflvalue/fantasy/shadow_kicker.py` is a **per-kicker empirical-rate baseline**:
distance-bucket attempt rates and make rates, shrunk toward the league rate by
fixed pseudo-counts, then Poisson attempts and Binomial makes, scored through
the league contract. That is all it is.

**Deliberately not implemented**, each of which §§3–8 below specify:

| Designed in this card | Status in code |
|---|---|
| Team scoring opportunity (implied total, spread, drive model) | **not implemented** — bucket attempt rates are per-week averages, unconditioned on the game |
| PAT linkage to offensive touchdowns | **not implemented** — PAT attempts are their own Poisson from a shrunk per-week rate |
| Weather, roof state, altitude | **not implemented** — no input is read |
| Blocked kicks as their own component | **not implemented** — folded into misses |
| Replacement-kicker path (`replacement: true`) | **not implemented** — the flag is hardcoded `False` |
| Parameter uncertainty on the shrunk rates | **not implemented** — point rates feed the draws |
| Hierarchical shrinkage via `hierarchy.py` | **not implemented** — fixed pseudo-counts instead |

Because attempt buckets are drawn independently and unconditioned, this
baseline understates the correlation between a kicker's volume and his team's
scoring, and its intervals are not the intervals §8 specifies.

**Therefore this is a baseline, not a completed preregistered model, and the
gates in §9 are UNRUN.** They now have code behind them
(`nflvalue/fantasy/k_audit.py`) and a declared threshold set
(`k_audit.GATES`), fixed before any run. Until `k_audit.gate()` returns a pass
on a real season-forward evaluation:

* `shadow_kicker.PROMOTION_STATUS["may_enter_lineup_objective"]` is `False`;
* no kicker is ranked into any lineup, waiver or trade objective;
* the K seat on the weekly card is its own `NO CURRENT PICK`, and it does not
  block the offensive lineup.

### Corrections to earlier claims in this card

* **§9 "Determinism: same seed + same inputs ⇒ byte-identical artifact"** was
  asserted and was false twice: the per-player seed came from Python's builtin
  `hash()`, which `PYTHONHASHSEED` salts, so the stream differed in every
  process; and `content_sha256` was taken over a dict including
  `model_run_at`, so the digest moved with the wall clock. Both are fixed
  (`stable_seed`, `NON_CONTENT_FIELDS`) and both now have tests, one of which
  runs in subprocesses under three different salts.
* **§2 "Fail closed: ... ⇒ `status: unavailable`"** was asserted and the code
  failed open: `active.get(pid, True)` read "nobody told me" as "he is
  playing". Unknown is now `unavailable`.
* **§1 "Every input carries `{source, retrieved_at, as_of}` or it is not
  used"** was asserted and never checked. It is enforced now, and
  `information_as_of` is derived as the minimum `as_of` across inputs rather
  than accepted on trust.
* **§0 B1** ("the exact scoring rules do not exist in this repository") is
  **stale**: `nflvalue/fantasy/espn_contract.py` now carries the live contract.

**Isolation commitment:** K is never added to `ModelConfig.positions`
(`nflvalue/fantasy/config.py:61`), never enters the frozen QB/RB/WR/TE training
or projection path, and never appears in a `PlayerProjectionSnapshot`.

Legend: **[OK]** verified present in this repo · **[GAP]** absent, must be built
or fetched · **[BLOCKED]** cannot proceed without a decision from Curtis.

---

## 0 · Two blocking inputs

**B1 — The exact scoring rules do not exist in this repository. [BLOCKED]**
`ScoringRules` (`nflvalue/fantasy/config.py:9-45`) has **no kicker fields at
all**: no field goal, no distance bucket, no PAT, no miss, no block. The league
is recorded as *"full PPR plus retained **custom** K and D/ST scoring"*
(`reports/espn_watchlist_2026_1111111111.json`, league 1111111111 -- the
league id is anonymised throughout this repo; the descriptive league name
recorded alongside it is not repeated here for the same reason). Custom
means the values are held in ESPN's league settings
and nowhere on disk. There is no versioned live league contract file: the only
ESPN-league code path is `nflvalue/fantasy/trade_planner.py:71-86`, which
constructs `espn_api.football.League` at runtime to pull **rosters** and never
reads, persists, or versions `scoringSettings`.

Consequence: the required test *"all K scoring buckets and boundaries are
exact"* cannot be written honestly. Against invented buckets it would assert
that the code matches a guess, which is worse than no test. **No scoring code is
written until the real table is supplied.**

**B2 — There is no local kicker event history. [BLOCKED for fitting]**
`historical/` holds only `download_history.py` (562 bytes). There are **zero
`.parquet` files anywhere in the repo** — no play-by-play cache, no
`rosters_weekly.parquet`. Field-goal attempts, distances, results and PATs are
nflverse play-by-play fields that are simply not on disk. Fitting distance-bucket
make rates therefore requires a network pull, which is a separate decision, not
something to slip into this change.

Everything below is designed so that when B1 and B2 clear, the build is
mechanical.

---

## 1 · Pregame / as-of data sources and timestamps

Every input carries `{source, retrieved_at, as_of}` or it is not used — the
project's existing fail-closed posture (`nflvalue/contracts.py`, header).

| Input | Source | Status |
|---|---|---|
| Kicker roster position | `nflvalue/sources/rosters.py` (nflreadpy `load_rosters_weekly`) | **[OK]** code; **[GAP]** cache absent |
| Injury / active status | `nflvalue/sources/availability.py` — dual clock: `wed` provisional, `t90` pre-kick roster `active` override | **[OK]** |
| FG/PAT event history | nflverse play-by-play | **[GAP]** not cached |
| Implied team totals | `nflvalue/sources/oddsapi.py` | **[OK]** |
| Weather | `nflvalue/sources/weather.py` (Open-Meteo, kickoff hour) | **[OK]** |
| Dome flag | `nflvalue.factors.STADIUMS[team]["dome"]` | **[OK]** |
| Altitude | — | **[GAP]** `STADIUMS` carries lat/lon, no elevation |
| Retractable-roof *state* | — | **[GAP]** dome is a static boolean; open/closed on the day is not modelled |

`information_as_of` is the minimum `as_of` across load-bearing inputs, mirroring
the snapshot contract. Any input whose timestamp is missing, stale, or later
than kickoff fails the run closed rather than defaulting.

## 2 · Kicker identity, availability, depth-chart resolution

- One kicker per team per game. Identity resolves from the weekly roster's
  listed `K`, matched to a stable id; unmatched rows are returned separately and
  never guessed (the `availability.py` UNMATCHED convention).
- Availability uses the `t90` pre-kick `active` flag when its freshness gate
  passes, else the Wednesday provisional read, with the clock recorded on the row.
- **Fail closed:** zero or more than one resolvable active K for a team ⇒ that
  team's kicker emits `status: "unavailable"` with a machine-readable reason and
  **no distribution**. An unknown kicker never inherits the incumbent's numbers.
- No depth-chart feed exists in this repo **[GAP]**; resolution is roster-listed
  position plus active flag only, and the card says so rather than implying a
  depth chart.

## 3 · Team scoring opportunity and implied scoring context

Kicker scoring is almost entirely a function of *how often the offense reaches
scoring range and how often it stalls there*. Inputs, all pregame:

- Implied team total and spread (`oddsapi`) → expected drives ending in a score.
- `data/league_priors.json` **[OK]** already carries the league-level drive
  outcome split (`td 0.2206`, `fg 0.1452`), drives per game (`11.19 ± 1.70`),
  and league PPG — the natural prior for a TD-vs-FG split under shrinkage.
- Game script: spread magnitude and direction shift both attempt volume and the
  4th-down/FG decision boundary. Available pregame **[OK]**.

## 4 · Field-goal attempts and makes by distance bucket

Buckets are the *modelling* grid and are stated here so they are not confused
with the scoring grid, which is B1's to define. Attempts are drawn per bucket;
makes are Bernoulli per attempt with a shrunk kicker-specific rate.

Model buckets: `0-19, 20-29, 30-39, 40-49, 50-59, 60+`.

**The scoring buckets must equal the league's, not these.** If Curtis's custom
scoring uses different edges, the scoring grid is redefined from his table and
the model grid is refined to be a strict subdivision of it — never the reverse.
Bucket edges are inclusive-low/inclusive-high on integer yards; the boundary
test asserts every edge yard individually (39/40, 49/50, 59/60).

Kicker make rates are hierarchically shrunk toward the league bucket rate — the
repo already has this machinery (`nflvalue/fantasy/hierarchy.py`), and a kicker
with a handful of 50+ attempts must not get a bespoke long-range rate.

## 5 · PAT attempts and makes

PAT attempts follow offensive TDs (which the existing team-total path already
produces); makes are Bernoulli at a shrunk rate. Two-point attempts remove a PAT
opportunity, so PAT attempts are modelled as TDs minus simulated 2-point tries,
not as TDs.

## 6 · Miss and block treatment

Modelled **only if Curtis's scoring assigns them a value** (B1). Where supported:
misses split into short/long by the same scoring grid; blocks are folded into
misses unless his settings price them separately. Play-by-play distinguishes
blocked kicks, so the split is available once B2 clears. If his league scores no
penalty, the miss branch still runs — it consumes an attempt — but contributes 0.

## 7 · Weather, roof, altitude, game script

- Wind, precipitation, temperature at the kickoff hour **[OK]**; dome ⇒ neutral.
- Altitude **[GAP]** and retractable-roof state **[GAP]** are declared unmodelled
  rather than approximated. Denver is the only material altitude venue and a
  one-venue effect fitted from thin data is a coin flip dressed as a factor.
- Effects enter as adjustments to long-bucket make rates only. **Any weather
  input that is missing does not silently become "neutral"**: the row is marked
  degraded and its band widens.

## 8 · Uncertainty and replacement behaviour

- Output is a **distribution**, not a point estimate: N simulations under an
  explicit seed, reported as mean, sd, p05/p25/p50/p75/p95, and P(0 points).
- Uncertainty is compounded, not assumed: team scoring volume, TD-vs-FG split,
  per-bucket attempt counts, and per-attempt make probability each carry their
  own variance, plus parameter uncertainty on the shrunk make rates.
- **Kicker change mid-week:** the incumbent's row is invalidated, never
  transferred. A replacement with no history falls back to the league bucket
  prior with parameter variance set at the league dispersion — visibly wider,
  and flagged `replacement: true`. A replacement who cannot be resolved is
  `unavailable`, not a league-average body.

## 9 · Calibration and promotion gates

Shadow → candidate → promoted. Nothing skips a step; K stays `shadow` in this
change regardless of results.

**Pre-registered before any fitting:** evaluation seasons and walk-forward
boundaries, the pregame-only information boundary, and the gates below.

| Gate | Bar |
|---|---|
| Leakage | Poisoning any post-kickoff or future-week field leaves earlier K forecasts bit-identical |
| Determinism | Same seed + same inputs ⇒ byte-identical artifact |
| Coverage | Empirical coverage of the 50% and 90% bands within tolerance of nominal, out of sample |
| Calibration | Reliability slope/intercept on P(make) by bucket; calibration-in-the-large ≈ 0 |
| Sharpness | Beats a league-average-kicker baseline on MAE *and* CRPS — beating MAE alone can be done by shrinking to the mean |
| n-gate | Per-bucket minimum sample before a kicker-specific rate is used at all |
| Isolation | Frozen QB/RB/WR/TE artifacts byte-identical before and after; no K row validates against the snapshot schema |

Failing any gate keeps K in shadow and is recorded as a failure, not retuned and
re-run inside the same registration.

## 10 · Artifact

`data/shadow/k_weekly_{season}_wk{week}.json`, written only through the versioned
league scoring contract once it exists:

```
status: "shadow"                     # literal, survives JSON + dashboard render
scoring_contract_version, scoring_hash   # hash of the exact rule table used
model_version, seed, simulations
retrieved_at, model_run_at, information_as_of
provenance: [{source, retrieved_at, as_of}, ...]
players: [{ kicker_id, name, team, opponent, game_id,
            status: "projected" | "unavailable",
            unavailable_reason, replacement,
            distribution: {mean, sd, p05, p25, p50, p75, p95, p_zero},
            components: {fg_attempts_by_bucket, fg_makes_by_bucket,
                         pat_attempts, pat_makes, misses, blocks},
            degraded, degraded_reason }]
```

Separate directory, separate file, separate schema. It is never merged into the
offensive artifact and never validated against
`schemas/player_projection_snapshot.schema.json` — whose `position` enum is
exactly `["QB","RB","WR","TE"]` with `additionalProperties: false`, so a K row
fails that contract by construction. That is the isolation test, and it is a
real assertion rather than a tautology.
