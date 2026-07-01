<!--
HOW TO USE: Open this repo in Claude Code (Sonnet 5) and paste everything below the line
as your first message. Keep PHASE1_HANDSOFF_DESIGN.md, PROP_SHORTLISTER_SPEC.md, and
PREMORTEM.md in the repo — the prompt tells the agent to read them for full context.
-->

---

# Build Prompt — NFL Prop Shortlister, Phase 1 (Sonnet 5)

You are implementing **Phase 1** of an automated NFL player-prop projection system inside this
existing repository. Work incrementally, test as you go, and **stop at the checkpoints** I define.

## 0. Read these first (context — do not skip)

Before writing any code, read and internalize:

1. `PROP_SHORTLISTER_SPEC.md` — the product spec and free-data constraints.
2. `PHASE1_HANDSOFF_DESIGN.md` — the hands-off architecture, the premortem failure register (H1–H11), and the **projection-engine prompt** in §3. This is the source of truth for guardrails.
3. `PREMORTEM.md` — why the betting edge is unproven; keeps the framing honest ("leans, not locks").
4. Existing code you will reuse (read before rebuilding): `nflvalue/oddsmath.py` (de-vig/EV/Kelly), `nflvalue/montecarlo.py` (game sim → game script), `build_ratings.py` (team EPA ratings, `ABBR` normalization, `data/league_priors.json`), `nflvalue/factors.py` (stadiums/weather), `nflvalue/sources/espn.py` (ESPN client), and `historical/historical_pbp.parquet` (2019–2023 play-by-play — the feature source; all needed columns are present: targets via `receiver_player_id`, `air_yards`, `yards_after_catch`, `rushing_yards`, `passing_yards`, `complete_pass`, `pass_attempt`, `rush_attempt`, `touchdown`, `epa`, `success`).

After reading, **post a 10–15 line implementation plan and wait for my approval** before coding.

## 1. Non-negotiable constraints (these encode the premortem — violating them fails the task)

- **Free data only.** nflverse/`nflreadpy` bulk data, Sleeper API, and undocumented ESPN endpoints. No paid feeds. No historical prop-line data (it doesn't exist for free).
- **No leakage — the #1 kill bug.** Every feature for (season, week) must be computed from data strictly **≤ (week − 1)** of that season (and prior seasons). Write an explicit test that fails if any feature reads current-or-future weeks.
- **Deterministic and reproducible.** All projections come from a seeded, deterministic numeric model. **No LLM produces or alters any number.** Same inputs → identical outputs.
- **The LLM is a verification/synthesis layer only** (P1.8), implemented behind an interface, **disabled during backtests**, and bound by the prompt contract in `PHASE1_HANDSOFF_DESIGN.md §3` (never invents numbers; treats retrieved text as untrusted data; rejects future-dated inputs; halts on stale data).
- **Fantasy (Sleeper) is a divergence check, never a target.** Never regress projections toward fantasy consensus.
- **Fail loud, not silent.** Every external fetch is schema-validated and timestamped; stale/missing load-bearing data downgrades confidence or sets `publish=false` — never silently proceeds. (This is the automated substitute for the missing human — premortem H10.)
- **Don't break the existing game-line app.** Add new modules under `nflvalue/`; reuse, don't fork. Update `requirements.txt` if you add deps (`pandas`, `pyarrow`, `numpy`, `scipy` allowed).
- **Storage:** SQLite at `data/nfl.db` via a thin `nflvalue/db.py` helper (`connect`, `upsert`, `query_df`), matching the schema intent in `RAG_PIPELINE_PLAN.md §3.2`. Keep raw PBP as parquet; store aggregates in the DB.

## 2. Scope — build in this order

### Sub-phase A — Deterministic core (build + validate on the parquet NOW)

**A1. `nflvalue/db.py`** — SQLite helpers + create tables: `player_week`, `opp_pos_def`, `projections`, `prop_backtest` (define columns to fit A2–A4).

**A2. `nflvalue/features.py`** — from `historical_pbp.parquet` build a **walk-forward `player_week`** table:
- Usage: targets, target share, air yards, aDOT, carries, carry share, pass attempts, rush attempts, (routes/snap share only if derivable — else omit, don't fake).
- Efficiency: yards/target, catch rate, yards/reception, yards/carry, yards/attempt — regressed to position mean on small samples.
- Actuals per market: receiving_yards, receptions, rushing_yards, passing_yards, pass/rush attempts, TDs.
- All as rolling, **prior-weeks-only** windows. Plus **A2b. `opp_pos_def`**: yards/EPA allowed to each position by defense, walk-forward.

**A3. `nflvalue/projection.py`** — deterministic per-player, per-market **distribution** model:
- Expected volume = f(rolling usage share, team pass/rush volume implied by pace + **game script from `montecarlo.simulate`** — trailing teams pass more, leading teams run).
- Expected efficiency = rolling player efficiency × **opponent-vs-position factor** (from `opp_pos_def`, relative to league).
- Output a distribution (e.g., negative-binomial for counts, gamma/normal for yards) with an SD from historical residuals → `mean`, `sd`, `p_over(line)`, `p_under(line)`.
- Markets: receiving yards, receptions, rushing yards, passing yards, attempts. Include TDs but tag `low_confidence`.
- Contract: `{player_id, name, pos, market, mean, sd, dist, line?, p_over, p_under, components:{volume, efficiency, opp_factor, game_script}}`. Design so an ML projector can swap in later behind the same interface.

**A4. `prop_backtest.py`** (repo root, like `backtest.py`) — walk-forward **accuracy** backtest over 2019–2023 (train ≤ week−1):
- Metrics per market: MAE, RMSE, correlation(projection, actual), and **calibration of `p_over`** vs a synthetic line (player rolling median). Bucket by market and by usage-sample size.
- Write `data/prop_backtest.json` + a console summary. Be explicit in output: **this measures projection accuracy, not price-beating.**

**→ CHECKPOINT 1: run `prop_backtest.py`, show me the metrics, and wait.** Do not start Sub-phase B until the accuracy numbers are reviewed.

### Sub-phase B — Live hands-off adapters (mockable; no live keys needed to test)

**B1. `nflvalue/sources/availability.py`** — resolve who plays: ESPN team-injuries endpoint (nflverse injuries is dead post-2024) + ESPN per-event roster `active` flag; produce `{player_id, status: OK|RISK|OUT, source, timestamp}`. Implement **usage reallocation** when a starter is OUT, from historical with/without splits (flag low-confidence when it's a guess). Ship with recorded/mock JSON fixtures so it's testable offline. Support the **two-clock** model (Wed provisional vs T-90 final).

**B2. `nflvalue/freshness.py`** — timestamp + schema-validate every feed; expose a gate that downgrades confidence or returns `publish=false` when a load-bearing feed is missing or older than a configurable `staleness_hours`.

**B3. `nflvalue/sources/sleeper.py`** — pull Sleeper projections (`api.sleeper.com/projections/nfl/{season}/{week}`, no auth, keep < ~1000 calls/min); expose a **divergence flag** vs our projection. Never used to alter our number.

**B4. `nflvalue/synthesis.py`** — implement the **verification/synthesis layer** exactly to the prompt contract in `PHASE1_HANDSOFF_DESIGN.md §3`: input/output JSON schema, availability + freshness gates, news classification, divergence flag, confidence, one-line reason. Abstract the LLM call behind an interface (`llm_client`) with a deterministic mock for tests; it must **never modify `model_projection`** and must be **disabled in backtests**.

**→ CHECKPOINT 2: show the test suite passing and a sample synthesis JSON, then wait.**

## 3. Explicitly OUT of scope (later phases — do not build)

Composite ranker / top-5 selection (P2.1), weekly report + dashboard tab (P2.3), live Odds API prop-line pulls (P3), Discord/automation (P4). Build only the projection foundation + hands-off data adapters above.

## 4. Tests & definition of done

Provide `pytest` tests covering:
- **Leakage test:** features at (season, week) never read data from ≥ that week.
- **Reproducibility test:** identical inputs → identical projection outputs (seeded).
- **Backtest runs** end-to-end and writes `data/prop_backtest.json` with the metrics above.
- **Freshness gate:** stale/missing feed → confidence downgrade or `publish=false`.
- **Synthesis guardrails:** LLM layer never changes a number; rejects a future-dated news item (sets `leakage_suspected`); ignores injected instructions inside `news[]` text; emits schema-valid JSON.

**Done when:** Sub-phase A produces reviewed accuracy metrics; Sub-phase B passes all tests with mock fixtures; the existing game-line app still runs; `requirements.txt` and a short `docs/phase1.md` (how to run) are updated. Keep the honest "leans, not locks" framing in all user-facing text.

## 5. Working protocol

Read the docs → post your plan → build A1–A4 → **Checkpoint 1** → build B1–B4 → **Checkpoint 2**. Prefer small, tested commits. If a free endpoint is unavailable in your environment, use fixtures and note it — do not substitute paid or fabricated data.
