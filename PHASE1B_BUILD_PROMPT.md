<!--
HOW TO USE: Run NEXT (Phase 1 Sub-phase A is already built + reviewed at Checkpoint 1).
Open the repo in Sonnet 5 and paste everything below the divider. Keep the design .md files present.
-->

---

# Build Prompt — Phase 1B: Checkpoint-1 hardening + hands-off adapters

**North star for the whole project:** an in-season **Wednesday Discord post that ranks the best
VALUE NFL props** from the current lines + your projections. This phase makes the projection layer
trustworthy and adds the "who's actually playing / is the data fresh" safety layer that a hands-off
system needs.

**State:** Phase 1 Sub-phase A is complete and merged — `nflvalue/db.py`, `nflvalue/features.py`
(`player_week`, `opp_pos_def`, `build_team_week`), `nflvalue/projection.py`, `prop_backtest.py`,
15 passing tests, `docs/phase1.md`, DB at `data/nfl_props.db`. Two known gaps from Checkpoint 1:
positions are **role-inferred** (passer→QB, rusher→RB, receiver→WR/TE **combined**), and **P(over)
calibration is loose** (shrinkage-to-role-mean lags real usage trends). Fix those first, then build the adapters.

## 0. Read first
- `PHASE1_HANDSOFF_DESIGN.md` (architecture, premortem H1–H11, and the **projection-engine prompt** in §3 — the contract for the synthesis layer).
- `PROP_SHORTLISTER_SPEC.md` §1 (free data), §5 (validation). `PREMORTEM.md` F9 (accuracy ≠ profit).
- Phase 1A code: `features.py`, `projection.py`, `db.py`, `prop_backtest.py`, `docs/phase1.md`.

Post a short plan and wait for approval before coding.

## 1. Constraints
All **Phase 1 non-negotiables still apply** (`PHASE1_BUILD_PROMPT.md §1`: no leakage, deterministic, LLM never alters a number, fail-loud, don't break the existing app, SQLite). Additionally: keep the deterministic numeric model backtestable; the synthesis LLM stays **disabled during backtests**.

## 2. Scope — build in order

### Part 1 — Hardening (do FIRST; these gate later phases)

**1B.0a Real positions.** Add a free position source — `nflreadpy`/`nfl_data_py` `load_rosters_weekly` (or `load_players`) — behind a small `nflvalue/sources/rosters.py` with a recorded fixture for offline tests. Replace role-inference; **split the REC bucket into WR vs TE**. Rebuild `opp_pos_def` on real positions and **re-run `prop_backtest.py`; report the accuracy deltas** vs the role-inferred baseline.

**1B.0b Cold-start exclusion.** Enforce a configurable minimum trailing sample (e.g., ≥ N games / min usage) so zero/low-history players are **excluded or hard-flagged `low_confidence`** and never eligible for the shortlist. Add a test.

**1B.0c Calibration fix.** Tighten `projection.py` P(over) calibration (levers: reduce shrinkage lag / weight recent usage more heavily / widen predictive SD to match residual variance). Report per-market calibration curves. **This is a prerequisite for Phase 3 edge math** — an edge is `model_prob − de-vigged line`, so bad calibration = fake edges.

**→ CHECKPOINT 1B-a: show accuracy deltas (real positions) + improved calibration curves, then wait.**

### Part 2 — Hands-off adapters

**1B.1 `nflvalue/sources/availability.py`** — who plays: ESPN team-injuries endpoint (nflverse injuries is dead post-2024) + ESPN per-event roster `active` flag → `{player_id, status: OK|RISK|OUT, source, timestamp}`. Usage **reallocation** when a starter is OUT (from historical with/without splits; flag low-confidence when it's a guess). Ship mock JSON fixtures. Support the **two-clock** model (Wed provisional vs T-90 final).

**1B.2 `nflvalue/freshness.py`** — timestamp + schema-validate every feed; a gate that downgrades confidence or returns `publish=false` when a load-bearing feed is missing or older than configurable `staleness_hours`. This is the automated substitute for the missing human (premortem H10).

**1B.3 `nflvalue/sources/sleeper.py`** — pull Sleeper projections (`api.sleeper.com/projections/nfl/{season}/{week}`, no auth, < ~1000 calls/min); expose a **divergence flag** vs our projection. Never used to alter our number (premortem H5).

**1B.4 `nflvalue/synthesis.py`** — the verification/synthesis layer exactly to `PHASE1_HANDSOFF_DESIGN.md §3`: input/output JSON schema, availability + freshness gates, news classification, divergence flag, confidence, one-line reason. Abstract the LLM behind an `llm_client` interface with a deterministic mock for tests; it **never modifies `model_projection`** and is **disabled in backtests**.

**→ CHECKPOINT 1B-b: tests passing + a sample synthesis JSON, then wait.**

## 3. Out of scope
Composite ranking (Phase 2), live prop-line pulls (Phase 3), dashboard/automation/Discord (Phase 4), RAG (Phase 5). Also: delete the stray empty `data/nfl.db`/`nfl.db-journal` if the environment allows.

## 4. Tests & definition of done
- **Positions test:** WR/TE split correctly; `opp_pos_def` uses real positions.
- **Calibration assertion:** P(over) calibration measurably improved vs the Sub-phase A baseline.
- **Cold-start test:** low-history players are excluded/flagged.
- **Freshness gate test:** stale/missing feed → downgrade or `publish=false`.
- **Synthesis guardrails:** LLM never changes a number; rejects a future-dated news item (`leakage_suspected`); ignores instructions embedded in `news[]`; emits schema-valid JSON.
- **Done when:** hardening deltas reviewed, adapters pass tests with fixtures, the numeric model stays reproducible + leak-free, and the existing app still runs.

## 5. Protocol
Read docs → post plan → Part 1 → Checkpoint 1B-a → Part 2 → Checkpoint 1B-b. Branch + small, tested commits.
