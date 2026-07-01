<!--
HOW TO USE: Run AFTER Phase 1 is merged. Open the repo in Sonnet 5 and paste everything
below the divider. Keep the design .md files in the repo for context.
-->

---

# Build Prompt — Phase 2: Composite ranker + top-5 leans + context panel + weekly report

You are implementing **Phase 2**. **Phase 1 is complete and merged** (deterministic
`nflvalue/projection.py`, `nflvalue/features.py`, `nflvalue/synthesis.py`, `nflvalue/db.py`,
`prop_backtest.py`) and Phase 1B (positions, calibration, availability, freshness, Sleeper, synthesis). Build the weekly product surface on top of it.

**North star (project end goal):** an in-season **Wednesday Discord post ranking the best VALUE NFL props** from current lines + your projections. This phase builds the ranking + report that post is made of; real lines arrive in Phase 3, delivery in Phase 4.

## 0. Read first
- `PROP_SHORTLISTER_SPEC.md` §3 (composite score), §4 (context panel), §6 (weekly output).
- `PHASE1_HANDSOFF_DESIGN.md` (guardrails, synthesis output contract).
- Phase 1 outputs: the `projection.py` contract and `synthesis.py` JSON (status, confidence, context_notes, divergence_flag).
- Reuse: `nflvalue/oddsmath.py` (de-vig → implied prob), `nflvalue/db.py`, existing `nflvalue/dashboard.py` render patterns.

Post a short plan and wait for approval before coding.

## 1. Constraints
All **Phase 1 non-negotiables still apply** (re-read `PHASE1_BUILD_PROMPT.md §1`: no leakage, deterministic, LLM never alters a number, fail-loud, don't break the existing app). Additionally:
- **Context flags carry ZERO ranking weight.** Personal/news context is displayed only. Write a test that proves the composite score is identical with and without any context notes.
- **Selection honesty:** every top-5 must record how many candidates were screened (`X of N`). Never hide the denominator (premortem: multiple comparisons).
- **Leans, not locks** in all user-facing text.

## 2. Scope — build in order

**2.1 `nflvalue/candidates.py`** — for a given (season, week), enumerate every eligible player-market candidate per game from `projection.py` (filter to a minimum usage-sample so scrubs don't appear).

**2.2 `nflvalue/composite.py`** — 0–100 composite from three components:
- `edge` — model `p_side` minus de-vigged implied prob from a prop line **(optional input; defaults to unavailable — Phase 3 supplies real lines).** When absent, drop this component and tag `no_market`.
- `confidence` — projection distance from the line in SD units, scaled.
- `matchup` — opponent-vs-position rank, pace, game-script fit.
Weights configurable in `config.json`. Deterministic. Output per candidate: `{composite, edge?, confidence, matchup, components}`. **Design the defaults so that once Phase 3 supplies real lines, `edge` becomes the dominant term** — "best" must mean best value vs the line, not highest projection. Only cold-start-eligible players (Phase 1B min-sample threshold) may appear.

**2.3 `nflvalue/shortlist.py`** — rank a game's candidates, take **top 5**, attach `screened: "5 of N"`. Pull per-player context (injury status + `personal_context` notes from `synthesis.py`) into a **context panel** that rides alongside but does not affect ranking.

**2.4 `reports/` generator (`nflvalue/report.py` → `reports/props_week_{S}_{W}.md`)** — per game: the 5 leans (player · market · line · side · projection · confidence · edge-or-`no_market` · composite · one-line reason · "5 of N"), then a **Context** block per game labeled "context only — not scored." This markdown doubles as the RAG context pack.

**→ CHECKPOINT: generate `reports/props_week_2023_10.md` from historical data, show it to me, and wait.**

## 3. Out of scope
Live prop-line pulls (Phase 3), interactive dashboard tab (Phase 4), automation/Discord (Phase 4), RAG (Phase 5).

## 4. Tests & definition of done
- **Context-neutrality test:** composite score unchanged with/without context notes.
- **Determinism test:** same inputs → same shortlist.
- **Screen-count test:** `N` equals actual candidates considered.
- **Report smoke test:** generates valid markdown for a historical week from fixtures.
- **Done when:** a historical week's report renders correctly, ranking is deterministic and context-neutral, and the existing app still runs.

## 5. Protocol
Read docs → post plan → build 2.1–2.4 → checkpoint. Small, tested commits on a branch.
