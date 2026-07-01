<!--
HOW TO USE: Run AFTER Phase 3 is merged. Paste everything below the divider into Sonnet 5
in the repo. Keep the design .md files present.
-->

---

# Build Prompt — Phase 4: Dashboard tab + weekly automation (two-clock) + optional Discord

You are implementing **Phase 4**. **Phases 1–3 are complete and merged.** Make the system run
itself and surface results — hands-off, but safe. **This is the phase that delivers the project's end goal: the in-season Wednesday Discord post of the best value props, given current lines + projections.**

## 0. Read first
- `PHASE1_HANDSOFF_DESIGN.md` §1 (two-clock design), the premortem (H1–H11: dead injury feed, T-90 inactives, silent breakage, no-human-gate).
- `PROP_SHORTLISTER_SPEC.md` §6 (output), §8 (guardrails); `PREMORTEM.md` F10 (tout/legal), F13 (responsible gambling).
- Reuse: existing `nflvalue/dashboard.py` + `dashboard.html`, `nflvalue/freshness.py` and `availability.py` (Phase 1), `run.py`/`weekly.py` patterns, `nflvalue/db.py`.

Post a plan and wait for approval.

## 1. Constraints
All prior non-negotiables apply. Additionally:
- **Two clocks, enforced:** a Wed **provisional** run, and a **T-90-minutes-per-game** refresh that re-pulls availability and **auto-voids/downgrades** any lean whose player is Out/inactive. Nothing is "final" before the T-90 gate.
- **Fail loud:** the freshness gate halts or marks `publish=false` on stale/missing feeds — the pipeline must never post confident picks on stale data (premortem H1/H10).
- **Secrets never committed** (Discord webhook, API key) — env vars / untracked `config.local.json`.
- **Discord is optional, personal, unmonetized:** every post carries a "leans, not locks" disclaimer and a `1-800-GAMBLER` footer; no affiliate links; keep the whole venture off any `ufl.edu` email/resource/brand.

## 2. Scope — build in order

**4.1 Dashboard Props tab** — extend `dashboard.py`/`dashboard.html` with a Props tab: top-5 leans per game, context panel, confidence, edge/`no_market`, and a rolling **CLV** summary (from Phase 3). Same honest framing as the existing Weekly tab.

**4.2 `pipeline_weekly.py`** (repo root) — orchestrate the full chain idempotently: ingest → features → availability → projections → synthesis → composite → shortlist → report → dashboard. Upsert by natural key; safe to re-run. Wire the **freshness gates** so a bad feed stops publication.

**4.3 Two-clock scheduler** — a Wed provisional job + game-day T-90 refresh jobs (Cowork scheduled task or cron). The refresh voids inactive players and regenerates the affected game's shortlist + report.

**4.4 `nflvalue/notify.py` (optional, flag-gated)** — post the weekly leans to a private Discord webhook as an embed (game, market, side, line, confidence, edge, one-line reason, top context flag), with the disclaimer + RG footer. Skip cleanly if no webhook configured.

**→ CHECKPOINT: dry-run `pipeline_weekly.py` end-to-end on a historical week (no posting), show the generated report + dashboard, then a single test Discord post to a private channel. Wait.**

## 3. Out of scope
RAG query layer (Phase 5). No automated bet placement or money movement — the tool informs, the human acts.

## 4. Tests & definition of done
- **Idempotency test:** re-running the pipeline yields the same DB state.
- **Two-clock test:** a player flipped to inactive at T-90 is removed/downgraded and the report regenerates.
- **Freshness-halt test:** a stale/missing feed sets `publish=false`.
- **No-secrets test:** repo scan finds no webhook/key committed.
- **Discord payload test:** embed formats correctly against a mock webhook; disclaimer + RG footer present.
- **Done when:** one command runs the weekly pipeline, the two-clock refresh works, the dashboard tab is live, Discord is optional and safe, and secrets stay untracked.

## 5. Protocol
Read docs → plan → 4.1–4.4 → checkpoint. Branch + small commits. Include a monthly-loss-cap reminder and the RG footer in user-facing surfaces.
