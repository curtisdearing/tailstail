<!--
HOW TO USE: Run AFTER Phase 2 is merged. Paste everything below the divider into Sonnet 5
in the repo. Keep the design .md files present.
-->

---

# Build Prompt — Phase 3: Live prop lines (free tier) + edge component + forward CLV logging

You are implementing **Phase 3**. **Phases 1–2 are complete and merged.** Add the live market
layer that supplies the `edge` component to the composite and starts the **only real edge test the
free data allows** — forward CLV logging.

## 0. Read first
- `PROP_SHORTLISTER_SPEC.md` §1 (free-data limits), §3 (edge component), §5 (validation, CLV, kill criteria).
- `PREMORTEM.md` (why CLV, not W/L, is the scoreboard; why props limit fast).
- `README.md` (The Odds API usage, `config.json` credits, `fetch_props`/`max_prop_games_per_run`).
- Reuse: `nflvalue/oddsmath.py` (de-vig, implied prob, EV), existing `nflvalue/sources/oddsapi.py`, `nflvalue/db.py`, the `composite.py` `edge` input from Phase 2.

Post a plan and wait for approval.

## 1. Constraints
All prior non-negotiables apply. Additionally:
- **Calibration gate (precondition):** do not wire edges into the ranker or publish them until the model's P(over) calibration passes the Phase-1B gate. An edge = `model_prob − de-vigged line`, so poor calibration manufactures fake edges. Verify calibration first, then make `edge` the dominant composite term (per Phase 2).
- **Never exceed the free tier (500 credits/mo).** A credit budgeter must hard-stop pulls; player props are a costly per-event endpoint — pull a **rotating subset** of games and tag the rest `no_market`.
- **CLV, not profit, is the success metric.** Log it honestly; do not cherry-pick.
- **No paid or scraped historical prop lines** — forward-only logging.
- **Personal use.** No affiliate links; if that ever changes, FTC disclosure is required (out of scope here).

## 2. Scope — build in order

**3.1 `nflvalue/sources/oddsapi_props.py`** — player-props client (per-event endpoint), credit-budgeted, rotating game selection. Persist snapshots to a `lines` table: `{ts, game_id, book, market, player_id, side, point, price}`. Idempotent upserts.

**3.2 Edge wiring** — de-vig each prop price (`oddsmath`), compute model-vs-market edge, feed it into `composite.py`'s `edge` component for games where a line was pulled; keep graceful `no_market` degradation elsewhere.

**3.3 `nflvalue/clv.py`** — forward CLV logger: for each posted lean, record the number/price available at decision time and the eventual closing number → `clv` table; compute per-lean and rolling average CLV. Approximate closing via the last snapshot before kickoff.

**3.4 `nflvalue/killcheck.py`** — after ~150 logged leans, report rolling CLV and whether leans beat a naive baseline; surface a clear **go / no-go** (per `PROP_SHORTLISTER_SPEC.md §5`).

**→ CHECKPOINT: pull a small live (or mock-fixture) sample, show an edge calculation, a `lines` row, and a `clv` entry, then wait.**

## 3. Out of scope
Dashboard/automation/Discord (Phase 4), RAG (Phase 5). Do not auto-place bets — ever.

## 4. Tests & definition of done
- **Budget test:** the credit budgeter never exceeds the configured monthly cap (simulate a month of runs).
- **De-vig test:** two-sided prop prices de-vig to sum-to-1 fair probs.
- **CLV math test:** CLV computed correctly for known open/close pairs.
- **Offline test:** all API calls mockable via fixtures (no key needed for tests).
- **Done when:** the edge component is live in the ranker, CLV logging works forward, the kill-check reports, and the free-tier budget is provably respected.

## 5. Protocol
Read docs → plan → 3.1–3.4 → checkpoint. Branch + small commits. If no API key in the environment, use fixtures and note it.
