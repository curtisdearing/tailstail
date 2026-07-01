# NFL Value Engine — RAG + ML + Automation Build Plan

Plan for evolving the current project into a data-warehoused, ML-scored,
RAG-queryable, self-updating mismatch pipeline. Nothing here is built yet — this
is the blueprint. Honest framing up front: the walk-forward backtest already
showed closing lines are efficient, so the goal is **calibrated projections,
prioritized research leans, and qualitative context** — not a promise of beating
the close. The durable edge is line-shopping soft prices + the qualitative tags.

---

## 0. Foundation already built (reuse, don't rebuild)

- **Data on disk:** `historical/` play-by-play (2019–2023, has `epa`, drives) +
  schedules with closing lines/results.
- **EPA-capable ratings:** `build_ratings.py` (walk-forward, opponent-adjusted,
  EPA-blended) → `data/ratings_current.json`, `backtest_games.json`.
- **Monte Carlo + backtest:** `nflvalue/montecarlo.py`, `backtest.py`.
- **Weekly projections on real lines:** `weekly.py` → `data/weekly.json`.
- **Live sources:** `sources/oddsapi.py` (lines), `sources/espn.py` (injuries),
  `sources/weather.py`; **stadium coordinates** in `factors.py` (reuse for travel).
- **Dashboard + scheduled tasks + Discord-style notify** hooks.

The new work is a **database + feature store + ML scorer + mismatch report +
RAG query layer + weekly Discord automation** layered on top.

---

## 1. Locked decisions

| Area | Choice | Notes |
|------|--------|-------|
| Storage | **SQLite** (`data/nfl.db`) | one file, easy joins, RAG/NL-SQL friendly |
| Data pulls | **nflreadpy** | current package; `nfl_data_py` is now **deprecated** |
| RAG | **Report-first**, NL-SQL optional later | compact weekly markdown → analyze here |
| Automation | **Cowork weekly (Wed) + Discord webhook** | your pick |

> **Package note:** your `download_history.py` uses `nfl_data_py`, which nflverse
> deprecated in 2024 in favor of **`nflreadpy`** (`load_pbp`, `load_schedules`,
> `load_player_stats`, `load_rosters_weekly`, `load_injuries`; Polars-based, use
> `.to_pandas()`). Plan targets `nflreadpy`; the old package still works if preferred.

---

## 2. Target data flow

```
                 nflreadpy (pbp, schedules, weekly stats, rosters, injuries)
                        │
The Odds API ──┐        ▼
ESPN injuries ─┼──►  ingest.py  ──►  SQLite  (data/nfl.db)
weather        │                     ├─ games         (schedule + lines + results)
manual notes ──┘                     ├─ team_week     (EPA off/def, pass/rush, pace)
                                     ├─ injuries      (weekly status by player)
                                     ├─ rosters       (player↔team↔pos)
                                     ├─ lines         (market snapshots over time)
                                     ├─ manual_notes  (revenge / scheme / motivation tags)
                                     ├─ mismatch      (weekly unit-vs-unit findings)
                                     └─ signals       (ML output + calibrated confidence)
                                              │
             features.py (walk-forward join) ─┤
                                              ▼
                    models.py  (XGBoost / Random Forest → calibrated prob)
                                              │
        ┌─────────────────────────────────────┼───────────────────────────┐
        ▼                                     ▼                           ▼
 mismatch.py → reports/week_S_W.md     notify.py → Discord (>60%)    dashboard (Signals tab)
        │
        └── compact report = the RAG context pack you analyze here
```

Weekly orchestrator (`pipeline_weekly.py`) runs the whole chain; a Cowork
scheduled task fires it every Wednesday.

---

## 3. Data layer

### 3.1 Ingestion — `ingest.py` (nflreadpy → SQLite)

- `load_pbp(seasons)` → trim to needed columns → aggregate to `team_week` (don't
  store 400-col raw PBP in SQLite; keep raw as parquet, store aggregates in DB).
- `load_schedules(seasons)` → `games` (schedule + closing lines + results + rest).
- `load_player_stats()` (weekly) → player weekly EPA/usage for props later.
- `load_rosters_weekly()` → `rosters`; `load_injuries()` → `injuries`.
- **Incremental & idempotent:** upsert by natural key (e.g. `game_id`,
  `(season,week,team)`); only pull seasons/weeks not already current.
- Team-abbreviation normalization reused from `build_ratings.ABBR`.

### 3.2 SQLite schema (sketch)

```sql
CREATE TABLE games (               -- one row per game
  game_id TEXT PRIMARY KEY, season INT, week INT, gameday TEXT,
  home TEXT, away TEXT, home_score REAL, away_score REAL,
  spread_line REAL, total_line REAL, home_ml INT, away_ml INT,
  home_rest INT, away_rest INT, div_game INT, roof TEXT, temp REAL, wind REAL);

CREATE TABLE team_week (           -- efficiency, walk-forward friendly
  season INT, week INT, team TEXT,
  off_epa REAL, def_epa REAL, off_pass_epa REAL, off_rush_epa REAL,
  def_pass_epa REAL, def_rush_epa REAL, pass_rate REAL, proe REAL,
  success_rate REAL, plays INT, sec_per_play REAL,
  PRIMARY KEY (season, week, team));

CREATE TABLE injuries (
  season INT, week INT, team TEXT, player_id TEXT, player TEXT, pos TEXT,
  report_status TEXT, practice_status TEXT, updated TEXT,
  PRIMARY KEY (season, week, player_id));

CREATE TABLE lines (               -- market snapshots to track line moves / CLV
  ts TEXT, game_id TEXT, book TEXT, market TEXT, side TEXT,
  point REAL, price INT);

CREATE TABLE manual_notes (        -- the qualitative / non-quantifiable table
  id INTEGER PRIMARY KEY, season INT, week INT, scope TEXT,   -- 'game'|'team'|'player'
  ref TEXT, tag TEXT,                                          -- 'revenge','scheme','motivation','weather','coaching'
  note TEXT, weight REAL DEFAULT 0.0, created_at TEXT);

CREATE TABLE mismatch (            -- weekly unit-vs-unit findings
  season INT, week INT, game_id TEXT, unit TEXT,              -- e.g. 'BUF pass_off vs MIA pass_def'
  off_metric REAL, def_rank INT, delta REAL, severity REAL);

CREATE TABLE signals (             -- ML output
  season INT, week INT, game_id TEXT, market TEXT, side TEXT,
  model_prob REAL, confidence REAL, edge_vs_line REAL,
  features_json TEXT, created_at TEXT,
  PRIMARY KEY (season, week, game_id, market, side));
```

Helpers in `nflvalue/db.py` (connect, `upsert`, `query_df`).

### 3.3 Efficiency metrics (EPA, not raw yards)

`team_week` is built from PBP with **EPA/play** for offense and defense, split by
pass/rush, plus pass-rate-over-expected (PROE), success rate, and pace. All
consumed as **rolling, prior-weeks-only** values so nothing leaks.

---

## 4. Mismatch analysis — `mismatch.py`

For each upcoming game, compute and rank:

1. **Unit EPA vs opponent rank:** offense pass EPA vs opponent **pass-defense EPA
   rank**; same for rush. Big positive delta vs a bottom-ranked defense = mismatch.
2. **Tendency vs vulnerability:** team pass/run split & PROE vs what the opponent
   defense allows most (e.g. run-heavy team into a stout run D but soft pass D).
3. **Schedule context:** rest-day diff, short week (Thu), off-bye, and **travel
   distance / time-zone shift** computed from stadium coordinates (`factors.STADIUMS`).
4. **Market context:** model projected spread/total (from the MC/ratings) minus
   the **current sportsbook line** → where the model most disagrees.
5. **Qualitative join:** attach any `manual_notes` tags (revenge, scheme, key
   injury) for that game/team/player.

**Output:** rows into `mismatch` + a compact `reports/week_{S}_{W}_mismatch.md`
(dense: top mismatches, deltas, context, tags) — this is the RAG context pack.

---

## 5. ML scoring — `models.py` (XGBoost / Random Forest)

- **Targets (start with 2, keep honest):**
  - `cover_home` — classification, calibrated probability → drives the >60% alert.
  - `margin` and `total` — regression, to derive projected line vs market edge.
- **Features (all walk-forward):** rolling off/def EPA (pass/rush), PROE, success
  rate, pace; matchup deltas (off − opp def); rest/short-week/travel; injury-impact
  flags (key player Out by position weight); weather; the market line itself.
- **Validation:** expanding-window walk-forward by season/week (train ≤ week−1);
  **probability calibration** (isotonic/Platt); report ROC-AUC, Brier, ATS ROI vs
  close, and calibration curve — the honest scoreboard (expect ≈ market-efficient).
- **Output:** `signals` rows with `model_prob`, calibrated `confidence`, and
  `edge_vs_line`. `confidence > 0.60` flags a high-conviction mismatch.
- Artifacts saved under `models/` (joblib); needs `xgboost scikit-learn`.

> Expectation: ML rarely beats closing spreads. Its value here is **calibrated
> confidence + a ranked shortlist** to combine with line-shopping and the
> qualitative tags — not a standalone money machine.

---

## 6. RAG / LLM query layer

**Primary (now):** the weekly compact markdown report (§4) + a small
`signals` summary = a tiny, high-signal context pack. You drop it in the folder
and I read/compare it here. Smallest context, highest accuracy — your stated goal.

**Optional (later phase):** `nflvalue/rag.py` with two modes:
- **NL-to-SQL (querychat-style):** a schema-aware prompt turns plain-English
  questions ("run-heavy teams facing bottom-10 run defenses on short weeks with a
  revenge tag") into SQL over `nfl.db`. Runs via Claude here or a small API loop.
- **Vector retrieval:** embed weekly reports + `manual_notes` into a local vector
  store (Chroma/FAISS) for semantic recall across seasons. Only worth it once many
  reports/notes accumulate.

---

## 7. Notifications — `notify.py` (Discord)

- `post_discord(webhook_url, week, signals)` posts a concise embed: game, market,
  side, confidence, edge vs line, top mismatch reason, any tag.
- Trigger: any `signal` with `confidence > 0.60` (configurable).
- Secret: `discord_webhook_url` via env var / untracked `config.local.json`
  (never commit the webhook).

---

## 8. Automation — Cowork weekly (Wednesday)

- One scheduled task, `cron 0 9 * * 3` (Wed 9am), runs `pipeline_weekly.py`:
  ingest new week → recompute `team_week`/ratings → mismatch → models → signals →
  report → Discord (>60%) → refresh dashboard.
- **Wednesday rationale:** by then the week's odds and first injury reports have
  solidified.
- Idempotent: safe to re-run; upserts by key.
- I can set this up here once Phase 4 lands (you already chose this path).

---

## 9. Dashboard integration

New **Signals / Mismatch** tab fed from `signals` + `mismatch`: ranked weekly
mismatches, model confidence, edge vs current line, and qualitative tags — with the
same honest "lean, not lock" framing as the Weekly tab.

---

## 10. Phased milestones

| Phase | Deliverable | Rough effort |
|------|-------------|--------------|
| **1. Warehouse** | `db.py` schema + `ingest.py` (nflreadpy → SQLite) + `team_week` EPA + `manual_notes` | small–med |
| **2. Mismatch report** | `mismatch.py` + weekly markdown report + dashboard tab | small |
| **3. ML signals** | `features.py` + `models.py` (XGB/RF, calibrated, walk-forward) → `signals` | med |
| **4. Automate + alert** | `notify.py` Discord + `pipeline_weekly.py` + Cowork Wed schedule | small |
| **5. RAG query (optional)** | NL-to-SQL helper and/or vector store over reports+notes | med |

Suggested start: **Phase 1 + 2** (get the DB, EPA warehouse, and the compact
mismatch report you can analyze here) — highest value, unblocks everything else.

---

## 11. Honest expectations & risks

- **Efficiency:** closing lines already price the EPA story; treat signals as
  calibrated research leans. Real edge = soft/early prices + CLV + qualitative tags.
- **Leakage:** every feature must be prior-weeks-only; enforce in `features.py`.
- **Data quirks:** injury granularity varies; abbreviation drift (OAK→LV, SD→LAC);
  props need player-level modeling and bigger samples.
- **Secrets:** Discord webhook + any paid API keys stay out of the repo.
- **Sandbox network:** confirm `nflreadpy` can fetch nflverse releases in the run
  environment; fall back to your uploaded parquet if blocked.

---

## 12. Open questions

1. Discord: do you have a webhook URL, or want me to document creating one?
2. ML target priority — cover probability first, or projected margin/total first?
3. Seasons to warehouse — just 2019–2023 for now, or pull 2024–2025 too (true "7 years")?
4. Manual notes — happy to enter via a tiny `note.py add` CLI, or edit SQL directly?
