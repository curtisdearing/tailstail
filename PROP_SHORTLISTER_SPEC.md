# NFL Prop Shortlister — de-risked build spec (free data)

*The constructive output of `PREMORTEM.md`. Goal: every week, for each game, surface the
**5 best-looking player-prop leans** by a composite score, plus a **context panel** of
injury/news/personal flags. Free data only. Leans, not locks — "the stats give you the lucky
bets," honestly labeled. Not financial advice; 1-800-GAMBLER.*

---

## 0. Your locked decisions (drive this whole spec)

| Decision | Choice | Consequence baked in below |
|---|---|---|
| Data | **Free sources only** | No paid prop-line history. Validate on 10+ yrs of player *results*; get live prop lines from The Odds API free tier (credit-budgeted). |
| Ranking | **Composite score** | Blend edge-vs-line + model confidence + matchup/situation into one 0–100 rank. |
| News/personal factors | **Mentioned with the 5, not scored** | Injury + matchup + usage drive the picks; birthday/revenge/bereavement/etc. appear in a context panel with **zero** weight on the ranking. |

---

## 1. What "free data only" actually gives you

**You CAN (free, 10+ years):**
- `nflreadpy`/`nfl_data_py`: play-by-play, weekly player stats, rosters, **snap counts** (~2012+), **Next Gen Stats** (air yards, targets, separation; ~2016+), injuries, schedules **with closing game lines** (spread/total/ML). This is enough to model prop *outcomes* deeply.
- ESPN (no key): live injuries/news. Open-Meteo (no key): weather. Already wired in `nflvalue/sources/`.
- The Odds API **free tier** (500 credits/mo): live game lines cheap; **player props are a per-event endpoint that costs more**, so you can only pull props for a handful of games per week.

**You CANNOT (without paying):**
- Historical player-prop **lines**. They start ~2019 (SportsDataIO) / May 2023 (The Odds API), both paid. So **you cannot backtest "did we beat the prop price" on 10 years.** You can backtest **projection accuracy** on 10 years, and measure price-beating only **going forward** on the games you can afford to pull. This is the single biggest honest constraint of the free build — design around it, don't pretend around it.

---

## 2. Projection model (per prop market, walk-forward)

Project each player's **stat distribution** (not just a point), then read off P(over)/P(under) for a line.

**Markets, in order of how honestly modelable they are (start at the top):**
1. Receiving yards, receptions, rushing yards, passing yards — volume-driven, stable, model these first.
2. Pass completions/attempts, rush attempts — very stable (usage).
3. TDs / anytime-TD — high variance, red-zone-dependent — flag as **low-confidence**, include last.

**Features (all rolling, prior-weeks-only — leakage is a kill bug):**
- **Usage/role:** targets, target share, air yards, carries, snap %, route participation (the biggest signal for props).
- **Efficiency:** yards/target, catch rate, yards/carry, aDOT — regressed to position mean on small samples.
- **Opponent vs position:** yards/EPA allowed to the position (WR/TE/RB), pressure/coverage tendencies.
- **Volume drivers:** team pace, PROE, and **projected game script** — pull the projected spread/total from the existing Monte Carlo (`nflvalue/montecarlo.py`); trailing teams pass more, leading teams run.
- **Injury-driven usage shift:** if a starter is Out, redistribute his targets/carries to the backup (this is where real prop edges hide, and it's free from the injury feed).
- **Weather** for passing/receiving (wind especially).

**Output per player-market:** projected mean + SD → `P(over line)`, `P(under line)`, and a projection z-distance from the line.

> Reuse: the drive-level MC gives you game script and totals; `factors.py` has weather/stadium; `oddsmath.py` has de-vig/EV. You're adding a **player layer** on top, not rebuilding.

---

## 3. Composite score (0–100) → top 5 per game

Three components, combined:

1. **Edge vs line** *(only when a live prop line was pulled)* — model P(side) minus the de-vigged implied prob from the prop price. This is the market-disagreement signal.
2. **Model confidence** — how far the projection sits from the line in SD units (z-score), scaled; tighter, higher-conviction distributions score higher.
3. **Matchup/situation** — opponent rank vs the position, pace/PROE fit, injury-vacated usage, game-script fit.

`composite = w1·edge + w2·confidence + w3·matchup`, normalized to 0–100. Rank all candidate props in a game, take the **top 5**.

**Two premortem guardrails wired into the score itself:**
- **Selection honesty:** report *how many* candidate props were screened to surface each top-5 (e.g., "5 of 41"). The more you screen, the more the top of the list is noise — show that number so you never fool yourself.
- **Graceful degradation:** if no live prop line exists for a game (free-tier credits ran out), drop component 1 and rank on confidence + matchup only, clearly tagged "no market pulled."

---

## 4. Context panel (mentioned, never scored)

Below each game's 5 picks, a **Context** block lists per-player qualitative flags, pulled from the existing `manual_notes` table + ESPN news:
- injury/practice status, snap-count trend, role change;
- revenge spot (vs former team), coordinator/scheme familiarity;
- personal/news items you want to know about — birthday, bereavement, contract/holdout, new baby, etc.

Labeled: **"Context only — not part of the composite score."** This is exactly your ask: you see the story next to the numbers, but a birthday never moves a bet. (Your engine already shrinks such factors to ~0 — this keeps them visible without letting them overfit.)

---

## 5. Validation on free data (the honest scoreboard)

1. **Accuracy backtest (10+ yrs, free):** walk-forward over player results — does the projection predict the actual stat? Report MAE, correlation, and **calibration of P(over)** against a *synthetic* line (e.g., the player's rolling median or the Vegas-implied number where the game line lets you infer it). This proves the projections are *sound*; it does **not** prove you beat prop prices.
2. **Forward price-test (the only real edge test on free data):** from week 1, log every top-5 lean + the live prop line pulled + the result. Approximate CLV where you have open→late lines. After ~100–150 logged props, check whether the leans beat the number.
3. **Kill criteria:** if forward leans don't beat a naive baseline (take the model's side at the posted number) after ~150 bets, the composite isn't finding real prop edges — revert to "projection/entertainment tool," stop staking.

---

## 6. Weekly output

- `reports/props_week_{S}_{W}.md` — per game: the 5 leans (player · market · line · side · projection · edge · confidence · composite · one-line matchup reason · "X of N screened"), then the Context panel. This doubles as the RAG context pack and the Discord-embed source.
- Dashboard **Props** tab fed from the same rows.
- Optional Discord post — **personal, unmonetized**; if you ever add an affiliate link, disclose it (FTC); keep it off any `ufl.edu` resource/email.

---

## 7. Phased build (all on free data already partly on disk)

| Phase | Deliverable | Notes |
|---|---|---|
| **P1** | Player projection layer + 10-yr **accuracy** backtest | Uses parquet on disk (2019–2023) now; extend seasons if `nflreadpy` fetch works in-env, else keep 5 and grow forward. |
| **P2** | Composite ranker → top-5 report + Context panel | `manual_notes` for personal flags; ESPN for injuries/news. |
| **P3** | Live prop lines via Odds API free tier (credit-budgeted) → edge component + forward CLV log | Prioritize/rotate games; tag "no market pulled" otherwise. |
| **P4** | Dashboard Props tab + optional personal Discord | Same "leans, not locks" framing. |

**Suggested start: P1 + P2** — you immediately get the weekly top-5 leans with context, and an honest accuracy scoreboard, before spending a single API credit.

---

## 8. Guardrails carried over from the premortem

- **Leans, not locks.** Free-data props = a ranked, well-contextualized shortlist; treat variance as variance ("it can be luck" — by design).
- **Selection bias is the main enemy** of a "top-5 of many" rule — pre-commit the market set, score out-of-sample, show the screen count.
- **If you ever stake:** quarter-to-half Kelly on a *shrunk* edge, hard per-bet cap, fixed monthly loss limit you can't override.
- **Props limit fast** — expect $15–$50 caps if you win; the value here is the research/shortlist, not scalable income.
