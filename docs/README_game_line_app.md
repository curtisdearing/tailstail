# NFL Value Engine

A self-running NFL value-betting dashboard. It pulls odds from multiple
sportsbooks, removes the bookmaker margin to estimate fair prices, hunts for
mispriced lines and player props, adjusts for situational factors
(injuries, weather, revenge spots, matchups), sizes bets with Kelly, and
**learns** from finished games by adjusting its own factor weights.

It runs with **zero setup** on realistic demo data, and switches to **live data**
when you paste in a free API key.

> **Reality check.** Sportsbook lines are very efficient. This tool finds and
> sizes *potential* edges and gets smarter as results come in — it is **not** a
> guaranteed money-maker. Bet only what you can afford to lose. Help: 1-800-GAMBLER.

---

## Quick start (demo, no keys)

From this folder:

```bash
python3 update_results.py --simulate-weeks 15   # learn from 15 simulated weeks, then build today's slate
```

Then open **`dashboard.html`** in any browser. That's it. The page auto-refreshes,
and re-running the pipeline updates what it shows.

To just rebuild the current slate without re-learning:

```bash
python3 run.py --demo
```

Nothing to install — it uses only the Python standard library (Python 3.8+).

---

## What you're looking at

The dashboard opens on the **Weekly Projections** board and has these tabs:

- **Weekly Projections** – the main view. Every game for a week, projected on its
  **real line**: projected score, win %, the market spread/total, and the model's
  picks (straight-up / against the spread / total). Pick a week from the dropdown;
  as results come in each week the picks are graded ✓/✗ and a **season-to-date
  record builds** (SU %, ATS, projection error) — exactly like fantasy projections.
- **Value Bets** – game-line picks with positive expected value, the best price
  across books, fair vs. projected probability, the Kelly stake, and why.
- **Player Props** – the same, for passing / rushing / receiving yards and TDs.
- **Monte Carlo** – each game simulated thousands of times from team ratings:
  win %, projected score, cover/over probabilities, and the margin distribution.
- **All Games** – every game with consensus lines plus weather and injury context.
- **Model & Learning** – ROI and calibration of recommended bets and the learned
  factor weights (they move as games finish — the model learning).
- **Backtest** – how the Monte Carlo did across 1,390 historical games.

---

## Weekly projections (the main view)

Built from your real historical lines (and live lines in season):

```bash
python3 build_ratings.py            # one-time: ratings from 2019-2023 data
python3 weekly.py                   # latest season (2023), all weeks graded
python3 weekly.py --season 2021     # any season in your data
python3 weekly.py --through 6       # show weeks 1-6 graded + week 7 as upcoming
python3 run.py --demo               # refresh the dashboard
```

Each week is projected with the Monte Carlo using **only data through the prior
week** (walk-forward — no peeking), then graded as the real results land, and the
season record grows. Step through the weeks with the dropdown to watch it build.

**In season**, point it at the live slate so it projects this week's real lines and
updates every week as games finish:

```bash
python3 weekly.py --live            # needs your odds_api_key + an active NFL week
```

Honest expectations carry over from the backtest: the model nails **~62% of
straight-up winners** and projects final margins to about **10–11 points** average
error, but against the spread it's ~50% — the closing line is sharp. Treat the
board as projections and lean signals, not guaranteed winners.

---

## How it works

1. **De-vig.** Each book's two-sided price implies > 100% — the margin. We strip
   it out to get that book's fair probability, then blend books into a consensus
   (weighting sharp books like Pinnacle more).
2. **Line shop.** We take the *best* price available across all books.
3. **Adjust.** Situational factors nudge the fair probability up or down. Each
   factor has a weight; the total nudge is capped so we never stray too far from
   a sharp market.
4. **Find value.** Expected value = `fair_prob × best_price − 1`. Anything above
   the threshold (default +3%) is a recommended bet, staked with fractional Kelly.
5. **Learn.** When games finish, `update_results.py` grades every prediction and
   runs one logistic-regression step per result, pushing each factor's weight
   toward whatever actually predicted winners. Useless factors (e.g. birthdays)
   shrink toward zero; predictive ones grow.

---

## Going live (free)

Three data sources, all with free tiers. Only the odds feed needs a key.

### 1. Odds — The Odds API (free key required)

1. Go to **https://the-odds-api.com/** and click *Get API Key* (the free
   "Starter" plan: **500 credits/month**, all sports, all markets).
2. Check your email for the key.
3. Open **`config.json`** and paste it:
   ```json
   "odds_api_key": "PASTE_YOUR_KEY_HERE"
   ```
4. Run it live:
   ```bash
   python3 run.py --live
   ```

The header badge flips from **DEMO** to **LIVE**.

> **NFL season note:** it's currently the offseason, so live NFL games won't
> appear until preseason/regular season. Use demo mode until then.

### 2. Weather — Open-Meteo (no key)
Automatic. Looks up wind/precip/temperature at each outdoor stadium; domes are
skipped.

### 3. Injuries & matchup — ESPN (no key)
Automatic. Pulls the league injury report and a light team-strength signal from
standings.

### Watch your credits
Game lines are cheap (~3–4 credits/refresh). **Player props use a per-event
endpoint and cost more**, so props are limited to a few games per run. With props
on, run roughly **once or twice a day** to stay under 500/month. Tune in `config.json`:

- `"fetch_props": false` – game lines only (cheapest)
- `"max_prop_games_per_run": 4` – how many games to pull props for
- `"prop_markets": [...]` – which prop types

---

## Monte Carlo simulation (real data)

A drive-by-drive Monte Carlo built from **5 seasons of real play-by-play**
(2019–2023, via `nfl_data_py`, in `historical/`). It builds team power ratings,
then simulates each game thousands of times — possession by possession, scoring
real integer points so the margin distribution spikes on **3 and 7** like real
NFL football.

```bash
pip install pandas pyarrow numpy        # one-time, for the MC components
python3 build_ratings.py                # ratings + league priors from history
python3 backtest.py --sims 6000         # walk-forward test vs 1,390 games
python3 run.py --demo                   # MC projections now show on the dashboard
```

Two new dashboard tabs appear:

- **Monte Carlo** – each game's simulated win probability, projected score, cover
  and over probabilities vs the market, and the margin distribution.
- **Backtest** – how the model did against 1,390 historical games with their
  closing lines.

### What the backtest honestly shows

The simulator's **win probabilities are well-calibrated** (Brier ≈ 0.23; when it
says 70%, those teams win ~72%). Its margin forecast correlates ~0.37 with actual
results — real signal.

**But it does not beat the closing line.** The closing spread correlates ~0.43
with results — sharper than the model — so betting *into* closing numbers loses
the vig (spread ROI ≈ −5%, totals/moneyline worse). Blending the model into the
market doesn't improve on the market alone. This is the honest, expected finding:
**NFL closing lines are extremely efficient.**

So the Monte Carlo is used as a **calibrated fair-value second opinion and
scenario tool** — not as a signal to bet against the market. Real edges come from
**line-shopping softer prices across books** (the Value Bets tab) and from
derivative/key-number spots, *not* from out-predicting the close. The tool tells
you this truth instead of hiding it.

> Walk-forward = each game is simulated using only ratings known *before* kickoff,
> so the backtest has no look-ahead bias. To extend to a true 7 seasons, add
> 2024–2025 to `historical/download_history.py`, re-run it, then `build_ratings.py`.

---

## Make it run itself

**The open dashboard already auto-refreshes** (every `refresh_seconds`, default 90).
Leave the tab open; whenever the pipeline reruns, the page shows fresh numbers.

To refresh the data automatically on a schedule, run these two commands on a timer:

```bash
python3 run.py                 # refresh odds + rebuild the slate/dashboard
python3 update_results.py      # grade finished games + learn (live)
```

Options:

- **Ask me (Cowork) to schedule it** — I can set up a daily task that runs both.
- **macOS/Linux cron** — e.g. refresh hourly, grade each morning:
  ```cron
  0 * * * *  cd "/path/to/nfl gambling" && /usr/bin/python3 run.py
  30 9 * * * cd "/path/to/nfl gambling" && /usr/bin/python3 update_results.py
  ```
- **Windows** — Task Scheduler running the same two commands.

---

## Configuration (`config.json`)

| Key | Meaning |
|-----|---------|
| `odds_api_key` | Your The Odds API key (blank = demo mode) |
| `ev_threshold` | Minimum EV to recommend a bet (0.03 = 3%) |
| `edge_shrinkage` | How much of the model's disagreement with the market to trust (0–1) |
| `kelly_multiplier` | Fraction of full Kelly to stake (0.15 = very conservative) |
| `max_stake_pct` | Hard cap on any single stake (% of bankroll) |
| `sharp_books` / `sharp_weight` | Books trusted more in the consensus |
| `learning_rate` / `l2` | How fast / how regularized the weight updates are |
| `refresh_seconds` | Dashboard auto-reload interval |
| `fetch_props`, `max_prop_games_per_run`, `prop_markets` | Player-prop controls |

You can also set the key via an `ODDS_API_KEY` environment variable.

---

## Project layout

```
run.py                 build the slate + dashboard
update_results.py      grade finished games + learn
build_ratings.py       team ratings + league priors from historical data  [needs pandas/numpy]
weekly.py              weekly projections board on real lines              [needs numpy]
backtest.py            walk-forward Monte Carlo backtest vs closing lines   [needs numpy]
config.json            your settings + API key
dashboard.html         the auto-refreshing dashboard (generated)
historical/            historical_pbp.parquet, historical_lines.parquet, download_history.py
data/                  weights/history/latest + ratings, league_priors, backtest (generated)
nflvalue/
  oddsmath.py          de-vig, EV, Kelly, odds conversions
  factors.py           stadiums, weather/injury/revenge/matchup signals
  model.py             builds candidate bets from odds + factors
  learn.py             grading, calibration, online weight updates
  montecarlo.py        drive-by-drive game simulator                        [needs numpy]
  dashboard.py         renders the HTML
  pipeline.py          ties it together
  sources/             oddsapi, weather, espn live clients + demo generator
```

The base app (demo, live odds, learning) is **standard-library only**. The Monte
Carlo / backtest add-ons need `pip install pandas pyarrow numpy`; if those aren't
installed, the app still runs and the MC tabs simply show a hint.

---

## Extending it

- **Revenge spots:** set `revenge_home` / `revenge_away` in a game's context
  (player vs. old team, coach revenge, etc.) — the weight is already learned.
- **Live prop grading:** wire real box scores into `oddsapi.fetch_scores`
  (`prop_actuals`) so props learn from live results too.
- **Better matchup signal:** replace the standings proxy in `sources/espn.py`
  with EPA / DVOA-style ratings.
- **Realism dial:** in `sources/demo.py`, raise `MARKET_EFFICIENCY` toward 0.9
  for a harder, more realistic simulation.
