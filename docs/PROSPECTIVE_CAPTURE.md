# Prospective capture: what counts as a pre-deadline forecast, and how it is proved

Written 2026-09-08 for the 2026 season. Governs the record behind the freeze's
rule (`docs/PROTOCOL_FREEZE_2026.md` §2) that a prediction counts only if its
decision snapshot predates kickoff.

## The record

Every weekly run writes one **immutable archive entry** to
`data/prospective_archive/<season>/projection_<season>_wk<ww>_<forecast stamp>.json`
(`nflvalue/fantasy/prospective_archive.py`). An entry carries:

| Field | Meaning |
|---|---|
| `season`, `week`, `model_version` | identity of the forecast (`model_version` is the commit the run executed) |
| `scoring.preset`, `scoring.rules` | the rules the scored numbers are under |
| `clocks.forecast_generated_at` | when the ensemble and simulation finished (UTC) |
| `clocks.information_as_of` | the retrieval time of the source tables the forecast was built from |
| `clocks.archived_at` | when this entry was written |
| `clocks.scheduled_for`, `run_context` | the cron the run fired on, the event, run id and attempt -- from the workflow's environment; `null` when unknown |
| `deadlines[game_id]` | that game's kickoff = its decision deadline, and whether **both** forecast clocks are strictly before it, with the reason when not |
| `eligibility` | `prospective_full_slate` / `prospective_partial_slate` / `not_prospective`, with game and player counts, including players whose game has no known kickoff (never counted as prospective) |
| `players[]` | per player: game, kickoff, prospective flag and reason, and the scored summary (mean, sd, p10, p50, p90, availability) |
| `projection_snapshot` | the full scoring-independent snapshot document and its hashes |
| `payload_sha256` | SHA-256 of the whole entry; verified on every load |

Beside the entries: `index.json` (append-only; a write that would drop an
indexed entry, or that finds an entry on disk the index no longer names, is
refused) and `latest.json` (a separate pointer, the only file that moves).

The public payload (`data/fantasy_public.json`, served as `fantasy_latest.json`)
carries a `prospective_capture` block with the label, counts, deadlines and the
entry's hash, so the published week can be tied to its archived entry. The
`Verify generated contracts` step of the workflow refuses to publish unless the
entry exists, is indexed, is what `latest.json` points at, matches the
snapshot about to be published, and passes the public-safety guard.

## The rule

A game is prospective only when `forecast_generated_at < kickoff` **and**
`information_as_of < kickoff`, strictly. Equality fails. A naive timestamp is
refused, not assumed UTC. A game without a known kickoff is not a met deadline.
The week never gets one label: a run after a Wednesday-night kickoff but
before Thursday's is a partial slate, and the Wednesday game's cohort is
recorded as missed. Nothing reconstructs a missing forecast; a missed cohort
begins at the next eligible window.

The ESPN comparison ledger applies the same rule to its rows
(`espn_compare.record_week`: both the ESPN retrieval time and the model
generation time must precede the row's kickoff, and a row whose game has no
kickoff time is skipped rather than judged against a defaulted 13:00). Each
ESPN capture also keeps four clocks apart: `requested_at`, `received_at`,
`provider_timestamp` (unknown for this endpoint and recorded as such) and
`archived_at`.

## Where it lives

The archive is Tailstail's own forecast, already published in full on the
Pages site, so it is public material. It travels between runs inside the
checksummed public state release (`scripts/state_store.py`, profile
`fantasy`) and is gitignored. It shares no path with the raw ESPN captures or
the row-level ledger, which stay in the private state repository.

## The schedule, and what it cannot promise

`fantasy-weekly.yml` fires Wednesday 23:35 UTC and Sunday 12:15 UTC
(September to January). GitHub started both September 2026 scheduled runs
late, by 1h47m and 2h49m; the Sunday cron was moved from 14:15 to 12:15 UTC so
one such delay cannot push it past the 17:00 UTC early window. A Wednesday-night
game -- 2026 Week 1 opened with one at 00:20 UTC Thursday -- is not covered by
the Wednesday cron. The archive does not infer from the schedule what a run
captured: each entry records what it actually preceded.

## Why the Week 1 runs had failed

nflreadpy 0.1.5's `get_current_season()` advances to the new year on the
Thursday after Labor Day, so before 2026-09-10 its roster, snap-count, injury
and expected-points loaders refused season 2026 with
`ValueError: Season must be between 2002 and 2025` -- while nflverse had
already published `roster_weekly_2026` and `injuries_2026`.
`nflvalue.fantasy.data.fetch_historical` now tries the library call unchanged
first (the frozen path) and, only when the library refuses, loads the seasons
it accepts from the library in one call and the newest season directly from the
same release asset through the library's own downloader. An in-season file that
does not yet exist (snap counts and expected points before Week 1) drops only
that season of an optional table; a required table with no current season still
fails the run. The manifest records which seasons each table carries and why.
