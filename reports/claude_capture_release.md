# Prospective capture and ESPN-comparison release — tailstail — 2026-09-08

Executor: Claude (Fable 5.1), release session 2026-09-08 (started 19:32 UTC, verified with `date -u`).
Manifest: `reports/claude_capture_release_manifest.json`. Logs: `reports/claude_capture_release_logs/`.
Isolated workspace: git worktree `agent/prospective-capture-2026-09` branched from `origin/main` `0b560d9`
at `/private/tmp/claude-501/…/scratchpad/tt-capture`; the main checkout (local `main` `80e742d`, 73 dirty paths)
was not modified. CI-parity environment: Homebrew Python 3.11.15 venv with `requirements.txt`, pytest, ruff 0.15.22
(nflreadpy 0.1.5, pandas 2.3.3, numpy 2.0.2, scikit-learn 1.6.1, pyarrow 21.0.0, polars 1.34.0).

## Verdict

| Item | Result |
|---|---|
| ESPN-compare branch | `origin/agent/espn-compare-2026-08` = `656f9b6`, already merged to `origin/main` by PR #10 (`94ccfde`, 2026-08-30) and hardened by PR #11 (`0b560d9`). Reviewed here; three defects fixed (below). |
| Why every Week 1 production run had failed | nflreadpy 0.1.5's season rule. Fixed. |
| Prospective capture | Immutable per-run archive with per-game deadlines built, tested, and exercised end-to-end. |
| Release | PR #12 merged to `origin/main` `c8071b7`; production run dispatched — see §7. |
| Role mixture | OFF and unchanged (§8). |
| Trends / 2025 | Nothing acted on; 2025 not read. |

## 1. Starting state (verified, not assumed)

| Fact | Evidence |
|---|---|
| Date | `date -u` → 2026-09-08 19:32:52 UTC. Week 1 has not started. |
| Local checkout | `main` = `80e742d`, **9 behind** `origin/main` `0b560d9`; 13 dirty tracked files + 60 untracked. The dirty tracked files are an *older* snapshot of what PR #11 committed (e.g. `scripts/fantasy_weekly.py` differs from origin by 330 lines); nothing in them is newer than origin. Left untouched. |
| ESPN-compare branch | `git branch -a`: `remotes/origin/agent/espn-compare-2026-08` at `656f9b6` (8 files, +1468/−2); `origin/main` log shows `94ccfde Merge pull request #10 …espn-compare-2026-08` and `0b560d9 Merge pull request #11 …finalize-simple-picks-2026`. `origin/agent/finalize-simple-picks-2026` was deleted on fetch (merged). |
| Tooling / auth | `gh` 2.96.0 logged in as the repo owner, scopes `repo, workflow` (token redacted). No ESPN credentials in the environment; the workflow's ESPN pull is the public, unauthenticated endpoint. Only repository secret: environment `fantasy-production` → `TAILSTAIL_STATE_SSH_KEY`. Private state repo `curtisdearing/tailstail-state` exists (PRIVATE, pushed 2026-08-31). |
| Pages | `gh api …/pages`: `https://curtisdearing.github.io/tailstail/`, `build_type: workflow`, public. Live requests at 19:5x UTC: `index.html` 200 (offseason landing, last-modified 2026-08-31), `model-audit.json` 200, **`player_projection_snapshot.json` 404, `fantasy_latest.json` 404**. The 404 is *current*, not historical: no production job had ever reached the Pages upload. |
| Workflow runs | `fantasy-weekly.yml`: 2026-09-03T01:22Z (schedule, **failure**), 2026-09-06T17:04Z (schedule, **failure**), three earlier `push` runs (landing only). Deployments to `github-pages`: only push-triggered landing deploys (latest 2026-08-31, `0b560d9`). No `fantasy-model-state` release existed (`gh release list`: only the prop `model-state`). |
| Private data in remote history | No league id in any tracked object (`git log --all -S1692768992` empty; the id is anonymised to `1111111111` in tracked docs/tests). Commit `6874610` did add `docs/K_SHADOW_MODEL_CARD.md` naming the league by its descriptive name ("Dearing fantasy football"); `cbf8ab3` redacted it two commits later. **That string remains in remote history at `6874610`.** Reported here; no history rewrite attempted (not authorised). Not a roster, id, cookie or token. |

## 2. Diagnosis of the failed production runs

Both scheduled runs reached `Build current projections` and died in the same place
(`reports/claude_capture_release_logs/run_34047448573.log:471-486`):

```
File "…/scripts/fantasy_weekly.py", line 315, in main
    fetch_historical(range(args.start_season, end + 1), data_dir)
File "…/nflvalue/fantasy/data.py", line 151, in <lambda>
    "rosters": lambda: nfl.load_rosters_weekly(seasons),
File "…/nflreadpy/load_rosters_weekly.py", line 43, in load_rosters_weekly
    raise ValueError(f"Season must be between 2002 and {current_season}")
ValueError: Season must be between 2002 and 2025
```

Root cause (nflreadpy 0.1.5 `utils_date.get_current_season`, read from the tagged source): the season year
advances only on the **Thursday following Labor Day** — 2026-09-10. Every run before that date asked for 2026
and was refused, although nflverse had already published `roster_weekly_2026` (HTTP 200, 2,955 week-1 rows,
2,005 with `espn_id`) and `injuries_2026` (200, 11 rows). `load_snap_counts`, `load_injuries` and
`load_ff_opportunity` validate the same way; `load_schedules` and `load_player_stats` do not. 0.1.5 is the newest
release on PyPI, so a version bump was not available. Consequences before the fix: rosters (required) → run
dies; and even after 2026-09-10, an optional table whose 2026 file does not yet exist (`snap_counts_2026`,
`ep_weekly_2026` → 404) would have been dropped **for every season**, not just 2026.

Both runs classified as `production=true` (ref `main`, event `schedule`), restored the private state
repository successfully, and found no prior public state (`no prior fantasy state release; starting fresh`).
Scheduled runs started 1h47m and 2h49m after their cron.

## 3. Schedule versus kickoffs (2026 Week 1, nflverse `schedules/games.parquet`, 272 REG games)

| Game | Kickoff (ET) | Kickoff (UTC) | Wednesday cron 23:35Z 09-09 | Sunday cron |
|---|---|---|---|---|
| NE @ SEA | Wed 09-09 20:20 | **2026-09-10T00:20Z** | 45 min before kickoff, *before* any observed start delay; and the run would have failed on the season rule anyway | — |
| SF @ LA | Thu 09-10 20:35 | 2026-09-11T00:35Z | 25 h margin | — |
| 9 early games | Sun 09-13 13:00 | 2026-09-13T17:00Z | — | 14:15Z cron: 2h45m margin, **less than the 2h49m delay observed on 09-06** → moved to 12:15Z (4h45m) |
| DEN @ KC | Mon 09-14 20:15 | 2026-09-15T00:15Z | — | Sunday run: 36 h |

The Wednesday-night opener's cohort is therefore not capturable by the cron schedule at all; it is
capturable only by a production run earlier in the week, which is what §7 does. The archive labels the
slate per game rather than assuming any of this.

## 4. Review of the ESPN-compare code on `origin/main`

Read in full: `nflvalue/sources/espn_projections.py` (340 lines), `nflvalue/fantasy/espn_compare.py`
(725), `scripts/fantasy_weekly.py`, `scripts/state_store.py`, `nflvalue/fantasy/private_state.py`,
`nflvalue/fantasy/private_boundary.py`, `.github/workflows/fantasy-weekly.yml`, `scripts/classify_run.sh`,
and `tests/test_espn_compare.py` (24 tests).

| Requirement | Finding |
|---|---|
| Player identity joins | gsis↔espn crosswalk from nflverse weekly rosters for the season, `keep="last"`, fail-loud on a missing `espn_id`/`gsis_id` column. Unmatched ESPN players (no crosswalk / not projected) are counted and named in `identity_report`, never dropped. Duplicate ESPN ids collapse to the latest roster row. ✔ |
| Scoring compatibility | ESPN raw stat lines re-scored through the model's own `score_components` under the same `ScoringRules`; applied-vs-rescored delta recorded per snapshot; snapshot carries `scoring.rules`. ✔ |
| Missing projections / DNP | Grading uses actual points under the same rules; rows absent from the stats map are `played=False`, excluded from `mae_espn`/`mae_model`, counted as `n_dnp`, with a separate `*_incl_dnp` figure. Coverage gaps are reported, not scored as zeros. ✔ |
| Timezone | `game_kickoffs_utc`: ET (`America/New_York`) → UTC via `zoneinfo`; test covers EDT. ✔ |
| **Defect 1** | A schedule row with an empty `gametime` was **defaulted to 13:00 ET** — a fabricated decision deadline. Fixed: such a row now has no kickoff and `record_week` counts it as `skipped_no_kickoff`. (nflverse nulls arrive as NaN and were already rejected; an empty string was not.) RED→GREEN test `test_kickoffs_without_a_gametime_are_omitted_not_defaulted`. |
| **Defect 2** | One `retrieved_at` stamped *after* parsing stood in for every clock. Fixed: `clocks.requested_at` (before the HTTP call), `received_at` (after; `retrieved_at` is now this), `provider_timestamp: null` with an explicit note that `kona_player_info` carries no publication time (unknown stays unknown), `archived_at` stamped at the immutable write; plus `run_context` (event, cron, run id, attempt) from the environment. Test `test_snapshot_separates_request_response_provider_and_archive_clocks`. |
| **Defect 3** (adjacent) | Equality at kickoff was untested in `is_prospective`; mutation `<`→`<=` survived. Test added; killed. |
| Pre-kickoff capture of ESPN | Every run snapshots ESPN before recording; `record_week` writes a row only when BOTH `espn_retrieved_at` and `model_generated_at` strictly precede that row's kickoff; a post-kickoff refresh leaves the earlier row standing; graded weeks are immutable and hash-checked. ✔ (pre-existing) |
| Raw ESPN storage | `data/espn_snapshots/*.json` + `data/espn_comparison_ledger.json` are gitignored, excluded from the state release, the workflow artifact and `_site`, and travel only via the private repository under `private_state.py`'s allow-list, which refuses credential shapes and roster/member keys. ✔ (pre-existing; `tests/test_workflow_trust_boundary.py`, 30 tests) |
| Published comparison artifacts | Positive allow-list (`private_boundary.PUBLIC_*`): status, season series, per-week aggregates, provenance scalars. No per-player ESPN row reaches `fantasy_public.json` or `fantasy.html`. ✔ |

## 5. What was built (branch `agent/prospective-capture-2026-09`, commit `d9ea7d0`, 14 files, +1,490/−24)

1. **`nflvalue/fantasy/data.py` — reach the current season.** `fetch_historical` tries the library call
   unchanged first (the frozen path, byte for byte). Only when the library refuses does it split: the seasons
   the library accepts still come from the library in one call; the newest season comes from the library if
   it accepts it, otherwise directly from the same release asset via the library's own downloader
   (`SEASONAL_ASSETS`: nflverse-data `weekly_rosters/…`, `snap_counts/…`, `injuries/…`; ffopportunity
   `latest-data/ep_weekly_…`). A not-yet-published in-season file drops **only that season** of an optional
   table; a required table with no current season still fails the run. Manifest records
   `nflreadpy_current_season`, `seasons_loaded`, `seasons_note`. Verified against the real library today
   (`e2e_fetch_2026_v2.log`): library says 2025; rosters 2019–2026 (330,659 rows, 2026 direct), injuries
   2019–2026 (2026 direct), snaps 2019–2025 (2026 404, skipped), expected points 2019–2025 (2026 404,
   skipped — previously the whole table was lost), 10.9 s.
2. **`nflvalue/fantasy/prospective_archive.py` (new, 408 lines).** One immutable entry per run at
   `data/prospective_archive/<season>/projection_<season>_wk<ww>_<forecast stamp>.json` carrying
   season/week/`model_version` (the commit), `scoring.preset` + rules, four clocks kept apart
   (`forecast_generated_at`, `information_as_of`, `archived_at`, `scheduled_for`), `run_context`, every
   game's kickoff as `decision_deadline_utc`, per-game `prospective` decided **strictly on both forecast
   clocks** with a reason, per-player rows (game, kickoff, flag, reason, mean/sd/p10/p50/p90/availability),
   the snapshot document's hash and path, and `payload_sha256` over the entry. The published snapshot
   document is archived **verbatim** beside it (`snapshot_…json`) and verified against the entry's hash on
   load. `index.json` is append-only (an index that drops an entry, or an entry on disk the index no longer
   names, is refused); `latest.json` is a separate pointer. Naive timestamps are refused. An unknown kickoff is
   never a met deadline. Labels: `prospective_full_slate` / `prospective_partial_slate` / `not_prospective`.
   Nothing reconstructs a missed cohort.
3. **`scripts/fantasy_weekly.py`.** Writes the archive entry immediately after the snapshot and before the
   ESPN step; passes `run_context` to the ESPN capture; adds `prospective_capture` (labels, counts,
   deadlines, hash, path) to the payload. **`private_boundary.PUBLIC_PAYLOAD_KEYS`** allow-lists that block.
4. **`scripts/state_store.py`.** Fantasy profile carries `data/prospective_archive/*.json` and
   `data/prospective_archive/*/*.json` (fixed depth: `_safe_member` uses `PurePath.match`, which has no `**`).
   The archive is Tailstail's own forecast, already published in full, so it is public material. It shares no
   path with raw ESPN state (test). `.gitignore` adds `data/prospective_archive/`.
5. **`.github/workflows/fantasy-weekly.yml`.** Build step receives `TAILSTAIL_RUN_EVENT/SCHEDULE/ID/ATTEMPT`
   through `env:` (never spliced into the script); the `Verify generated contracts` step refuses to publish
   unless the entry exists, is indexed, is what `latest.json` points at, hash-matches, ties to the snapshot
   being published, its archived document equals that snapshot, and it passes `assert_public_safe`. Sunday
   cron `15 14` → `15 12`. Comments record the Wednesday-night limitation. Pages copy list unchanged
   (`index.html`, `fantasy_latest.json`, `player_projection_snapshot.json`); the archive history is not served.
6. **`docs/PROSPECTIVE_CAPTURE.md`** — the rule, the record, where it lives, what the schedule cannot promise.

Not touched: `nflvalue/fantasy/simulation.py`, `models.py`, `features.py`, `config.py`,
`nflvalue/projection_snapshot.py`, `analysis/accuracy_protocol.json`
(SHA-256 `733e370c2dac3c7355827174f1e36def9bb1a38e6e317ec02b8980a56bd1ce42`, matches the freeze).
A test (`test_the_frozen_center_files_are_not_touched_by_the_capture_work`) pins that none of the frozen
files mentions the archive.

## 6. Tests, commands, exit codes

All in the worktree with the venv; logs copied to `reports/claude_capture_release_logs/`.

| Step | Command | Exit | Result |
|---|---|---|---|
| Baseline at `origin/main` `0b560d9` | `pytest -q -m offline` | 0 | 1,266 passed, 6 skipped (`baseline_offline.log`); `ruff check .` clean |
| RED (tests written first) | `pytest … test_prospective_archive.py test_fantasy_fetch_seasons.py test_prospective_capture_boundary.py + 2 espn tests` | 1 | 14 failed, 7 passed, 1 collection error (`red_new_tests.log`) |
| GREEN | same | 0 | 116 passed incl. adjacent suites (`green_new_tests.log`) |
| Lint | `ruff check .` (ruff.toml, 0.15.22) | 0 | clean |
| Full offline (CI selection) | `pytest -q -m offline` | 0 | **1,306 passed, 6 skipped, 35 deselected** (`full_offline_final.log`) |
| Excluded suites still import | `pytest -q --collect-only -m "needs_history or needs_network"` | 0 | (`collect_excluded.log`) |
| Mutation battery | `scratchpad/mutations.py` (copy of the tree per mutation) | 0 | **10/10 killed**: M1 accept snapshot AT deadline (`>=`→`>`), M1b ignore `information_as_of`, M1c ESPN ledger `<`→`<=`, M2 overwrite immutable entry, M2b index accepts lost history, M3 private field in entry, M3b `my_team` in the public allow-list, M4 role mixture default ON, M5 silent drop of the lagging season, M6 defaulted 13:00 kickoff (`mutation_tests.log`). Three of these survived the first draft of the tests and the tests were strengthened until they did not. |
| Real fetch 2019–2026 | `fetch_historical(range(2019, 2027), …, force=True)` with nflreadpy 0.1.5 | 0 | §5 item 1 (`e2e_fetch_2026_v2.log`) |
| End-to-end weekly, fast | `scripts/fantasy_weekly.py --no-fetch --data-dir …/e2e/historical/fantasy --fast --simulations 2000` (`TAILSTAIL_RUN_EVENT=local_e2e`) | 0 | 256 players, 2026 wk 1; `[prospective-archive] prospective_full_slate: 16/16 games before their deadline`; ESPN comparison `ok`; my_team `no_current_pick` (no league snapshot in the worktree); 101 s (`e2e_weekly_fast_v2.log`) |
| Workflow verify step, replayed locally | the exact Python heredoc extracted from the yml | 0 | `prospective archive: prospective_full_slate (16/16 …)` (`e2e_verify_step.log`). **The first replay failed**: nesting the snapshot document in the entry carried `source_manifest.quality.tables.rosters` into it and `assert_public_safe` refused the key name. Fixed by archiving the document as a sibling file; regression test added. |
| State pack | `state_store.py pack --profile fantasy` | 0 | 7 files incl. entry, snapshot document, index, latest (`e2e_pack.log`) |
| End-to-end weekly, full fidelity | same without `--fast`, 10,000 sims, separate copy | 0 | 123 s wall (with concurrent load) — the 75-minute job timeout is not a risk (`e2e_weekly_full_timing.log`) |
| CI on PR #12 | `lint`, `unit` on push and pull_request | pass | runs 34273150105 / 34273156464, unit 4m34s / 4m37s |

## 7. Release

All gates in §6 passed before any remote write. Sequence, with evidence:

| Step | Evidence |
|---|---|
| Branch pushed | `origin/agent/prospective-capture-2026-09` = `d9ea7d0` (`git push -u`, no force). |
| PR opened | https://github.com/curtisdearing/tailstail/pull/12 |
| CI on the PR | `lint` pass (10 s / 11 s), `unit` pass (4m34s / 4m37s) on both the push and pull_request runs (34273150105, 34273156464). |
| Merged via the normal path | `gh pr merge 12 --merge` → merge commit **`c8071b7b85b6c5990b41d789de4af1d652c6398c`**; `git merge-base --is-ancestor d9ea7d0 origin/main` true; `gh api …/commits/main` = `c8071b7`. The push to `main` triggered the workflow's `push` job (landing deploy, run 34273670938, success) as designed. |
| Production run | `gh workflow run fantasy-weekly.yml --ref main` (no inputs) → run **34273697642**, event `workflow_dispatch`, `run mode: production=true ref=refs/heads/main`, created 2026-09-08T20:15:07Z, completed 20:22:40Z, conclusion **success**. Every step green: private-state checkout, restore and save (raw ESPN state committed to the private repo as `dc3ce3a`); `no prior fantasy state release; starting fresh` (correct — none existed); Build 20:15:39→20:22:08 (6m29s); Verify `prospective archive: prospective_full_slate (16/16 games before deadline) -> 2026/projection_2026_wk01_20260908T202113.json`; state published as `fantasy-state-34273697642-1.tar.gz`, SHA-256 `1a177e94c3ff4ce9cf935c420a59a2ffbdce2c5fbc8fbb56fd0520e8d7e1b1d9` (release `fantasy-model-state`, created 20:14:48Z, pointer body verified); `deploy-dashboard` success 20:22:29→20:22:39; github-pages deployment `6336400042` at `c8071b7`. |
| Deployed snapshot, requested after the deploy | `https://curtisdearing.github.io/tailstail/player_projection_snapshot.json` → **HTTP 200**, `application/json`, `Last-Modified: Tue, 08 Sep 2026 20:22:33 GMT`, 794,712 bytes; `schema_version` 1; **season 2026, week 1**; `generated_at 2026-09-08T20:21:13.624843+00:00`; `information_as_of 2026-09-08T20:15:40.625920+00:00`; `model_version c8071b7b85b6c5990b41d789de4af1d652c6398c`; 256 players; 10,000 simulations; `players_canonical_csv_sha256 8723faef77bd4836…`. `fantasy_latest.json` → 200, allow-listed payload with `prospective_capture` (`prospective_full_slate`, 16/16, `payload_sha256 e07f91cc7a2cce66…`), `espn_comparison.status: ok` (511 ESPN players, 251 matched, 10 without crosswalk, 250 not projected by the model — reported, not scored), no `my_team`. `index.html` → 200, the 2026 Week 1 dashboard. |
| Archived pre-deadline entry (the prospective-readiness claim) | Downloaded the release asset, SHA-256 matches the pointer; contents: `player_projection_snapshot.json`, `espn_comparison_history.json` (kind `espn-comparison-history/1`, no weeks yet), `fantasy_model.joblib`, `prospective_archive/{index,latest}.json`, `prospective_archive/2026/projection_2026_wk01_20260908T202113.json`, `…/snapshot_2026_wk01_20260908T202113.json`. `load_entry` verifies `payload_sha256 e07f91cc7a2cce669f24b1507d92590b3673965449e41ca6789877352b88df2b`; `load_document` verifies the archived document, which **equals the deployed snapshot**; `model_version c8071b7…`; `run_context {event: workflow_dispatch, run_id: 34273697642, attempt: 1, schedule: null}`; clocks `forecast_generated_at 20:21:13Z`, `information_as_of 20:15:40Z`, `archived_at 20:22:07Z`; `assert_public_safe` passes on the released entry. |

**Prospective eligibility, 2026 Week 1, by game** (from the released entry; rule: both forecast clocks strictly before kickoff):

| Kickoff (UTC) | Games | Prospective |
|---|---|---|
| 2026-09-10T00:20Z | NE@SEA | yes (forecast 28h earlier) |
| 2026-09-11T00:35Z | SF@LA | yes |
| 2026-09-13T17:00Z | ATL@PIT, BAL@IND, BUF@HOU, CHI@CAR, CLE@JAX, NO@DET, NYJ@TEN, TB@CIN | yes |
| 2026-09-13T20:25Z | ARI@LAC, GB@MIN, MIA@LV, WAS@PHI | yes |
| 2026-09-14T00:20Z | DAL@NYG | yes |
| 2026-09-15T00:15Z | DEN@KC | yes |

Label `prospective_full_slate`: 16/16 games, 256/256 players, 0 without a kickoff. There was no earlier cohort to
miss or repair: no production run had ever produced a snapshot, and nothing was reconstructed. The ESPN side of
the comparison was captured at 20:22:07Z, also before every Week 1 kickoff; its rows and raw capture live only
in the private state repository.

The scheduled Wednesday run (23:35Z on 09-09) will now succeed on the season rule; if it starts before
00:20Z it refreshes every game's row and archives a second full-slate entry, and if GitHub delays it past the
NE@SEA kickoff it archives a `prospective_partial_slate` entry and the ledger keeps this run's NE@SEA rows.
Either way the record is explicit.

## 8. Frozen model, role mixture, trends, 2025

- `SimulationConfig.role_scenario_mixture` defaults `False` (`config.py` untouched); `scripts/fantasy_weekly.py`
  never sets it; the workflow never passes `--shadow-output`; mutation M4 (default `True`) is killed by
  `test_the_role_mixture_stays_off_by_default_and_the_weekly_run_never_enables_it`. The rejected challenger
  (`reports/claude_mc_grade.md`, FAIL) is untouched.
- No trend was acted on; none is referenced by any code in this release.
- 2025 was not read. The only historical data touched were the source tables the production run refreshes anyway.
- The forecast centre is produced by the same code as at `0b560d9`; this release adds a record beside it.

## 9. Privacy and production isolation

- The exact diff of the 14 files was grepped for the league id, the team name, cookie shapes and the
  redacted league name: no match. `git check-ignore` confirms `data/prospective_archive/`,
  `data/espn_snapshots/`, `private/`, `data/fantasy_public.json` are ignored in the worktree.
- Staging was by explicit path (no `git add -A`); `git status` after the commit shows nothing else staged.
- The archive entry passes `assert_public_safe`; the ESPN snapshots and ledger remain outside every public sink
  (`test_no_raw_espn_path_appears_in_any_public_sink`, `test_no_state_profile_names_a_raw_espn_path`,
  `test_the_archive_never_shares_a_path_with_raw_espn_state`).
- The two local private-data files in the main checkout (`reports/espn_watchlist_2026_1692768992.json`,
  untracked; `docs/K_SHADOW_MODEL_CARD.md`, an older draft) were not touched, not copied into the worktree, and
  not committed. Nothing was deleted.
- Remote history: see §1 (league descriptive name at `6874610`, redacted at `cbf8ab3`; no rewrite).

## 10. Remaining blockers and open items

- **Local `main` checkout is stale and dirty.** `~/Documents/Side-Projects-Vault/20-projects/tailstail/tailstail`
  is still at `80e742d` with 13 dirty tracked files that are older versions of what origin already carries,
  plus 60 untracked paths (league tooling, shadow work, reports). A fast-forward would conflict; not attempted.
  Curtis's call whether to discard the dirty tracked edits (`git checkout -- <files>` then `git merge --ff-only
  origin/main`) or keep them. The untracked `reports/espn_watchlist_2026_1692768992.json` must never be added.
- **Remote history carries the league's descriptive name** at `6874610` (redacted at `cbf8ab3`). Not a roster,
  id, cookie or token. A rewrite was not authorised and was not attempted.
- **Wednesday-night games are outside the cron schedule.** Any future Wednesday kickoff needs a production
  dispatch earlier that week; the archive will label the slate partial otherwise. Only Week 1 has one in 2026.
- **The Sunday run still depends on GitHub starting it within ~4h45m.** Delays beyond that are recorded, not
  prevented; the ledger and the archive lock per game.
- **ESPN identity coverage is 49%** (251 of 511 ESPN offensive players matched; 250 are players the model did
  not project this week, 10 have no nflverse crosswalk). Coverage gaps are counted and named, not scored.
  Whether to widen the model's projected set is a modelling decision, not part of this release.
- **`snap_counts_2026` and `ep_weekly_2026` did not exist yet** at run time; those features are absent for
  2026 Week 1 rows only (manifest records this) and will fill in as nflverse publishes them.
- **No lockfile** (pre-existing, freeze §1): the production run resolved nflreadpy 0.1.5, Python 3.11.16 on the
  runner; the exact package set lives in run 34273697642's `pip install` log.
- **This report and manifest** are committed in a follow-up documentation PR after the release (the release PR
  could not contain evidence that did not yet exist). Logs: `reports/claude_capture_release_logs/`.
