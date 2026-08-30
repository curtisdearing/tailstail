---
type: protocol-freeze
project: tailstail
status: SIGNED — committed to tailstail `docs/` and tagged freeze-2026-wk1
created: 2026-08-10
signed: 2026-08-10
freeze_deadline: 2026-09-05  # per docs/ACCURACY_PROTOCOL.md; Week 1 kickoff ~2026-09-10
---

# Tailstail 2026 season protocol freeze

Instantiates `docs/FROZEN_PROTOCOL_TEMPLATE.md` at the SEASON level: these are
the rules that govern every promotion decision from Week 1 through the end of
the 2026 season. Individual levers still get their own frozen template copy
before running (the template stays per-experiment; this document pins what no
experiment may renegotiate). Once signed, this file is immutable — a change
requires a new dated freeze note that supersedes it, never an edit.

## 1 · Identity

- Product/track: **tailstail — fantasy** (fablesfable/betting is governed by its
  own freeze; the shared projection contract binds both, see §3).
- Decision owner/reviewer: **Curtis** (TBD: countersignature if any).
- Frozen at git commit/tag: **`a9e9765` — tag `freeze-2026-wk1`** (post-merge
  `main`, pushed to `origin/main` 2026-08-10). Both pending branches resolved
  before the freeze commit was chosen, per the rule above:
  - `agent/quality-tests-rebased` (`c310d52`) — merged via GitHub PR #9
    (merge commit `ab18675`) before this freeze; carries the
    `MAX_MEAN_BIAS = 1.5` release-gate fix. 40 new tests, all passing.
  - `agent/draft-trade-planner-2026-07` (tip `f164172`, 7 commits) — merged
    locally (`--no-ff`, no conflicts — disjoint file set) and pushed to
    `main` as part of this freeze. Draft/trade tooling, grading-irrelevant
    per the rule above. 21 new tests, all passing.
  - Full suite at the freeze commit: 446 passed, 4 skipped, 4 failed / 14
    errors — all failures/errors are pre-existing local-environment gaps
    (missing gitignored `*.parquet` history files, e.g.
    `historical_lines.parquet`), not from either merged branch; neither
    branch touches the affected test files or data paths.
- `analysis/accuracy_protocol.json` SHA-256 (re-verified at freeze commit
  `a9e9765`, unchanged by either merge — neither branch touches this file):
  `733e370c2dac3c7355827174f1e36def9bb1a38e6e317ec02b8980a56bd1ce42`
- Runtime/package versions: CI runner (`.github/workflows/ci.yml`) pins
  **Python 3.11**. `requirements.txt` at the freeze commit uses open lower
  bounds (`pandas>=1.5`, `pyarrow>=10.0`, `numpy>=1.23`, `scipy>=1.9`,
  `nflreadpy>=0.1`, `scikit-learn>=1.3`, `joblib>=1.3`) — there is no lock
  file, so the exact resolved versions are only pinned by the specific
  GitHub Actions run log for commit `a9e9765`, not reproducible from the
  repo alone. Flagged as an open gap, not fabricated here (fail-closed per
  §5). Local verification above ran on Python 3.9.6 (dev machine, not the
  CI runner) — sufficient to confirm the merges are conflict-free and the
  new tests pass, not a substitute for the CI record.

## 2 · Truth windows and grading — frozen

- Development folds: expanding season-forward through **2024** only.
- **2025 is a locked regression checkpoint** — inspected only at declared
  checkpoints, never described as untouched (it has been audited once).
- **2026 prospective predictions are the final judge. Prospective grading
  only:** no promotion may cite a retrospective 2026 re-grade, a re-simulated
  week, or any number produced after the outcomes were known. A prediction
  counts only if its decision snapshot (Phase A immutable snapshots) predates
  kickoff.
- Scorecard (unchanged from `docs/ACCURACY_PROTOCOL.md`): MAE/RMSE by
  position, Spearman, interval coverage + width, distribution scores where
  samples are retained. Baseline to beat, as reproduced from source on
  2026-08-30: **MAE 5.0941 / RMSE 6.7215 / Spearman 0.6240 / 80%-interval
  coverage 82.41%** on an 11,482 player-week replay (mean CRPS 3.693). The
  figure this replaces — MAE 5.09 / RMSE 6.72 / Spearman 0.625 / coverage
  82.24% on 11,481 rows — was produced from an earlier nflverse pull in an
  environment that recorded only numpy and pandas versions; the delta is
  ~0.003 MAE and both runs are kept in `data/accuracy_registry.json`.
- Synthetic lines (`floor(prior trailing mean)+0.5`, TD=0.5) support trend and
  regression tests ONLY — never profit, ROI, edge, or CLV claims. Forward CLV
  (internal ledger; recorded boundary deviation) claims nothing until
  **150 resolved entries, mean CLV > 0, ≥52% beating the same-side consensus
  close**.

## 3 · Factor promotion — frozen

- **Every factor promotion requires the matched-control audit.** Minimums:
  exposed n ≥ 100 AND matched unexposed n ≥ 100; matching on
  position/market, depth role, season phase, and game script
  (favored/neutral/underdog); uncertainty clustered by team-season;
  Benjamini-Hochberg q < 0.05 across the registered family; and a later
  season-forward replication. Any missing field ⇒ **research-only**, no
  exceptions mid-season.
- The retracted depth-confounded cascade numbers stay retracted; the
  matched-control audit (e.g. RB1-out→RB2 +31.8pp) is the canonical pattern
  evidence for both projects.
- Narrative factors (birthday/revenge/primetime/referee/etc.) stay shadow
  features under the same gate; even after passing, live effect capped at 3%
  until prospective replication.
- Absence/roster cohorts: short-notice outs and structural long-term
  incumbent vacancies are **separate estimands** — the 2026 vacancy-cohort
  rebuild may not pool them.

## 4 · Experiment discipline — frozen

- One-lever contract: immutable ledger entry (hypothesis, track, expected
  delta, feature clock, sample definition, compute budget, accept/reject
  gate) BEFORE the run. Each lever gets its own copy of
  `docs/FROZEN_PROTOCOL_TEMPLATE.md`.
- Fantasy and props may run one lever each in parallel; anything touching the
  shared projection core serializes both tracks and requires both suites.
- **Three consecutive rejected levers ⇒ stop**; one preregistered
  ensemble/blend challenger; no further combination search on the benchmark.
- After every candidate: `analysis/sanity_diff.py BASE CANDIDATE`; top-10
  identity overlap ≥ 50% unless the ledger predicted and explained the churn.
- Adjusting any gate after a run has started = protocol violation by
  construction (same rule fablesfable's challengers registered).

## 5 · Engineering invariants — frozen

- Immutable decision snapshots + closing-line capture + precommitted kill bar
  (Phase A) stay on for every published week.
- Explicit MC seeds: published weekly/dashboard numbers are deterministic.
- No silent synthetic-game fallback: `pipeline.run` raises unless
  `allow_demo_fallback=True` (audit crons before the first slate — open op).
- CLV stays INTERNAL to tailstail (recorded boundary deviation); a CLV field
  in the shared `PlayerProjectionSnapshot` contract still fails validation.
- Release audit runs with `MAX_MEAN_BIAS = 1.5` — confirmed: `agent/quality-tests-rebased`
  merged before freeze (§1, PR #9, commit `c310d52`); the +5-pt-biased-projection
  escape is closed as of the freeze commit.
- Week 8 bye-week audit: data/schema, calibration, drift, compute; retraining
  does not waive promotion gates.
- Fail closed whenever identity, clock, or provenance is ambiguous.

## 6 · Registered 2026 lever queue (each needs its own template before running)

Priority order from `hot.md` + `fantasy-accuracy-notes` (error lives in TD
variance and role/usage shocks); this list registers INTENT, not gates:

1. Long-term incumbent-vacancy cohort rebuild → rerun the frozen audit
   (separate estimand rule, §3).
2. Strictly-prior season-forward role-shock probability model
   (roster/depth/injury/transaction signals only; feature clock enforced).
3. Market-prop shrinkage blend (roadmap #1 — public baseline provenance must
   be complete per protocol before any claim).
4. Availability/workload split (roadmap #2).
5. Raw passing volume + hierarchical TD allocation repair (prerequisite for
   any event-center blend).
6. Port fablesfable condition book as start/sit CONTEXT (not a lever; no
   grading claim).

## 7 · Sign-off

- [x] Pending branches merged or excluded; freeze commit chosen and tagged:
      both branches merged, freeze commit `a9e9765`, tag `freeze-2026-wk1`
      (pushed to origin 2026-08-10).
- [x] Protocol JSON hash re-verified at the freeze commit:
      `733e370c2dac3c7355827174f1e36def9bb1a38e6e317ec02b8980a56bd1ce42`
      (matches, file untouched by either merge).
- [x] Runtime versions recorded: CI runner pinned to Python 3.11
      (`.github/workflows/ci.yml`); `requirements.txt` has no lock file —
      exact resolved versions live only in the GitHub Actions run log for
      `a9e9765` (open gap, flagged in §1, not fabricated).
- [x] Week 8 checkpoint owner: **Curtis**.
- [x] Signed (owner, date): **Curtis, 2026-08-10**.

*Drafted 2026-08-10 by the cowork agent from `docs/ACCURACY_PROTOCOL.md`,
`docs/FROZEN_PROTOCOL_TEMPLATE.md`, the 2026-07-16 hardening checkpoint, the
shipped-model audit, and `hot.md`. Signed 2026-08-10 at commit `a9e9765`
(tag `freeze-2026-wk1`), ahead of the 2026-09-05 deadline. Nothing in this
freeze changes a gate — it pins the already-registered rules to the 2026
season. Per the immutability rule above, any future change requires a new
dated freeze note that supersedes this one, never an edit to this file.*
