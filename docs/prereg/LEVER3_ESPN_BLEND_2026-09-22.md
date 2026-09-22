# Frozen protocol — 2026-09-22 / registered freeze lever 3 (market-prop shrinkage blend)

Copy of `docs/FROZEN_PROTOCOL_TEMPLATE.md`. **Status: REGISTERED, NOT RUN.**
This document opens the measurement and fixes the gate in advance. No promotion,
no claim, and no model change is authorized by it. Do not edit a frozen copy;
supersede it with a new dated copy if anything below has to change.

## Identity

- Product/track: tailstail (fantasy). Not fablesfable; fantasy and props may run one lever each in parallel.
- Decision owner/reviewer: Curtis (owner). Preregistered by Claude Opus 5 on his standing instruction to advance model accuracy.
- Git commit and tag: registered against `main` `46f6b4d`; season freeze `freeze-2026-wk1` at `a9e9765`, signed doc `docs/PROTOCOL_FREEZE_2026.md`.
- `analysis/accuracy_protocol.json` SHA-256: unchanged by this registration; this lever adds measurement only.
- Input canonical hashes and registry path: each graded week is bound to its own `projections_sha256` in the private ESPN comparison ledger (`espn_comparison_ledger.json`, private repo `curtisdearing/tailstail-state`). Those digests are the sample's identity.
- Runtime/package versions: production workflow pins Python 3.11 and `requirements.txt`; no lockfile (known gap, recorded 2026-08-10).

## Registered experiment

- Lever and feature flag: freeze lever 3, "market-prop shrinkage blend". Measured arm only — `espn_compare.BLEND_WEIGHTS` with `blend_arms` written into the PRIVATE ledger grading block. There is no serving flag and no published surface; `public_aggregate`'s positive allow-list drops the arm from the published history by construction.
- Causal/mechanical hypothesis: ESPN's weekly projection embeds depth-chart, touch-share and game-script information that tailstail's frozen feature frame does not carry, most visibly at running back where role changes (a back demoted, traded, or newly inheriting a lead role) leave our `pre_*` volume features describing a role the player no longer has. Shrinking our projection toward ESPN's should therefore reduce error, and should reduce it MOST at RB and least at QB.
- Expected delta and primary metric: **primary metric is pooled played-only MAE** on paired pre-kickoff rows. Directional expectation from the 2026 wk1-2 retro (n=437, exploratory, not part of the confirmatory sample): model alone 5.55, ESPN alone 5.33, w=0.5 5.37, w=0.75 5.33. Expectation registered here: the minimising w lies in [0.5, 1.0] and beats w=0 by at least 0.15 MAE.
- Development folds: the 2026 wk1-2 retro is DEVELOPMENT and is excluded from the confirmatory sample. It is reported in `20-projects/tailstail/_grade-2026-09-22/retro_corrected_vs_espn_REPORT.md` and in `accuracy_ledger.md`.
- Locked checkpoint opened: 2026-09-22. The confirmatory sample is **2026 weeks 3 and later**, graded prospectively by the production run. Weeks 1-2 are permanently excluded: they were produced by the defective serving path (same-week missingness flags) and cannot be regenerated prospectively.
- Population, exclusions, exact exposed/control n: every player-week with BOTH a pre-kickoff tailstail projection and a pre-kickoff ESPN projection, matched by gsis id, as recorded by `record_week` under the existing prospective rule (no backfill; a locked row cannot be touched after kickoff). Primary analysis is played-only; incl-DNP is reported as secondary because DNP rows measure availability, not projection skill. n accrues at roughly 230-260 rows per week; the checkpoint below sets the minimum.
- Matching keys, strata, and cluster unit: matched on gsis `player_id` within week. Strata: position (QB/RB/WR/TE). **Cluster unit is the week** — player-weeks inside one week share slate-level shocks, so intervals are computed by clustered/blocked bootstrap over weeks, not over rows.
- Multiplicity family: the five weights in `BLEND_WEIGHTS` plus four position strata. The primary decision reads the pooled played-only metric at the preregistered candidate weights only; position strata are descriptive and carry no independent promotion right.
- Random seed, simulations, wall time: no simulation. The bootstrap at decision time uses seed 20260922 with 10,000 week-block resamples.

## Decision rule — fixed in advance

- **Checkpoint:** the first production grading run at which the confirmatory sample reaches **≥ 6 graded weeks AND ≥ 1,200 played paired rows**. Not earlier, whatever the interim numbers look like.
- **Accept** (promote a blend into serving, as a separate registered change with its own review): the best preregistered weight `w*` beats `w=0` on pooled played-only MAE by ≥ 0.15 points AND the week-clustered 95% bootstrap interval on that paired difference excludes 0 AND the same sign holds in ≥ 5 of the graded weeks.
- **Reject:** the interval includes 0, or the gain is < 0.15, or the sign flips in ≥ 2 weeks.
- **Stop-and-report** either way; a rejected lever is recorded in `accuracy_ledger.md` with n, metric, gate and commit, exactly like every negative result before it.
- Adjusting any threshold, the weight grid, or the sample definition after the first confirmatory week is graded is a protocol violation by construction. The grid is frozen in code beside a pointer to this file.

## Results and decision

- Baseline metrics: TBD at checkpoint (w=0 arm, the incumbent).
- Candidate metrics and paired interval/probability: TBD at checkpoint.
- Calibration: not applicable to a point-estimate blend; the model's own 80% interval coverage is tracked separately (2026 wk1-2 retro: 0.79 corrected vs 0.68 production).
- Forward CLV: not applicable — this is a fantasy projection track, not a priced market.
- Top-10 sanity diff path and explanation: `analysis/sanity_diff.py BASE CANDIDATE` is required before any promotion, with ≥ 50% top-10 identity overlap unless this ledger entry predicts and explains the churn.
- Accepted, rejected, or research-only: **NOT RUN — measurement open, decision deferred to the checkpoint.**
- Reviewer sign-off and date: pending Curtis.

## Operations

- Data/provider and schema checks: ESPN snapshots carry all five protocol provenance fields and are hash-verified; nflverse supplies actuals under the same `ScoringRules` as both projections.
- Roster/identity freshness: unchanged; identity is the nflverse weekly-roster gsis↔espn crosswalk, and unmatched players are reported rather than guessed.
- Rollback commit/flag: the measurement is additive and private. Removing `blend_arms` from `_aggregate` reverts it; no serving path reads it.
- Week 8 bye-week checkpoint owner: Curtis — bye weeks shrink the weekly sample and change the position mix, which is why the gate counts weeks AND rows.
- Known caveats shown to users: none. Nothing here reaches the public page or the decision card, so there is nothing to disclose to a reader yet. If the lever is ever accepted, the decision card must say that a published projection is part market.
