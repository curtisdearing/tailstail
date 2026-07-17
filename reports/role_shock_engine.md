# Role-shock engine — pregame role-state probabilities + scenario-mixture MC

Status: **built, gate NOT passed, ships flag-OFF (shadow).** The predeclared
gate required +3pp 80%-coverage improvement on both role cohorts; the mixture
delivered +1.0pp (role-up) and +1.3pp (role-down). Honest verdict: directionally
right, below threshold. No production behavior changes while
`fantasy.role_state_mixture` stays false (default).

## Gate comparison (frozen 11,481 player-weeks, 54 week blocks, 1,000 draws)

| Cohort | Baseline repro cov80 | Mixture cov80 | Delta | Gate (+3pp) |
|---|---:|---:|---:|---|
| role_increase_5_plus (n=1,676) | 72.0% | 73.0% | +1.0pp | FAIL |
| role_decrease_5_plus (n=1,258) | 71.6% | 72.9% | +1.3pp | FAIL |
| stable_role_abs_lt_3 (n=6,247) | 88.3% | 86.6% | −1.7pp | OK (≥86%) |
| injury_questionable (n=470) | 90.9% | 90.9% | 0.0 | — |

Direct-ensemble center unchanged by construction (MAE 5.106 = baseline repro;
paired week-block delta +0.000, tie probability 100%). Raw event simulator
improved as a side effect (5.451 → 5.414 MAE) but remains research-only.
Full mixture replay tables: `reports/role_shock_replay_mixture.md`.

## What was built (all strictly pregame, shift-then-roll, roster-of-record)

- Feature frame v2 with snap/route shares, depth-chart rank as-of, practice
  progression (DNP/Limited/Full), roster transitions, teammate-vacancy by
  exact seat (short-notice vs long-term ≥2wk separated; early vs established
  replacement windows), returning-incumbent flag. Loader in
  `nflvalue/fantasy/role_state.py`.
- Season-forward multinomial role-state model (≤2022→2023, ≤2023→2024,
  ≤2024→2025), predictions in /tmp/exp_roleshock/role_probs_{2023,2024,2025}.parquet.
- Scenario-mixture layer in the simulator behind `fantasy.role_state_mixture`
  (default OFF): per-draw role-state sampling, state-conditional reallocation
  summing to team volume, same-team correlations preserved.

## Known gaps (registered follow-ups)

1. `tests/test_role_state.py` was never written (agent interrupted) — as-of
   safety of frame v2 and sum-to-team invariants are asserted in code but not
   yet under CI. Existing simulation/leakage suites pass (29 tests).
2. Reallocation magnitudes are conservative; composing the mixture with a
   PROE-based team-volume node (see `reports/factor_families_audit.md` — team
   pass-att MAE 6.008 vs 6.761 current node) is the most promising path to
   clearing the +3pp gate.
3. Cascade-audit tables and role-state calibration-by-cohort tables live in
   /tmp/exp_roleshock checkpoints; they must be regenerated into a committed
   report before promotion.
