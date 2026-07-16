# Professional projection methods and iterative model audit

This document separates what public projection providers say they do from
what this repository can reproduce and validate. Commercial implementations
are proprietary; their public descriptions are not enough to reconstruct an
internal model exactly.

## What the public methods consistently do

1. **Project the team before the player.** ESPN's Mike Clay describes a
   team-by-team process followed by player dropback, carry, and target shares.
   His published examples also use play-caller history, personnel allocation,
   teammate availability, and touchdown regression.
2. **Allocate constrained player opportunity.** Carries, targets, and QB
   dropbacks must reconcile to plausible team totals. A projection that gives
   every teammate his unconstrained trailing share double-counts volume.
3. **Separate opportunity from efficiency.** The open ffopportunity model
   estimates the points an average player would score from each play's
   situation. This helps distinguish repeatable workload from realized yards
   and touchdowns.
4. **Use game environment and matchup at the correct level.** Establish The
   Run publicly describes pace, pass/run rates, stat- and position-specific
   matchups, recent-performance weighting, and manual checks of shares. PFF
   publicly identifies grades, schedule, teammate quality, and outcome
   distributions as inputs to its fantasy tooling.
5. **Publish a distribution, not false certainty.** ETR describes median,
   floor, and ceiling estimates based on player volatility, archetype, and
   game situation. Consensus systems reduce single-expert risk, but consensus
   is a separate source and must retain timestamp lineage.
6. **Update when the information set changes.** Roster, injury, depth-chart,
   market, and weather changes can be more important than another historical
   efficiency split.

Public references:

- ESPN/Mike Clay, 2024 process and examples:
  <https://www.espn.com/fantasy/football/story/_/id/39813723/2024-fantasy-football-projections-draft-trends-carry-target-shares>
- ESPN/Mike Clay, 2026 examples:
  <https://www.espn.com/fantasy/football/story/_/id/48276085/2026-fantasy-football-projections-draft-rankings-trends-carry-target-shares>
- Establish The Run projection-process summary:
  <https://cdn.establishtherun.com/wp-content/uploads/2021/09/05125301/ETR-Episode-239-Summary.pdf>
- PFF fantasy tool description:
  <https://www.pff.com/fantasy/draft/mock-draft-simulator?step=settings>
- FantasyPros consensus and accuracy methodology:
  <https://www.fantasypros.com/about/faq/football-draft-accuracy-methodology/>
- ffopportunity expected-points model:
  <https://github.com/ffverse/ffopportunity>
- NFL Next Gen Stats tracking description:
  <https://operations.nfl.com/game-operations-logistics/technology/performance-tracking-data-next-gen-stats>
- nflverse participation-data timing and provenance:
  <https://nflreadr.nflverse.com/reference/load_participation.html>

## Translation into this engine

The point projection remains a position-specific OOF stack of Bayesian ridge,
gradient boosting, random forest, and a small neural network. The admitted
feature architecture now adds:

- lagged expected passing, rushing, and receiving points;
- lagged expected touchdowns, yards, receptions, completions, interceptions,
  and first downs;
- short-versus-long opportunity, snap, target-share, and carry-share momentum;
- prior-only coach pace, pass rate, RB/WR/TE target allocation, and QB rushing
  allocation;
- active-roster reconciliation of QB attempt, target, and carry shares;
- top-down attempts, targets, carries, and total opportunity from team volume;
- explicit role rank, share-overflow, missing-history, injury, teammate-out,
  market, weather, stadium, surface, referee, primetime, and QB-pair context.

All outcome-derived values are shifted before rolling. The feature contract
and leakage tests verify that changing a current-week outcome or expected
opportunity cannot change that same week's model inputs.

The Monte Carlo layer still owns discrete touchdown, availability, team pace,
pass/rush mix, and correlated teammate allocation scenarios. A median point
model should not pretend to know which discrete touchdown outcome will occur.

## Iterative season-forward results

The current evaluation uses 11,481 eligible player-weeks in untouched 2023–
2025 outer seasons. Comparisons to previous production use the 11,477 rows
common to both models. Model selection happens inside each earlier cutoff.

| Iteration | MAE | RMSE | Spearman | Decision |
|---|---:|---:|---:|---|
| Previous production | 5.1010 | 6.7320 | 0.6222 | Baseline |
| Rich factors, large learners | 5.1035 | 6.7244 | 0.6241 | Rejected: MAE regressed |
| Coach/share-only, large learners | 5.1056 | 6.7247 | 0.6238 | Rejected: MAE regressed |
| Rich factors, compact 3-season stack (pre-audit) | 5.0917 | 6.7224 | 0.6246 | Superseded: schedule-missing history found |
| Final schedule-safe compact stack | **5.0914** | **6.7184** | **0.6246** | Admitted |
| Added structural-only learner | 5.0960 (fast screen) | 6.7237 | 0.6244 | Rejected |
| Three-season recency weighting | 5.0990 (fast screen) | 6.7286 | 0.6238 | Rejected |

On the rows shared with previous production, the admitted change improves MAE
by 0.0084 points per player-week, RMSE by 0.0126, and Spearman correlation by
0.0020. A paired week-block bootstrap gives a 95% two-sided interval of -0.0024
to 0.0197 and a 6.4% probability of no MAE improvement. This is a small
directional gain, not proof of a universal improvement. It improved 30 of 54
weeks.

The capacity result matters: larger forests, boosting runs, and neural nets
fit the expanded feature space more aggressively and worsened absolute error.
The production learners therefore use the validated compact capacity. “More
model” was not better.

The last audit also found 49 historical roster rows with no matched game
schedule. They are now excluded from eligibility and masked out of player,
team, and coach rolling history. Keeping that correctness fix reduced the
headline MAE gain; the more flattering pre-fix number is documented above and
not presented as the final result.

## Factors not admitted to the point model

- **FantasyPros historical consensus:** valuable as an external benchmark and
  future optional prior, but not admitted until every archive snapshot has a
  verifiable timestamp before the player's kickoff. Ambiguous lineage can
  create postgame leakage.
- **Play-level route and formation participation:** analytically valuable, but
  nflverse documents that 2023+ participation data is delivered after the
  postseason. It can support historical research, not a live weekly dependency
  unless an in-season licensed source is configured.
- **Raw player-specific referee/stadium/turf narratives:** retained only as
  strongly shrunk priors. Small samples do not justify unpooled adjustments.
- **Birthdays, revenge games, and similar quirks:** candidates for preregistered
  audits, never automatic boosts. They need enough predeclared opportunities,
  a plausible mechanism, and season-forward replication.
- **Manual overrides:** not used silently. If added later, each override must
  record author, timestamp, pregame evidence, original value, adjusted value,
  and retrospective error.

## Next accuracy work, in order

1. Add a timestamped professional-consensus benchmark and test whether a
   shrinkage blend improves errors without leakage.
2. Model active/inactive and conditional workload as separate probabilistic
   stages using final practice and game-status timestamps.
3. Add in-season depth-chart snapshots with an as-of timestamp and explicit
   source-change handling for 2025 onward.
4. Study a separate long-term incumbent-vacancy cohort for reserve/IR and
   multi-week roster absences. It must exclude traded/cut players, distinguish
   the early and established replacement windows, and earn promotion through a
   season-forward ablation rather than borrowing the short-notice injury effect.
5. Evaluate NGS passing, receiving, and rushing metrics as lagged talent priors,
   with missing-source controls and position-specific ablations.
6. Score decisions, not only projections: lineup points lost versus the best
   available pregame choice, calibration by start threshold, and trade value
   under replacement-level uncertainty.

No factor is promoted because it sounds football-smart. It is promoted only
when it is pregame reproducible, survives a future-season test, and improves a
declared metric without breaking calibration or a position-level release gate.
