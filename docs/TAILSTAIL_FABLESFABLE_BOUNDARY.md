# Tailstail / fablesfable projection boundary

Both products may use the same football evidence, but neither product is the
other's dependency. The exchange format is schema version 1 of
`PlayerProjectionSnapshot` plus its correlated component-sample artifact.

## Shared

- stable player/game identity;
- source and information-as-of provenance;
- availability state;
- passing, rushing, and receiving event-component samples;
- canonical content hashes and Parquet integrity hashes; and
- model/simulation version metadata; and
- explicit component-validation status (`approved`, `research_only`, or
  `unvalidated`) with optional market overrides.

## Excluded from the shared contract

- fantasy scoring and league settings;
- fantasy rosters, lineup, waiver, and trade decisions;
- sportsbook lines, prices, books, vig, edge, and CLV; and
- either consumer's dashboard or release state.

The JSON summary is suitable for inspection and light consumers. The Parquet
sample artifact preserves the same simulation row across all players and
components, so game and teammate correlations survive rescoring.

Tailstail applies league scoring to the component samples. Fablesfable maps the
same component columns to stat markets, but refuses to emit a betting
probability unless the producer marks that market `approved`. Integrity is not
accuracy: a canonical hash proves which draws arrived, not that the draws beat
history. A consumer-specific field appearing in the shared snapshot fails
validation.

The contract is frozen by:

- `schemas/player_projection_snapshot.schema.json`;
- `nflvalue/projection_snapshot.py`; and
- the producer and consumer projection-contract tests.

Neither repository should track the other repository's default branch. Once
the contract has survived prospective use, its implementation can move to a
small tagged `nflvalue-core` release and both consumers can pin that version.

## Recorded deviation: tailstail's internal CLV ledger (2026-07)

Phase A added an internal decision-snapshot ledger to tailstail — immutable
pick snapshots, closing-line capture, forward CLV, and a precommitted kill bar.
This was user-directed and is a deliberate exception to the split above, which
had treated lines, prices, and CLV as the betting consumer's (fablesfable's)
domain. To be precise about what did and did not change:

- CLV remains **excluded from the shared contract**: it is never part of
  `PlayerProjectionSnapshot` or the component-sample artifact, and a CLV field
  appearing in the shared snapshot still fails validation.
- The exception is scope, not schema. Tailstail now *computes and stores* CLV
  internally to grade its own decision quality; it does not *export* it.
  `tests/test_product_boundary.py` enforces the shared-schema boundary only —
  it does not police this internal build.
- Recorded here, in `docs/decisions_p3-5.md`, and in the runtime decision
  ledger (`nflvalue/decision_ledger.py`). Shipped to `main` as PR #5.
