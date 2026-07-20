# Development workflow

Local setup and the checks CI enforces, so a change lands green the first time.

## Environment

```bash
pip install -r requirements.txt pytest ruff
```

The base app (`run.py`, `update_results.py`, dashboard, learning) is
standard-library only. `pandas`/`numpy`/`scipy`/`scikit-learn` are needed for the
projection engine, Monte Carlo, and the fantasy package; `nflreadpy` is needed
only to refresh official data.

## Lint

CI runs `ruff check .` as a required job. The ruleset in `ruff.toml` is curated
for genuine defects (unused imports/variables, undefined names, empty f-strings,
a few correctness lints) — not style churn. Rules that would change runtime
semantics (e.g. `B905` `zip(strict=)`) are intentionally **not** enabled.
Research scripts under `analysis/` and the test suite keep exploratory dead
assignments (`F841` is ignored there).

```bash
ruff check .          # what CI runs
ruff check --fix .    # apply the safe autofixes
```

## Tests

CI runs a **data-independent** subset — the reproducible tests that need no
historical Parquet cache and no network. That explicit allow-list lives in
`.github/workflows/ci.yml` under the `unit` job. **When you add a new
data-independent test file, add it to that list** or CI will not run it.

```bash
# Run exactly what CI runs:
CI_TESTS=$(sed -n '/pytest -q/,/^$/p' .github/workflows/ci.yml | grep 'tests/' | tr -d ' ')
pytest -q $CI_TESTS

# Full suite (the data-dependent tests need historical/ Parquet caches +
# the local nflverse loader; without them they fail/error, which is expected):
pytest -q
```

Prefer data-independent tests built on small **synthetic** frames (see
`tests/test_fantasy_cli.py`, `tests/test_fantasy_audit_dashboard.py`,
`tests/test_fantasy_season_tooling.py`, `tests/test_role_multipliers.py`) so the
behaviour they pin runs in CI without a data dependency.

## Regenerating frozen constants

The role-shock simulator's `DEFAULT_STATE_MULTIPLIERS` table is frozen on
2019–2022 training data. Regenerate it from committed code when the feature
frame definition changes:

```bash
python -m scripts.build_role_multipliers --max-season 2022 --print
```

## Accuracy discipline

No factor or model change ships on a good pooled score alone. Promotions go
through the frozen protocol (`docs/ACCURACY_PROTOCOL.md`) and the release gate
in `nflvalue/fantasy/audit.py` (interval calibration, per-position coverage,
paired-bootstrap improvement, calibration slope). Record every tested lever —
including rejections — in the vault `accuracy_ledger`.
