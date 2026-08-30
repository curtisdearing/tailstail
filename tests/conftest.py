"""Shared fixtures for the prop-shortlister test suite.

Uses a 2-season slice (2019-2020) rather than the full 2019-2023 parquet so
the suite runs quickly; the leakage/reproducibility properties being tested
don't depend on how many seasons are loaded.
"""

from __future__ import annotations

import pathlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue.features import load_pbp

FAST_SEASONS = [2019, 2020]


@pytest.fixture(scope="session")
def pbp_fast():
    df = load_pbp()
    return df[df["season"].isin(FAST_SEASONS)].copy()


@pytest.fixture(scope="session")
def pbp_tiny():
    """A small (single-season) slice for tests where determinism, not data
    volume, is what's being checked -- keeps the suite fast."""
    df = load_pbp()
    return df[df["season"] == 2019].copy()


@pytest.fixture(scope="session")
def backtest_report_fast(tmp_path_factory):
    """One shared run of the backtest (single season, for speed) -- reused by
    every smoke-test assertion instead of each test re-running the pipeline.
    All generated files live outside the repository checkout."""
    import prop_backtest
    directory = tmp_path_factory.mktemp("prop-backtest")
    report = prop_backtest.run(
        seasons=[2019], output_path=str(directory / "prop_backtest.json"),
        db_path=str(directory / "prop_backtest.db"),
    )
    report["_test_output_path"] = str(directory / "prop_backtest.json")
    return report


# --------------------------------------------------------------------------- #
# CI test selection
# --------------------------------------------------------------------------- #
# CI used to name the tests it ran in a 35-entry file list inside ci.yml. A
# list like that is wrong the moment someone adds a file and forgets to append
# it -- which is how `test_espn_compare.py` shipped in a merged PR without CI
# ever running it, and how two genuinely failing suites stayed green for weeks.
#
# The rule is inverted here: every test is `offline` and runs in CI unless it
# is registered below as needing something CI has not got. A new test file is
# therefore selected by default, and excluding one is a visible, reasoned edit
# to this registry rather than an omission. `tests/test_ci_selection.py`
# enforces both halves.

NEEDS_HISTORY: dict[str, str] = {
    "test_backtest_smoke.py":
        "runs prop_backtest over historical/*.parquet, which is gitignored",
    "test_factor_families.py":
        "reads historical/fantasy/schedules.parquet directly",
    "test_leakage.py":
        "leakage properties are asserted over the real pbp corpus",
    "test_positions.py":
        "asserts real roster positions from the downloaded player-week frame",
    "test_reproducibility.py":
        "determinism is checked by re-running the pipeline over the pbp corpus",
}

NEEDS_NETWORK: dict[str, str] = {
    "test_ingest.py":
        "calls nflreadpy against the live nflverse endpoints; hangs without egress",
}


def pytest_collection_modifyitems(items):
    """Mark by module: registered files get their reason marker, all else offline."""
    for item in items:
        name = pathlib.Path(str(item.fspath)).name
        if name in NEEDS_HISTORY:
            item.add_marker(pytest.mark.needs_history)
        elif name in NEEDS_NETWORK:
            item.add_marker(pytest.mark.needs_network)
        else:
            item.add_marker(pytest.mark.offline)
