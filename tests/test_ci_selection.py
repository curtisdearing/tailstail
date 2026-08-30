"""CI must run every offline test, and excluding one must be a visible choice.

The failure this guards against already happened twice on this repository: a
hand-maintained list of test files inside `ci.yml` silently stopped covering
`test_espn_compare.py` when that suite was merged, and two genuinely failing
modules stayed out of CI for weeks because nobody remembered to add them.

So the selection rule is inverted (see `tests/conftest.py`): everything is
`offline` unless registered as needing history or network. These tests keep
the mechanism honest -- that CI still selects by marker rather than by name,
that the exclusion registry is small, real and reasoned, and that the suites
this release depends on cannot be quietly dropped into it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from conftest import NEEDS_HISTORY, NEEDS_NETWORK  # pytest puts tests/ on sys.path

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"
CI = REPO / ".github" / "workflows" / "ci.yml"

# Suites the reconciliation is accountable for. Excluding any of these is the
# specific mistake this file exists to catch, so they are named, not inferred.
REQUIRED_OFFLINE = {
    "test_espn_compare.py",          # merged in PR #10 and never CI-selected
    "test_espn_league_adapter.py",
    "test_espn_league_contract.py",
    "test_waiver_planner.py",
    "test_my_team_contract.py",
    "test_my_team_dashboard.py",
    "test_league_trades.py",
    "test_league_simulation.py",
    "test_shadow_kicker.py",
    "test_fantasy_watchlist.py",
    "test_fantasy_scoring.py",
    "test_draft_and_trades.py",
}


def _test_modules() -> set[str]:
    return {p.name for p in TESTS.glob("test_*.py")}


def _ci_pytest_lines() -> list[str]:
    text = CI.read_text(encoding="utf-8")
    return [line for line in text.splitlines() if "pytest" in line]


def test_ci_selects_by_marker_not_by_file_list():
    """The pytest invocation must be marker-driven."""
    lines = _ci_pytest_lines()
    assert lines, "ci.yml runs no pytest command at all"
    joined = "\n".join(lines)
    assert re.search(r"-m\s+[\"']?offline", joined), (
        "ci.yml must select tests with `-m offline`; found:\n" + joined)


def test_ci_names_no_individual_test_files():
    """A file list is the failure mode; it must not come back."""
    text = CI.read_text(encoding="utf-8")
    named = re.findall(r"tests/test_[A-Za-z0-9_]+\.py", text)
    assert not named, (
        "ci.yml enumerates test files again, which is how a new test stops "
        f"running: {sorted(set(named))}")


def test_every_test_module_is_either_offline_or_registered():
    """No module may be unclassified, and no registry entry may be a ghost."""
    modules = _test_modules()
    registered = set(NEEDS_HISTORY) | set(NEEDS_NETWORK)
    missing = registered - modules
    assert not missing, f"exclusion registry names files that do not exist: {sorted(missing)}"
    # Everything else is offline by construction (conftest marks the default),
    # so this assertion is really about the registry staying a closed set.
    assert modules - registered, "every test module is excluded; CI would run nothing"


@pytest.mark.parametrize("name", sorted(REQUIRED_OFFLINE))
def test_release_critical_suites_are_not_excluded(name):
    modules = _test_modules()
    assert name in modules, f"{name} is missing from tests/"
    assert name not in NEEDS_HISTORY, f"{name} was moved into the history-excluded registry"
    assert name not in NEEDS_NETWORK, f"{name} was moved into the network-excluded registry"


def test_every_exclusion_carries_a_reason():
    for registry, label in ((NEEDS_HISTORY, "NEEDS_HISTORY"), (NEEDS_NETWORK, "NEEDS_NETWORK")):
        for name, reason in registry.items():
            assert reason and len(reason) > 20, (
                f"{label}[{name!r}] needs a real reason, not {reason!r}")


def test_conftest_still_applies_the_default_marker():
    """The inversion itself -- lose this hook and every guard above is theatre."""
    source = (TESTS / "conftest.py").read_text(encoding="utf-8")
    assert "def pytest_collection_modifyitems" in source
    assert "pytest.mark.offline" in source
