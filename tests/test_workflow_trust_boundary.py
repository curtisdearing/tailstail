"""Where the weekly workflow is allowed to put raw ESPN state, and who is trusted.

The previous fix moved the row-level ESPN ledger and the pre-kickoff captures
out of the public Pages payload, the workflow artifact and the public release
asset -- and into an `actions/cache` entry, with a comment claiming a cache is
"published to nobody". That claim is false. GitHub documents that a workflow
triggered by a pull request can restore caches created on the default branch,
and says not to store sensitive information in one:
https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching

So there are two questions here, and they are different. *Where* may raw state
be written -- answered by an allow-list of destinations, none of which is a
cache. And *who* may reach the credential that opens the private store --
answered by a classifier that requires the run to be on the default branch,
not merely to have blank dispatch inputs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORKFLOW_PATH = ROOT / ".github" / "workflows" / "fantasy-weekly.yml"
CLASSIFIER = ROOT / "scripts" / "classify_run.sh"

#: Every path that is raw ESPN per-player state.
RAW_ESPN_PATHS = ("data/espn_comparison_ledger.json", "data/espn_snapshots")
#: Destinations that are, or can become, readable outside a trusted run.
PUBLIC_SINKS = ("actions/cache", "upload-artifact", "upload-pages-artifact",
                "deploy-pages", "_site", "gh release upload", "state_store.py pack")


@pytest.fixture(scope="module")
def workflow() -> str:
    return WORKFLOW_PATH.read_text()


def steps(workflow: str) -> list[str]:
    """The workflow split into steps, each carrying the comments written above it.

    A naive split on the `- ` marker puts a step's explanatory comment block at
    the end of the PREVIOUS step, which is exactly backwards for reading a step
    as a unit -- and would let an `env:` block be checked against the wrong
    comment.
    """
    lines = workflow.splitlines()
    # A step starts with `- name:` or `- uses:` at step indentation. Matching a
    # bare `      - ` also catches the `on.push.paths` list items, which shifts
    # every block by one and quietly makes these assertions read the wrong step.
    starts = [index for index, line in enumerate(lines)
              if line.startswith(("      - name:", "      - uses:"))]
    blocks = []
    for position, start in enumerate(starts):
        head = start
        while head > 0 and lines[head - 1].strip().startswith("#"):
            head -= 1
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        while end - 1 > start and lines[end - 1].strip().startswith("#"):
            end -= 1
        blocks.append("\n".join(lines[head:end]))
    return blocks


# --------------------------------------------------------------------------- #
# C1. Raw ESPN state reaches no public sink, and no cache at all
# --------------------------------------------------------------------------- #
def test_the_workflow_uses_no_cache_for_espn_state(workflow):
    for block in steps(workflow):
        if "actions/cache" not in block:
            continue
        for raw in RAW_ESPN_PATHS:
            assert raw not in block, f"raw ESPN path {raw} in a cache step"


def test_no_raw_espn_path_appears_in_any_public_sink(workflow):
    for block in steps(workflow):
        if not any(sink in block for sink in PUBLIC_SINKS):
            continue
        for raw in RAW_ESPN_PATHS:
            assert raw not in block, f"raw ESPN path {raw} in a published step"


def test_the_pages_payload_is_the_allow_listed_public_file(workflow):
    assert "cp data/fantasy_public.json _site/fantasy_latest.json" in workflow
    assert "cp data/fantasy_latest.json _site/" not in workflow


def test_the_public_grading_history_is_what_survives_between_runs(workflow):
    assert "data/espn_comparison_history.json" in workflow


# --------------------------------------------------------------------------- #
# C2/C3. Trust: the classifier, executed rather than read
# --------------------------------------------------------------------------- #
def classify(**env) -> dict[str, str]:
    """Run the real classifier the workflow runs, and read its GITHUB_OUTPUT."""
    import os
    import tempfile

    with tempfile.NamedTemporaryFile("w+", suffix=".out", delete=False) as handle:
        output_path = handle.name
    environment = {
        **os.environ, "GITHUB_OUTPUT": output_path,
        "IN_SEASON": "", "IN_WEEK": "", "IN_FAST": "false",
        "GH_REF": "refs/heads/main", "GH_EVENT": "schedule",
        **{key: str(value) for key, value in env.items()},
    }
    completed = subprocess.run(["bash", str(CLASSIFIER)], env=environment,
                               capture_output=True, text=True, check=False)
    parsed = dict(line.split("=", 1) for line in Path(output_path).read_text().splitlines()
                  if "=" in line)
    Path(output_path).unlink()
    parsed["_exit"] = str(completed.returncode)
    parsed["_stderr"] = completed.stderr
    return parsed


def test_a_scheduled_run_on_main_is_production():
    assert classify()["production"] == "true"


def test_a_dispatch_on_main_with_blank_inputs_is_production():
    assert classify(GH_EVENT="workflow_dispatch")["production"] == "true"


def test_a_dispatch_from_another_branch_with_blank_inputs_is_not_production():
    """Blank optional inputs used to be the whole test. A branch is not trust."""
    result = classify(GH_EVENT="workflow_dispatch", GH_REF="refs/heads/agent/anything")
    assert result["production"] == "false"


@pytest.mark.parametrize("ref", [
    "refs/heads/agent/finalize-simple-picks-2026", "refs/pull/11/merge",
    "refs/heads/mainline", "refs/tags/v1", "refs/heads/Main", "",
])
def test_no_ref_other_than_the_default_branch_is_production(ref):
    assert classify(GH_REF=ref)["production"] == "false"


@pytest.mark.parametrize("override", [
    {"IN_FAST": "true"}, {"IN_SEASON": "2024"}, {"IN_WEEK": "3"},
])
def test_an_overridden_run_on_main_is_still_not_production(override):
    assert classify(**override)["production"] == "false"


def test_the_classifier_still_rejects_a_malformed_season_or_week():
    assert classify(IN_SEASON="20x4")["_exit"] != "0"
    assert classify(IN_WEEK="99")["_exit"] != "0"


def test_the_workflow_passes_ref_and_event_through_the_environment(workflow):
    """Expression text spliced into `run:` is executed by bash before parsing."""
    classifier_step = next(block for block in steps(workflow) if "classify_run.sh" in block)
    assert "GH_REF: ${{ github.ref }}" in classifier_step
    assert "GH_EVENT: ${{ github.event_name }}" in classifier_step
    for line in classifier_step.splitlines():
        stripped = line.strip()
        if stripped.startswith(("bash ", "./scripts/", "python ")) or stripped == "run: |":
            assert "${{" not in stripped, line


# --------------------------------------------------------------------------- #
# C4. Only a trusted production run can reach the private store
# --------------------------------------------------------------------------- #
def private_steps(workflow: str) -> list[str]:
    return [block for block in steps(workflow)
            if "TAILSTAIL_STATE_SSH_KEY" in block or "private_state.py" in block
            or "tailstail-state" in block]


def test_every_private_state_step_is_gated_on_a_production_run(workflow):
    blocks = private_steps(workflow)
    assert blocks, "the workflow has no private-state steps at all"
    for block in blocks:
        assert "steps.mode.outputs.production == 'true'" in block, block[:200]


def test_the_deploy_key_is_named_only_inside_a_gated_step(workflow):
    for block in steps(workflow):
        if "TAILSTAIL_STATE_SSH_KEY" not in block:
            continue
        assert "steps.mode.outputs.production == 'true'" in block


def test_the_private_checkout_lands_in_a_gitignored_path(workflow):
    assert "path: .tailstail-state" in workflow
    assert ".tailstail-state/" in (ROOT / ".gitignore").read_text()


def test_a_private_state_failure_warns_and_never_falls_back(workflow):
    blocks = private_steps(workflow)
    assert any("continue-on-error: true" in block for block in blocks)
    assert "::warning::" in workflow
    # No private-state step may reach for a cache. Checked against the action it
    # would have to `uses:`, not against the word -- the comment above these
    # steps explains why a cache is not the boundary, which is the point.
    for block in blocks:
        for line in block.splitlines():
            assert not line.strip().startswith("uses: actions/cache"), line


def test_the_workflow_keeps_least_privilege_and_writes_nothing_to_espn(workflow):
    assert "permissions:\n  contents: read" in workflow
    # Count the permission itself, not the sentence in the header comment that
    # says which jobs hold it.
    granted = [line for line in workflow.splitlines()
               if line.strip() == "pages: write" and not line.strip().startswith("#")]
    assert len(granted) == 2, granted
    for verb in ("curl -X POST", "curl -X PUT", "espn_client", "--espn-s2", "--swid"):
        assert verb not in workflow


# --------------------------------------------------------------------------- #
# C5. No surviving claim that an Actions cache is private
# --------------------------------------------------------------------------- #
FALSE_CACHE_CLAIMS = (
    "published to nobody", "repository-scoped", "repo-scoped",
    "scoped to the repository and published",
)


@pytest.mark.parametrize("relative", [
    ".github/workflows/fantasy-weekly.yml", "scripts/state_store.py",
    "docs/DECISION_CARD.md", "README.md",
    "nflvalue/fantasy/private_boundary.py", "nflvalue/fantasy/private_state.py",
])
def test_no_file_still_claims_an_actions_cache_is_private(relative):
    text = (ROOT / relative).read_text().lower()
    for claim in FALSE_CACHE_CLAIMS:
        assert claim not in text, f"{relative} still claims: {claim}"


# --------------------------------------------------------------------------- #
# D. The frozen forecast centre is untouched by any of this
# --------------------------------------------------------------------------- #
#: The 2026 freeze's forecast centre. None of the public/private split, the
#: grading-history contract or the private-state boundary is a lever, so none of
#: them may move a projection. Pinned by content digest rather than by "nothing
#: looked different" (scripts/neutrality_hashes.py makes the same argument for
#: the production frames).
FROZEN_CENTRE = (
    "nflvalue/fantasy/scoring.py",
    "nflvalue/fantasy/models.py",
    "nflvalue/fantasy/features.py",
    "nflvalue/fantasy/hierarchy.py",
    "nflvalue/fantasy/role_state.py",
    "nflvalue/fantasy/config.py",
    "nflvalue/fantasy/simulation.py",
    "nflvalue/projection_snapshot.py",
)


def test_no_new_module_reaches_into_the_frozen_forecast_centre():
    """The new code may read a contract; it may not import the model."""
    frozen_names = {Path(relative).stem for relative in FROZEN_CENTRE}
    for relative in ("nflvalue/fantasy/private_state.py",
                     "nflvalue/fantasy/decision_card.py",
                     "nflvalue/fantasy/private_boundary.py"):
        source = (ROOT / relative).read_text()
        for name in frozen_names - {"config"}:
            assert f"import {name}" not in source, (relative, name)


def test_generic_ppr_scoring_is_unchanged():
    """One arithmetic check on the frozen scorer, longhand from the rule values."""
    from nflvalue.fantasy.config import ScoringRules
    from nflvalue.fantasy.scoring import score_components

    rules = ScoringRules.preset("ppr")
    components = {
        "passing_yards": 300.0, "passing_tds": 2.0, "passing_interceptions": 1.0,
        "rushing_yards": 40.0, "rushing_tds": 1.0,
        "receiving_yards": 85.0, "receiving_tds": 1.0, "receptions": 7.0,
        "fumbles_lost": 1.0,
    }
    expected = (300.0 * rules.passing_yard + 2.0 * rules.passing_td
                + 1.0 * rules.interception
                + 40.0 * rules.rushing_yard + 1.0 * rules.rushing_td
                + 85.0 * rules.receiving_yard + 1.0 * rules.receiving_td
                + 7.0 * rules.reception
                + 1.0 * rules.fumble_lost)
    assert score_components(components, rules) == pytest.approx(expected)
    assert score_components(components, rules) == pytest.approx(47.5)


def test_a_component_name_the_scorer_does_not_know_is_worth_zero():
    """Pinned because it caught this file's own author.

    `scoring._value` reads an absent component as 0.0, so a misspelled key --
    `interceptions` for `passing_interceptions` -- is scored as "the event did
    not happen" rather than raising. That is the documented reason an "exact
    custom scoring" label is refused over simulator output, and it is worth an
    assertion rather than a comment.
    """
    from nflvalue.fantasy.config import ScoringRules
    from nflvalue.fantasy.scoring import score_components

    rules = ScoringRules.preset("ppr")
    assert score_components({"interceptions": 5.0}, rules) == pytest.approx(0.0)
    assert score_components({"passing_interceptions": 5.0}, rules) == pytest.approx(
        5.0 * rules.interception)


def test_the_projection_pipeline_never_imports_the_private_boundary_modules():
    """A projection must not be able to depend on whether raw state arrived."""
    for relative in FROZEN_CENTRE:
        source = (ROOT / relative).read_text()
        for module in ("private_state", "private_boundary", "decision_card",
                       "decision_page", "espn_compare"):
            assert module not in source, (relative, module)


# --------------------------------------------------------------------------- #
# E. The public PAGE, not just the public JSON
# --------------------------------------------------------------------------- #
# The JSON boundary was built and tested first, and it was the wrong half. The
# ESPN per-player rows were never only in `data/fantasy_latest.json` -- the
# weekly dashboard rendered them into a table, and `fantasy.html` is copied
# verbatim to `_site/index.html` and served to the public internet. A grep for
# path literals in workflow steps cannot see a leak that travels inside a file
# the workflow legitimately publishes.
def comparison_with_rows() -> dict:
    return {
        "status": "ok", "season": 2026, "current_week": 1, "disclaimer": "d",
        "espn_provenance": {"retrieved_at": "2026-09-09T12:00:00+00:00",
                            "source": {"name": "ESPN Fantasy API"},
                            "scoring": {"rescored_vs_applied_mean_abs_delta": 0.02}},
        "identity": {"espn_players": 400, "matched": 380, "coverage_pct": 95.0,
                     "unmatched_no_crosswalk_count": 10,
                     "unmatched_model_not_projected_count": 10},
        "current_week_rows": [{"abs_delta_rank": 1, "position": "RB",
                               "player_name": "R. Bell", "team": "PHI",
                               "espn_pts": 18.4, "model_pts": 14.9, "delta": -3.5}],
        "season_series": [{"week": 1, "n_played": 300, "mae_espn": 5.4, "mae_model": 5.1,
                           "espn_closer": 128, "model_closer": 150}],
    }


def summaries_frame():
    import pandas as pd

    return pd.DataFrame([{
        # Values deliberately distinct from the ESPN row fixture below, so a
        # match in the rendered page can only have come from the ESPN rows.
        "position": "RB", "player_name": "W. Gray", "team": "MIN", "mean": 22.3,
        "median": 21.7, "event_simulator_mean": 21.1, "p10": 11.2, "p90": 31.6,
        "prob_15_plus": 0.61, "prob_20_plus": 0.42, "availability_probability": 0.97,
        "component_model_disagreement": False}])


def test_the_public_dashboard_refuses_row_level_espn_data(tmp_path):
    """Fail closed at the renderer, so no caller can publish rows by accident."""
    from nflvalue.fantasy.dashboard import render_fantasy_dashboard
    from nflvalue.fantasy.private_boundary import PrivateDataLeak

    with pytest.raises(PrivateDataLeak):
        render_fantasy_dashboard(
            summaries_frame(), tmp_path / "fantasy.html", season=2026, week=1,
            generated_at="2026-09-09T12:00:00+00:00",
            espn_comparison=comparison_with_rows())


def test_the_public_dashboard_renders_the_aggregate_grading(tmp_path):
    """The redacted comparison still publishes the scoreboard, without players."""
    from nflvalue.fantasy.dashboard import render_fantasy_dashboard
    from nflvalue.fantasy.private_boundary import public_espn_comparison

    out = tmp_path / "fantasy.html"
    render_fantasy_dashboard(
        summaries_frame(), out, season=2026, week=1,
        generated_at="2026-09-09T12:00:00+00:00",
        espn_comparison=public_espn_comparison(comparison_with_rows()))
    document = out.read_text()

    assert "Season grading" in document
    assert "5.40" in document and "5.10" in document          # the aggregate MAEs
    for leaked in ("R. Bell", "18.4", "14.9", "ESPN (PPR)", "Model − ESPN", "Δ rank"):
        assert leaked not in document, leaked


def test_the_text_guard_catches_a_per_player_espn_table():
    """The guard must fire on the markup itself, not only on league identity."""
    from nflvalue.fantasy.private_boundary import PrivateDataLeak, assert_public_text_safe

    page = ("<table><thead><tr><th>Δ rank</th><th>Pos</th><th>Player</th>"
            "<th>ESPN (PPR)</th><th>Model (PPR)</th><th>Model − ESPN</th></tr></thead></table>")
    with pytest.raises(PrivateDataLeak):
        assert_public_text_safe(page, what="fantasy.html")


def test_the_pipeline_hands_the_page_only_the_redacted_comparison():
    source = (ROOT / "scripts" / "fantasy_weekly.py").read_text()
    call = source[source.index("render_fantasy_dashboard("):]
    call = call[:call.index(")")]
    assert "espn_comparison=public_payload[\"espn_comparison\"]" in call, call


# --------------------------------------------------------------------------- #
# G. Durability of the one public copy, and a server-side trust gate
# --------------------------------------------------------------------------- #
def test_a_failed_public_state_restore_is_fatal_on_a_production_run(workflow):
    """The history is gitignored: the release asset is its only durable copy.

    Continuing after a failed restore loads an empty history, saves it, and
    repoints the release at the empty archive. One transient network error and
    the published season is gone.
    """
    restore = next(block for block in steps(workflow)
                   if "Restore prior fantasy state" in block)
    assert "exit 1" in restore
    assert "::error::" in restore


def test_the_prior_season_archive_travels_in_the_public_state_profile():
    from scripts import state_store

    fantasy = state_store.STATE_PROFILES["fantasy"]
    assert "data/espn_comparison_history.json" in fantasy
    assert any("espn_comparison_history." in pattern and pattern.endswith(".json")
               and "*" in pattern for pattern in fantasy), fantasy


def test_the_private_state_secret_is_gated_by_a_github_environment(workflow):
    """A step-level `if:` is a convention; secrets resolve per JOB.

    An environment with a deployment-branch rule is enforced by GitHub, not by
    a script read out of the dispatched ref, so it is the only gate here that a
    branch cannot edit its way past.
    """
    project = workflow.split("  project:")[1].split("\n  deploy-")[0]
    assert "environment: fantasy-production" in project
    assert "TAILSTAIL_STATE_SSH_KEY" in project


def test_every_private_state_step_also_checks_the_ref_directly(workflow):
    """Belt and braces: the classifier script comes from the dispatched ref."""
    for block in private_steps(workflow):
        assert "github.ref == 'refs/heads/main'" in block, block[:160]


def test_the_private_restore_step_cannot_fail_the_run(workflow):
    """Named explicitly: `any(...)` across the private steps proved nothing."""
    restore = next(block for block in private_steps(workflow)
                   if "actions/checkout@v6" in block and "tailstail-state" in block)
    assert "continue-on-error: true" in restore
