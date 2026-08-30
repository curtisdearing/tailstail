"""K shadow lane — research only, and provably outside the frozen path.

These tests are the promotion gate's floor: they do not ask whether the kicker
numbers are *good*, they ask whether the lane is honest — that it scores only
through the live league contract, that it refuses rather than guesses, that it
cannot touch the frozen QB/RB/WR/TE artifacts, and that it is reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nflvalue.fantasy import shadow_kicker as SK
from nflvalue.fantasy.config import ModelConfig
from nflvalue.fantasy.espn_contract import from_settings_payload

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = json.loads(
    (ROOT / "tests/fixtures/espn_league_settings_2026_recorded.json").read_text())


@pytest.fixture(scope="module")
def contract():
    return from_settings_payload(PAYLOAD)


PROVENANCE = [{"source": "nflverse pbp", "retrieved_at": "2026-08-29T10:00:00Z",
               "as_of": "2026-08-29T09:00:00Z"}]


def _history(player="K1", team="BUF", weeks=8, season=2025):
    """A kicker with a real, boring workload."""
    return pd.DataFrame([{
        "season": season, "week": w, "player_id": player, "player_display_name": "Test Kicker",
        "position": "K", "team": team,
        "fg_made_0_19": 0, "fg_made_20_29": 1, "fg_made_30_39": 1,
        "fg_made_40_49": 1, "fg_made_50_59": 0, "fg_made_60_": 0,
        "fg_missed_0_19": 0, "fg_missed_20_29": 0, "fg_missed_30_39": 0,
        "fg_missed_40_49": 1, "fg_missed_50_59": 0, "fg_missed_60_": 0,
        "pat_made": 2, "pat_att": 2,
    } for w in range(1, weeks + 1)])


# --------------------------------------------------------------------- #
# Isolation — the whole reason this is a separate lane
# --------------------------------------------------------------------- #
def test_k_is_not_in_the_frozen_position_set():
    assert "K" not in ModelConfig().positions
    assert "D/ST" not in ModelConfig().positions


def test_a_kicker_row_cannot_validate_against_the_snapshot_contract():
    """PlayerProjectionSnapshot is scoring-independent and offence-only."""
    schema = json.loads((ROOT / "schemas/player_projection_snapshot.schema.json").read_text())
    enum = schema["properties"]["players"]["items"]["properties"]["position"]["enum"]
    assert enum == ["QB", "RB", "WR", "TE"]
    assert schema["properties"]["players"]["items"]["additionalProperties"] is False


def test_shadow_module_never_imports_the_frozen_model_path():
    """An import edge is the only way a shadow number could reach production.

    Checked over the parsed import graph, not the file text -- a docstring that
    merely *mentions* `simulation` is not an import, and a test that cannot
    tell the difference would fail for the wrong reason forever.
    """
    import ast

    tree = ast.parse((ROOT / "nflvalue/fantasy/shadow_kicker.py").read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(f"{node.module or ''}.{a.name}" for a in node.names)

    forbidden = {"models", "simulation", "season", "hierarchy", "draft",
                 "trade_planner", "nflvalue.projection_snapshot",
                 "projection_snapshot"}
    hits = {name for name in imported
            if name.lstrip(".").split(".")[0] in forbidden or name in forbidden}
    assert not hits, f"shadow lane imports the frozen path: {sorted(hits)}"


# --------------------------------------------------------------------- #
# Scoring flows through the contract, never through a constant
# --------------------------------------------------------------------- #
def test_points_come_from_the_contract_not_from_code(contract):
    """Mutating the contract must move the answer; a hardcoded 3.0 would not."""
    line = {"fg_made_0_39": 1, "pat_made": 1}
    base = SK.score_line(line, contract)

    louder = json.loads(json.dumps(PAYLOAD))
    for item in louder["settings"]["scoringSettings"]["scoringItems"]:
        if item["statId"] == SK.FG_0_39_STAT_ID:
            item["points"] = 99.0
    bumped = from_settings_payload(louder)
    assert bumped.scoring_hash != contract.scoring_hash, "payload edit had no effect"
    assert SK.score_line(line, bumped) != base


def test_contract_bucket_edges_are_exact(contract):
    """49 and 50 are worth different points in this league."""
    assert contract.points("fg_made_40_49") == 4.0
    assert contract.points("fg_made_50_59") == 5.0
    for distance, key in ((39, "fg_made_0_39"), (40, "fg_made_40_49"),
                          (49, "fg_made_40_49"), (50, "fg_made_50_59"),
                          (59, "fg_made_50_59"), (60, "fg_made_60_plus")):
        assert SK.bucket_for(distance) == key, f"{distance} yards"


# --------------------------------------------------------------------- #
# Distributions, not point estimates
# --------------------------------------------------------------------- #
def test_projection_is_a_distribution(contract):
    row = SK.project(_history(), "K1", contract, season=2026, week=1,
                     simulations=2000, seed=7, active=True)
    dist = row["distribution"]
    for key in ("mean", "sd", "p05", "p25", "p50", "p75", "p95", "p_zero"):
        assert key in dist
    assert dist["sd"] > 0, "a point estimate dressed as a distribution"
    assert dist["p05"] <= dist["p50"] <= dist["p95"]
    assert row["simulations"] == 2000


def test_same_seed_reproduces_the_same_numbers(contract):
    a = SK.project(_history(), "K1", contract, season=2026, week=1,
                   simulations=1000, seed=11, active=True)
    b = SK.project(_history(), "K1", contract, season=2026, week=1,
                   simulations=1000, seed=11, active=True)
    assert a["distribution"] == b["distribution"]
    c = SK.project(_history(), "K1", contract, season=2026, week=1,
                   simulations=1000, seed=12, active=True)
    assert c["distribution"] != a["distribution"], "seed is not actually used"


# --------------------------------------------------------------------- #
# Fail closed
# --------------------------------------------------------------------- #
def test_unknown_kicker_is_unavailable_not_league_average(contract):
    row = SK.project(_history(), "NOBODY", contract, season=2026, week=1,
                     simulations=500, seed=3)
    assert row["status"] == "unavailable"
    assert row["distribution"] is None
    assert row["unavailable_reason"]


def test_inactive_kicker_is_unavailable(contract):
    row = SK.project(_history(), "K1", contract, season=2026, week=1,
                     simulations=500, seed=3, active=False)
    assert row["status"] == "unavailable"
    assert row["distribution"] is None


def test_history_after_the_target_week_is_ignored(contract):
    """A pregame projection may not see its own or any later week."""
    past = _history(season=2025, weeks=8)
    future = _history(season=2026, weeks=4)
    future.loc[:, "fg_made_50_59"] = 9          # implausible, and in the future
    clean = SK.project(past, "K1", contract, season=2026, week=1,
                       simulations=1500, seed=5)
    poisoned = SK.project(pd.concat([past, future], ignore_index=True), "K1",
                          contract, season=2026, week=1, simulations=1500, seed=5)
    assert clean["distribution"] == poisoned["distribution"]


# --------------------------------------------------------------------- #
# Artifact
# --------------------------------------------------------------------- #
def test_artifact_is_labelled_shadow_and_survives_json(tmp_path, contract):
    out = tmp_path / "k_weekly_2026_wk1.json"
    art = SK.build_artifact(_history(), ["K1"], contract, season=2026, week=1,
                            simulations=500, seed=9, out_path=out,
                            active={"K1": True}, provenance=PROVENANCE)
    assert art["status"] == "shadow"

    loaded = json.loads(out.read_text())
    assert loaded["status"] == "shadow"
    assert loaded["scoring_hash"] == contract.scoring_hash
    assert loaded["roster_slot_hash"] == contract.roster_slot_hash
    for key in ("model_run_at", "information_as_of", "simulations", "seed",
                "provenance", "players", "model_version", "content_sha256",
                "promotion"):
        assert key in loaded, f"artifact missing {key}"
    assert loaded["players"], "empty artifact is not a candidate"
    assert loaded["promoted"] is False


def test_artifact_never_carries_an_offensive_position(tmp_path, contract):
    art = SK.build_artifact(_history(), ["K1"], contract, season=2026, week=1,
                            simulations=200, seed=9, out_path=tmp_path / "a.json",
                            active={"K1": True}, provenance=PROVENANCE)
    for player in art["players"]:
        assert player["position"] == "K"
