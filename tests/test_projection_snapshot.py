import json

import numpy as np
import pandas as pd
import pytest

from nflvalue.fantasy.config import ScoringRules
from nflvalue.fantasy.scoring import score_components
from nflvalue.projection_snapshot import (
    COMPONENT_NAMES,
    build_projection_snapshot,
    load_component_samples,
    validate_projection_snapshot,
    write_component_samples,
    write_projection_snapshot,
)


def _components():
    ids = ["p1", "p2"]
    values = {}
    for index, name in enumerate(COMPONENT_NAMES):
        values[name] = pd.DataFrame(
            np.arange(8, dtype=float).reshape(4, 2) / (index + 2), columns=ids
        )
    return values


def _snapshot(tmp_path):
    components = _components()
    sample_artifact = write_component_samples(components, tmp_path / "samples.parquet")
    players = pd.DataFrame([
        {"player_id": "p1", "player_name": "One", "position": "QB", "team": "A",
         "opponent_team": "B", "game_id": "g", "status_inactive": 0, "injury_out": 0},
        {"player_id": "p2", "player_name": "Two", "position": "WR", "team": "A",
         "opponent_team": "B", "game_id": "g", "status_inactive": 0, "injury_out": 0},
    ])
    summaries = pd.DataFrame([
        {"player_id": "p1", "availability_probability": 1.0},
        {"player_id": "p2", "availability_probability": 0.75},
    ])
    snapshot = build_projection_snapshot(
        players, summaries, components, season=2026, week=1,
        generated_at="2026-09-09T20:00:00+00:00",
        information_as_of="2026-09-09T19:55:00+00:00", model_version="abc123",
        simulation_metadata={"simulations": 4, "random_seed": 7, "players": 2,
                             "games": 1, "calibration": "test", "scoring": {"ppr": 1}},
        sample_artifact=sample_artifact, source_manifest={"source": "fixture"},
        component_validation={"status": "research_only", "evaluated_through": "2025-18"},
    )
    return snapshot, components


def test_projection_snapshot_is_consumer_neutral_and_round_trips_samples(tmp_path):
    snapshot, components = _snapshot(tmp_path)
    output = tmp_path / "snapshot.json"
    write_projection_snapshot(snapshot, output)
    loaded_snapshot = json.loads(output.read_text())
    validate_projection_snapshot(loaded_snapshot)
    assert "scoring" not in loaded_snapshot["simulation"]
    assert "fantasy_points" not in output.read_text()
    assert loaded_snapshot["component_validation"]["status"] == "research_only"

    loaded = load_component_samples(tmp_path / "samples.parquet")
    for name in COMPONENT_NAMES:
        assert np.allclose(
            loaded[name].sort_index(axis=1), components[name].sort_index(axis=1)
        )
    ppr = score_components(loaded, ScoringRules.preset("ppr"))
    standard = score_components(loaded, ScoringRules.preset("standard"))
    assert not np.allclose(ppr, standard)


def test_projection_snapshot_rejects_consumer_specific_fields(tmp_path):
    snapshot, _ = _snapshot(tmp_path)
    snapshot["players"][0]["edge"] = 0.08
    with pytest.raises(ValueError, match="consumer-specific"):
        validate_projection_snapshot(snapshot)


def test_projection_snapshot_rejects_unknown_market_validation_status(tmp_path):
    snapshot, _ = _snapshot(tmp_path)
    snapshot["component_validation"]["markets"] = {"passing_yards": "trust_me"}
    with pytest.raises(ValueError, match="market validation"):
        validate_projection_snapshot(snapshot)


def test_projection_snapshot_rejects_unordered_component_quantiles(tmp_path):
    snapshot, _ = _snapshot(tmp_path)
    snapshot["players"][0]["components"]["attempts"]["p10"] = 999
    with pytest.raises(ValueError, match="quantiles are unordered"):
        validate_projection_snapshot(snapshot)


def test_json_schema_names_every_component():
    from pathlib import Path

    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "schemas/player_projection_snapshot.schema.json").read_text()
    )
    assert set(schema["$defs"]["components"]["required"]) == set(COMPONENT_NAMES)
