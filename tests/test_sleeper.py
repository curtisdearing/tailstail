"""Sleeper cross-check (nflvalue/sources/sleeper.py): divergence is a FLAG,
never a number (PHASE1_HANDSOFF_DESIGN.md H5). Offline, fixture-driven."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue.sources import sleeper  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def proj_fixture():
    return json.loads((FIXTURES / "sleeper_projections_recorded.json").read_text())


@pytest.fixture(scope="module")
def map_fixture():
    return json.loads((FIXTURES / "sleeper_players_map_recorded.json").read_text())


def test_parse_projections_shapes_markets(proj_fixture):
    df = sleeper.parse_projections(proj_fixture["payload"])
    assert not df.empty
    assert set(df.columns) == {"sleeper_id", "name", "team", "pos", "market", "sleeper_proj"}
    wr_rows = df[(df["pos"] == "WR") & (df["market"] == "receiving_yards")]
    assert len(wr_rows) > 0
    assert (wr_rows["sleeper_proj"] > 0).all()
    # anytime_td aggregates rush_td + rec_td into one expected-TD number
    assert "anytime_td" in set(df["market"])


def test_parse_projections_rejects_garbage():
    with pytest.raises(sleeper.SleeperSchemaError):
        sleeper.parse_projections({"not": "a list"})
    with pytest.raises(sleeper.SleeperSchemaError):
        sleeper.parse_projections([{"totally": "wrong"}, {"shape": 1}])


def test_parse_player_map(map_fixture):
    df = sleeper.parse_player_map(map_fixture["payload"])
    assert not df.empty
    assert set(df.columns) == {"sleeper_id", "gsis_id", "full_name", "team", "position"}
    assert df["gsis_id"].str.startswith("00-").all()


def test_attach_gsis_keeps_unmatched_visible(proj_fixture, map_fixture):
    proj = sleeper.parse_projections(proj_fixture["payload"])
    pmap = sleeper.parse_player_map(map_fixture["payload"])
    out = sleeper.attach_gsis(proj, pmap)
    assert len(out) == len(proj)                       # nothing dropped
    assert out["gsis_id"].notna().sum() > 0            # some matched
    # unmatched rows stay, flagged by NaN -- never name-guessed
    assert "gsis_id" in out.columns


def test_divergence_flags_big_gap_not_small():
    small = sleeper.divergence(model_mean=60.0, sleeper_proj=55.0, market="receiving_yards")
    big = sleeper.divergence(model_mean=95.0, sleeper_proj=45.0, market="receiving_yards")
    assert small["divergence_flag"] is False
    assert big["divergence_flag"] is True
    assert big["abs_diff"] == 50.0


def test_divergence_no_fantasy_ref():
    d = sleeper.divergence(model_mean=60.0, sleeper_proj=None, market="receiving_yards")
    assert d["divergence_flag"] is False
    assert d["note"] == "no_fantasy_ref"


def test_divergence_never_returns_a_replacement_number():
    d = sleeper.divergence(model_mean=95.0, sleeper_proj=45.0, market="receiving_yards")
    # the result carries diffs and a flag -- no key could be mistaken for a projection
    assert set(d) == {"divergence_flag", "abs_diff", "rel_diff", "threshold", "note"}


def test_divergence_is_pure(proj_fixture):
    payload = copy.deepcopy(proj_fixture["payload"])
    sleeper.parse_projections(payload)
    assert payload == proj_fixture["payload"]          # input untouched
