"""Phase 1B: real positions (nflvalue/sources/rosters.py) replace Phase 1A's
role-inference heuristic. These tests check the WR/TE split actually landed
and that opp_pos_def is built from real receiver positions, not the old
combined REC bucket."""

from __future__ import annotations

from nflvalue.features import build_opp_pos_def, build_player_week
from nflvalue.sources import rosters as rostersmod


def test_player_week_has_four_real_position_buckets(pbp_tiny):
    pw = build_player_week(pbp_tiny)
    assert set(pw["role"].unique()) <= {"QB", "RB", "WR", "TE"}
    # a 1-season slice should have real examples of all four
    assert set(pw["role"].unique()) == {"QB", "RB", "WR", "TE"}


def test_most_rows_use_real_roster_position_not_fallback(pbp_tiny):
    pw = build_player_week(pbp_tiny)
    assert "position_source" in pw.columns
    roster_frac = (pw["position_source"] == "roster").mean()
    assert roster_frac > 0.9, f"expected >90% real roster positions, got {roster_frac:.1%}"


def test_known_players_get_correct_real_position(pbp_tiny):
    """Spot-check a few unambiguous, well-known players rather than trusting
    the aggregate stats alone."""
    pw = build_player_week(pbp_tiny)
    checks = {
        "00-0019596": "QB",  # Tom Brady
        "00-0031381": "WR",  # Davante Adams
    }
    for player_id, expected in checks.items():
        rows = pw[pw["player_id"] == player_id]
        if rows.empty:
            continue  # player not in this season slice; skip rather than fail
        assert (rows["role"] == expected).all(), f"{player_id} expected {expected}, got {rows['role'].unique()}"


def test_opp_pos_def_has_separate_wr_and_te_rows(pbp_tiny):
    opd = build_opp_pos_def(pbp_tiny)
    assert set(opd["role"].unique()) == {"QB", "RB", "WR", "TE"}
    # WR and TE factors for the same (defteam, week) should generally differ --
    # they're built from disjoint sets of plays (different targeted-receiver
    # positions), so it would be a bug if they were always identical.
    wr = opd[opd["role"] == "WR"].set_index(["season", "week", "defteam"])["roll_ypt_allowed_factor"]
    te = opd[opd["role"] == "TE"].set_index(["season", "week", "defteam"])["roll_ypt_allowed_factor"]
    common = wr.index.intersection(te.index)
    assert len(common) > 0
    assert not (wr.loc[common] == te.loc[common]).all()


def test_rosters_fixture_loads_and_has_expected_positions():
    fixture = rostersmod.load_fixture()
    assert len(fixture) > 0
    assert set(fixture["position"].unique()) <= {"QB", "RB", "WR", "TE"}
    assert {"season", "week", "team", "player_id", "full_name"}.issubset(fixture.columns)
