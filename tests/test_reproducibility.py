"""Same inputs -> identical outputs, every time (PHASE1_HANDSOFF_DESIGN.md:
the projection engine must be deterministic and backtestable; the LLM layer,
when it exists, is not allowed anywhere near these numbers). Nothing in
``features.py`` or ``projection.py`` uses randomness today, so this is a
strict equality check, not a statistical one -- if it ever fails, either a
source of nondeterminism (unordered groupby, dict iteration, wall-clock) has
crept in, or it is genuinely a numeric bug.
"""

from __future__ import annotations

import pandas as pd

from nflvalue.features import build_opp_pos_def, build_player_week, build_team_week
from nflvalue.projection import game_script_multipliers, project


def test_player_week_is_deterministic(pbp_tiny):
    a = build_player_week(pbp_tiny)
    b = build_player_week(pbp_tiny)
    pd.testing.assert_frame_equal(a, b)


def test_opp_pos_def_is_deterministic(pbp_tiny):
    a = build_opp_pos_def(pbp_tiny)
    b = build_opp_pos_def(pbp_tiny)
    pd.testing.assert_frame_equal(a, b)


def test_team_week_is_deterministic(pbp_tiny):
    a = build_team_week(pbp_tiny)
    b = build_team_week(pbp_tiny)
    pd.testing.assert_frame_equal(a, b)


def test_project_is_deterministic_given_identical_rows(pbp_tiny):
    pw = build_player_week(pbp_tiny)
    tw = build_team_week(pbp_tiny)
    opd = build_opp_pos_def(pbp_tiny)

    rec = pw[pw["role"].isin(["WR", "TE"])].iloc[10].to_dict()
    team_row = tw[(tw["team"] == rec["team"]) & (tw["season"] == rec["season"]) & (tw["week"] == rec["week"])]
    team_row = team_row.iloc[0].to_dict() if len(team_row) else None
    opp_row = opd[(opd["defteam"] == rec["defteam"]) & (opd["role"] == rec["role"])
                  & (opd["season"] == rec["season"]) & (opd["week"] == rec["week"])]
    opp_row = opp_row.iloc[0].to_dict() if len(opp_row) else None

    r1 = project(rec, "receiving_yards", team_row=team_row, opp_row=opp_row, line=55.5, sd=25.0)
    r2 = project(rec, "receiving_yards", team_row=team_row, opp_row=opp_row, line=55.5, sd=25.0)
    assert r1 == r2

    # game_script_multipliers is a pure function of a numeric margin -- same in, same out
    assert game_script_multipliers(-3.5) == game_script_multipliers(-3.5)
    assert game_script_multipliers(6.0) == game_script_multipliers(6.0)


def test_project_seed_argument_does_not_change_output(pbp_tiny):
    """All current markets are closed-form (no simulation), so the ``seed``
    kwarg on ``project()`` must be a no-op today -- verified explicitly so a
    future change can't silently make results seed-dependent without this
    test being updated too."""
    pw = build_player_week(pbp_tiny)
    row = pw[pw["role"] == "QB"].iloc[5].to_dict()
    r1 = project(row, "passing_yards", line=220.5, sd=50.0, seed=1)
    r2 = project(row, "passing_yards", line=220.5, sd=50.0, seed=999)
    assert r1 == r2
