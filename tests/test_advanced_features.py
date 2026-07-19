"""Advanced features: the AsOfLookup anti-leak primitive, walk-forward
tendencies, contract-year math, neutral degradation, panel items."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import advanced_features as af


# --------------------------------------------------------------------------- #
# AsOfLookup: the missingness-leak guard
# --------------------------------------------------------------------------- #
def test_asof_is_strictly_before():
    """The 88%-OOS incident regression test: a value AT the candidate week
    must never be readable — only strictly-prior rows."""
    df = pd.DataFrame([
        {"player_id": "P1", "season": 2025, "week": 3, "v": 0.10},
        {"player_id": "P1", "season": 2025, "week": 5, "v": 0.30},
        {"player_id": "P1", "season": 2025, "week": 9, "v": 0.50},
    ])
    lk = af.AsOfLookup(df, ["v"])
    assert lk.get("P1", 2025, 5) == (0.10,)      # week 5 row invisible AT week 5
    assert lk.get("P1", 2025, 6) == (0.30,)      # visible one week later
    assert lk.get("P1", 2025, 12) == (0.50,)     # carries forward over gaps
    assert np.isnan(lk.get("P1", 2025, 3)[0])    # nothing strictly before wk3
    assert np.isnan(lk.get("GHOST", 2025, 9)[0])
    assert lk.get("P1", 2026, 1) == (0.50,)      # season boundary carries


def test_asof_presence_is_not_current_week_information():
    """Two players, one with red-zone work THIS week, one without: at this
    week both must resolve from prior history only — identical treatment."""
    df = pd.DataFrame([
        {"player_id": "A", "season": 2025, "week": 4, "v": 0.2},
        {"player_id": "A", "season": 2025, "week": 8, "v": 0.9},  # busy THIS week
        {"player_id": "B", "season": 2025, "week": 4, "v": 0.2},  # idle THIS week
    ])
    lk = af.AsOfLookup(df, ["v"])
    assert lk.get("A", 2025, 8) == lk.get("B", 2025, 8) == (0.2,)


# --------------------------------------------------------------------------- #
# Team tendencies: walk-forward + neutral filter
# --------------------------------------------------------------------------- #
def _mini_pbp():
    rows = []
    for wk in (1, 2, 3):
        for i in range(12):
            rows.append(dict(
                season=2025, week=wk, game_id=f"2025_{wk:02d}_AAA_BBB",
                season_type="REG", posteam="AAA", defteam="BBB",
                epa=0.1 * (1 if i % 2 else -1),
                pass_attempt=1 if i % 2 else 0, rush_attempt=0 if i % 2 else 1,
                complete_pass=0, pass_touchdown=0, rush_touchdown=0,
                air_yards=5.0, yards_after_catch=2.0, passing_yards=6.0,
                rushing_yards=4.0, receiver_player_id="R1", receiver_player_name="R",
                rusher_player_id="B1", rusher_player_name="B",
                passer_player_id="Q1", passer_player_name="Q",
                down=1 if i < 8 else 3, ydstogo=10, yardline_100=50 - i,
                score_differential=3, qtr=2, wp=0.55,
                xpass=0.5, pass_oe=(10.0 if i % 2 else -10.0), cpoe=2.0,
                shotgun=1, no_huddle=0,
                game_seconds_remaining=3600 - wk * 40 - i * 30,
                sack=0, qb_hit=0, fixed_drive=1 + i // 6,
            ))
    df = pd.DataFrame(rows)
    df["pass"] = df["pass_attempt"]
    df["rush"] = df["rush_attempt"]
    return df


def test_tendencies_are_walk_forward():
    tt = af.build_team_tendencies(_mini_pbp())
    wk1 = tt[(tt.team == "AAA") & (tt.week == 1)]
    wk2 = tt[(tt.team == "AAA") & (tt.week == 2)]
    assert np.isnan(wk1.iloc[0]["team_neutral_proe"])   # nothing before week 1
    assert not np.isnan(wk2.iloc[0]["team_neutral_proe"])  # week 1 informs week 2


def test_tendencies_ignore_non_neutral_plays():
    p = _mini_pbp()
    poisoned = p.copy()
    # rewrite week 1's NON-neutral plays (down 3) to absurd pass_oe: neutral
    # PROE at week 2 must not move
    poisoned.loc[(poisoned.week == 1) & (poisoned.down == 3), "pass_oe"] = 500.0
    a = af.build_team_tendencies(p)
    b = af.build_team_tendencies(poisoned)
    va = a[(a.team == "AAA") & (a.week == 2)].iloc[0]["team_neutral_proe"]
    vb = b[(b.team == "AAA") & (b.week == 2)].iloc[0]["team_neutral_proe"]
    assert va == pytest.approx(vb)


# --------------------------------------------------------------------------- #
# Contract year + neutral fill + panel
# --------------------------------------------------------------------------- #
def test_contract_year_math(tmp_path, monkeypatch):
    con = pd.DataFrame([
        {"gsis_id": "P1", "player": "One", "position": "WR", "year_signed": 2022,
         "years": 4, "value": 80.0, "apy": 20.0, "is_active": True},   # ends 2025
        {"gsis_id": "P2", "player": "Two", "position": "RB", "year_signed": 2024,
         "years": 1, "value": 2.0, "apy": 2.0, "is_active": True},     # ends 2024
    ])
    path = tmp_path / "contracts.parquet"
    con.to_parquet(path, index=False)
    lut = af.contract_year_lookup(str(path))
    assert lut[("P1", 2025)] == 1 and lut[("P1", 2024)] == 0
    assert lut[("P2", 2024)] == 1
    # walk-forward: a 2024-signed deal is unknown in 2023
    assert lut.get(("P2", 2023), 0) == 0


def test_attach_neutral_fills_every_feature():
    cands = pd.DataFrame([{"player_id": "P", "season": 2025, "week": 1,
                           "team": "AAA", "defteam": "BBB", "game_id": "G",
                           "market": "receiving_yards"}])
    out = af.attach_neutral(cands)
    assert all(f in out.columns for f in af.FEATURES)
    assert out.iloc[0]["oline_outs"] == 0 and out.iloc[0]["is_contract_year"] == 0


def test_panel_items():
    lean = {"is_contract_year": 1, "oline_outs": 3, "wind": 22.0, "qb_continuity": 0.1}
    items = af.panel_items(lean)
    assert any("contract year" in i for i in items)
    assert any("O-line lists 3" in i for i in items)
    assert any("wind 22" in i for i in items)
    assert any("trailing attempts" in i for i in items)
    assert af.panel_items({"wind": 5.0, "qb_continuity": 0.95}) == []
