"""Deterministic context features: birthday/revenge/defensive-outs math,
walk-forward discipline, neutral degradation, panel + ledger wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue import context_features as cf  # noqa: E402


def _pack():
    meta = pd.DataFrame([
        {"player_id": "P1", "birth_date": "1998-11-12"},
        {"player_id": "P2", "birth_date": "2000-01-02"},   # year-boundary case
    ])
    rosters = pd.DataFrame([
        # P1: 2022 on AAA (5 wks), 2023 wk1-2 on BBB, then CCC
        *[{"season": 2022, "week": w, "team": "AAA", "position": "WR",
           "player_id": "P1", "full_name": "P One"} for w in range(1, 6)],
        *[{"season": 2023, "week": w, "team": "BBB", "position": "WR",
           "player_id": "P1", "full_name": "P One"} for w in (1, 2)],
        *[{"season": 2023, "week": w, "team": "CCC", "position": "WR",
           "player_id": "P1", "full_name": "P One"} for w in range(3, 10)],
    ])
    injuries = pd.DataFrame([
        {"season": 2023, "week": 9, "team": "DDD", "gsis_id": f"D{i}",
         "position": pos, "report_status": st, "full_name": f"D {i}"}
        for i, (pos, st) in enumerate([
            ("CB", "Out"), ("S", "Out"), ("LB", "Doubtful"),
            ("CB", "Questionable"),          # not counted (plays)
            ("WR", "Out"),                   # offense: not counted
        ])
    ])
    return cf.ContextPack(rosters, [2022, 2023], opd=None,
                          players_meta=meta, injuries=injuries)


def test_birthday_window_and_year_boundary():
    p = _pack()
    assert p.is_birthday_week("P1", "2023-11-10") == 1     # 2 days early
    assert p.is_birthday_week("P1", "2023-11-17") == 1     # 5 days after
    assert p.is_birthday_week("P1", "2023-10-01") == 0
    assert p.is_birthday_week("P2", "2023-12-30") == 1     # Jan-2 birthday, Dec game
    assert p.is_birthday_week("UNKNOWN", "2023-11-12") == 0  # no DOB -> 0, never guess


def test_revenge_is_walk_forward_and_excludes_current_team():
    p = _pack()
    # week 9 2023: P1 (now CCC) faces AAA (5 wks in 2022 -> revenge)
    assert p.revenge_game("P1", 2023, 9, "CCC", "AAA") == 1
    # BBB was only a 2-week stint (< min 3): not revenge
    assert p.revenge_game("P1", 2023, 9, "CCC", "BBB") == 0
    # current team can never be a revenge opponent
    assert p.revenge_game("P1", 2023, 9, "CCC", "CCC") == 0
    # in 2022 week 3, his FUTURE stints (BBB/CCC) don't exist yet
    assert p.former_teams("P1", 2022, 3, "AAA") == set()


def test_defense_outs_counts_def_only_and_out_statuses():
    p = _pack()
    total, dbs = p.defense_outs(2023, 9, "DDD")
    assert (total, dbs) == (3, 2)          # CB+S+LB out/doubtful; Questionable + WR excluded
    assert p.defense_outs(2023, 9, "ZZZ") == (0, 0)


def test_attach_neutral_without_pack():
    cands = pd.DataFrame([{"player_id": "P1", "season": 2023, "week": 9,
                           "team": "CCC", "defteam": "AAA", "pos": "WR",
                           "market": "receiving_yards", "gameday": "2023-11-12"}])
    out = cf.attach(cands, None)
    assert out.iloc[0]["is_birthday_week"] == 0
    assert out.iloc[0]["revenge_game"] == 0
    assert np.isnan(out.iloc[0]["def_out_total"])


def test_attach_with_pack_stamps_all_features():
    p = _pack()
    cands = pd.DataFrame([{"player_id": "P1", "season": 2023, "week": 9,
                           "team": "CCC", "defteam": "AAA", "pos": "WR",
                           "market": "receiving_yards", "gameday": "2023-11-12"}])
    out = cf.attach(cands, p)
    assert out.iloc[0]["is_birthday_week"] == 1
    assert out.iloc[0]["revenge_game"] == 1


def test_panel_items_and_ledger_tags():
    lean = {"is_birthday_week": 1, "revenge_game": 1,
            "def_out_total": 3, "def_out_db": 2}
    items = cf.panel_items(lean)
    assert any("birthday" in i for i in items)
    assert any("revenge" in i for i in items)
    assert any("3 Out/Doubtful" in i and "secondary" in i for i in items)
    # the ledger's keyword tagger picks these up as measurable tags
    from nflvalue.context_study import tags_from_text
    assert "birthday" in tags_from_text(items[0])
    assert "revenge" in tags_from_text(items[1])


def test_quiet_defense_not_flagged():
    assert cf.panel_items({"def_out_total": 1, "def_out_db": 0}) == []
