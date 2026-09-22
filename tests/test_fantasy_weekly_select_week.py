"""`select_week` must pick the week by the real kickoff clock, not the calendar date.

Regression for the Wednesday-after-Monday-night bug: the 23:35 UTC Wednesday
cron, started before midnight UTC, used to return the week that had just
finished (2026-09-23 -> Week 2 again) because the Monday-night gameday was
still inside a two-day calendar grace window.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_weekly():
    spec = importlib.util.spec_from_file_location("fantasy_weekly_under_test", ROOT / "scripts" / "fantasy_weekly.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


weekly = _load_weekly()


def _schedule() -> pd.DataFrame:
    # 2026 Weeks 2-4, ET local times as nflverse publishes them.
    rows = [
        (2026, 2, "2026_02_DET_BUF", "2026-09-17", "20:15"),
        (2026, 2, "2026_02_CAR_ATL", "2026-09-20", "13:00"),
        (2026, 2, "2026_02_IND_KC", "2026-09-20", "20:20"),
        (2026, 2, "2026_02_NYG_LA", "2026-09-21", "20:15"),   # MNF, kicks 2026-09-22T00:15Z
        (2026, 3, "2026_03_A_B", "2026-09-24", "20:15"),      # TNF, kicks 2026-09-25T00:15Z
        (2026, 3, "2026_03_C_D", "2026-09-27", "13:00"),
        (2026, 3, "2026_03_E_F", "2026-09-28", "20:15"),
        (2026, 4, "2026_04_G_H", "2026-10-01", "20:15"),
        (2026, 4, "2026_04_I_J", "2026-10-04", "13:00"),
    ]
    frame = pd.DataFrame(rows, columns=["season", "week", "game_id", "gameday", "gametime"])
    frame["game_type"] = "REG"
    return frame


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        ("2026-09-20T12:15:00", (2026, 2)),   # Sunday cron, early window ahead
        ("2026-09-21T23:00:00", (2026, 2)),   # Monday evening, MNF not yet kicked
        ("2026-09-22T00:14:59", (2026, 2)),   # one second before MNF kickoff
        ("2026-09-22T00:15:01", (2026, 3)),   # MNF underway: nothing left to project in week 2
        ("2026-09-23T23:35:00", (2026, 3)),   # THE BUG: Wednesday cron before midnight UTC
        ("2026-09-24T02:22:00", (2026, 3)),   # Wednesday cron started late (Thursday UTC)
        ("2026-09-28T23:00:00", (2026, 3)),   # week-3 MNF still ahead
        ("2026-09-30T23:35:00", (2026, 4)),
    ],
)
def test_select_week_follows_kickoff_clock(now, expected):
    assert weekly.select_week(_schedule(), None, None, now=_utc(now)) == expected


def test_select_week_after_last_kickoff_returns_latest_week():
    assert weekly.select_week(_schedule(), None, None, now=_utc("2026-10-05T12:00:00")) == (2026, 4)


def test_select_week_override_is_validated_and_returned():
    assert weekly.select_week(_schedule(), 2026, 3, now=_utc("2026-09-20T12:15:00")) == (2026, 3)
    with pytest.raises(ValueError, match="no 2026 week 9"):
        weekly.select_week(_schedule(), 2026, 9)
    with pytest.raises(ValueError, match="together"):
        weekly.select_week(_schedule(), 2026, None)


def test_select_week_without_kickoff_times_falls_back_to_gameday():
    frame = _schedule()
    frame.loc[frame["week"].eq(3), "gametime"] = None
    # Week 2 is judged on its clock (finished); week 3 has no clock and its
    # last gameday (09-28) is still today or later -> week 3.
    assert weekly.select_week(frame, None, None, now=_utc("2026-09-23T23:35:00")) == (2026, 3)
    assert weekly.select_week(frame, None, None, now=_utc("2026-09-28T23:00:00")) == (2026, 3)
    assert weekly.select_week(frame, None, None, now=_utc("2026-09-29T00:00:00")) == (2026, 4)


def test_select_week_ignores_postseason_and_needs_aware_clock():
    frame = _schedule()
    frame.loc[len(frame)] = [2026, 19, "2026_19_X_Y", "2027-01-10", "16:30", "WC"]
    assert weekly.select_week(frame, None, None, now=_utc("2026-10-05T12:00:00")) == (2026, 4)
    with pytest.raises(ValueError, match="aware"):
        weekly.select_week(frame, None, None, now=datetime(2026, 9, 23, 23, 35))
