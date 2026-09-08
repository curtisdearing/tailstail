"""The weekly fetch must reach the season that is about to be played.

Both scheduled production runs before 2026 Week 1 (GitHub Actions runs
33703355606 and 34047448573) died in `fetch_historical` with

    ValueError: Season must be between 2002 and 2025

raised by nflreadpy 0.1.5: its `get_current_season()` only advances to the new
year on the Thursday after Labor Day, so every run before 2026-09-10 asked the
library for 2026 weekly rosters and was refused -- although nflverse had
already published `roster_weekly_2026` and `injuries_2026`. The library's date
rule is not a data-availability check, and the Wednesday run of Week 1 is the
one that has to succeed.

These tests fix the contract: the library's own loaders stay the path for every
season they accept (the frozen behaviour, byte for byte); a season the library
has not "reached" is downloaded through the library's own downloader from the
same nflverse asset; and an in-season file that legitimately does not exist yet
(snap counts before Week 1) drops only that season of an optional table, never
the whole table and never the run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue.fantasy import data as data_mod


class _Downloader:
    def __init__(self, missing=()):
        self.calls: list[tuple[str, str]] = []
        self.missing = set(missing)

    def download(self, repository, path, **kwargs):
        self.calls.append((repository, path))
        if path in self.missing:
            raise ConnectionError(f"Failed to download {path}: 404")
        season = int(path.rsplit("_", 1)[-1])
        return pd.DataFrame({"season": [season], "week": [1], "gsis_id": ["p"],
                             "position": ["QB"], "team": ["NE"], "espn_id": [1],
                             "player_id": ["p"], "pfr_player_id": ["p"]})


class _FakeNfl:
    """Enough of nflreadpy to exercise the season rule: loaders that refuse a
    season past `current_season`, and the downloader they would have used."""

    __version__ = "0.1.5-fake"

    def __init__(self, current_season: int, missing=()):
        self.current_season = current_season
        self.downloader = self
        self._downloader = _Downloader(missing)
        self.loader_calls: list[tuple[str, list[int]]] = []
        self.utils_date = self

    def get_current_season(self, roster: bool = False) -> int:
        return self.current_season

    def get_downloader(self):
        return self._downloader

    def _frame(self, seasons):
        return pd.DataFrame({"season": list(seasons), "week": [1] * len(seasons),
                             "gsis_id": ["p"] * len(seasons), "position": ["QB"] * len(seasons),
                             "team": ["NE"] * len(seasons), "espn_id": [1] * len(seasons),
                             "player_id": ["p"] * len(seasons),
                             "pfr_player_id": ["p"] * len(seasons)})

    def _seasonal(self, name, seasons):
        seasons = [int(s) for s in ([seasons] if isinstance(seasons, int) else seasons)]
        self.loader_calls.append((name, seasons))
        if max(seasons) > self.current_season:
            raise ValueError(f"Season must be between 2002 and {self.current_season}")
        return self._frame(seasons)

    def load_rosters_weekly(self, seasons):
        return self._seasonal("rosters", seasons)

    def load_snap_counts(self, seasons):
        return self._seasonal("snaps", seasons)

    def load_injuries(self, seasons):
        return self._seasonal("injuries", seasons)

    def load_player_stats(self, season, summary_level="week"):
        return pd.DataFrame({"season": [season], "week": [1], "player_id": ["p"],
                             "position": ["QB"], "team": ["NE"]})

    def load_schedules(self, seasons):
        return pd.DataFrame({"season": list(seasons), "week": [1] * len(seasons),
                             "game_id": [f"{s}_01_NE_SEA" for s in seasons],
                             "home_team": ["SEA"] * len(seasons), "away_team": ["NE"] * len(seasons)})

    def load_ff_opportunity(self, seasons, stat_type="weekly"):
        return self._seasonal("expected_points", seasons)


def fetch(tmp_path, nfl, seasons=(2024, 2025, 2026)):
    return data_mod.fetch_historical(seasons, tmp_path, force=True, nfl=nfl)


def test_the_library_path_is_used_unchanged_when_it_accepts_every_season(tmp_path):
    nfl = _FakeNfl(current_season=2026)
    fetch(tmp_path, nfl)
    assert ("rosters", [2024, 2025, 2026]) in nfl.loader_calls
    assert nfl._downloader.calls == []
    rosters = pd.read_parquet(tmp_path / "weekly_rosters.parquet")
    assert sorted(rosters["season"].unique()) == [2024, 2025, 2026]


def test_a_season_the_library_rule_has_not_reached_is_downloaded_directly(tmp_path):
    nfl = _FakeNfl(current_season=2025)      # the state of nflreadpy 0.1.5 on 2026-09-09
    manifest = fetch(tmp_path, nfl)
    rosters = pd.read_parquet(tmp_path / "weekly_rosters.parquet")
    assert sorted(rosters["season"].unique()) == [2024, 2025, 2026]
    assert ("nflverse-data", "weekly_rosters/roster_weekly_2026") in nfl._downloader.calls
    assert ("nflverse-data", "injuries/injuries_2026") in nfl._downloader.calls
    # expected points come from the ffopportunity release, not nflverse-data
    assert ("ffopportunity", "latest-data/ep_weekly_2026") in nfl._downloader.calls
    # the seasons the library accepts still come from the library, in one call
    assert ("rosters", [2024, 2025]) in nfl.loader_calls
    note = manifest["tables"]["rosters"]["seasons_note"]
    assert "2026" in note and "directly" in note
    assert manifest["nflreadpy_current_season"] == 2025


def test_a_missing_in_season_file_drops_only_that_season_of_an_optional_table(tmp_path):
    nfl = _FakeNfl(current_season=2025, missing={"snap_counts/snap_counts_2026"})
    manifest = fetch(tmp_path, nfl)
    snaps = pd.read_parquet(tmp_path / "snap_counts.parquet")
    assert sorted(snaps["season"].unique()) == [2024, 2025]
    assert manifest["tables"]["snaps"]["seasons_loaded"] == [2024, 2025]
    assert "2026" in manifest["tables"]["snaps"]["seasons_note"]
    assert manifest["tables"]["snaps"].get("available", True) is True


def test_a_missing_current_season_of_a_required_table_still_fails_the_run(tmp_path):
    nfl = _FakeNfl(current_season=2025, missing={"weekly_rosters/roster_weekly_2026"})
    with pytest.raises(Exception, match="roster_weekly_2026"):
        fetch(tmp_path, nfl)


def test_the_manifest_records_which_seasons_each_table_carries(tmp_path):
    nfl = _FakeNfl(current_season=2025)
    fetch(tmp_path, nfl)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["tables"]["injuries"]["seasons_loaded"] == [2024, 2025, 2026]
    assert manifest["tables"]["rosters"]["seasons_loaded"] == [2024, 2025, 2026]
