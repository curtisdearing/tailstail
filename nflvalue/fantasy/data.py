"""Official-data ingestion and roster-first fantasy dataset contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

DATA_FILES = {
    "stats": "player_stats.parquet",
    "rosters": "weekly_rosters.parquet",
    "schedules": "schedules.parquet",
    "snaps": "snap_counts.parquet",
    "injuries": "injuries.parquet",
    "expected_points": "expected_points.parquet",
}

REQUIRED_COLUMNS = {
    "stats": {"season", "week", "player_id", "position", "team"},
    "rosters": {"season", "week", "gsis_id", "position", "team"},
    "schedules": {"season", "week", "game_id", "home_team", "away_team"},
}


def _pandas(frame) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    raise TypeError(f"unsupported dataframe type: {type(frame)!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class HistoricalData:
    stats: pd.DataFrame
    rosters: pd.DataFrame
    schedules: pd.DataFrame
    snaps: pd.DataFrame | None = None
    injuries: pd.DataFrame | None = None
    expected_points: pd.DataFrame | None = None

    def validate(self) -> dict[str, object]:
        tables = {
            "stats": self.stats,
            "rosters": self.rosters,
            "schedules": self.schedules,
            "snaps": self.snaps,
            "injuries": self.injuries,
            "expected_points": self.expected_points,
        }
        errors: list[str] = []
        report: dict[str, object] = {"tables": {}}
        for name, frame in tables.items():
            if frame is None:
                report["tables"][name] = {"present": False, "rows": 0}
                continue
            missing = sorted(REQUIRED_COLUMNS.get(name, set()) - set(frame.columns))
            if missing:
                errors.append(f"{name} missing {missing}")
            season_values = frame["season"] if "season" in frame else pd.Series(dtype=float)
            seasons = sorted(pd.to_numeric(season_values, errors="coerce").dropna().astype(int).unique())
            report["tables"][name] = {
                "present": True,
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "seasons": seasons,
            }
        if self.stats.duplicated(["season", "week", "player_id"]).any():
            errors.append("stats has duplicate player-week keys")
        roster_keys = self.rosters[self.rosters.get("gsis_id", "").fillna("").astype(str).ne("")]
        if roster_keys.duplicated(["season", "week", "gsis_id"]).any():
            report["roster_duplicate_keys"] = int(
                roster_keys.duplicated(["season", "week", "gsis_id"], keep=False).sum()
            )
        report["valid"] = not errors
        report["errors"] = errors
        if errors:
            raise ValueError("; ".join(errors))
        return report

    @classmethod
    def load(cls, directory: str | Path) -> "HistoricalData":
        directory = Path(directory)
        loaded: dict[str, pd.DataFrame | None] = {}
        for name, filename in DATA_FILES.items():
            path = directory / filename
            loaded[name] = pd.read_parquet(path) if path.exists() else None
        missing = [name for name in ("stats", "rosters", "schedules") if loaded[name] is None]
        if missing:
            raise FileNotFoundError(
                f"missing required fantasy cache tables {missing} in {directory}; run the fetch command"
            )
        bundle = cls(**loaded)  # type: ignore[arg-type]
        bundle.validate()
        return bundle


def fetch_historical(
    seasons: Iterable[int], directory: str | Path, *, force: bool = False
) -> dict[str, object]:
    """Download and cache the official tables needed by every model family.

    nflreadpy returns Polars frames.  Cache boundaries are pandas Parquet so
    the modeling code stays independent of the downloader implementation.
    """

    seasons = sorted({int(season) for season in seasons})
    if not seasons:
        raise ValueError("at least one season is required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        import nflreadpy as nfl
    except ImportError as exc:
        raise RuntimeError("nflreadpy is required for fetch; install requirements.txt") from exc

    def available_player_stats():
        frames = []
        for season in seasons:
            try:
                frames.append(_pandas(nfl.load_player_stats(season, summary_level="week")))
            except Exception:
                # Before week 1 the current-season stats file can legitimately
                # be absent. Historical rows still provide every prior feature.
                if season != max(seasons):
                    raise
        if not frames:
            raise RuntimeError("no player-stat season was available")
        return pd.concat(frames, ignore_index=True, sort=False)

    loaders = {
        "stats": available_player_stats,
        "rosters": lambda: nfl.load_rosters_weekly(seasons),
        "schedules": lambda: nfl.load_schedules(seasons),
        "snaps": lambda: nfl.load_snap_counts(seasons),
        "injuries": lambda: nfl.load_injuries(seasons),
        "expected_points": lambda: nfl.load_ff_opportunity(seasons, stat_type="weekly"),
    }
    manifest: dict[str, object] = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "seasons": seasons,
        "nflreadpy_version": getattr(nfl, "__version__", "unknown"),
        "tables": {},
    }
    optional = {"snaps", "injuries", "expected_points"}
    for name, loader in loaders.items():
        path = directory / DATA_FILES[name]
        try:
            use_cache = False
            if path.exists() and not force:
                cached_frame = pd.read_parquet(path)
                cached_seasons = set(
                    pd.to_numeric(cached_frame.get("season"), errors="coerce").dropna().astype(int)
                ) if "season" in cached_frame else set()
                now = datetime.now(timezone.utc)
                current_nfl_season = now.year if now.month >= 3 else now.year - 1
                # Immutable past seasons can be reused; the current season must
                # refresh because rosters, injuries and stats change each week.
                use_cache = set(seasons).issubset(cached_seasons) and max(seasons) < current_nfl_season
            if use_cache:
                frame = cached_frame
                cached = True
            else:
                frame = _pandas(loader())
                frame.to_parquet(path, index=False)
                cached = False
        except Exception as exc:
            if name not in optional:
                raise
            manifest["tables"][name] = {
                "path": path.name,
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue
        manifest["tables"][name] = {
            "path": path.name,
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "sha256": _sha256(path),
            "reused_cache": cached,
        }
    bundle = HistoricalData.load(directory)
    manifest["quality"] = bundle.validate()
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def materialize_projection_week(
    data: HistoricalData, season: int, week: int
) -> HistoricalData:
    """Create a target-week roster snapshot when the weekly feed lags schedule.

    Only roster identity/static context is carried. No stats, injury report, or
    outcome is copied. The resulting row is visibly tagged so consumers can
    lower confidence until the official target-week roster appears.
    """

    season = int(season)
    week = int(week)
    rosters = data.rosters.copy()
    existing = rosters[
        pd.to_numeric(rosters["season"], errors="coerce").eq(season)
        & pd.to_numeric(rosters["week"], errors="coerce").eq(week)
    ]
    if not existing.empty:
        if "snapshot_carried_forward" not in rosters:
            rosters["snapshot_carried_forward"] = False
        return HistoricalData(
            stats=data.stats, rosters=rosters, schedules=data.schedules,
            snaps=data.snaps, injuries=data.injuries, expected_points=data.expected_points,
        )
    prior = rosters[
        pd.to_numeric(rosters["season"], errors="coerce").eq(season)
        & pd.to_numeric(rosters["week"], errors="coerce").lt(week)
    ]
    if prior.empty:
        raise ValueError(f"no prior {season} roster snapshot can seed week {week}")
    latest = int(pd.to_numeric(prior["week"], errors="coerce").max())
    carried = prior[pd.to_numeric(prior["week"], errors="coerce").eq(latest)].copy()
    carried["week"] = week
    carried["snapshot_carried_forward"] = True
    rosters["snapshot_carried_forward"] = rosters.get("snapshot_carried_forward", False)
    rosters = pd.concat([rosters, carried], ignore_index=True)
    return HistoricalData(
        stats=data.stats, rosters=rosters, schedules=data.schedules,
        snaps=data.snaps, injuries=data.injuries, expected_points=data.expected_points,
    )
