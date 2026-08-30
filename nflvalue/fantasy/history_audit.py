"""Strict, hash-recorded rebuild of the canonical historical player-week frame.

``data.fetch_historical`` is deliberately forgiving: it treats snaps, injuries
and expected opportunity as optional so a weekly production run can still
publish when one nflverse feed is late.  That tolerance is wrong for an
accuracy audit.  Every one of those "optional" feeds supplies a *named*
feature of the frozen champion (``pre_offense_pct_ewm4`` needs snap counts,
``injury_out``/``practice_dnp`` need the injury report, and
``pre_expected_points_ewm4`` -- one of the two published baselines -- needs
``load_ff_opportunity``).  A frame built without them is a different frame,
and scoring it would report a missing cache as a model result.

This module therefore rebuilds from source under an explicit contract:

* every required season must be present in every required feed,
* a missing or short feed is a hard failure, never a warning,
* the manifest records file digests, a writer-independent content digest,
  package versions, interpreter, platform and the git commit, so a scorecard
  can be tied to the exact inputs that produced it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from ..reproducibility import CANONICAL_CSV_VERSION, canonical_csv_sha256
from .config import ScoringRules
from .data import DATA_FILES, HistoricalData, fetch_historical
from .features import build_feature_frame, frame_quality_report, model_features

#: Feeds an accuracy audit may not run without, and the champion feature each
#: one is load-bearing for.  Keep the justification next to the requirement so
#: nobody quietly demotes a feed back to "optional" to make a run go green.
REQUIRED_FEEDS: dict[str, str] = {
    "stats": "fantasy_points label and every pre_* rolling history feature",
    "rosters": "roster-first row universe, position, team, age, draft_number",
    "schedules": "game context: spread, total, rest, roof, surface, referee",
    "snaps": "pre_offense_pct_ewm4 and the snaps_missing indicator",
    "injuries": "injury_out / injury_questionable / practice_dnp features",
    "expected_points": "pre_expected_points_ewm4, the published expected-points baseline",
}

#: Frame columns whose presence proves the corresponding feed actually landed
#: rather than being silently zero-filled by ``build_feature_frame``.
FEED_WITNESS: dict[str, str] = {
    "snaps": "snaps_missing",
    "expected_points": "expected_points_missing",
}

#: A feed is only "present" for a season if it carries at least this many rows.
#: One stray row satisfies a set-membership check while carrying no signal.
MIN_ROWS_PER_SEASON = 100

#: Row keys that uniquely identify a feature-frame row.
FRAME_KEYS = ["season", "week", "player_id"]

PACKAGES = (
    "numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "joblib",
    "nflreadpy", "polars",
)


class HistoryRebuildError(RuntimeError):
    """Raised when the rebuilt corpus does not satisfy the audit contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions(names: Iterable[str] = PACKAGES) -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "absent"
    return out


def environment_fingerprint() -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
        "canonical_csv_version": CANONICAL_CSV_VERSION,
    }


def git_commit(repo: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-c", "gc.auto=0", "rev-parse", "HEAD"],
            cwd=repo, capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _seasons_of(frame: pd.DataFrame) -> dict[int, int]:
    if "season" not in frame:
        return {}
    values = pd.to_numeric(frame["season"], errors="coerce").dropna().astype(int)
    return {int(season): int(count) for season, count in values.value_counts().items()}


def check_feeds(
    directory: str | Path, required_seasons: Sequence[int]
) -> tuple[dict[str, object], list[str]]:
    """Verify every required feed exists and covers every required season."""

    directory = Path(directory)
    required = sorted({int(season) for season in required_seasons})
    tables: dict[str, object] = {}
    failures: list[str] = []
    for feed, reason in REQUIRED_FEEDS.items():
        path = directory / DATA_FILES[feed]
        if not path.exists():
            failures.append(f"{feed}: {path.name} is absent (needed for {reason})")
            tables[feed] = {"present": False, "reason": reason}
            continue
        frame = pd.read_parquet(path)
        counts = _seasons_of(frame)
        missing = [season for season in required if season not in counts]
        short = [
            season for season in required
            if season in counts and counts[season] < MIN_ROWS_PER_SEASON
        ]
        if missing:
            failures.append(f"{feed}: no rows for season(s) {missing} (needed for {reason})")
        if short:
            failures.append(
                f"{feed}: fewer than {MIN_ROWS_PER_SEASON} rows for season(s) {short}"
            )
        tables[feed] = {
            "present": True,
            "reason": reason,
            "path": path.name,
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "sha256": _sha256_file(path),
            "rows_by_season": {str(k): counts[k] for k in sorted(counts)},
            "missing_seasons": missing,
            "short_seasons": short,
        }
    return tables, failures


def check_frame(frame: pd.DataFrame, required_seasons: Sequence[int]) -> list[str]:
    """Verify the built frame really carries every required season and feed."""

    required = sorted({int(season) for season in required_seasons})
    failures: list[str] = []
    counts = _seasons_of(frame)
    missing = [season for season in required if not counts.get(season)]
    if missing:
        failures.append(f"feature frame has no rows for season(s) {missing}")
    eligible = frame[frame["model_eligible"].fillna(False)]
    for season in required:
        rows = int(pd.to_numeric(eligible["season"], errors="coerce").eq(season).sum())
        if rows < MIN_ROWS_PER_SEASON:
            failures.append(f"feature frame has only {rows} eligible rows for season {season}")
    for feed, witness in FEED_WITNESS.items():
        if witness not in frame:
            failures.append(f"{feed}: witness column {witness} absent from the frame")
            continue
        rate = float(pd.to_numeric(frame[witness], errors="coerce").fillna(1.0).mean())
        if rate >= 1.0:
            failures.append(
                f"{feed}: {witness} is 1.0 for every row -- the feed did not reach the frame"
            )
    absent = sorted(set(model_features()) - set(frame.columns))
    if absent:
        failures.append(f"feature frame missing model features: {absent}")
    return failures


def rebuild(
    *,
    seasons: Sequence[int],
    data_dir: str | Path,
    frame_path: str | Path,
    manifest_path: str | Path,
    quality_path: str | Path | None = None,
    scoring: ScoringRules | None = None,
    fetch: bool = True,
    force: bool = False,
    repo: str | Path | None = None,
) -> dict[str, object]:
    """Fetch, validate and rebuild the canonical frame, or fail loudly."""

    required = sorted({int(season) for season in seasons})
    if not required:
        raise HistoryRebuildError("at least one required season must be given")
    data_dir = Path(data_dir)
    rules = scoring or ScoringRules()
    started = datetime.now(timezone.utc)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "kind": "canonical-history-rebuild",
        "started_at": started.isoformat(),
        "required_seasons": required,
        "scoring": rules.to_dict(),
        "environment": environment_fingerprint(),
        "git_commit": git_commit(Path(repo) if repo else Path.cwd()),
        "required_feeds": dict(REQUIRED_FEEDS),
        "min_rows_per_season": MIN_ROWS_PER_SEASON,
    }

    fetch_error: str | None = None
    if fetch:
        try:
            manifest["fetch"] = fetch_historical(required, data_dir, force=force)
        except Exception as exc:  # surfaced as a contract failure below
            fetch_error = f"{type(exc).__name__}: {exc}"
            manifest["fetch"] = {"failed": fetch_error}

    tables, failures = check_feeds(data_dir, required)
    manifest["feeds"] = tables
    if fetch_error:
        failures.insert(0, f"fetch failed: {fetch_error}")

    if failures:
        manifest["ok"] = False
        manifest["failures"] = failures
        _write_json(manifest_path, manifest)
        raise HistoryRebuildError(
            "canonical history rebuild failed the audit contract:\n  - "
            + "\n  - ".join(failures)
        )

    bundle = HistoricalData.load(data_dir)
    frame = build_feature_frame(bundle, rules)
    frame_failures = check_frame(frame, required)
    if frame_failures:
        manifest["ok"] = False
        manifest["failures"] = frame_failures
        _write_json(manifest_path, manifest)
        raise HistoryRebuildError(
            "rebuilt feature frame failed the audit contract:\n  - "
            + "\n  - ".join(frame_failures)
        )

    frame_path = Path(frame_path)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(frame_path, index=False)
    quality = frame_quality_report(frame)
    if quality_path:
        _write_json(quality_path, quality)

    eligible = frame[frame["model_eligible"].fillna(False)]
    manifest["frame"] = {
        "path": str(frame_path),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "feature_count": len(model_features()),
        "seasons": sorted(_seasons_of(frame)),
        "rows_by_season": {str(k): v for k, v in sorted(_seasons_of(frame).items())},
        "eligible_rows": int(len(eligible)),
        "eligible_rows_by_season": {
            str(k): v for k, v in sorted(_seasons_of(eligible).items())
        },
        "parquet_sha256": _sha256_file(frame_path),
        "content_sha256": canonical_csv_sha256(frame, row_keys=FRAME_KEYS),
        "feature_content_sha256": canonical_csv_sha256(
            frame[FRAME_KEYS + model_features()], row_keys=FRAME_KEYS
        ),
    }
    manifest["quality_summary"] = {
        "rows": quality["rows"],
        "eligible_rows": quality["eligible_rows"],
        "active_participant_rows": quality["active_participant_rows"],
        "positions": quality["positions"],
        "worst_missing_features": sorted(
            (
                (column, rate)
                for column, rate in quality["feature_missing_rate"].items()
                if rate > 0
            ),
            key=lambda item: -item[1],
        )[:10],
    }
    manifest["ok"] = True
    manifest["failures"] = []
    finished = datetime.now(timezone.utc)
    manifest["finished_at"] = finished.isoformat()
    manifest["wall_seconds"] = round((finished - started).total_seconds(), 3)
    _write_json(manifest_path, manifest)
    return manifest


def _write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _season_range(value: str) -> list[int]:
    if ":" in value:
        start, end = map(int, value.split(":", 1))
        if end < start:
            raise argparse.ArgumentTypeError("season range end must not precede start")
        return list(range(start, end + 1))
    return sorted({int(item) for item in value.split(",") if item.strip()})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=_season_range, required=True)
    parser.add_argument("--data-dir", default="historical/fantasy")
    parser.add_argument("--frame", default="historical/fantasy/feature_frame.parquet")
    parser.add_argument("--manifest", default="reports/history_rebuild_manifest.json")
    parser.add_argument("--quality", default="reports/fantasy_data_quality.json")
    parser.add_argument("--scoring", choices=["ppr", "half_ppr", "standard"], default="ppr")
    parser.add_argument("--no-fetch", action="store_true", help="validate and build from the cache on disk")
    parser.add_argument("--force", action="store_true", help="ignore any cached feed and refetch")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = rebuild(
            seasons=args.seasons,
            data_dir=args.data_dir,
            frame_path=args.frame,
            manifest_path=args.manifest,
            quality_path=args.quality,
            scoring=ScoringRules.preset(args.scoring),
            fetch=not args.no_fetch,
            force=args.force,
            repo=Path.cwd(),
        )
    except HistoryRebuildError as exc:
        print(str(exc), file=sys.stderr)
        print(f"manifest written to {args.manifest}", file=sys.stderr)
        return 2
    printable = {key: value for key, value in manifest.items() if key != "fetch"}
    print(json.dumps(printable, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
