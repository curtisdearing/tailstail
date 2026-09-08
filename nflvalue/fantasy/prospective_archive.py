"""Immutable, timestamped archive of each weekly prediction payload.

Why this exists
---------------
The 2026 freeze (docs/PROTOCOL_FREEZE_2026.md §2) makes prospective 2026
predictions the final judge and says a prediction counts only if its decision
snapshot predates kickoff. Before this module the weekly run wrote ONE file,
``data/player_projection_snapshot.json``, overwrote it on the next run, and
published it to a Pages URL. A URL cannot show when a forecast existed, and an
overwritten file cannot show what an earlier run said. Neither can say which
games a forecast preceded.

What one entry records
----------------------
* identity: season, week, model revision, the players and their games;
* the scoring rules the scored numbers are under, and the per-player scored
  summary (mean / sd / p10 / p50 / p90 / availability);
* four clocks, kept apart: when the forecast was generated, the as-of time of
  its source data, when it was archived, and -- when the run was scheduled --
  the schedule it was meant to fire on (unknown stays ``None``);
* the decision deadline of every game in the slate (its kickoff) and whether
  BOTH forecast clocks strictly precede it, so a run that lands after one
  kickoff is labelled a partial slate rather than the week getting one label;
* the SHA-256 of the scoring-independent projection snapshot document, which
  is archived beside the entry as its own immutable file (it is the artifact
  published on the Pages site, kept verbatim), and a SHA-256 of the whole
  entry so tampering is detectable.

Immutability
------------
An entry is named by its forecast time and is never overwritten. An index of
every entry is append-only: a write that would drop an indexed entry is
refused. The ``latest`` pointer is a separate file and is the only thing that
moves. Nothing here reconstructs a missing forecast: a run after a deadline
records the cohort as missed.

Privacy
-------
The entry is Tailstail's own forecast, already published in full on the Pages
site. It carries no ESPN projection, no league, no roster. It is stored under
``data/prospective_archive`` -- a path shared with nothing on the private side
-- and travels between runs in the checksummed PUBLIC state release.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .config import ScoringRules

ARCHIVE_SCHEMA_VERSION = 1
ARCHIVE_KIND = "prospective-projection-archive/1"
INDEX_KIND = "prospective-projection-index/1"
LATEST_KIND = "prospective-projection-latest/1"
DEFAULT_DIRECTORY = "data/prospective_archive"
INDEX_FILE = "index.json"
LATEST_FILE = "latest.json"

#: Environment variables the workflow passes through (never interpolated into
#: a shell script). Absent means unknown, and unknown is recorded as ``None``.
RUN_CONTEXT_ENV = {
    "event": "TAILSTAIL_RUN_EVENT",
    "schedule": "TAILSTAIL_RUN_SCHEDULE",
    "run_id": "TAILSTAIL_RUN_ID",
    "run_attempt": "TAILSTAIL_RUN_ATTEMPT",
}

LABEL_FULL = "prospective_full_slate"
LABEL_PARTIAL = "prospective_partial_slate"
LABEL_NONE = "not_prospective"

SCORED_FIELDS = ("mean", "sd", "p10", "p50", "p90", "availability_probability")


class ArchiveIntegrityError(RuntimeError):
    """A stored entry or the index no longer says what it said when written."""


# --------------------------------------------------------------------------- #
# Clocks
# --------------------------------------------------------------------------- #

def parse_instant(value: Any, *, what: str) -> datetime:
    """An aware datetime, or a refusal. A naive stamp has no provenance and is
    never assumed to be UTC: that is how a post-deadline forecast would pass."""
    if isinstance(value, datetime):
        stamp = value
    else:
        try:
            stamp = datetime.fromisoformat(str(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{what} is not an ISO-8601 timestamp: {value!r}") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError(f"{what} carries no timezone; refusing to assume one: {value!r}")
    return stamp.astimezone(timezone.utc)


def _iso(stamp: datetime) -> str:
    return stamp.astimezone(timezone.utc).isoformat()


def run_context_from_env(environ: Mapping[str, str]) -> dict[str, str | None]:
    """The run's provenance, from the environment only. Missing is ``None``."""
    return {
        key: (environ.get(name) or None) for key, name in RUN_CONTEXT_ENV.items()
    }


# --------------------------------------------------------------------------- #
# Eligibility
# --------------------------------------------------------------------------- #

def game_eligibility(
    generated_at: datetime, information_as_of: datetime, kickoffs_utc: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    """Per game: its deadline (kickoff) and whether both forecast clocks
    strictly precede it. The reason names the clock that failed."""
    deadlines: dict[str, dict[str, Any]] = {}
    for game_id in sorted(kickoffs_utc):
        kickoff = parse_instant(kickoffs_utc[game_id], what=f"kickoff of {game_id}")
        if generated_at >= kickoff:
            prospective, reason = False, "forecast_generated_at_or_after_kickoff"
        elif information_as_of >= kickoff:
            prospective, reason = False, "information_as_of_at_or_after_kickoff"
        else:
            prospective, reason = True, "both_clocks_before_kickoff"
        deadlines[str(game_id)] = {
            "kickoff_utc": _iso(kickoff),
            "decision_deadline_utc": _iso(kickoff),
            "prospective": prospective,
            "reason": reason,
        }
    return deadlines


def _label(games_total: int, games_prospective: int) -> str:
    if games_total and games_prospective == games_total:
        return LABEL_FULL
    if games_prospective:
        return LABEL_PARTIAL
    return LABEL_NONE


# --------------------------------------------------------------------------- #
# Building an entry
# --------------------------------------------------------------------------- #

def _canonical_sha256(document: Mapping[str, Any]) -> str:
    body = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def build_entry(
    projection_snapshot: Mapping[str, Any],
    summaries: pd.DataFrame,
    *,
    scoring: ScoringRules,
    scoring_preset: str | None = None,
    kickoffs_utc: Mapping[str, str],
    run_context: Mapping[str, str | None],
    archived_at: str | datetime | None = None,
) -> dict[str, Any]:
    """One archive entry from this run's snapshot and scored summaries."""
    generated = parse_instant(projection_snapshot["generated_at"], what="generated_at")
    as_of = parse_instant(projection_snapshot["information_as_of"], what="information_as_of")
    archived = parse_instant(archived_at or datetime.now(timezone.utc), what="archived_at")
    deadlines = game_eligibility(generated, as_of, kickoffs_utc)

    scored = summaries.copy()
    scored["player_id"] = scored["player_id"].astype(str)
    scored = scored.drop_duplicates("player_id").set_index("player_id")

    players: list[dict[str, Any]] = []
    counts = {"players_prospective": 0, "players_missed": 0, "players_no_kickoff": 0}
    for record in projection_snapshot["players"]:
        player_id = str(record["player_id"])
        game_id = str(record.get("game_id", "unknown"))
        deadline = deadlines.get(game_id)
        if deadline is None:
            prospective, reason, kickoff = False, "kickoff_unknown", None
            counts["players_no_kickoff"] += 1
        else:
            prospective, reason, kickoff = (
                deadline["prospective"], deadline["reason"], deadline["kickoff_utc"])
            counts["players_prospective" if prospective else "players_missed"] += 1
        row: dict[str, Any] = {
            "player_id": player_id,
            "player_name": str(record.get("player_name", player_id)),
            "position": str(record.get("position", "unknown")),
            "team": str(record.get("team", "unknown")),
            "game_id": game_id,
            "kickoff_utc": kickoff,
            "prospective": prospective,
            "reason": reason,
        }
        if player_id in scored.index:
            for field in SCORED_FIELDS:
                if field in scored.columns:
                    value = scored.loc[player_id, field]
                    row[field] = None if pd.isna(value) else float(value)
        players.append(row)

    games_total = len(deadlines)
    games_prospective = sum(1 for d in deadlines.values() if d["prospective"])
    entry: dict[str, Any] = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "kind": ARCHIVE_KIND,
        "season": int(projection_snapshot["season"]),
        "week": int(projection_snapshot["week"]),
        "model_version": str(projection_snapshot.get("model_version", "unknown")),
        "scoring": {"preset": scoring_preset, "rules": scoring.to_dict()},
        "clocks": {
            "forecast_generated_at": _iso(generated),
            "information_as_of": _iso(as_of),
            "archived_at": _iso(archived),
            "scheduled_for": run_context.get("schedule"),
        },
        "run_context": dict(run_context),
        "deadlines": deadlines,
        "eligibility": {
            "label": _label(games_total, games_prospective),
            "rule": ("a game is prospective only when forecast_generated_at AND "
                     "information_as_of are strictly before its kickoff; an unknown "
                     "kickoff is never a met deadline"),
            "games_total": games_total,
            "games_prospective": games_prospective,
            "games_missed": games_total - games_prospective,
            **counts,
        },
        "projection_snapshot": {
            "players_canonical_csv_sha256": projection_snapshot["players_canonical_csv_sha256"],
            "document_sha256": _canonical_sha256(projection_snapshot),
            "document_path": _document_name(projection_snapshot, generated),
        },
        "players": players,
    }
    entry["payload_sha256"] = _canonical_sha256(entry)
    return entry


def _stamp(generated_at_iso: str) -> str:
    return (generated_at_iso.replace("-", "").replace(":", "").split(".")[0]
            .replace("+0000", ""))


def _document_name(projection_snapshot: Mapping[str, Any], generated: datetime) -> str:
    """The archived snapshot document sits beside its entry, under the season,
    with a prefix the entry glob never matches."""
    return (f"{int(projection_snapshot['season'])}/snapshot_{int(projection_snapshot['season'])}"
            f"_wk{int(projection_snapshot['week']):02d}_{_stamp(_iso(generated))}.json")


# --------------------------------------------------------------------------- #
# Writing, indexing, loading
# --------------------------------------------------------------------------- #

def entry_path(directory: str | Path, entry: Mapping[str, Any]) -> Path:
    stamp = _stamp(entry["clocks"]["forecast_generated_at"])
    name = f"projection_{entry['season']}_wk{int(entry['week']):02d}_{stamp}.json"
    return Path(directory) / str(entry["season"]) / name


def load_document(directory: str | Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    """The archived snapshot document an entry names, verified against the
    hash the entry recorded for it."""
    path = Path(directory) / entry["projection_snapshot"]["document_path"]
    document = json.loads(path.read_text())
    if _canonical_sha256(document) != entry["projection_snapshot"]["document_sha256"]:
        raise ArchiveIntegrityError(
            f"{path} does not match the document_sha256 its entry recorded; "
            "archived snapshot documents are immutable")
    return document


def _verify(entry: Mapping[str, Any], *, where: str) -> None:
    body = {k: v for k, v in entry.items() if k not in ("payload_sha256", "archived_path")}
    if _canonical_sha256(body) != entry.get("payload_sha256"):
        raise ArchiveIntegrityError(
            f"{where} does not match its stored payload_sha256; archive entries are immutable")


def load_entry(path: str | Path) -> dict[str, Any]:
    entry = json.loads(Path(path).read_text())
    if entry.get("kind") != ARCHIVE_KIND:
        raise ArchiveIntegrityError(f"{path} is not a prospective archive entry")
    _verify(entry, where=str(path))
    return entry


def load_index(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / INDEX_FILE
    if not path.exists():
        return {"schema_version": ARCHIVE_SCHEMA_VERSION, "kind": INDEX_KIND, "entries": []}
    index = json.loads(path.read_text())
    if index.get("kind") != INDEX_KIND or not isinstance(index.get("entries"), list):
        raise ArchiveIntegrityError(f"{path} is not a prospective archive index")
    return index


def _index_row(entry: Mapping[str, Any], relative: str) -> dict[str, Any]:
    eligibility = entry["eligibility"]
    return {
        "path": relative,
        "season": entry["season"],
        "week": entry["week"],
        "forecast_generated_at": entry["clocks"]["forecast_generated_at"],
        "archived_at": entry["clocks"]["archived_at"],
        "payload_sha256": entry["payload_sha256"],
        "label": eligibility["label"],
        "games_total": eligibility["games_total"],
        "games_prospective": eligibility["games_prospective"],
        "games_missed": eligibility["games_missed"],
    }


def write_entry(
    entry: Mapping[str, Any],
    directory: str | Path = DEFAULT_DIRECTORY,
    *,
    projection_snapshot: Mapping[str, Any] | None = None,
) -> Path:
    """Persist one entry, append it to the index, and move the latest pointer.

    Refuses to overwrite an entry, refuses to shorten the index, and verifies
    every entry the index already names before adding to it. When the
    snapshot document is supplied it is archived verbatim beside the entry,
    at the path and hash the entry already names; a document that does not
    match is refused.
    """
    directory = Path(directory)
    _verify(entry, where="entry to be written")
    path = entry_path(directory, entry)
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable archive entry {path}; a new forecast "
            "must produce a new timestamped entry")
    relative = path.relative_to(directory).as_posix()
    document_path = directory / entry["projection_snapshot"]["document_path"]
    if projection_snapshot is not None:
        if _canonical_sha256(projection_snapshot) != entry["projection_snapshot"]["document_sha256"]:
            raise ArchiveIntegrityError(
                "the snapshot document offered for archiving is not the one this entry hashed")
        if document_path.exists():
            raise FileExistsError(
                f"refusing to overwrite immutable archived snapshot {document_path}")

    index = load_index(directory)
    known = [row["path"] for row in index["entries"]]
    for row in index["entries"]:
        stored = directory / row["path"]
        if not stored.exists():
            raise ArchiveIntegrityError(
                f"index names {row['path']} but it is missing; the archive is append-only")
        if load_entry(stored)["payload_sha256"] != row["payload_sha256"]:
            raise ArchiveIntegrityError(
                f"{row['path']} no longer matches the index; the archive is append-only")
    on_disk = sorted(
        p.relative_to(directory).as_posix()
        for p in directory.glob("*/projection_*.json")
    )
    if any(existing not in known for existing in on_disk):
        raise ArchiveIntegrityError(
            "the index has lost entries that are still on disk; the archive is append-only "
            "and an index that drops history is refused")

    path.parent.mkdir(parents=True, exist_ok=True)
    if projection_snapshot is not None:
        document_path.write_text(json.dumps(projection_snapshot, indent=2, sort_keys=True) + "\n")
    stored_entry = {**entry, "archived_path": relative}
    path.write_text(json.dumps(stored_entry, indent=2, sort_keys=True) + "\n")

    index["entries"] = [*index["entries"], _index_row(entry, relative)]
    (directory / INDEX_FILE).write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    latest = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "kind": LATEST_KIND,
        **_index_row(entry, relative),
    }
    (directory / LATEST_FILE).write_text(json.dumps(latest, indent=2, sort_keys=True) + "\n")
    return path


def public_summary(entry: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    """What the public payload says about this run's capture: labels, counts,
    deadlines and the hash -- not the players (they are already the payload)."""
    return {
        "label": entry["eligibility"]["label"],
        "rule": entry["eligibility"]["rule"],
        "games_total": entry["eligibility"]["games_total"],
        "games_prospective": entry["eligibility"]["games_prospective"],
        "games_missed": entry["eligibility"]["games_missed"],
        "players_prospective": entry["eligibility"]["players_prospective"],
        "players_missed": entry["eligibility"]["players_missed"],
        "players_no_kickoff": entry["eligibility"]["players_no_kickoff"],
        "clocks": dict(entry["clocks"]),
        "deadlines": {
            game: {"kickoff_utc": d["kickoff_utc"], "prospective": d["prospective"]}
            for game, d in entry["deadlines"].items()
        },
        "archived_path": Path(path).as_posix(),
        "payload_sha256": entry["payload_sha256"],
    }
