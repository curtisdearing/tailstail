"""Pregame availability gate: the latest official status, read at serving time.

2026 Week 2 (Zay Flowers, RJ Harvey): the Thursday card was built before the
official report listed either player; nothing between that card and Sunday
kickoff re-read the report, and the lineup layer only knew the ESPN league's
own ``injury_status``, which does not block a DOUBTFUL player at all.  Both
were declared Doubtful/Questionable on Friday and inactive on Sunday; the set
lineup carried Flowers at 14.4 projected points and scored zero.

This module is the operational half of the fix.  It reads two official feeds
the pipeline already caches -- the nflverse injury report and the weekly
roster -- for ONE (season, week), and states per player whether the latest
official word is Out or Doubtful.  It is a serving-time gate on who may be
recommended, not a model input: no projection number is changed here, and the
same vocabulary the feature frame uses (``features.py``: ``injury_out`` is a
full match on "out", ``injury_doubtful`` a substring match on "doubt",
``status_inactive`` a roster status in :data:`INACTIVE_ROSTER_CODES`) is
applied so the gate and the simulation never disagree about a player.

Refreshing it is cheap: ``scripts/fantasy_weekly.py --no-fit`` re-reads the
current season's feeds, re-simulates from the saved model and regenerates the
private card, without a refit.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .features import INACTIVE_ROSTER_CODES

#: The availability the simulation itself assigns to a Doubtful report
#: (``simulation._availability`` caps it at 0.30).  Stated here so the card
#: can say what "Doubtful" is worth without computing anything.
DOUBTFUL_AVAILABILITY = 0.30

#: Report statuses that gate a player.  Anything else on the report
#: (Questionable, a practice-only listing) is carried for display and left to
#: the simulation's availability draw.
GATE_OUT = "out"
GATE_DOUBTFUL = "doubtful"

SOURCE = "nflverse injuries + weekly_rosters (official club reports)"

#: Marker `data.materialize_projection_week` puts on a roster row it copied
#: forward because the weekly feed has not reached the projection week yet.
#: Its docstring says the row "is visibly tagged so consumers can lower
#: confidence"; this gate is such a consumer, and it MUST not report a copied
#: row as though a club had filed it for the projection week.
CARRIED_COLUMN = "snapshot_carried_forward"


def classify_report(report_status: Any) -> str | None:
    """``out`` / ``doubtful`` / None from a report status, features.py semantics."""
    text = str(report_status or "").strip().lower()
    if not text or text == "nan":
        return None
    if text == "out":
        return GATE_OUT
    if "doubt" in text:
        return GATE_DOUBTFUL
    return None


def _week_rows(frame: pd.DataFrame | None, season: int, week: int) -> pd.DataFrame:
    if frame is None or frame.empty or not {"season", "week"}.issubset(frame.columns):
        return pd.DataFrame()
    mask = (
        pd.to_numeric(frame["season"], errors="coerce").eq(int(season))
        & pd.to_numeric(frame["week"], errors="coerce").eq(int(week))
    )
    return frame[mask]


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def injury_report_published(injuries: pd.DataFrame | None, season: int, week: int) -> bool:
    """Has a club actually filed an injury report for this (season, week) yet?

    Clubs file from Wednesday. Before that the feed simply has no rows for the
    week, and the gate's silence on a player means "nobody has said anything
    yet", not "he is healthy".
    """
    return not _week_rows(injuries, season, week).empty


def official_statuses(
    injuries: pd.DataFrame | None,
    rosters: pd.DataFrame | None,
    *,
    season: int,
    week: int,
) -> dict[str, dict[str, Any]]:
    """``{player_id: status}`` for every player the official feeds say something about.

    A player appears when the target week's injury report lists him, or the
    target week's roster row carries a non-active status.  Healthy, unlisted
    players are absent -- absence means "no official concern", never "cleared".

    Each status carries ``gate`` (``"out"``, ``"doubtful"`` or ``None``), the
    raw report/practice/roster strings it was read from, a one-line ``reason``
    the lineup layer can print verbatim, and the feed it came from.

    **Staleness is stated, not hidden.**  Before the weekly feed reaches the
    projection week, `materialize_projection_week` copies the previous week's
    roster rows forward and stamps them with the projection week, so a player
    who was inactive LAST week looks inactive THIS week.  Such a status is
    marked ``stale`` with the week it actually came from, and its reason says
    so.  It still gates -- a player who was inactive and has no newer word is
    not someone to start, and the freeze's rule is to fail closed when
    provenance is ambiguous -- but the card must never call last week's
    absence this week's ruling.  That conflation is the same mistake that
    produced the 2026 week 1-2 serving defect.
    """
    statuses: dict[str, dict[str, Any]] = {}

    injury_rows = _week_rows(injuries, season, week)
    if not injury_rows.empty:
        id_column = "gsis_id" if "gsis_id" in injury_rows else "player_id"
        # keep="last" matches features._merge_optional: the newest row for a
        # player-week is the latest report.
        injury_rows = injury_rows.drop_duplicates([id_column], keep="last")
        for row in injury_rows.to_dict("records"):
            player_id = _text(row.get(id_column))
            if not player_id.startswith("00-"):
                continue
            report = _text(row.get("report_status"))
            practice = _text(row.get("practice_status"))
            injury = _text(row.get("report_primary_injury")) or _text(row.get("practice_primary_injury"))
            gate = classify_report(report)
            statuses[player_id] = {
                "season": int(season), "week": int(week),
                "report_status": report or None,
                "practice_status": practice or None,
                "injury": injury or None,
                "roster_status": None,
                "gate": gate,
                "reason": (f"official injury report: {report}" + (f" ({injury})" if injury else "")
                           if report else None),
                "source": SOURCE,
            }

    roster_rows = _week_rows(rosters, season, week)
    # Which week the roster rows really describe. A copied row keeps every
    # other field of its source week, so the newest genuine week in the frame
    # is the week its status was actually true of.
    carried_source_week: int | None = None
    if not roster_rows.empty and CARRIED_COLUMN in roster_rows:
        genuine = rosters[
            pd.to_numeric(rosters["season"], errors="coerce").eq(int(season))
            & ~rosters[CARRIED_COLUMN].fillna(False).astype(bool)
        ]
        weeks = pd.to_numeric(genuine.get("week"), errors="coerce").dropna()
        weeks = weeks[weeks.lt(int(week))]
        if not weeks.empty:
            carried_source_week = int(weeks.max())
    if not roster_rows.empty and "status" in roster_rows and "gsis_id" in roster_rows:
        if "game_type" in roster_rows:
            roster_rows = roster_rows[roster_rows["game_type"].fillna("REG").eq("REG")]
        for row in roster_rows.to_dict("records"):
            player_id = _text(row.get("gsis_id"))
            status = _text(row.get("status")).upper()
            if not player_id.startswith("00-") or not status:
                continue
            stale = bool(row.get(CARRIED_COLUMN)) and carried_source_week is not None
            entry = statuses.get(player_id)
            if entry is not None:
                entry["roster_status"] = status
            if status not in INACTIVE_ROSTER_CODES:
                continue
            if entry is None:
                entry = statuses[player_id] = {
                    "season": int(season), "week": int(week),
                    "report_status": None, "practice_status": None, "injury": None,
                    "roster_status": status, "gate": None, "reason": None, "source": SOURCE,
                }
            # A roster placement (IR, PUP, suspension, inactive) is an Out,
            # whatever the report said earlier in the week.
            entry["gate"] = GATE_OUT
            if stale:
                entry["stale"] = True
                entry["as_of_week"] = carried_source_week
                entry["reason"] = (
                    f"roster status {status} in week {carried_source_week}; no official "
                    f"week {int(week)} roster has been published yet"
                )
            else:
                entry["reason"] = f"official roster status {status}"
    for entry in statuses.values():
        entry.setdefault("stale", False)
        entry.setdefault("as_of_week", int(week))
    return statuses


def stale(statuses: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Gated players whose only evidence is a week that has already been played."""
    return {pid: dict(status) for pid, status in gated(statuses).items()
            if status.get("stale")}


def gated(statuses: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Only the players the gate actually blocks."""
    return {pid: dict(status) for pid, status in statuses.items()
            if status.get("gate") in (GATE_OUT, GATE_DOUBTFUL)}
