"""What may be published, decided by an allow-list rather than a blocklist.

Two kinds of thing come out of a weekly run and only one of them may leave
this machine.

*Public*: Tailstail's own projections and the aggregate model audit.  These are
the project's own output, they name no league and no person, and publishing
them is the point of the site.

*Private*: the league snapshot and everything derived from it — rosters, team
names, manager identity, the league id, and the personalised decision card.
The league is private, and an ESPN ``members[].id`` **is** the SWID cookie.
Also private: the raw per-player rows of the ESPN external-challenger
comparison, because the recorded terms under which they were fetched grant no
redistribution right.  The season's aggregate grading (how close each side was,
week by week) carries no ESPN projection and is published in their place.

Why an allow-list
-----------------
The leak this module exists to close was not a mistake anybody made in a hurry.
``fantasy.html`` embedded the eight-section personal contract, the workflow
copied that file to ``_site/index.html``, and ``data/fantasy_latest.json`` — the
whole payload, ``my_team`` included — was copied next to it.  Every step was
reasonable on its own and the result published a private league's rosters every
week.  A blocklist would have needed somebody to think of ``my_team`` in
advance; an allow-list publishes a named set and nothing else, so the next
section somebody adds is private until it is deliberately made public.

:func:`assert_public_safe` is the second half: the allow-list decides what is
copied, and the assertion refuses to let anything else through even if the
copying is wrong.  Both run in the weekly pipeline, not only in tests.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

#: Top-level keys of the weekly payload that may be published.  Anything not
#: named here stays on disk.
PUBLIC_PAYLOAD_KEYS = (
    "generated_at", "season", "week", "data_quality", "model_card", "simulation",
    "projection_snapshot", "players",
)

#: ESPN comparison fields that carry no per-player ESPN projection.
PUBLIC_COMPARISON_KEYS = (
    "status", "error", "season", "current_week", "disclaimer", "prospective_rule",
    "season_series", "latest_graded_week",
)
PUBLIC_PROVENANCE_KEYS = ("retrieved_at", "players_sha256", "redistribution_rights", "coverage")
PUBLIC_IDENTITY_KEYS = (
    "espn_players", "matched", "coverage_pct",
    "unmatched_no_crosswalk_count", "unmatched_model_not_projected_count",
)

#: Key names that only ever appear on private objects.  Finding one inside
#: something about to be published is a defect, never a false positive: no
#: public artifact has a roster, a member or an ESPN player id in it.
PRIVATE_KEY_NAMES = frozenset({
    "my_team", "rosters", "roster", "members", "member", "espn_player_id",
    "current_week_rows", "espn_pts", "unresolved_identities", "lineup_changes",
    "shadow_seats", "waiver_plan", "start_sit", "optimal_lineup", "kicker_shadow",
    "dst_shadow", "team_name", "team_abbrev", "league_name", "swid", "espn_s2",
})

#: Schema strings that identify a private contract wherever they appear.
PRIVATE_SCHEMA_MARKERS = ("decision-card/", "my_team/", "espn-league/")

WITHHELD_NOTE = (
    "This file carries Tailstail's own projections and the aggregate model grading only. "
    "The league this project is pointed at is private, so its rosters, team names, league id "
    "and the personalised decision card are written to a gitignored local path and are never "
    "published. The ESPN external-challenger comparison is published as week-by-week aggregate "
    "grading; its per-player rows are not redistributed."
)


class PrivateDataLeak(AssertionError):
    """Something private reached, or was about to reach, a public artifact."""


def public_espn_comparison(comparison: Mapping[str, Any] | None) -> dict | None:
    """Aggregate grading, with every per-player ESPN row dropped."""
    if not comparison:
        return None
    out: dict[str, Any] = {key: comparison.get(key) for key in PUBLIC_COMPARISON_KEYS}
    provenance = comparison.get("espn_provenance") or {}
    out["espn_provenance"] = ({key: provenance.get(key) for key in PUBLIC_PROVENANCE_KEYS}
                              if provenance else None)
    identity = comparison.get("identity") or {}
    out["identity"] = ({key: identity.get(key) for key in PUBLIC_IDENTITY_KEYS}
                       if identity else None)
    out["rows_published"] = False
    out["rows_withheld_reason"] = (
        "the recorded terms of the ESPN pull grant no redistribution right, so the per-player "
        "rows stay local and only the week-by-week grading is published")
    return out


def public_weekly_payload(payload: Mapping[str, Any]) -> dict:
    """The published JSON, assembled from the allow-list and nothing else."""
    public = {key: payload.get(key) for key in PUBLIC_PAYLOAD_KEYS if key in payload}
    public["espn_comparison"] = public_espn_comparison(payload.get("espn_comparison"))
    public["visibility"] = "public"
    public["withheld"] = WITHHELD_NOTE
    assert_public_safe(public, what="public weekly payload")
    return public


def _private_strings(league_id: Any, names: Sequence[str]) -> list[str]:
    needles = [str(name).strip() for name in names if str(name or "").strip()]
    if league_id not in (None, ""):
        needles.append(str(league_id))
    return needles


def assert_public_safe(obj: Any, *, what: str = "artifact", league_id: Any = None,
                       names: Sequence[str] = ()) -> None:
    """Refuse an object that carries private structure or private strings.

    Raises on the first finding with the path to it, because a leak report that
    says only "something leaked" is one somebody has to reproduce before they
    can fix it.
    """
    needles = _private_strings(league_id, names)
    for path, key, value in _walk(obj):
        if key is not None and str(key).lower() in PRIVATE_KEY_NAMES:
            raise PrivateDataLeak(f"{what}: private key {key!r} at {path}")
        if isinstance(value, str):
            for marker in PRIVATE_SCHEMA_MARKERS:
                if marker in value:
                    raise PrivateDataLeak(
                        f"{what}: private contract marker {marker!r} at {path}")
            for needle in needles:
                if needle in value:
                    raise PrivateDataLeak(f"{what}: private string at {path}")
        elif isinstance(value, int) and not isinstance(value, bool) and needles:
            if str(value) in needles:
                raise PrivateDataLeak(f"{what}: private identifier at {path}")


def assert_public_text_safe(text: str, *, what: str = "artifact", league_id: Any = None,
                            names: Sequence[str] = ()) -> None:
    """The same guard for a rendered document, where structure is gone."""
    from .decision_page import PRIVACY_BANNER

    if PRIVACY_BANNER[:24] in text:
        raise PrivateDataLeak(f"{what}: carries the private page's banner")
    for marker in PRIVATE_SCHEMA_MARKERS:
        if marker in text:
            raise PrivateDataLeak(f"{what}: carries the private contract marker {marker!r}")
    for needle in _private_strings(league_id, names):
        if needle in text:
            raise PrivateDataLeak(f"{what}: carries a private string")


def _walk(node: Any, path: str = "$") -> Iterable[tuple[str, Any, Any]]:
    if isinstance(node, Mapping):
        for key, value in node.items():
            child = f"{path}.{key}"
            yield child, key, value
            yield from _walk(value, child)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            child = f"{path}[{index}]"
            yield child, None, value
            yield from _walk(value, child)
