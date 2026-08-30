"""The network boundary for the read-only ESPN league adapter.

This is the only module that touches the network or the process environment.
Everything it returns is raw view payloads; :mod:`nflvalue.fantasy.espn_league`
turns those into a validated snapshot. Keeping the split sharp is what lets the
whole normalization contract be tested offline.

Read-only by construction
-------------------------
Every request here is a GET of a league *read* view. There is no code path in
this module that submits a lineup, a waiver claim, a drop, or a trade, and the
test suite asserts that by reading this file's own source.

Credentials
-----------
A private league needs the ``espn_s2`` and ``SWID`` cookies. They are read from
the process environment and nowhere else — never from a command-line argument
(shell history, and every other process's view of ``ps``), never from a config
file in the vault, never from a function parameter a caller might log.

:class:`EspnCredentials` renders as ``<redacted>`` under ``repr`` and ``str``,
so a traceback, a log line, or an f-string cannot spill it. The cookie header
is assembled inside the request call and is never returned, stored, or logged;
failures surface the URL and the status, never the request headers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..sources._http import get_json

REDACTED = "<redacted>"

#: ESPN's read host for league views.
READ_HOST = "https://lm-api-reads.fantasy.espn.com"
LEAGUE_PATH = "/apis/v3/games/ffl/seasons/{season}/segments/0/leagues/{league_id}"

#: The read views the adapter consumes. All of these are league *read* views.
LEAGUE_VIEWS: tuple[str, ...] = (
    "mSettings", "mTeam", "mRoster", "mMatchup",
    "mDraftDetail", "mStandings", "mTransactions2", "mPendingTransactions",
)

#: The free-agent pool lives behind a separate view and a filter header.
PLAYER_POOL_VIEW = "kona_player_info"

DEFAULT_TIMEOUT = 20.0


@dataclass(frozen=True)
class EspnCredentials:
    """ESPN session cookies for a private league.

    The values are deliberately unprintable: every rendering is redacted, so a
    stack trace or a debug log cannot leak the session.
    """

    espn_s2: str
    swid: str

    def __repr__(self) -> str:      # pragma: no cover - exercised via str()
        return f"EspnCredentials({REDACTED})"

    def __str__(self) -> str:
        return f"EspnCredentials({REDACTED})"


def credentials_from_env(environ: Mapping[str, str] | None = None) -> EspnCredentials | None:
    """Read ESPN cookies from the environment, or return ``None``.

    Both values are required: half a session is not a session, and a partial
    cookie jar produces a confusing 401 rather than an honest "no credentials".
    """
    source = os.environ if environ is None else environ
    s2 = (source.get("ESPN_S2") or "").strip()
    swid_value = (source.get("SWID") or "").strip()
    if not s2 or not swid_value:
        return None
    return EspnCredentials(espn_s2=s2, swid=swid_value)


def build_league_url(league_id: int, season: int) -> str:
    """The league read URL. Views ride as query parameters."""
    return READ_HOST + LEAGUE_PATH.format(season=int(season), league_id=int(league_id))


def _cookie_header(credentials: EspnCredentials | None) -> dict[str, str]:
    """Build the request header. Never returned to a caller, never logged."""
    if credentials is None:
        return {}
    return {"Cookie": f"espn_s2={credentials.espn_s2}; SWID={credentials.swid}"}


def fetch_league_views(league_id: int, season: int, *,
                       views: Sequence[str] = LEAGUE_VIEWS,
                       credentials: EspnCredentials | None = None,
                       timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """GET each read view once and return ``{view_name: payload}``.

    One request per view keeps a single failing view from costing the whole
    read, and keeps the returned mapping honest about which views were actually
    answered — a view that is absent was not read, which is a different claim
    from a view that came back empty.
    """
    url = build_league_url(league_id, season)
    headers = _cookie_header(credentials)
    answered: dict[str, Any] = {}
    for view in views:
        answered[view] = get_json(url, {"view": view}, timeout=timeout,
                                  headers=dict(headers), source="espn-fantasy")
    return answered
