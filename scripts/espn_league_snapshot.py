#!/usr/bin/env python3
"""Write one immutable, redacted snapshot of an ESPN fantasy league.

Read-only: this reads league views and writes a local JSON file. It submits
nothing to ESPN -- no lineup, no waiver claim, no drop, no trade, no watchlist
change.

Credentials are never accepted as arguments. A private league needs the ESPN
session cookies exported in the environment before the command runs; a public
league needs nothing. Nothing about their values is printed, and the snapshot
records only whether credentials were used, never what they were.

    ESPN_S2=... SWID=... python scripts/espn_league_snapshot.py \\
        --league-id 1111111111 --season 2026 --team-id 1 \\
        --team-name "Team One" --expect-teams 8

Identity is asserted, not discovered: wrong league, wrong season, wrong team,
or the wrong number of teams is a non-zero exit and no file, because a snapshot
of the wrong league is worse than none.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nflvalue.fantasy import espn_client, espn_league

DEFAULT_OUT = Path("reports") / "espn"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--league-id", type=int, required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--team-id", type=int, required=True)
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--expect-teams", type=int, required=True,
                        help="league size the caller asserts; a mismatch is a hard failure")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--include-free-agents", action="store_true",
                        help="also read the player pool view")
    parser.add_argument("--timeout", type=float, default=espn_client.DEFAULT_TIMEOUT)
    args = parser.parse_args(argv)

    credentials = espn_client.credentials_from_env()
    print(f"[espn] credentials: {'present' if credentials else 'absent (public read)'}")

    views = list(espn_client.LEAGUE_VIEWS)
    if args.include_free_agents:
        views.append(espn_client.PLAYER_POOL_VIEW)

    try:
        payloads = espn_client.fetch_league_views(
            args.league_id, args.season, views=views, credentials=credentials,
            timeout=args.timeout)
    except Exception as exc:                       # reported below, never swallowed
        print(f"[espn] read failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    retrieved_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    expected = espn_league.ExpectedIdentity(
        league_id=args.league_id, season=args.season, team_id=args.team_id,
        team_name=args.team_name, team_count=args.expect_teams)

    try:
        snapshot = espn_league.normalize_league(
            payloads, expected=expected, retrieved_at=retrieved_at,
            source_urls=[espn_client.build_league_url(args.league_id, args.season)],
            credentialed=credentials is not None)
    except espn_league.EspnAdapterError as exc:
        print(f"[espn] refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    path = espn_league.write_snapshot(snapshot, args.out)
    print(f"[espn] league {snapshot.league.league_id} season {snapshot.league.season}: "
          f"{snapshot.league.size} teams, draft {snapshot.draft.status}, "
          f"rosters {snapshot.roster_state}")
    for warning in snapshot.warnings:
        print(f"[espn] WARN {warning}")
    print(f"[espn] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
