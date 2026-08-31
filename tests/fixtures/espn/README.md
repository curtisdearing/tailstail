# ESPN fantasy fixtures

Hand-built, fully synthetic recordings of the ESPN fantasy-football league
views the read-only adapter consumes (`nflvalue/fantasy/espn_client.py`,
`nflvalue/fantasy/espn_league.py`). Generated once and committed; regenerate by
hand, never by pointing a script at the live league.

**No real secrets are in these files, and none may ever be added.**
`tests/test_espn_league_adapter.py::test_fixtures_contain_no_real_credentials`
enforces that on every run.

Two things here look like credentials and are not:

* `members[].id` — ESPN really does use the SWID cookie value as a member id,
  so every GUID in these files is invented (`{AAAAAAAA-1111-…}`). The adapter
  hashes member ids into `member:<digest>` keys, so a real one could not reach
  a snapshot either.
* `raw_view_with_fake_secrets.json` `_debug_echo` — an invented block shaped
  like a request echo (`espn_s2=FAKEs2VALUE…`, a `Cookie` header, a bare
  `swid`). It exists so the redactor has something to strip; the values are
  literal and fake.

| file | what it represents |
| --- | --- |
| `league_predraft_2026.json` | The league as it stands before the 2026-09-05 draft: `draftDetail.drafted = false`, no picks, every roster empty. This is the state the adapter must represent honestly rather than filling in. |
| `league_inseason_2026.json` | The same league after the draft: 16-round snake picks, full rosters (including an IR-slotted player), three completed matchup periods, executed and pending transactions, standings. |
| `players_free_agents_2026.json` | A `kona_player_info`-shaped free-agent/waiver pool. |
| `raw_view_with_fake_secrets.json` | Pre-draft league plus the fake `_debug_echo` block described above. |
| `league_trade_scan_2026.json` | The in-season league the trade scan reads: eight realistically named teams whose skill players are real rows from `data/draft_board_2026.csv`, so identity mapping is exercised against the board it will meet in production. Team 1 is at the 16-man cap and carries on IR one player the board has never heard of (`Dontae Whitfield`, so unmatched reporting has something to report; he is on IR because a *startable* unidentified player now blocks his whole team from the scan); team 3 sits one under the cap so a 1-for-2 is legal for them and not for a full roster; every team carries a K and a D/ST (shadow positions); and a pending waiver locks one of team 1's bench players. Team 1 is long at RB and short at WR with team 5 the mirror, because a snake-balanced league contains no positive-sum trade and could only ever exercise the hold path. |

Team and league identity (league 1111111111, season 2026, team 1
"Team One", 8 teams, 16-round snake from slot 1) mirrors Curtis's
real league so the identity gate is exercised against the values it will meet
in production. That identity is not secret; the credentials that read it are,
and they live only in the process environment.
