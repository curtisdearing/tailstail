"""Canonical content hashes of every production frame Phase B must not change.

Phase B is a hardening phase: fail-loud I/O, explicit seeds, schema contracts,
lint, and profiling. None of that is allowed to move a single model number.
"Nothing looked different" is not evidence, so this script pins the evidence:
it rebuilds each production frame from local parquet and prints the canonical
CSV-content hash (``reproducibility.canonical_csv_sha256`` -- content, not
Parquet bytes, so a writer-version change cannot masquerade as a numeric one).

Run before the change, run after, diff the JSON. Any differing hash is a
behaviour change and belongs in the accuracy ledger, not in Phase B.

    python3 scripts/neutrality_hashes.py --out /tmp/before.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from nflvalue import ingest
from nflvalue.features import build_opp_pos_def, build_player_week, build_team_week
from nflvalue.projection import game_script_multipliers, project
from nflvalue.reproducibility import canonical_csv_sha256

SEASON, WEEK = 2023, 10


def _hash(frame: pd.DataFrame, keys) -> Dict:
    return {"sha256": canonical_csv_sha256(frame, row_keys=list(keys)),
            "rows": int(len(frame)), "cols": int(frame.shape[1])}


def collect(rosters_path: str | None = None) -> Dict:
    timings: Dict[str, float] = {}

    def timed(label, fn, *a, **kw):
        start = time.perf_counter()
        out = fn(*a, **kw)
        timings[label] = round(time.perf_counter() - start, 3)
        return out

    pbp = timed("load_all_pbp", ingest.load_all_pbp)
    rosters = pd.read_parquet(rosters_path) if rosters_path else None

    player_week = timed("build_player_week", build_player_week, pbp, rosters)
    opp_pos_def = timed("build_opp_pos_def", build_opp_pos_def, pbp, rosters)
    team_week = timed("build_team_week", build_team_week, pbp)

    out = {
        "player_week": _hash(player_week, ["season", "week", "player_id"]),
        "opp_pos_def": _hash(opp_pos_def, ["season", "week", "defteam", "role"]),
        "team_week": _hash(team_week, ["season", "week", "team"]),
    }

    # Closed-form projection over a fixed, sorted slice -- pins the market math.
    rows = player_week[(player_week["season"] == SEASON) & (player_week["week"] == WEEK)]
    rows = rows.sort_values("player_id").head(200)
    projections = []
    start = time.perf_counter()
    for record in rows.to_dict("records"):
        for market, line, sd in (("receiving_yards", 55.5, 25.0),
                                 ("rushing_yards", 45.5, 22.0),
                                 ("passing_yards", 240.5, 55.0)):
            result = project(record, market, line=line, sd=sd)
            if result:
                projections.append({"player_id": record["player_id"], "market": market,
                                    **{k: v for k, v in result.items()
                                       if isinstance(v, (int, float, str, bool)) or v is None}})
    timings["project_600"] = round(time.perf_counter() - start, 3)
    out["projections"] = _hash(pd.DataFrame(projections), ["player_id", "market"])

    out["game_script"] = {"sha256": canonical_csv_sha256(
        pd.DataFrame([{"margin": m, **game_script_multipliers(m)}
                      for m in (-10.0, -3.5, 0.0, 3.5, 6.0, 14.0)]), row_keys=["margin"])}

    out["_timings_sec"] = timings
    out["_pandas"] = pd.__version__
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--rosters", default=None)
    args = parser.parse_args()
    result = collect(args.rosters)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
