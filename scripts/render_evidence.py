"""Render the Phase C evidence payload + prose for selections in a weekly run.

    python3 scripts/render_evidence.py --player 00-0037238 --market receiving_yards

Reads the persisted weekly props artifact and the player_week table, builds the
deterministic evidence payload, and runs the plain-language translator over it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from nflvalue import (
    evidence,
    evidence_prose,
    ingest,
)
from nflvalue.features import build_player_week


def load_leans(path: str):
    with open(path) as handle:
        payload = json.load(handle)
    return payload, [lean for game in payload["games"] for lean in game["leans"]]


def player_week_row(pw: pd.DataFrame, lean: dict):
    match = pw[(pw["player_id"] == lean["player_id"]) &
               (pw["season"] == lean["season"]) & (pw["week"] == lean["week"])]
    return match.iloc[0].to_dict() if len(match) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weekly", default=os.path.join("data", "weekly_props.json"))
    parser.add_argument("--player", action="append", default=[])
    parser.add_argument("--market", action="append", default=[])
    parser.add_argument("--json", action="store_true", help="dump the payload, not the prose")
    args = parser.parse_args()

    payload, leans = load_leans(args.weekly)
    pw = build_player_week(ingest.load_all_pbp())
    wanted = list(zip(args.player, args.market)) if args.player else None

    for lean in leans:
        if wanted and (lean["player_id"], lean["market"]) not in wanted:
            continue
        built = evidence.build_evidence(
            lean, player_week_row=player_week_row(pw, lean), as_of=payload.get("as_of"))
        if args.json:
            print(json.dumps(built, indent=2, sort_keys=True))
        else:
            print(evidence_prose.translate(built)["prose"])
        print("\n" + "=" * 78 + "\n")


if __name__ == "__main__":
    main()
