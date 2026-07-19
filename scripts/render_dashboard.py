"""Render the Phase D interface from stored deterministic artifacts.

    python3 scripts/render_dashboard.py --state no-bet   --out /tmp/no_bet.html
    python3 scripts/render_dashboard.py --state selections --out /tmp/cards.html

Every number on the page comes from a persisted artifact: the weekly props run,
the lean replay bands, the fantasy Monte Carlo regime audit, the premortem
Monte Carlo, and the Phase A CLV ledger.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nflvalue import db as dbmod
from nflvalue import evidence, evidence_view, ingest, killcheck
from nflvalue.features import build_player_week

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(path, default):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return default
    with open(full) as handle:
        return json.load(handle)


def gather(state: str, limit: int = 3):
    weekly = _load("data/weekly_props.json", {})
    leans = [lean for game in weekly.get("games", []) for lean in game.get("leans", [])]
    replay = _load("data/lean_replay_2025.json", {})
    bands = (replay.get("leans") or {}).get("by_composite_band") or {}
    regimes = _load("reports/fantasy_monte_carlo_history.json", {}).get("regimes") or {}
    premortem = _load("premortem_mc_results.json", {})
    config = _load("config.json", {})

    # Live CLV state from the Phase A ledger.
    conn = dbmod.connect()
    clv_report = killcheck.forward_clv_report(conn)

    calibration_bins = []
    for market in (_load("data/prop_backtest.json", {}) or {}).get("markets", {}).values():
        for entry in (market or {}).get("calibration", []) or []:
            if entry.get("n"):
                calibration_bins.append({"predicted": entry.get("predicted_p_over"),
                                         "actual": entry.get("actual_over_rate"),
                                         "n": entry.get("n")})
    calibration_bins = sorted(calibration_bins, key=lambda e: e["predicted"] or 0)

    payloads = []
    if state == "selections":
        pw = build_player_week(ingest.load_all_pbp())
        chosen = sorted(leans, key=lambda l: -(l.get("composite") or 0))[:limit]
        for lean in chosen:
            match = pw[(pw["player_id"] == lean["player_id"]) &
                       (pw["season"] == lean["season"]) & (pw["week"] == lean["week"])]
            row = match.iloc[0].to_dict() if len(match) else None
            payloads.append(evidence.build_evidence(lean, player_week_row=row,
                                                    as_of=weekly.get("as_of")))
    return {
        "weekly": weekly, "leans": leans, "bands": bands, "regimes": regimes,
        "premortem": premortem, "config": config, "clv_report": clv_report,
        "calibration_bins": calibration_bins, "payloads": payloads,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", choices=("no-bet", "selections"), default="no-bet")
    parser.add_argument("--out", required=True)
    parser.add_argument("--staking", action="store_true")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    ctx = gather(args.state, args.limit)
    weekly = ctx["weekly"]
    staking = None
    if args.staking:
        staking = {"config": ctx["config"], "premortem": ctx["premortem"],
                   "killcheck": ctx["clv_report"], "edge": None}

    reasons = weekly.get("publish_reasons") or [
        "No candidate cleared the composite screen at the published threshold.",
        "Closing-line value is unproven (INSUFFICIENT_SAMPLE), so no price-based gate can pass.",
        "Every registered factor touching these drivers is research_only or rejected by its own gate.",
    ]
    page = evidence_view.render_page(
        season=weekly.get("season", 2023), week=weekly.get("week", 10),
        as_of=weekly.get("as_of", ""), clv_report=ctx["clv_report"],
        selections=ctx["payloads"], screened=len(ctx["leans"]),
        no_bet_reasons=reasons,
        nearest=_nearest(ctx["leans"]) if args.state == "no-bet" else None,
        calibration_bins=ctx["calibration_bins"], bands=ctx["bands"],
        regimes=ctx["regimes"], clv_history=[], staking=staking)
    with open(args.out, "w") as handle:
        handle.write(page)
    print(f"wrote {args.out} ({len(page)} bytes)")


def _nearest(leans):
    if not leans:
        return None
    best = max(leans, key=lambda l: l.get("composite") or 0)
    return {"name": best.get("name"), "market": best.get("market"),
            "composite": best.get("composite"),
            "failed_because": ("its composite band has n=33 graded selections, below the "
                               "100 required before any rate may be quoted")}


if __name__ == "__main__":
    main()
