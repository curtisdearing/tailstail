"""Pregame availability gate (nflvalue/fantasy/availability_gate.py) and its
serving-time wiring through my_team -> decision_card -> decision_page.

The regression it pins is 2026 Week 2: Zay Flowers (BAL WR) was carried at
14.4 projected points on the Thursday card, listed Doubtful (hamstring,
limited practice) on Friday's official report, inactive on Sunday, and scored
zero in a set lineup.  The rows below are the real nflverse shapes for that
week, reduced to the players that matter.
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nflvalue.fantasy import availability_gate, decision_card, decision_page, my_team  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "my_team"
NOW = "2026-08-29T03:00:00+00:00"

FLOWERS, HARVEY, GESICKI, IR_PLAYER = "00-0039064", "00-0040730", "00-0034829", "00-0099999"


def week2_injuries() -> pd.DataFrame:
    """nflverse injuries_2026 rows as they read after Friday's report."""
    return pd.DataFrame([
        # Week 1: on the report with a full practice and no game designation.
        {"season": 2026, "week": 1, "team": "BAL", "gsis_id": FLOWERS, "position": "WR",
         "full_name": "Zay Flowers", "report_status": None, "practice_status":
         "Full Participation in Practice", "report_primary_injury": None,
         "practice_primary_injury": "Hamstring"},
        {"season": 2026, "week": 2, "team": "BAL", "gsis_id": FLOWERS, "position": "WR",
         "full_name": "Zay Flowers", "report_status": "Doubtful", "practice_status":
         "Limited Participation in Practice", "report_primary_injury": "Hamstring",
         "practice_primary_injury": "Hamstring"},
        {"season": 2026, "week": 2, "team": "DEN", "gsis_id": HARVEY, "position": "RB",
         "full_name": "RJ Harvey", "report_status": "Questionable", "practice_status":
         "Limited Participation in Practice", "report_primary_injury": "Hamstring",
         "practice_primary_injury": "Hamstring"},
        # A lowercase "out" and an "Out" with a trailing note both mean out.
        {"season": 2026, "week": 2, "team": "KC", "gsis_id": "00-0088888", "position": "WR",
         "full_name": "Some Player", "report_status": "out", "practice_status":
         "Did Not Participate In Practice", "report_primary_injury": "Knee",
         "practice_primary_injury": "Knee"},
    ])


def week2_rosters() -> pd.DataFrame:
    """Weekly roster rows as they read BEFORE kickoff: everyone still ACT."""
    return pd.DataFrame([
        {"season": 2026, "week": 2, "team": "BAL", "gsis_id": FLOWERS, "position": "WR",
         "status": "ACT", "game_type": "REG", "espn_id": 4429615},
        {"season": 2026, "week": 2, "team": "DEN", "gsis_id": HARVEY, "position": "RB",
         "status": "ACT", "game_type": "REG", "espn_id": 4568490},
        {"season": 2026, "week": 2, "team": "CIN", "gsis_id": GESICKI, "position": "TE",
         "status": "ACT", "game_type": "REG", "espn_id": 3116164},
        {"season": 2026, "week": 2, "team": "NYJ", "gsis_id": IR_PLAYER, "position": "RB",
         "status": "RES", "game_type": "REG", "espn_id": 1},
        {"season": 2026, "week": 1, "team": "NYJ", "gsis_id": IR_PLAYER, "position": "RB",
         "status": "ACT", "game_type": "REG", "espn_id": 1},
    ])


# --------------------------------------------------------------------------- #
# The gate itself
# --------------------------------------------------------------------------- #
def test_week2_report_gates_flowers_doubtful_and_leaves_harvey_to_the_draw():
    statuses = availability_gate.official_statuses(
        week2_injuries(), week2_rosters(), season=2026, week=2)
    flowers = statuses[FLOWERS]
    assert flowers["gate"] == "doubtful"
    assert flowers["report_status"] == "Doubtful"
    assert flowers["practice_status"].startswith("Limited")
    assert "Doubtful" in flowers["reason"] and "Hamstring" in flowers["reason"]
    assert flowers["roster_status"] == "ACT"
    harvey = statuses[HARVEY]
    assert harvey["gate"] is None                      # Questionable is the draw's call
    assert harvey["report_status"] == "Questionable"
    # Nothing official was ever said about Gesicki: absent, not "cleared".
    assert GESICKI not in statuses
    assert set(availability_gate.gated(statuses)) == {FLOWERS, "00-0088888", IR_PLAYER}


def test_week1_row_does_not_leak_into_week2():
    statuses = availability_gate.official_statuses(
        week2_injuries(), week2_rosters(), season=2026, week=1)
    assert statuses[FLOWERS]["gate"] is None
    assert statuses[FLOWERS]["report_status"] is None
    assert IR_PLAYER not in statuses


def test_out_matches_features_semantics_and_roster_placement_is_out():
    assert availability_gate.classify_report("out") == "out"
    assert availability_gate.classify_report("Out") == "out"
    assert availability_gate.classify_report("Doubtful") == "doubtful"
    assert availability_gate.classify_report("Questionable") is None
    assert availability_gate.classify_report(None) is None
    assert availability_gate.classify_report(float("nan")) is None
    statuses = availability_gate.official_statuses(
        week2_injuries(), week2_rosters(), season=2026, week=2)
    assert statuses["00-0088888"]["gate"] == "out"
    assert statuses[IR_PLAYER] == {
        "season": 2026, "week": 2, "report_status": None, "practice_status": None,
        "injury": None, "roster_status": "RES", "gate": "out",
        "reason": "official roster status RES", "source": availability_gate.SOURCE,
    }


def test_missing_feeds_gate_nobody():
    assert availability_gate.official_statuses(None, None, season=2026, week=2) == {}
    assert availability_gate.official_statuses(pd.DataFrame(), pd.DataFrame(), season=2026, week=2) == {}


# --------------------------------------------------------------------------- #
# Serving-time wiring: the Flowers case on the post_draft fixture
# --------------------------------------------------------------------------- #
def fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def model(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.model.json").read_text())


def samples_for(model_side: dict, *, n: int = 400, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    return {pid: rng.normal(float(p["mean"]), max(1.0, (float(p["p90"]) - float(p["p10"])) / 2.56), n)
            for pid, p in model_side["projections"].items()}


#: W. Gray is the post_draft fixture's WR1: set at WR, best WR mean (17.8).
#: He plays Zay Flowers here.
GRAY, GRAY_ESPN = "00-0031", 31


def doubtful(player_id: str) -> dict:
    return {player_id: {
        "season": 2026, "week": 2, "report_status": "Doubtful",
        "practice_status": "Limited Participation in Practice", "injury": "Hamstring",
        "roster_status": "ACT", "gate": "doubtful",
        "reason": "official injury report: Doubtful (Hamstring)",
        "source": availability_gate.SOURCE,
    }}


def contract(statuses: dict, *, drop_from_summaries: set[str] = frozenset(), tmp_path: Path) -> dict:
    import fantasy_weekly

    shutil.copy(FIXTURES / "post_draft.json", tmp_path / "snap.json")
    side = model("post_draft")
    rows = [{"player_id": pid, **values, "availability_probability": 0.99}
            for pid, values in side["projections"].items() if pid not in drop_from_summaries]
    return fantasy_weekly.run_my_team(
        pd.DataFrame(rows), generated_at=NOW, snapshot_dir=str(tmp_path),
        espn_crosswalk={int(k): v for k, v in side["crosswalk"].items()},
        samples=samples_for(side), official_statuses=statuses)


def test_a_doubtful_starter_is_never_started_and_the_seat_is_refilled(tmp_path):
    payload = contract(doubtful(GRAY), tmp_path=tmp_path)
    lineup = payload["optimal_lineup"]
    assert lineup["status"] == "ok"
    assert GRAY not in {s["player_id"] for s in lineup["starters"]}
    excluded = {e["player_id"]: e for e in lineup["excluded"]}
    assert excluded[GRAY]["code"] == "official_doubtful"
    assert "Doubtful" in excluded[GRAY]["reason"] and "Hamstring" in excluded[GRAY]["reason"]
    assert excluded[GRAY]["lineup_slot"] == "WR"
    # The roster entry carries the official word and the model's availability.
    gray = next(p for p in payload["roster"] if p["player_id"] == GRAY)
    assert gray["official_status"]["gate"] == "doubtful"
    assert gray["projection"]["availability_probability"] == 0.99
    assert "official_status" not in gray["projection"]
    # The set lineup had him at WR, so start/sit names who replaces him.
    sits = {(d.get("sit") or {}).get("player_id") for d in payload["start_sit"]["decisions"]}
    assert GRAY in sits


def test_the_card_shouts_out_doubtful_and_marks_the_change_forced(tmp_path):
    payload = contract(doubtful(GRAY), tmp_path=tmp_path)
    card = decision_card.build(payload, now=NOW, model_version="abc1234")
    decision_card.validate(card)
    alerts = [a for a in card["alerts"] if a["kind"] == "availability"]
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["severity"] == "warning"
    assert alert["text"].startswith("OUT / DOUBTFUL")
    assert "1 of them in the lineup you have set" in alert["text"]
    assert any("W. Gray" in p and "currently set at WR" in p and "Doubtful" in p
               for p in alert["players"])
    assert all((d.get("subject") or {}).get("player_id") != GRAY for d in card["decisions"])
    forced = [d for d in card["decisions"] if d.get("forced")
              and (d.get("alternative") or {}).get("player_id") == GRAY]
    assert forced, "replacing a Doubtful starter is a consequence, not a judgement"
    assert "W. Gray cannot play" in forced[0]["headline"]
    assert forced[0]["mean_delta"] is None
    assert "Doubtful" in forced[0]["reason"]["text"]
    html = decision_page.render(card, my_team=payload)
    assert 'class="avail"' in html and "OUT / DOUBTFUL" in html and "W. Gray" in html


def test_an_out_player_the_model_dropped_is_a_stated_zero_not_a_gap(tmp_path):
    statuses = {GRAY: {**doubtful(GRAY)[GRAY], "report_status": "Out", "gate": "out",
                       "reason": "official injury report: Out (Hamstring)"}}
    payload = contract(statuses, drop_from_summaries={GRAY}, tmp_path=tmp_path)
    gray = next(p for p in payload["roster"] if p["player_id"] == GRAY)
    assert gray["projection"] == {"mean": 0.0, "p10": 0.0, "p90": 0.0,
                                  "availability_probability": 0.0}
    excluded = {e["player_id"]: e for e in payload["optimal_lineup"]["excluded"]}
    assert excluded[GRAY]["code"] == "official_out"
    assert excluded[GRAY]["reason"] == "official injury report: Out (Hamstring)"
    card = decision_card.build(payload, now=NOW, model_version="abc1234")
    assert any(a["kind"] == "availability" for a in card["alerts"])


def test_questionable_is_carried_but_not_gated(tmp_path):
    statuses = {GRAY: {**doubtful(GRAY)[GRAY], "report_status": "Questionable", "gate": None,
                       "reason": "official injury report: Questionable (Hamstring)"}}
    payload = contract(statuses, tmp_path=tmp_path)
    assert GRAY in {s["player_id"] for s in payload["optimal_lineup"]["starters"]}
    gray = next(p for p in payload["roster"] if p["player_id"] == GRAY)
    assert gray["official_status"]["report_status"] == "Questionable"
    card = decision_card.build(payload, now=NOW, model_version="abc1234")
    assert not any(a["kind"] == "availability" for a in card["alerts"])


def test_without_statuses_the_contract_is_unchanged_except_for_the_new_fields(tmp_path):
    payload = contract({}, tmp_path=tmp_path)
    assert payload["optimal_lineup"]["excluded"] == []
    assert all(p["official_status"] is None for p in payload["roster"])
    card = decision_card.build(payload, now=NOW, model_version="abc1234")
    assert not any(a["kind"] == "availability" for a in card["alerts"])


def test_espn_doubtful_now_blocks_with_a_stated_reason():
    snap = copy.deepcopy(fixture("post_draft"))
    entry = next(e for e in snap["rosters"]["1"] if e["player_id"] == GRAY_ESPN)
    entry["injury_status"] = "DOUBTFUL"
    side = model("post_draft")
    payload = my_team.build(snap, now=NOW,
                            crosswalk={int(k): v for k, v in side["crosswalk"].items()},
                            projections=side["projections"], byes=side["byes"])
    excluded = {e["player_id"]: e for e in payload["optimal_lineup"]["excluded"]}
    assert excluded[GRAY]["code"] == "espn_status"
    assert excluded[GRAY]["reason"] == "injury status DOUBTFUL"
    card = decision_card.build(payload, now=NOW, model_version="abc1234")
    assert any(a["kind"] == "availability" and "W. Gray" in " ".join(a["players"])
               for a in card["alerts"])


def test_no_fit_refuses_without_a_saved_model(tmp_path):
    import fantasy_weekly

    with pytest.raises(FileNotFoundError, match="--no-fit"):
        fantasy_weekly.main(["--no-fit", "--no-fetch", "--data-dir", str(tmp_path),
                             "--model", str(tmp_path / "absent.joblib")])
