"""``decision-card/1``: four answers, three actions, and nothing it cannot back.

The card is the layer between a contract that states everything and a person
who has ten minutes on a Sunday morning, so almost every property here is about
what the card *refuses* to say:

  * a stale, future-dated or unprovenanced run says one thing and stops;
  * a swap whose spread was never measured is not recommended on its mean;
  * a seat whose occupant is on bye is a consequence, not a judgement, and is
    not reported as a swap worth its arithmetic difference;
  * cited news can add a driver and can never move a number or an order;
  * K and D/ST stay visibly unpromoted;
  * waivers and trades are absent until their own gates pass.

The inputs are the canonical `espn-league/1` snapshots, including one built by
running the real adapter over the recorded ESPN views, because a card built
from a fixture shaped to suit the card proves only that the fixture was
well chosen.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nflvalue.fantasy import (  # noqa: E402
    decision_card,
    espn_contract,
    espn_league,
    my_team,
    waivers,
)

FIXTURES = ROOT / "tests" / "fixtures" / "my_team"
ESPN_FIXTURES = ROOT / "tests" / "fixtures" / "espn"
NOW = "2026-08-29T03:00:00+00:00"
MODEL = "e3f1c0d"

LEAGUE_ID = 1111111111
SEASON = 2026
TEAM_ID = 1
TEAM_NAME = "Team One"
TEAM_COUNT = 8
RETRIEVED_AT = "2026-08-29T12:00:00Z"
ADAPTER_NOW = "2026-08-29T13:00:00Z"


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def side(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.model.json").read_text())


def snapshot(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def samples_for(model_side: dict, *, n: int = 400, seed: int = 11) -> dict:
    """Paired draws: one row per simulated week, every player on the same row.

    Drawn from the fixture's own p10/p90 so a swap's spread is a real spread of
    the same weeks, which is exactly the input the card requires before it will
    call a lineup change actionable.
    """
    rng = np.random.default_rng(seed)
    return {
        player_id: rng.normal(float(p["mean"]), max(1.0, (float(p["p90"]) - float(p["p10"])) / 2.56), n)
        for player_id, p in model_side["projections"].items()
    }


def contract_payload(name: str, *, snap: dict | None = None, now: str = NOW, **kwargs) -> dict:
    model_side = side(name)
    return my_team.build(
        snap if snap is not None else snapshot(name), now=now,
        crosswalk={int(k): v for k, v in model_side["crosswalk"].items()},
        projections=model_side["projections"], byes=model_side["byes"], **kwargs)


def card_for(name: str, *, with_samples: bool = True, now: str = NOW, snap: dict | None = None,
             model_version: str | None = MODEL, context=(), **kwargs) -> dict:
    payload = contract_payload(
        name, snap=snap, now=now,
        samples=samples_for(side(name)) if with_samples else None, **kwargs)
    return decision_card.build(payload, now=now, model_version=model_version, context=context)


ALL_FIXTURES = ("pre_draft", "draft_in_progress", "post_draft", "stale_snapshot",
                "unmatched_player", "bye", "injury", "illegal_roster", "no_action")


# --------------------------------------------------------------------------- #
# Actual adapter output — not a shape written to please this reader
# --------------------------------------------------------------------------- #
def _views(raw, players):
    views = dict.fromkeys(("mSettings", "mTeam", "mRoster", "mMatchup",
                           "mDraftDetail", "mStandings", "mTransactions2"), raw)
    views["kona_player_info"] = players
    return views


@pytest.fixture
def adapter_snapshot():
    raw = json.loads((ESPN_FIXTURES / "league_inseason_2026.json").read_text())
    players = json.loads((ESPN_FIXTURES / "players_free_agents_2026.json").read_text())
    expected = espn_league.ExpectedIdentity(
        league_id=LEAGUE_ID, season=SEASON, team_id=TEAM_ID, team_name=TEAM_NAME,
        team_count=TEAM_COUNT)
    return espn_league.snapshot_to_dict(espn_league.normalize_league(
        _views(raw, players), expected=expected, retrieved_at=RETRIEVED_AT,
        source_urls=["https://lm-api-reads.fantasy.espn.com/<redacted>"]))


@pytest.fixture
def adapter_contract():
    return espn_contract.from_settings_payload(
        json.loads((ESPN_FIXTURES / "league_inseason_2026.json").read_text()))


def test_card_builds_from_real_adapter_output(adapter_snapshot, adapter_contract):
    """No translation layer between the adapter and the page."""
    payload = my_team.build(adapter_snapshot, now=ADAPTER_NOW, contract=adapter_contract)
    card = decision_card.build(payload, now=ADAPTER_NOW, model_version=MODEL)

    assert card["schema_version"] == "decision-card/1"
    assert card["source_contract"] == my_team.SCHEMA_VERSION
    assert card["visibility"] == "private"
    assert card["provenance"]["scoring_hash"] == adapter_contract.scoring_hash
    assert card["provenance"]["snapshot_hash"] == adapter_snapshot["hashes"]["roster"]
    assert card["provenance"]["freshness_state"] in {"fresh", "aging"}
    decision_card.validate(card)


def test_card_refuses_a_contract_it_does_not_read(adapter_snapshot):
    payload = my_team.build(adapter_snapshot, now=ADAPTER_NOW)
    payload["schema_version"] = "my_team/9.9.9"
    with pytest.raises(decision_card.CardRejected):
        decision_card.build(payload, now=ADAPTER_NOW, model_version=MODEL)


# --------------------------------------------------------------------------- #
# Every state produces a card, and every card passes its own validator
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_every_state_builds_and_validates(name):
    card = card_for(name)
    decision_card.validate(card)
    assert card["state"] in {"ok", "no_current_pick"}
    assert len([d for d in card["decisions"]
                if d["status"] in decision_card.ACTIONABLE_STATUSES]) <= 3


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_a_blocked_card_states_exactly_one_reason(name):
    card = card_for(name)
    if card["state"] == "ok":
        pytest.skip("this fixture produces a lineup")
    assert card["reason"]["text"].strip()
    assert card["decisions"] == []
    assert card["current_lineup"]["slots"] == []


# --------------------------------------------------------------------------- #
# Draft states
# --------------------------------------------------------------------------- #
def test_pre_draft_says_there_is_no_roster_rather_than_no_lineup():
    card = card_for("pre_draft")
    assert card["state"] == "no_current_pick"
    assert card["reason"]["code"] == "no_roster"
    assert "draft has not happened" in card["reason"]["text"]


def test_post_draft_produces_a_lineup_and_at_least_one_decision():
    card = card_for("post_draft")
    assert card["state"] == "ok"
    assert card["current_lineup"]["legal"] is True
    assert card["current_lineup"]["slots"]
    assert card["decisions"]


def test_draft_in_progress_cannot_field_a_lineup_and_says_so():
    card = card_for("draft_in_progress")
    assert card["state"] == "no_current_pick"
    assert any(alert["kind"] == "legality" for alert in card["alerts"])


# --------------------------------------------------------------------------- #
# Freshness: stale, missing and future all block, and none reuses a prior card
# --------------------------------------------------------------------------- #
def test_a_stale_snapshot_blocks_with_one_reason():
    card = card_for("stale_snapshot")
    assert card["state"] == "no_current_pick"
    assert card["reason"]["code"] == "snapshot_not_current"
    assert card["decisions"] == []
    assert any(a["kind"] == "freshness" and a["severity"] == "blocking" for a in card["alerts"])


def test_a_capture_dated_in_the_future_is_not_treated_as_fresh():
    """`age < FRESH_HOURS` is true for a negative age, which is the whole trap."""
    snap = snapshot("post_draft")
    snap["retrieved_at"] = "2026-08-30T09:00:00Z"       # six hours after NOW
    payload = contract_payload("post_draft", snap=snap)
    assert payload["freshness"]["state"] == "future"

    card = decision_card.build(payload, now=NOW, model_version=MODEL)
    assert card["state"] == "no_current_pick"
    assert "ahead of the clock" in card["reason"]["text"]
    assert card["decisions"] == []


def test_a_missing_capture_time_blocks():
    snap = snapshot("post_draft")
    snap.pop("retrieved_at")
    card = decision_card.build(contract_payload("post_draft", snap=snap),
                               now=NOW, model_version=MODEL)
    assert card["state"] == "no_current_pick"
    assert card["decisions"] == []


def test_a_blocked_card_carries_this_week_and_no_earlier_content():
    """The refusal is built from the current contract, never from a prior one."""
    fresh = card_for("post_draft")
    stale = card_for("stale_snapshot")
    fresh_names = {slot["name"] for slot in fresh["current_lineup"]["slots"]}
    assert fresh_names
    rendered = json.dumps(stale, default=str)
    assert not any(name and name in rendered for name in fresh_names if name)


# --------------------------------------------------------------------------- #
# Provenance is required, in full, before anything is recommended
# --------------------------------------------------------------------------- #
def test_no_model_version_means_no_decisions():
    card = card_for("post_draft", model_version=None)
    assert card["state"] == "no_current_pick"
    assert card["reason"]["code"] == "provenance_incomplete"
    assert "model_version" in card["provenance"]["missing"]
    assert card["decisions"] == []


def test_no_scoring_identity_means_no_decisions():
    snap = snapshot("post_draft")
    snap["hashes"] = {"league": snap["hashes"]["league"]}
    card = decision_card.build(contract_payload("post_draft", snap=snap),
                               now=NOW, model_version=MODEL)
    assert card["state"] == "no_current_pick"
    assert card["reason"]["code"] == "provenance_incomplete"
    assert {"scoring_hash", "snapshot_hash"} <= set(card["provenance"]["missing"])


def test_every_decision_carries_all_four_stamps():
    card = card_for("post_draft")
    for decision in card["decisions"]:
        stamps = decision["provenance"]
        assert stamps["model_version"] == MODEL
        assert stamps["scoring_hash"]
        assert stamps["snapshot_hash"]
        assert stamps["freshness_state"] in {"fresh", "aging"}


# --------------------------------------------------------------------------- #
# Only changes, and only three of them
# --------------------------------------------------------------------------- #
def test_no_action_reports_a_hold_and_no_changes():
    card = card_for("no_action")
    assert card["lineup_changes"] == []
    assert [d["status"] for d in card["decisions"]] == ["hold"]
    assert card["decisions"][0]["invalidation_trigger"]
    assert all(slot["already_set"] for slot in card["current_lineup"]["slots"])


def test_a_change_is_listed_only_where_the_seat_differs():
    card = card_for("post_draft")
    changed = {row["slot"] for row in card["lineup_changes"]}
    for slot in card["current_lineup"]["slots"]:
        if slot["already_set"]:
            assert not any(row["start"]["espn_player_id"] == slot["espn_player_id"]
                           for row in card["lineup_changes"])
    assert changed


def test_hold_is_never_claimed_while_a_change_is_pending():
    """Two contradictory answers on one page is worse than either alone."""
    for name in ALL_FIXTURES:
        card = card_for(name)
        statuses = {d["status"] for d in card["decisions"]}
        if card["lineup_changes"]:
            assert "hold" not in statuses, name


def test_more_than_three_actions_are_held_back_and_counted():
    card = card_for("post_draft")
    extra = copy.deepcopy(card["decisions"][0]) if card["decisions"] else None
    if extra is None or extra["status"] not in decision_card.ACTIONABLE_STATUSES:
        pytest.skip("this fixture has no actionable decision to clone")
    card["decisions"] = [copy.deepcopy(extra) for _ in range(4)]
    with pytest.raises(decision_card.CardRejected):
        decision_card.validate(card)


def test_the_budget_holds_over_a_long_waiver_plan():
    plan = [
        waivers.Recommendation(
            add_espn_id=900 + i, add_name=f"F{900 + i}", add_position="RB",
            drop_espn_id=25, drop_name="R. Frost", drop_state="selected",
            status="ok", shadow_reason=None, confidence="medium",
            rationale="projects above the worst legal drop",
            invalidation_trigger="a status change to either player",
            priority_implications={}, replacement_effect={}, opponent_opportunity_impact={},
            lineup_delta={"own_optimal_lineup_delta": 9.0 - i, "median": 8.8 - i,
                          "p10": 1.0 - i, "p90": 17.0 - i,
                          "model_relative_prob_improves": 0.71, "simulations": 400,
                          "basis": "paired joint simulation rows, both lineups solved per row"},
            lineup_delta_status="ok",
            data_timestamps={}, degraded=False, faab=None)
        for i in range(6)
    ]
    card = card_for("no_action", waiver_plan=plan)
    actionable = [d for d in card["decisions"]
                  if d["status"] in decision_card.ACTIONABLE_STATUSES]
    assert len(actionable) == 3
    assert card["withheld"]
    assert any(a["kind"] == "budget" for a in card["alerts"])
    # Ranked, not truncated arbitrarily: the biggest gains survive.
    assert [d["mean_delta"] for d in actionable] == sorted(
        (d["mean_delta"] for d in actionable), reverse=True)


# --------------------------------------------------------------------------- #
# Availability: a forced seat is a consequence, not a swap
# --------------------------------------------------------------------------- #
def test_a_bye_week_seat_is_forced_and_carries_no_arithmetic_delta():
    card = card_for("bye")
    forced = [d for d in card["decisions"] if d.get("forced")]
    assert forced, "the bye fixture must produce a forced change"
    row = forced[0]
    assert row["status"] == "start"
    assert row["mean_delta"] is None and row["median_delta"] is None
    assert row["interval"]["status"] == "not_applicable"
    assert row["model_relative_probability"] is None
    assert "cannot play" in row["headline"]
    assert row["risk"]["text"] and row["invalidation_trigger"]


def test_an_inactive_player_is_excluded_and_the_seat_is_refilled():
    card = card_for("injury")
    assert card["state"] == "ok"
    assert card["decisions"]
    assert all(d["forced"] or d["status"] == "no_current_pick" for d in card["decisions"])


def test_an_unmeasured_swap_is_not_recommended_on_its_mean():
    card = card_for("post_draft", with_samples=False)
    judgements = [d for d in card["decisions"] if not d["forced"]]
    assert judgements, "the fixture must produce a judgement call"
    for decision in judgements:
        assert decision["status"] == "no_current_pick"
        assert decision["reason"]["code"] == "no_joint_simulation"
        assert decision["mean_delta"] is not None      # the number is still shown
        assert decision["interval"]["status"] == "unavailable"


def test_a_measured_swap_carries_mean_median_interval_and_a_labelled_probability():
    card = card_for("post_draft")
    starts = [d for d in card["decisions"] if d["status"] == "start" and not d["forced"]]
    assert starts
    decision = starts[0]
    assert isinstance(decision["mean_delta"], float)
    assert isinstance(decision["median_delta"], float)
    assert decision["interval"]["status"] == "ok"
    assert decision["interval"]["p10"] <= decision["interval"]["p90"]
    assert decision["interval"]["simulations"] == 400
    probability = decision["model_relative_probability"]
    assert 0.0 <= probability["value"] <= 1.0
    assert "model-relative" in probability["qualifier"]
    assert "not a calibrated confidence" in probability["qualifier"]


# --------------------------------------------------------------------------- #
# Identity and legality
# --------------------------------------------------------------------------- #
def test_incomplete_identity_is_alerted_not_silently_dropped():
    card = card_for("unmatched_player")
    alert = next(a for a in card["alerts"] if a["kind"] == "identity")
    assert alert["players"]
    assert "counted as zero" in alert["text"]


def test_no_crosswalk_at_all_blocks_rather_than_fielding_an_empty_team():
    payload = my_team.build(snapshot("post_draft"), now=NOW,
                            projections=side("post_draft")["projections"])
    card = decision_card.build(payload, now=NOW, model_version=MODEL)
    assert card["state"] == "no_current_pick"
    assert card["decisions"] == []


def test_an_illegal_roster_blocks_and_names_the_seat_it_cannot_fill():
    card = card_for("illegal_roster")
    assert card["state"] == "no_current_pick"
    assert card["current_lineup"]["legal"] is False
    alert = next(a for a in card["alerts"] if a["kind"] == "legality")
    assert any("WR" in violation for violation in alert["violations"])


# --------------------------------------------------------------------------- #
# Shadow seats and the gates in front of waivers and trades
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_kicker_and_defence_are_always_visibly_unpromoted(name):
    card = card_for(name)
    positions = {seat["position"]: seat for seat in card["shadow_seats"]}
    assert set(positions) == {"K", "D/ST"}
    for seat in positions.values():
        assert seat["status"] == "shadow"
        assert seat["promoted"] is False
        assert seat["reason"]["text"]
        assert seat["invalidation_trigger"]


def test_a_missing_kicker_never_becomes_a_lineup_decision():
    snap = snapshot("post_draft")
    snap["rosters"]["1"] = [p for p in snap["rosters"]["1"] if p["default_position"] != "K"]
    card = card_for("post_draft", snap=snap)
    assert card["current_lineup"]["status"] == "ok"
    assert not any((d.get("slot") or "") == "K" for d in card["decisions"])
    assert any(seat["position"] == "K" for seat in card["shadow_seats"])


def test_a_shadow_seat_never_consumes_the_decision_budget():
    card = card_for("no_action")
    assert all(d["status"] != "shadow" for d in card["decisions"])


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_waivers_are_absent_until_the_planner_gate_passes(name):
    card = card_for(name)
    assert not any(d["kind"] == "waiver" for d in card["decisions"])


def _waiver(**overrides):
    fields = dict(
        add_espn_id=901, add_name="F. Waiver", add_position="RB",
        drop_espn_id=25, drop_name="R. Frost", drop_state="selected",
        status="ok", shadow_reason=None, confidence="medium",
        rationale="projects above the worst legal drop",
        invalidation_trigger="a status change to either player",
        priority_implications={}, replacement_effect={}, opponent_opportunity_impact={},
        lineup_delta={"own_optimal_lineup_delta": 2.3, "median": 2.1, "p10": -1.4, "p90": 6.2,
                      "model_relative_prob_improves": 0.71, "simulations": 400,
                      "basis": "paired joint simulation rows, both lineups solved per row"},
        lineup_delta_status="ok", data_timestamps={}, degraded=False, faab=None)
    fields.update(overrides)
    return waivers.Recommendation(**fields)


def test_a_measured_waiver_add_is_actionable_with_its_range():
    card = card_for("no_action", waiver_plan=[_waiver()])
    row = next(d for d in card["decisions"] if d["kind"] == "waiver")
    assert row["status"] == "add"
    assert row["mean_delta"] == 2.3 and row["median_delta"] == 2.1
    assert row["interval"] == {
        "status": "ok", "reason": None,
        "basis": "paired joint simulation rows, both lineups solved per row",
        "simulations": 400, "p10": -1.4, "p90": 6.2,
    }
    assert row["model_relative_probability"]["value"] == 0.71


def test_a_waiver_delta_with_no_percentiles_is_not_actionable():
    """`lineup_delta_status: ok` is not itself a measurement.

    The planner publishes the range under `p10`/`p90`/`simulations`. Trusting
    the status word alone produced a row that claimed an interval and rendered
    it as two em dashes -- a recommendation with a hole where its evidence was.
    """
    card = card_for("no_action",
                    waiver_plan=[_waiver(lineup_delta={"own_optimal_lineup_delta": 2.3})])
    row = next(d for d in card["decisions"] if d["kind"] == "waiver")
    assert row["status"] == "no_current_pick"
    assert row["reason"]["code"] == "no_joint_simulation"
    assert row["interval"]["status"] == "unavailable"
    assert row["mean_delta"] == 2.3


def test_a_degraded_waiver_plan_produces_no_waiver_row():
    plan = [waivers.Recommendation(
        add_espn_id=901, add_name="F901", add_position="RB",
        drop_espn_id=25, drop_name="R. Frost", drop_state="selected",
        status="shadow", shadow_reason="stale free-agent pool", confidence="none",
        rationale="inputs are degraded", invalidation_trigger="a fresh pool",
        priority_implications={}, replacement_effect={}, opponent_opportunity_impact={},
        lineup_delta=None, lineup_delta_status="unavailable",
        data_timestamps={}, degraded=True, faab=None)]
    card = card_for("no_action", waiver_plan=plan)
    assert not any(d["kind"] == "waiver" for d in card["decisions"])


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_trades_never_appear(name):
    card = card_for(name)
    assert not any(d["kind"] == "trade" for d in card["decisions"])


# --------------------------------------------------------------------------- #
# Cited context: a driver, a risk, an invalidation — never a number
# --------------------------------------------------------------------------- #
def _numbers_and_order(card: dict) -> str:
    """Everything context is forbidden to touch, as one comparable string."""
    trimmed = copy.deepcopy(card)
    for decision in trimmed["decisions"]:
        decision["drivers"] = None
        decision["risk"] = None
    trimmed["alerts"] = None
    return json.dumps(trimmed, sort_keys=True, default=str)


def _context_for(card: dict, **extra) -> list[dict]:
    ids = [slot["espn_player_id"] for slot in card["current_lineup"]["slots"]]
    return [{"text": "Named the starter in Wednesday's depth chart.",
             "source": "Club depth chart", "as_of": "2026-08-28T15:00:00Z",
             "espn_player_ids": ids, **extra}]


def test_context_changes_no_number_and_no_order():
    plain = card_for("post_draft")
    cited = card_for("post_draft", context=_context_for(plain))
    assert _numbers_and_order(plain) == _numbers_and_order(cited)


def test_a_matched_note_becomes_a_driver_with_its_source_and_timestamp():
    plain = card_for("post_draft")
    cited = card_for("post_draft", context=_context_for(plain))
    drivers = [d for decision in cited["decisions"] for d in decision["drivers"]]
    assert drivers
    for driver in drivers:
        assert driver["source"] and driver["as_of"]


def test_at_most_two_drivers_survive():
    plain = card_for("post_draft")
    ids = [slot["espn_player_id"] for slot in plain["current_lineup"]["slots"]]
    context = [{"text": f"Note {i}", "source": "Club depth chart",
                "as_of": f"2026-08-2{i}T15:00:00Z", "espn_player_ids": ids}
               for i in range(5)]
    cited = card_for("post_draft", context=context)
    for decision in cited["decisions"]:
        assert len(decision["drivers"]) <= 2


def test_a_note_without_a_source_or_timestamp_is_refused_and_counted():
    context = [{"text": "Looked good in practice."},
               {"text": "Also good.", "source": "A tip", "as_of": ""}]
    card = card_for("post_draft", context=context)
    alert = next(a for a in card["alerts"] if a["kind"] == "context")
    assert len(alert["refused"]) == 2
    assert all(d["drivers"] == [] for d in card["decisions"])


def test_a_refusal_never_echoes_the_note_it_refused():
    """Otherwise banned wording reaches the page through the apology."""
    context = [{"text": "the ensemble likes him", "source": "a blog",
                "as_of": "2026-08-28T15:00:00Z"}]
    card = card_for("post_draft", context=context)
    assert "ensemble" not in json.dumps(card, default=str)
    decision_card.validate(card)


def test_counter_evidence_replaces_the_generic_risk_and_is_cited():
    plain = card_for("post_draft")
    starts = [d for d in plain["decisions"] if d["status"] == "start"]
    if not starts:
        pytest.skip("no actionable start in this fixture")
    subject = starts[0]["subject"]["espn_player_id"]
    context = [{"text": "Limited in Friday's practice with an ankle.",
                "source": "Club injury report", "as_of": "2026-08-28T20:00:00Z",
                "espn_player_ids": [subject], "counter_evidence": True}]
    cited = card_for("post_draft", context=context)
    row = next(d for d in cited["decisions"] if d["status"] == "start")
    assert row["risk"]["text"] == "Limited in Friday's practice with an ankle."
    assert row["risk"]["source"] == "Club injury report"
    assert row["risk"]["as_of"] == "2026-08-28T20:00:00Z"


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_no_card_uses_model_internal_vocabulary(name):
    card = card_for(name)
    for path, text in decision_card._walk_prose(card):
        assert not decision_card.prose_violations(text), (path, text)


@pytest.mark.parametrize("word", ["composite", "ML score", "ensemble", "conformal",
                                  "random forest", "z-score", "expected value"])
def test_the_validator_catches_a_banned_word_anywhere(word):
    card = card_for("no_action")
    card["decisions"][0]["invalidation_trigger"] = f"a change in the {word}"
    with pytest.raises(decision_card.CardRejected):
        decision_card.validate(card)


def test_unqualified_confidence_is_refused_but_the_disclaimer_is_not():
    card = card_for("no_action")
    card["decisions"][0]["invalidation_trigger"] = "our confidence drops"
    with pytest.raises(decision_card.CardRejected):
        decision_card.validate(card)
    card["decisions"][0]["invalidation_trigger"] = (
        "the share of winning weeks falls — it is not a calibrated confidence")
    decision_card.validate(card)


def test_the_word_confidence_appears_nowhere_unqualified():
    for name in ALL_FIXTURES:
        rendered = json.dumps(card_for(name), default=str).lower()
        assert rendered.count("confidence") == rendered.count("not a calibrated confidence")


# --------------------------------------------------------------------------- #
# Validator: the structural rules, asserted directly
# --------------------------------------------------------------------------- #
def test_an_actionable_row_without_a_risk_is_refused():
    card = card_for("post_draft")
    row = next(d for d in card["decisions"] if d["status"] in decision_card.ACTIONABLE_STATUSES)
    row["risk"] = None
    with pytest.raises(decision_card.CardRejected):
        decision_card.validate(card)


def test_an_actionable_row_without_an_invalidation_trigger_is_refused():
    card = card_for("post_draft")
    row = next(d for d in card["decisions"] if d["status"] in decision_card.ACTIONABLE_STATUSES)
    row["invalidation_trigger"] = "  "
    with pytest.raises(decision_card.CardRejected):
        decision_card.validate(card)


def test_a_driver_without_provenance_is_refused():
    card = card_for("no_action")
    card["decisions"][0]["drivers"] = [{"text": "he is good", "source": "", "as_of": ""}]
    with pytest.raises(decision_card.CardRejected):
        decision_card.validate(card)


def test_an_unknown_status_is_refused():
    card = card_for("no_action")
    card["decisions"][0]["status"] = "buy"
    with pytest.raises(decision_card.CardRejected):
        decision_card.validate(card)


def test_a_card_may_not_be_relabelled_public():
    card = card_for("no_action")
    card["visibility"] = "public"
    with pytest.raises(decision_card.CardRejected):
        decision_card.validate(card)


# --------------------------------------------------------------------------- #
# Determinism, and the optional rewrite that cannot change a number
# --------------------------------------------------------------------------- #
def test_the_default_path_uses_no_language_model():
    card = card_for("post_draft")
    assert card["prose_rewrite"] == {
        "applied": False,
        "reason": "the default path composes every sentence in code",
    }


def test_two_builds_of_the_same_inputs_are_identical():
    a = json.dumps(card_for("post_draft"), sort_keys=True, default=str)
    b = json.dumps(card_for("post_draft"), sort_keys=True, default=str)
    assert a == b


def test_a_rewrite_that_keeps_the_numbers_is_accepted():
    card = card_for("post_draft")
    rewritten = decision_card.apply_prose_rewrite(card, lambda text: text + " Check inactives.")
    assert rewritten["prose_rewrite"]["applied"] is True
    assert rewritten["prose_rewrite"]["rewritten"] > 0
    assert rewritten["prose_rewrite"]["rejected"] == []
    for before, after in zip(card["decisions"], rewritten["decisions"]):
        assert before["mean_delta"] == after["mean_delta"]
        assert before["subject"] == after["subject"]


def test_a_rewrite_that_moves_a_number_is_rejected_and_the_original_survives():
    card = card_for("post_draft")
    rewritten = decision_card.apply_prose_rewrite(card, lambda text: text.replace("4", "9"))
    assert rewritten["prose_rewrite"]["rejected"]
    assert all("number" in item["reason"] or "identifier" in item["reason"]
               for item in rewritten["prose_rewrite"]["rejected"])
    for before, after in zip(card["decisions"], rewritten["decisions"]):
        assert before["headline"] == after["headline"]


def test_a_rewrite_that_drops_a_player_name_is_rejected():
    card = card_for("post_draft")
    starts = [d for d in card["decisions"] if d["status"] == "start"]
    if not starts:
        pytest.skip("no actionable start in this fixture")
    name = starts[0]["subject"]["name"]
    rewritten = decision_card.apply_prose_rewrite(card, lambda text: text.replace(name, "him"))
    assert any("identifier" in item["reason"] for item in rewritten["prose_rewrite"]["rejected"])


def test_a_rewrite_that_introduces_banned_wording_is_rejected():
    card = card_for("no_action")
    rewritten = decision_card.apply_prose_rewrite(
        card, lambda text: text + " per the ensemble.")
    assert any("wording this card does not show" in item["reason"]
               for item in rewritten["prose_rewrite"]["rejected"])
    # And the rejection itself does not smuggle the word onto the card.
    assert "ensemble" not in json.dumps(rewritten, default=str)
    decision_card.validate(rewritten)


def test_a_rewriter_that_raises_is_not_fatal():
    card = card_for("no_action")

    def boom(_text):
        raise RuntimeError("no model available")

    rewritten = decision_card.apply_prose_rewrite(card, boom)
    assert rewritten["prose_rewrite"]["rejected"]
    assert rewritten["decisions"][0]["headline"] == card["decisions"][0]["headline"]


# --------------------------------------------------------------------------- #
# Recommendation only
# --------------------------------------------------------------------------- #
def test_the_card_layer_contains_no_platform_write_path():
    for module in ("decision_card.py", "decision_page.py"):
        source = (ROOT / "nflvalue" / "fantasy" / module).read_text()
        assert "espn_client" not in source
        assert "requests" not in source
        for verb in ("urlopen", "http.client", ".post(", ".put(", ".delete("):
            assert verb not in source, (module, verb)


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_every_card_says_it_writes_nothing(name):
    assert "never" in card_for(name)["espn_use"].lower()
