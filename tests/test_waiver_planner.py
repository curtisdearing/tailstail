"""Waiver / free-agent planner — legality, provenance and refusal contracts.

Written test-first. The planner is RECOMMENDATION-ONLY: it never executes a
claim, never writes to ESPN, and every record it emits says so. These tests
pin the rules that make that safe rather than merely intended:

  * the live league payload is canonical — inverse-standings priority is the
    default only when the payload does not state a waiver order, and that
    assumption is recorded rather than silently applied;
  * FAAB fields exist only when the league actually uses FAAB;
  * a recommendation may never add a rostered or unavailable player, and may
    never drop a locked or undroppable one;
  * when nothing is legally droppable the record says `no_legal_drop` instead
    of inventing one;
  * stale free-agent data degrades the output visibly instead of producing
    confident-looking adds;
  * K/D-ST candidates stay labelled shadow and never become the objective;
  * no benefit ⇒ no recommendation.

Nothing here promotes anything into production lineup optimization.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from nflvalue.fantasy import espn_league as EL
from nflvalue.fantasy import waiver_rules as LC
from nflvalue.fantasy import waivers as WV

UTC = timezone.utc
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "leagues")
NOW = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


def payload(name):
    with open(os.path.join(FIXTURES, f"{name}.json")) as fh:
        return json.load(fh)


#: The league fixtures here are raw settings payloads. Waiver rules are now a
#: *view over the canonical snapshot*, so the fixture is projected into that
#: shape rather than parsed a second time — the point of the change being that
#: there is exactly one parser, and it lives in the adapter.
def snapshot(name="inverse_standings"):
    raw = payload(name)
    settings = raw["settings"]
    acquisition = settings.get("acquisitionSettings") or {}
    counts = {EL.SLOT_NAMES[int(slot)]: int(count)
              for slot, count in settings["rosterSettings"]["lineupSlotCounts"].items()}
    uses_faab = bool(acquisition.get("isUsingAcquisitionBudget"))
    reset = acquisition.get("waiverOrderReset")
    mode = (EL.WAIVER_MODE_FAAB if uses_faab
            else EL.WAIVER_MODE_INVERSE if reset
            else EL.WAIVER_MODE_ROLLING)
    return {
        "schema_version": EL.SCHEMA_VERSION,
        "retrieved_at": NOW.isoformat().replace("+00:00", "Z"),
        "league": {"league_id": raw["id"], "season": raw["seasonId"],
                   "current_scoring_period": raw["scoringPeriodId"],
                   "name": settings.get("name"), "size": len(raw.get("teams") or [])},
        "roster_settings": {"lineup_slot_counts": counts},
        "waivers": {
            "mode": mode,
            "order_reset": bool(reset),
            "uses_acquisition_budget": uses_faab,
            "acquisition_budget": float(acquisition.get("acquisitionBudget") or 0.0),
            "acquisition_limit": acquisition.get("acquisitionLimit", -1),
            "team_priority": {str(t["id"]): int(t.get("waiverRank") or 0)
                              for t in raw.get("teams") or []},
            "transaction_deadline": (raw.get("status") or {}).get("transactionDeadline"),
        },
        "transactions": {
            "pending": [
                {"transaction_id": str(txn.get("id") or ""), "type": txn.get("type"),
                 "status": txn.get("status"), "team_id": int(txn.get("teamId") or 0),
                 "items": list(txn.get("items") or [])}
                for txn in raw.get("pendingTransactions") or []
            ],
            "completed": [],
        },
        "hashes": {"league": "league-hash", "scoring": "scoring-hash", "roster": "roster-hash"},
    }


def contract(name="inverse_standings"):
    return LC.from_snapshot(snapshot(name), as_of=NOW)


# --------------------------------------------------------------------------- #
# Roster / pool helpers — stable ESPN IDs throughout
# --------------------------------------------------------------------------- #
def roster(*entries):
    return [WV.RosterEntry(**e) for e in entries]


def _r(espn_id, position, **over):
    base = dict(espn_id=espn_id, name=f"P{espn_id}", position=position,
                slot="BE", locked=False, undroppable=False,
                injury_status="ACTIVE", on_ir=False)
    base.update(over)
    return base


def _p(espn_id, position, **over):
    base = dict(espn_id=espn_id, name=f"F{espn_id}", position=position,
                availability="freeagent", waiver_process_time=None,
                as_of=NOW)
    base.update(over)
    return base


def pool(*entries):
    return [WV.PoolEntry(**e) for e in entries]


FULL_ROSTER = roster(
    _r(1, "QB", slot="QB"), _r(2, "RB", slot="RB"), _r(3, "RB", slot="RB"),
    _r(4, "WR", slot="WR"), _r(5, "WR", slot="WR"), _r(6, "TE", slot="TE"),
    _r(7, "K", slot="K"), _r(8, "D/ST", slot="D/ST"),
    _r(9, "RB"), _r(10, "WR"), _r(11, "TE"), _r(12, "QB"), _r(13, "WR"),
)


# --------------------------------------------------------------------------- #
# 1. League contract — the live payload is canonical
# --------------------------------------------------------------------------- #
def test_inverse_standings_priority_is_read_from_the_payload():
    c = contract("inverse_standings")
    assert c.waiver_mode == LC.WAIVER_INVERSE_STANDINGS
    assert c.waiver_mode_assumed is False, \
        "the payload states the order — nothing may be assumed"
    assert c.uses_faab is False


def test_priority_order_is_worst_record_first():
    c = contract("inverse_standings")
    # team 1 is 0-2, team 3 is 1-1, team 2 is 2-0
    assert c.priority_order == (1, 3, 2)


def test_priority_ties_break_deterministically_and_are_flagged():
    c = contract("priority_tie")
    assert c.priority_order == (1, 2, 3), "0-2 teams ahead of the 1-1 team"
    assert 1 in c.priority_tied_teams and 2 in c.priority_tied_teams, \
        "an unbroken tie must be surfaced, not silently ordered"
    a = LC.from_snapshot(snapshot("priority_tie"), as_of=NOW)
    b = LC.from_snapshot(snapshot("priority_tie"), as_of=NOW)
    assert a.priority_order == b.priority_order, "tie-break must be deterministic"


def test_faab_league_overrides_the_inverse_default():
    c = contract("faab")
    assert c.uses_faab is True
    assert c.faab_budget == 100
    assert c.waiver_mode == LC.WAIVER_FAAB


def test_a_snapshot_that_states_no_waiver_mode_is_refused_not_defaulted():
    """The old behaviour guessed inverse standings and set an `assumed` flag.

    Nothing read the flag. A snapshot now always states its mode because the
    adapter refuses a payload whose `waiverOrderReset` is absent, so the only
    way to reach this code without one is a hand-built snapshot — and that is
    refused rather than defaulted.
    """
    modeless = snapshot("waiver_order_absent")
    modeless["waivers"]["mode"] = ""
    with pytest.raises(LC.ContractError, match="mode"):
        LC.from_snapshot(modeless, as_of=NOW)


def test_contract_hashes_are_stable_and_separate():
    a, b = contract(), contract()
    assert a.scoring_hash == b.scoring_hash
    assert a.roster_hash == b.roster_hash
    assert a.scoring_hash != a.roster_hash
    assert contract("faab").scoring_hash == a.scoring_hash, \
        "FAAB changes acquisition, not scoring"


def test_transaction_deadline_is_parsed_and_enforced():
    c = contract("deadline_passed")
    assert c.transaction_deadline == datetime(2026, 9, 1, 18, 0, tzinfo=UTC)
    assert c.transactions_open(NOW) is False
    assert contract().transactions_open(NOW) is True


def test_competing_pending_claims_are_visible_per_player():
    c = contract("competing_claims")
    assert set(c.pending_claims_for(9001)) == {2, 3}
    assert c.pending_claims_for(1234) == ()


# --------------------------------------------------------------------------- #
# 2. Legality — the rules that make recommendation-only safe
# --------------------------------------------------------------------------- #
def test_never_recommends_adding_a_rostered_player():
    c = contract()
    avail = WV.addable(c, pool(_p(9, "RB", availability="rostered")),
                       FULL_ROSTER, now=NOW)
    assert avail == []


def test_a_player_on_waivers_is_claimable_but_is_not_an_immediate_add():
    """The distinction the planner has to keep, not collapse.

    This previously asserted that a player whose waivers clear tomorrow could
    not be acted on at all. That is backwards: the pending window is exactly
    when a claim is placed. What he is *not* is an immediate add, and that is
    what the two lists below separate.
    """
    c = contract()
    claimable = pool(_p(500, "WR", availability="waivers",
                        waiver_process_time=NOW + timedelta(days=1)))
    assert WV.addable(c, claimable, FULL_ROSTER, now=NOW) == claimable
    assert WV.waiver_claims(claimable) == claimable
    assert WV.immediate_free_agents(claimable) == []


def test_a_rostered_player_is_never_addable():
    c = contract()
    assert WV.addable(c, pool(_p(501, "WR", availability="onteam")),
                      FULL_ROSTER, now=NOW) == []


def test_no_adds_at_all_once_the_transaction_deadline_passes():
    c = contract("deadline_passed")
    assert WV.addable(c, pool(_p(500, "WR")), FULL_ROSTER, now=NOW) == []


def test_locked_and_undroppable_players_are_never_drop_candidates():
    c = contract()
    r = roster(_r(1, "QB", slot="QB"), _r(2, "RB", locked=True),
               _r(3, "WR", undroppable=True), _r(4, "TE"))
    ids = [e.espn_id for e in WV.droppable(c, r, now=NOW)]
    assert 2 not in ids and 3 not in ids
    assert ids == [1, 4]


def test_no_legal_drop_is_reported_rather_than_invented():
    c = contract()
    r = roster(_r(1, "QB", locked=True), _r(2, "RB", undroppable=True))
    recs = WV.plan(c, roster=r, pool=pool(_p(500, "WR")), now=NOW,
                   distributions=None)
    assert recs, "a full-but-undroppable roster still yields a reported option"
    assert recs[0].drop_espn_id is None
    assert recs[0].drop_state == WV.NO_LEGAL_DROP


def test_ir_stash_is_only_legal_for_an_ir_eligible_status():
    c = contract()
    eligible = WV.ir_eligible(c, WV.RosterEntry(**_r(20, "WR", injury_status="OUT")))
    not_eligible = WV.ir_eligible(c, WV.RosterEntry(**_r(21, "WR", injury_status="QUESTIONABLE")))
    assert eligible is True and not_eligible is False


def test_a_player_on_ir_is_not_a_drop_candidate_by_default():
    c = contract()
    r = roster(_r(1, "QB"), _r(30, "WR", on_ir=True, injury_status="OUT"))
    assert [e.espn_id for e in WV.droppable(c, r, now=NOW)] == [1]


def test_roster_limit_forces_a_drop_when_the_roster_is_full():
    c = contract()
    assert WV.roster_is_full(c, FULL_ROSTER) is True
    assert WV.roster_is_full(c, FULL_ROSTER[:5]) is False


# --------------------------------------------------------------------------- #
# 3. Recommendation record — shape, honesty, provenance
# --------------------------------------------------------------------------- #
def test_every_recommendation_is_marked_recommendation_only():
    recs = WV.plan(contract(), roster=FULL_ROSTER, pool=pool(_p(500, "WR")),
                   now=NOW, distributions=None)
    assert recs
    for rec in recs:
        d = rec.to_dict()
        assert d["recommendation_only"] is True
        for key in ("add_espn_id", "drop_espn_id", "drop_state", "confidence",
                    "rationale", "invalidation_trigger", "data_timestamps",
                    "priority_implications"):
            assert key in d, f"record missing {key}"


def test_faab_fields_are_absent_unless_the_league_uses_faab():
    plain = WV.plan(contract(), roster=FULL_ROSTER, pool=pool(_p(500, "WR")),
                    now=NOW, distributions=None)[0].to_dict()
    assert "faab_bid" not in plain and "faab_budget_remaining" not in plain

    faab = WV.plan(contract("faab"), roster=FULL_ROSTER,
                   pool=pool(_p(500, "WR")), now=NOW,
                   distributions=None)[0].to_dict()
    assert "faab_bid" in faab


def test_missing_distributions_yield_an_explicit_unavailable_not_a_zero():
    rec = WV.plan(contract(), roster=FULL_ROSTER, pool=pool(_p(500, "WR")),
                  now=NOW, distributions=None)[0]
    d = rec.to_dict()
    assert d["lineup_delta"] is None
    assert "unavailable" in d["lineup_delta_status"].lower()
    assert rec.confidence == "none", \
        "no distribution means no confidence, never a default"


def test_opponent_impact_is_secondary_and_never_the_objective():
    rec = WV.plan(contract(), roster=FULL_ROSTER, pool=pool(_p(500, "WR")),
                  now=NOW, distributions=None)[0]
    d = rec.to_dict()
    assert "opponent_opportunity_impact" in d
    assert d["objective"] == "own_optimal_lineup_delta", \
        "denying an opponent may never become the ranking objective"


def test_k_and_dst_candidates_stay_labelled_shadow():
    recs = WV.plan(contract(), roster=FULL_ROSTER,
                   pool=pool(_p(600, "K"), _p(601, "D/ST")),
                   now=NOW, distributions=None)
    assert recs
    for rec in recs:
        d = rec.to_dict()
        assert d["status"] == "shadow"
        assert d["shadow_reason"]
    plain = WV.plan(contract(), roster=FULL_ROSTER, pool=pool(_p(500, "WR")),
                    now=NOW, distributions=None)[0].to_dict()
    # Without joint samples the add cannot be valued, so it is not a
    # recommendation either -- see the gate tests below.
    assert plain["status"] == "no_current_pick"


def test_stale_free_agent_data_degrades_visibly_and_recommends_nothing():
    stale_as_of = NOW - timedelta(hours=WV.MAX_POOL_AGE_HOURS + 1)
    recs = WV.plan(contract(), roster=FULL_ROSTER,
                   pool=pool(_p(500, "WR", as_of=stale_as_of)),
                   now=NOW, distributions=None)
    assert len(recs) == 1
    d = recs[0].to_dict()
    assert d["degraded"] is True
    assert d["add_espn_id"] is None
    assert "stale" in d["rationale"].lower()


def test_no_benefit_returns_no_recommendation():
    recs = WV.plan(contract(), roster=FULL_ROSTER, pool=pool(), now=NOW,
                   distributions=None)
    assert recs == []


def test_priority_implications_reference_the_real_order():
    rec = WV.plan(contract(), roster=FULL_ROSTER, pool=pool(_p(500, "WR")),
                  now=NOW, distributions=None, my_team_id=1)[0]
    d = rec.to_dict()
    assert d["priority_implications"]["my_priority"] == 1
    assert d["priority_implications"]["mode"] == LC.WAIVER_INVERSE_STANDINGS


def test_competing_claims_are_surfaced_on_the_recommendation():
    c = contract("competing_claims")
    rec = WV.plan(c, roster=FULL_ROSTER, pool=pool(_p(9001, "WR")), now=NOW,
                  distributions=None, my_team_id=1)[0]
    assert rec.to_dict()["priority_implications"]["competing_claims"] == [2, 3]


# --------------------------------------------------------------------------- #
# 4. The planner cannot write to ESPN — structural, not aspirational
# --------------------------------------------------------------------------- #
def test_no_write_capable_call_exists_in_the_planner_modules():
    import inspect
    banned = ("requests.post", "requests.put", "urlopen", "add_player",
              "drop_player", "execute", "submit", "claim(", "espn_request")
    for module in (WV, LC):
        src = inspect.getsource(module)
        for token in banned:
            assert token not in src, f"{module.__name__} references {token!r}"


def test_plan_is_pure_and_leaves_no_files_behind(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    WV.plan(contract(), roster=FULL_ROSTER, pool=pool(_p(500, "WR")),
            now=NOW, distributions=None)
    assert list(tmp_path.iterdir()) == []


def test_repeated_planning_is_deterministic():
    kwargs = dict(roster=FULL_ROSTER, pool=pool(_p(500, "WR"), _p(501, "RB")),
                  now=NOW, distributions=None, my_team_id=1)
    a = [r.to_dict() for r in WV.plan(contract(), **kwargs)]
    b = [r.to_dict() for r in WV.plan(contract(), **kwargs)]
    assert a == b


@pytest.mark.parametrize("missing", ["roster", "pool"])
def test_missing_team_state_refuses_rather_than_guessing(missing):
    kwargs = dict(roster=FULL_ROSTER, pool=pool(_p(500, "WR")), now=NOW,
                  distributions=None)
    kwargs[missing] = None
    with pytest.raises(WV.PlannerError, match="required"):
        WV.plan(contract(), **kwargs)


def test_a_freeagent_flagged_pool_row_that_is_already_rostered_is_still_refused():
    """The availability string is not the defence — the ID is.

    ESPN's pool can carry a row still marked `freeagent` for a player who is
    already on the roster (stale or duplicated feed). Filtering only on the
    availability label passes the obvious test and ships the bug, so this
    exercises the ID path directly.
    """
    c = contract()
    already_rostered_but_looks_free = _p(9, "RB", availability="freeagent")
    assert WV.addable(c, pool(already_rostered_but_looks_free),
                      FULL_ROSTER, now=NOW) == []

    recs = WV.plan(c, roster=FULL_ROSTER,
                   pool=pool(already_rostered_but_looks_free), now=NOW,
                   distributions=None)
    assert recs == [], "a rostered player may never surface as an add"


# --------------------------------------------------------------------------- #
# The gate: four conditions, all of them required
# --------------------------------------------------------------------------- #
def _samples(mapping, simulations=800, seed=11):
    """Joint draws: one shared week factor so teammates move together."""
    import numpy as np

    rng = np.random.default_rng(seed)
    shared = rng.normal(0.0, 1.0, simulations)
    return {int(pid): mean + 3.0 * shared + rng.normal(0.0, 2.0, simulations)
            for pid, mean in mapping.items()}


def _roster_samples(extra=None, **over):
    means = {int(e.espn_id): 9.0 for e in FULL_ROSTER}
    means.update(over)
    if extra:
        means.update(extra)
    return _samples(means)


def test_a_row_exists_only_when_the_joint_delta_clears_the_declared_gate():
    """An upgrade big enough to matter, valued from paired rows."""
    star = _p(700, "WR")
    draws = _roster_samples(extra={700: 22.0})
    recs = WV.plan(contract(), roster=FULL_ROSTER, pool=pool(star), now=NOW,
                   distributions=draws)
    picked = [r for r in recs if r.status == "recommendation"]
    assert picked, "a clear upgrade should clear the gate"
    delta = picked[0].lineup_delta
    assert delta["own_optimal_lineup_delta"] >= WV.WAIVER_GATE["min_mean_lineup_delta"]
    assert delta["model_relative_prob_improves"] >= WV.WAIVER_GATE["min_prob_improves"]
    assert delta["basis"] == "paired joint simulation rows, both lineups solved per row"
    assert picked[0].drop_espn_id is not None, "a row must name the drop it requires"


def test_a_marginal_add_does_not_become_a_recommendation():
    """Legal, addable, and not worth doing — which is not a recommendation."""
    spare = _p(701, "WR")
    draws = _roster_samples(extra={701: 1.0})
    recs = WV.plan(contract(), roster=FULL_ROSTER, pool=pool(spare), now=NOW,
                   distributions=draws)
    assert all(r.status != "recommendation" for r in recs)
    assert any("below the declared gate" in r.rationale for r in recs)


def test_the_drop_is_the_cheapest_legal_cut_not_the_highest_player_id():
    """The drop used to be `max(bench, key=espn_id)`, which values nothing."""
    star = _p(702, "WR")
    weakest = min((e for e in FULL_ROSTER if e.slot == "BE"), key=lambda e: int(e.espn_id))
    means = {int(e.espn_id): 14.0 for e in FULL_ROSTER}
    means[int(weakest.espn_id)] = 0.5
    draws = _samples({**means, 702: 21.0})
    recs = WV.plan(contract(), roster=FULL_ROSTER, pool=pool(star), now=NOW,
                   distributions=draws)
    picked = [r for r in recs if r.status == "recommendation"]
    assert picked
    assert picked[0].drop_espn_id == int(weakest.espn_id), (
        "the drop should be the player whose loss costs the lineup least")


def test_without_samples_no_drop_is_nominated_at_all():
    recs = WV.plan(contract(), roster=FULL_ROSTER, pool=pool(_p(703, "WR")), now=NOW,
                   distributions=None)
    assert recs[0].drop_state == WV.NO_VALUED_DROP
    assert recs[0].drop_espn_id is None
    assert "arbitrary cut" in recs[0].rationale


def test_the_gate_is_declared_ahead_of_any_run():
    for key in ("min_mean_lineup_delta", "min_prob_improves", "min_simulations"):
        assert key in WV.WAIVER_GATE
