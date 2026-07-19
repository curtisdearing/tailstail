"""Phase C invariants: the evidence layer explains, and never decides.

The failure mode this suite exists to prevent is an explanation layer that
quietly becomes a second model — by re-weighting, by inventing a strength
score, by filling an empty calibration bucket with a plausible number, or by
letting a language model round a figure into existence.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nflvalue import evidence, evidence_prose  # noqa: E402
from nflvalue.evidence_prose import TranslationRejected  # noqa: E402


def _lean(**overrides):
    lean = {
        "player_id": "00-A1", "name": "Test Receiver", "pos": "WR",
        "market": "receiving_yards", "side": "over", "line": 54.5,
        "mean": 67.027, "sd": 30.662, "p_over": 0.6109, "p_under": 0.3891,
        "composite": 37.9, "season": 2023, "week": 10,
        "game_id": "2023_10_ATL_ARI", "team": "ATL", "defteam": "ARI", "home": False,
        "spread_line": -2.0, "total_line": 43.5, "roll_games": 8.0,
        "line_source": "synthetic_trailing_mean",
        "proj_components": {"volume": 7.233, "efficiency": 7.1005,
                            "opp_factor": 1.3051, "game_script": 0.9815},
    }
    lean.update(overrides)
    return lean


def _player_week(**overrides):
    row = {"roll_targets": 6.5, "roll_carries": 1.0, "roll_pass_attempts": 0.0,
           "roll_games": 8.0, "role": "WR"}
    row.update(overrides)
    return row


CALIBRATION = {
    "bands": {"<35": {"n": 531, "hit_rate": 0.5782},
              "35-45": {"n": 648, "hit_rate": 0.5556},
              "45-55": {"n": 148, "hit_rate": 0.5743},
              "55+": {"n": 33, "hit_rate": 0.4848}},
    "overall": {"n": 1360, "hit_rate": 0.5647},
    "baseline": {"hit_rate": 0.4696},
    "framing": "directional grading at synthetic lines",
    "source": "lean_replay_2025.json",
}


# --------------------------------------------------------------------------- #
# C.1 -- decomposition
# --------------------------------------------------------------------------- #
def test_decomposition_reconstructs_the_reported_mean():
    out = evidence.decompose(_lean(), player_week_row=_player_week())
    assert out["reconstruction_error"] == pytest.approx(0.0, abs=1e-3)
    assert not [w for w in out["warnings"] if "does not reconstruct" in w]


def test_decomposition_flags_an_unreconstructable_mean():
    """An adjustment applied without a recorded multiplier must be visible."""
    lean = _lean(mean=99.0)
    out = evidence.decompose(lean, player_week_row=_player_week())
    assert any("does not reconstruct" in w for w in out["warnings"])


def test_optional_adjustment_multipliers_enter_the_decomposition():
    lean = _lean(realloc_mult=1.18, mean=67.027 * 1.18)
    out = evidence.decompose(lean, player_week_row=_player_week())
    ids = [d["id"] for d in out["drivers"]]
    assert "realloc_mult" in ids
    assert out["reconstruction_error"] == pytest.approx(0.0, abs=1e-3)


def test_drivers_are_ranked_by_absolute_log_contribution():
    out = evidence.decompose(_lean(), player_week_row=_player_week())
    magnitudes = [abs(d["log_contribution"]) for d in out["drivers"]]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert [d["rank"] for d in out["drivers"]] == list(range(1, len(out["drivers"]) + 1))


def test_direction_is_relative_to_the_side_taken():
    over = evidence.decompose(_lean(side="over"), player_week_row=_player_week())
    under = evidence.decompose(_lean(side="under"), player_week_row=_player_week())
    opp_over = next(d for d in over["drivers"] if d["id"] == "opp_factor")
    opp_under = next(d for d in under["drivers"] if d["id"] == "opp_factor")
    assert opp_over["direction"] == "supports"
    assert opp_under["direction"] == "opposes"


def test_level_only_driver_is_not_called_neutral():
    """Unknown contribution must not masquerade as a measured zero."""
    out = evidence.decompose(_lean(), player_week_row=_player_week())
    efficiency = next(d for d in out["drivers"] if d["id"] == "efficiency")
    assert efficiency["direction"] == "not_ranked"
    assert efficiency["notes"]


def test_clipped_opponent_factor_is_flagged_as_censored():
    lean = _lean(proj_components={"volume": 7.233, "efficiency": 7.1005,
                                  "opp_factor": 1.6, "game_script": 0.9815})
    out = evidence.decompose(lean, player_week_row=_player_week())
    opp = next(d for d in out["drivers"] if d["id"] == "opp_factor")
    assert any("clip bound" in note for note in opp["notes"])


def test_anytime_td_declares_that_it_cannot_be_separated():
    lean = _lean(market="anytime_td",
                 proj_components={"volume": 12.0, "efficiency": 0.05,
                                  "opp_factor": 1.0, "game_script": 1.0},
                 mean=0.6)
    out = evidence.decompose(lean, player_week_row=_player_week())
    assert any("cannot separate" in w for w in out["warnings"])


def test_missing_components_refuses_rather_than_recomputing():
    with pytest.raises(evidence.EvidenceUnavailable, match="recompute the model"):
        evidence.decompose({"player_id": "x", "market": "receiving_yards"})


def test_implied_team_total_matches_the_standard_convention():
    assert evidence.implied_team_total(_lean(total_line=43.5, spread_line=-2.0, home=False)) == \
        pytest.approx(43.5 / 2 + 2.0 / 2)


# --------------------------------------------------------------------------- #
# C.2 -- evidence strength comes from the registries, not from here
# --------------------------------------------------------------------------- #
def test_zero_sample_proposed_factor_is_never_attached_as_evidence():
    """A `proposed` study with an empty cohort is a plan, not evidence."""
    registries = [{"id": "fam_A03_opp_wr_points_soft", "status": "proposed",
                   "n_raw": {"exposed": 0, "control": 0}, "n_effective": 0,
                   "component_node": "d_fp", "_registry": "families"}]
    driver = {"id": "opp_factor"}
    out = evidence.strength_for_driver(driver, registries, _player_week(), "WR")
    assert out["status"] == "NO_REGISTERED_EVIDENCE"


def test_registered_factor_reports_n_effective_posterior_and_forward():
    registries = [{"id": "fam_A03_opp_wr_points_soft", "status": "research_only",
                   "n_raw": {"exposed": 4765, "control": 4263}, "n_effective": 1279,
                   "effect": {"point": 1.2, "ci95": [0.8, 1.6], "p": 0.001},
                   "posterior": {"mean": 1.1, "ci95": [0.7, 1.5]},
                   "multiplicity_q": 0.006,
                   "season_forward": {"sign_agreement_2023_25": "3/3",
                                      "magnitude_order_holds": True},
                   "component_node": "d_fp", "_registry": "families"}]
    out = evidence.strength_for_driver({"id": "opp_factor"}, registries, _player_week(), "WR")
    assert out["status"] == "REGISTERED"
    assert out["n_raw"] == {"exposed": 4765, "control": 4263}
    assert out["n_effective"] == 1279
    assert out["posterior"]["ci95"] == [0.7, 1.5]
    assert out["gates"]["season_forward_replicates"] is True
    assert out["strength_label"] in ("strong", "suggestive")


def test_a_big_effect_on_a_tiny_sample_reads_weak_not_strong():
    """The C.2 requirement, stated directly."""
    registries = [{"id": "fam_A03_opp_wr_points_soft", "status": "research_only",
                   "n_raw": {"exposed": 30, "control": 28}, "n_effective": 12,
                   "effect": {"point": 9.9, "ci95": [-2.0, 21.0], "p": 0.31},
                   "posterior": {"mean": 0.4, "ci95": [-1.9, 2.7]},
                   "multiplicity_q": 0.42,
                   "season_forward": {"sign_agreement_2023_25": "1/3",
                                      "magnitude_order_holds": False},
                   "component_node": "d_fp", "_registry": "families"}]
    out = evidence.strength_for_driver({"id": "opp_factor"}, registries, _player_week(), "WR")
    assert out["strength_label"] in ("weak", "unsupported")
    assert out["gates"]["meets_protocol_n"] is False
    assert out["gates"]["posterior_ci_excludes_zero"] is False


def test_thin_player_history_reads_thin_after_shrinkage():
    thin = evidence.strength_for_driver({"id": "efficiency"}, [], _player_week(roll_games=3.0))
    thick = evidence.strength_for_driver({"id": "efficiency"}, [], _player_week(roll_games=8.0))
    assert thin["strength_label"] == "thin"
    assert thick["strength_label"] == "adequate"
    assert thin["n_effective"] < thick["n_effective"]
    assert thin["shrinkage"]["k"] == evidence.EFFICIENCY_SHRINK_K


def test_non_player_driver_does_not_borrow_the_player_sample():
    """Opponent strength has no player-games interpretation."""
    out = evidence.strength_for_driver({"id": "game_script"}, [], _player_week(), "WR")
    assert out["status"] == "NO_REGISTERED_EVIDENCE"
    assert "n_raw" not in out


# --------------------------------------------------------------------------- #
# C.5 -- calibration band, or an explicit refusal
# --------------------------------------------------------------------------- #
def test_measured_band_reports_rate_with_an_interval():
    band = evidence.calibration_band(_lean(composite=37.9), CALIBRATION)
    assert band["status"] == "MEASURED"
    assert band["band"] == "35-45"
    assert band["n"] == 648
    assert band["ci95"][0] < band["hit_rate"] < band["ci95"][1]


def test_thin_bucket_refuses_to_report_a_rate():
    band = evidence.calibration_band(_lean(composite=60.0), CALIBRATION)
    assert band["status"] == "UNMEASURED_BUCKET"
    assert band["n"] == 33
    assert "hit_rate" not in band and "ci95" not in band
    assert str(evidence.MIN_BUCKET_N) in band["reason"]


def test_composite_outside_every_measured_band_is_unmeasured():
    band = evidence.calibration_band(_lean(composite=None), CALIBRATION)
    assert band["status"] == "UNMEASURED_BUCKET"


# --------------------------------------------------------------------------- #
# C.3 / C.4 -- counter-evidence and falsifiers
# --------------------------------------------------------------------------- #
def test_counter_evidence_renders_even_when_nothing_is_wrong():
    payload = evidence.build_evidence(_lean(), player_week_row=_player_week(),
                                      registries=[], calibration=CALIBRATION, regimes={})
    assert isinstance(payload["counter_evidence"], list)
    prose = evidence_prose.translate(payload)["prose"]
    assert "What argues against it:" in prose


def test_unmeasured_bucket_appears_as_counter_evidence():
    payload = evidence.build_evidence(_lean(composite=60.0), player_week_row=_player_week(),
                                      registries=[], calibration=CALIBRATION, regimes={})
    kinds = {item["kind"] for item in payload["counter_evidence"]}
    assert "unmeasured_bucket" in kinds


def test_contradicting_driver_is_listed_against_the_selection():
    lean = _lean(side="under")   # every up-driver now argues the other way
    payload = evidence.build_evidence(lean, player_week_row=_player_week(),
                                      registries=[], calibration=CALIBRATION, regimes={})
    kinds = [item["kind"] for item in payload["counter_evidence"]]
    assert "contradicting_driver" in kinds


def test_rejected_registry_factor_becomes_counter_evidence():
    registries = [{"id": "fam_A03_opp_wr_points_soft", "status": "rejected",
                   "status_reason": "adjustment gate failed",
                   "n_raw": {"exposed": 4765, "control": 4263}, "n_effective": 1279,
                   "posterior": {"mean": 1.1, "ci95": [0.7, 1.5]}, "multiplicity_q": 0.006,
                   "season_forward": {"sign_agreement_2023_25": "2/3"},
                   "component_node": "d_fp", "_registry": "families"}]
    payload = evidence.build_evidence(_lean(), player_week_row=_player_week(),
                                      registries=registries, calibration=CALIBRATION, regimes={})
    kinds = {item["kind"] for item in payload["counter_evidence"]}
    assert "rejected_factor" in kinds
    assert "failed_replication" in kinds


def test_missing_freshness_is_itself_counter_evidence():
    payload = evidence.build_evidence(_lean(), player_week_row=_player_week(),
                                      registries=[], calibration=CALIBRATION, regimes={})
    assert any(item["kind"] == "freshness_unknown" for item in payload["counter_evidence"])


def test_stale_feed_is_reported_with_its_age_and_limit():
    from nflvalue.freshness import Feed
    feeds = [Feed(name="inactives", timestamp="2026-09-10T00:00:00Z", n_records=5)]
    payload = evidence.build_evidence(_lean(), player_week_row=_player_week(),
                                      registries=[], calibration=CALIBRATION, regimes={},
                                      feeds=feeds, as_of="2026-09-14T00:00:00Z")
    gaps = [i for i in payload["counter_evidence"] if i["kind"] == "freshness_gap"]
    assert gaps and gaps[0]["feed"] == "inactives"
    assert gaps[0]["age_hours"] > gaps[0]["limit_hours"]


def test_falsifiers_are_always_present_and_name_an_observable():
    payload = evidence.build_evidence(_lean(realloc_mult=1.18, mean=67.027 * 1.18),
                                      player_week_row=_player_week(),
                                      registries=[], calibration=CALIBRATION, regimes={})
    assert payload["falsifiers"]
    assert all(item.get("observable_by") for item in payload["falsifiers"])
    realloc = next(i for i in payload["falsifiers"] if i["driver"] == "realloc_mult")
    assert "activated" in realloc["statement"]
    assert realloc["gap"]          # the absent teammate's identity is not recorded


# --------------------------------------------------------------------------- #
# The layer must not decide anything
# --------------------------------------------------------------------------- #
def test_building_evidence_does_not_mutate_the_selection():
    lean = _lean()
    before = copy.deepcopy(lean)
    evidence.build_evidence(lean, player_week_row=_player_week(),
                            registries=[], calibration=CALIBRATION, regimes={})
    assert lean == before


def test_payload_declares_it_does_not_feed_live_scoring():
    payload = evidence.build_evidence(_lean(), player_week_row=_player_week(),
                                      registries=[], calibration=CALIBRATION, regimes={})
    assert payload["provenance"]["live_scoring_from_registries"] is False


# --------------------------------------------------------------------------- #
# C.6 -- the translator is verified, not trusted
# --------------------------------------------------------------------------- #
@pytest.fixture()
def payload():
    return evidence.build_evidence(_lean(), player_week_row=_player_week(),
                                   registries=[], calibration=CALIBRATION, regimes={})


def test_rule_based_prose_passes_both_guards(payload):
    result = evidence_prose.translate(payload)
    assert result["source"] == "rule_based"
    assert result["verified"] == {"numerals_grounded": True, "driver_order_preserved": True}


def test_every_numeral_in_the_prose_appears_in_the_payload(payload):
    """The C.6 test, stated literally."""
    prose = evidence_prose.translate(payload)["prose"]
    allowed = evidence_prose.payload_numerals(payload)
    for token in evidence_prose._numerals(prose):
        assert evidence_prose._normalize(token) in allowed, f"ungrounded numeral {token!r}"


def test_a_fabricated_number_is_rejected(payload):
    def lying_client(_prompt):
        return "The model projects 67.027 and this hits 73% of the time."
    with pytest.raises(TranslationRejected, match="absent from the evidence payload"):
        evidence_prose.translate(payload, client=lying_client)


def test_reordering_the_drivers_is_rejected(payload):
    labels = [d["label"] for d in payload["decomposition"]["drivers"]]

    def reordering_client(_prompt):
        return " ".join(reversed(labels))
    with pytest.raises(TranslationRejected, match="reorders the drivers"):
        evidence_prose.translate(payload, client=reordering_client)


def test_a_faithful_client_is_accepted(payload):
    def honest_client(_prompt):
        labels = [d["label"] for d in payload["decomposition"]["drivers"]]
        return "Drivers in order: " + ", ".join(labels) + "."
    assert evidence_prose.translate(payload, client=honest_client)["source"] == "client"


def test_percentage_renderings_of_a_payload_value_are_grounded(payload):
    """0.1305 may legitimately render as 13.1%; 73% may not."""
    allowed = evidence_prose.payload_numerals(payload)
    assert evidence_prose._normalize("30.51") in allowed     # from opp_factor 1.3051
    assert evidence_prose._normalize("73") not in allowed


def test_prompt_sends_the_payload_and_forbids_new_numbers(payload):
    prompt = evidence_prose._default_prompt(payload)
    assert "Use ONLY numbers that appear in the JSON" in prompt
    assert "Do not re-rank" in prompt
    assert json.dumps(payload["selection"]["player_id"]) in prompt
