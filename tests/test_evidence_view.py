"""Phase D invariants: the interface can say no, and cannot oversell.

Each test corresponds to a way a screening tool degrades into a tipping service:
an empty week rendered as a table with the least-bad option in it, counter-
evidence tucked behind a toggle, a dismissible caveat, a stake size with no loss
distribution, or copy that claims an edge the data cannot support.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nflvalue import evidence, evidence_view, language_guard  # noqa: E402
from nflvalue.evidence_view import StakingRefused  # noqa: E402

INSUFFICIENT = {"verdict": "INSUFFICIENT_SAMPLE", "n_resolved": 0, "n_unresolved": 4,
                "min_sample": 150, "precommitment_id": "tailstail-clv-killcheck-2026-v1",
                "detail": "0 resolved pairs of the 150 precommitted."}
BANDS = {"<35": {"n": 531, "hit_rate": 0.5782}, "35-45": {"n": 648, "hit_rate": 0.5556},
         "45-55": {"n": 148, "hit_rate": 0.5743}, "55+": {"n": 33, "hit_rate": 0.4848}}
REGIMES = {"role_increase_5_plus": {"calibrated_monte_carlo":
                                    {"n": 1676, "mae": 6.2, "coverage80": 0.72}},
           "stable_role_abs_lt_3": {"calibrated_monte_carlo":
                                    {"n": 6247, "mae": 4.9, "coverage80": 0.812}}}


def _lean(**overrides):
    lean = {"player_id": "00-A1", "name": "Test Receiver", "pos": "WR",
            "market": "receiving_yards", "side": "over", "line": 54.5, "mean": 67.027,
            "sd": 30.662, "p_over": 0.6109, "p_under": 0.3891, "composite": 37.9,
            "season": 2023, "week": 10, "game_id": "g", "team": "ATL", "defteam": "ARI",
            "home": False, "spread_line": -2.0, "total_line": 43.5, "roll_games": 8.0,
            "line_source": "synthetic_trailing_mean",
            "proj_components": {"volume": 7.233, "efficiency": 7.1005,
                                "opp_factor": 1.3051, "game_script": 0.9815}}
    lean.update(overrides)
    return lean


def _payload(**overrides):
    calibration = {"bands": BANDS, "overall": {"n": 1360, "hit_rate": 0.5647},
                   "baseline": {"hit_rate": 0.4696}, "source": "lean_replay_2025.json"}
    return evidence.build_evidence(_lean(**overrides),
                                   player_week_row={"roll_targets": 6.5, "roll_games": 8.0,
                                                    "role": "WR"},
                                   registries=[], calibration=calibration, regimes={})


def visible(markup: str) -> str:
    return language_guard.strip_markup(markup)


# --------------------------------------------------------------------------- #
# D.1 -- the no-bet state
# --------------------------------------------------------------------------- #
def test_no_bet_state_states_the_reason_and_is_not_an_empty_table():
    page = evidence_view.render_page(
        season=2026, week=3, as_of="2026-09-20T12:00:00Z", clv_report=INSUFFICIENT,
        selections=[], screened=214,
        no_bet_reasons=["No candidate cleared the composite screen."],
        bands=BANDS, regimes=REGIMES)
    text = visible(page)
    assert "No qualifying selections this week." in text
    assert "214 candidates were screened" in text
    assert "No candidate cleared the composite screen." in text
    assert "successful week" in text
    assert "<table" not in page.split('id="selections"')[1].split("</section>")[0]


def test_no_bet_state_does_not_promote_a_fallback_selection():
    page = evidence_view.render_page(
        season=2026, week=3, as_of="t", clv_report=INSUFFICIENT, selections=[],
        screened=214, no_bet_reasons=["nothing cleared"],
        nearest={"name": "A Player", "market": "receiving_yards", "composite": 55.76,
                 "failed_because": "its band has n=33"},
        bands=BANDS, regimes=REGIMES)
    text = visible(page)
    assert "not a selection" in text
    assert "not offered as a fallback call" in text
    assert "class=\"card\"" not in page      # no selection card is emitted


def test_selection_state_replaces_the_no_bet_screen():
    page = evidence_view.render_page(
        season=2023, week=10, as_of="t", clv_report=INSUFFICIENT,
        selections=[_payload()], screened=70, bands=BANDS, regimes=REGIMES)
    assert "No qualifying selections" not in visible(page)
    assert 'class="card"' in page


# --------------------------------------------------------------------------- #
# D.2 -- the selection card
# --------------------------------------------------------------------------- #
def test_card_top_line_carries_band_n_interval_and_denominator():
    card = evidence_view.selection_card(_payload(), screened=70, as_of="2023-11-08T12:00:00Z")
    text = visible(card)
    assert "band 35-45" in text and "n=648" in text and "95% CI" in text
    assert "70 screened" in text


def test_card_ranks_drivers_and_shows_sample_size_per_driver():
    card = evidence_view.selection_card(_payload(), screened=70)
    text = visible(card)
    assert "Drivers, ranked by contribution" in text
    assert "player history 8 games" in text
    assert "effective n" in text


def test_counter_evidence_is_not_collapsed_behind_a_toggle():
    """The requirement, enforced structurally: no <details> anywhere."""
    card = evidence_view.selection_card(_payload(), screened=70)
    assert "<details" not in card.lower()
    assert "What argues against this" in visible(card)
    counter_block = card.split('class="counter"')[1]
    assert "<li" in counter_block          # rendered inline, in the document flow


def test_falsifiers_render_on_every_selection():
    card = evidence_view.selection_card(_payload(), screened=70)
    text = visible(card)
    assert "What would flip it" in text
    assert "observable by:" in text


def test_unmeasured_bucket_is_visually_marked_on_the_card():
    card = evidence_view.selection_card(_payload(composite=60.0), screened=70)
    assert "UNMEASURED_BUCKET" in visible(card)
    assert "chip-unmeasured" in card


# --------------------------------------------------------------------------- #
# D.3 -- the CLV banner
# --------------------------------------------------------------------------- #
def test_unproven_clv_banner_states_directional_skill_only():
    banner = evidence_view.clv_banner(INSUFFICIENT)
    text = visible(banner)
    assert "Directional skill only" in text
    assert "no closing-line edge is established" in text
    assert "INSUFFICIENT_SAMPLE" in text
    assert "0 of the 150" in text


def test_the_banner_cannot_be_dismissed():
    """Not a convention: the page ships no script, so dismissal is unexpressible."""
    page = evidence_view.render_page(
        season=2026, week=3, as_of="t", clv_report=INSUFFICIENT, selections=[],
        screened=10, no_bet_reasons=["none"], bands=BANDS, regimes=REGIMES)
    banner = page.split('class="banner')[1].split("</div>")[0]
    for dismissal in ("onclick", "close", "dismiss", "hidden", "aria-hidden=\"true\""):
        assert dismissal not in banner.lower()
    assert "<script" not in page.lower()
    assert "cannot be dismissed" in visible(page)


def test_clv_panel_reports_insufficient_sample_without_a_point_estimate():
    panel = evidence_view.clv_panel(INSUFFICIENT)
    text = visible(panel)
    assert "INSUFFICIENT_SAMPLE" in text
    assert "Mean probability CLV" not in text     # no estimate below the gate
    assert "no CLV series" in text                # empty, not a flat line at zero


def test_proven_clv_swaps_the_banner_tone():
    proven = {"verdict": "GO", "detail": "mean CLV positive over 150 pairs",
              "n_resolved": 150, "min_sample": 150, "mean_clv_prob": 0.01,
              "ci95": [0.004, 0.017], "beat_close_rate": 0.55}
    banner = evidence_view.clv_banner(proven)
    assert "banner-ok" in banner
    assert "Directional skill only" not in visible(banner)


# --------------------------------------------------------------------------- #
# D.4 -- trend views must include the bad news
# --------------------------------------------------------------------------- #
def test_trends_include_all_four_views():
    panel = evidence_view.trends_panel(
        calibration_bins=[{"predicted": 0.45, "actual": 0.44, "n": 120},
                          {"predicted": 0.55, "actual": 0.58, "n": 200}],
        bands=BANDS, regimes=REGIMES, clv_history=[])
    text = visible(panel)
    assert "Calibration: predicted vs realized" in text
    assert "Hit rate by bucket" in text
    assert "CLV over time" in text
    assert "Interval coverage by regime" in text


def test_worst_regime_is_sorted_first_and_named():
    panel = evidence_view.trends_panel(calibration_bins=[], bands=BANDS,
                                       regimes=REGIMES, clv_history=[])
    text = visible(panel)
    assert "Worst measured regime: role_increase_5_plus" in text
    first_label = re.search(r'class="axis end">([a-z_0-9]+)</text>', panel)
    assert first_label.group(1) == "role_increase_5_plus"


def test_bucket_chart_marks_unmeasured_buckets_and_draws_breakeven():
    svg = evidence_view.svg_bucket_hit_rates(BANDS)
    assert "bar-unmeasured" in svg          # the n=33 bucket
    assert "breakeven" in svg
    assert "n=33 · unmeasured" in svg


def test_empty_clv_series_is_drawn_as_empty_not_as_zero():
    assert "no CLV series exists" in visible(evidence_view.svg_clv_over_time([]))


# --------------------------------------------------------------------------- #
# D.5 -- staking discipline
# --------------------------------------------------------------------------- #
CONFIG = {"kelly_multiplier": 0.15, "max_stake_pct": 0.03, "bankroll_units": 100.0}
PREMORTEM = {"constants": {"trials": 20000},
             "A_flat": [{"p": 0.5, "nbets": 2000, "unit_frac": 0.01, "median": 9.1,
                         "p5": -61.5, "p95": 79.7, "prob_down_50pct": 0.8324,
                         "prob_ruin": 0.4956}]}


def test_stake_is_never_rendered_without_the_loss_distribution():
    with pytest.raises(StakingRefused, match="loss distribution"):
        evidence_view.staking_panel(config=CONFIG, premortem={}, killcheck=INSUFFICIENT,
                                    edge=0.05)


def test_stake_requires_a_kill_check_status():
    with pytest.raises(StakingRefused, match="kill-check"):
        evidence_view.staking_panel(config=CONFIG, premortem=PREMORTEM, killcheck={},
                                    edge=0.05)


def test_no_stake_is_sized_while_the_kill_check_is_not_go():
    panel = evidence_view.staking_panel(config=CONFIG, premortem=PREMORTEM,
                                        killcheck=INSUFFICIENT, edge=0.05)
    text = visible(panel)
    assert "No stake is sized" in text
    assert "INSUFFICIENT_SAMPLE" in text
    assert "prob_ruin" not in text and "0.4956" in text     # the fan is still shown


def test_sized_stake_is_bounded_by_the_precommitted_fraction():
    proven = {"verdict": "GO", "detail": "ok"}
    panel = evidence_view.staking_panel(config=CONFIG, premortem=PREMORTEM,
                                        killcheck=proven, edge=0.80)
    text = visible(panel)
    assert "0.0300" in text            # 0.80 * 0.15 = 0.12, capped at max_stake_pct
    assert "0.12" not in text


def test_drawdown_fan_leads_with_the_downside():
    panel = evidence_view.staking_panel(config=CONFIG, premortem=PREMORTEM,
                                        killcheck=INSUFFICIENT)
    text = visible(panel)
    assert "5th pct" in text and "P(ruin)" in text
    assert "5th percentile column is the one to read first" in text


# --------------------------------------------------------------------------- #
# D.6 -- language enforcement in CI
# --------------------------------------------------------------------------- #
def test_banned_list_is_read_from_the_frozen_protocol():
    claims = language_guard.protocol_forbidden_claims()
    assert claims == ["profit", "ROI", "market edge", "closing-line value"]


def test_certainty_and_unearned_claims_are_caught():
    bad = "This is a lock — guaranteed profit, it beats the closing line, +EV all day."
    kinds = {v.kind for v in language_guard.scan_text(bad)}
    assert "certainty_language" in kinds
    assert "protocol_forbidden_claim" in kinds
    assert "unearned_claim" in kinds


def test_denying_a_claim_is_not_making_it():
    honest = ("Closing-line value is unproven: INSUFFICIENT_SAMPLE, so no market edge "
              "is established and this is not a profit claim.")
    assert language_guard.scan_text(honest) == []


def test_rendered_no_bet_page_passes_the_language_gate():
    page = evidence_view.render_page(
        season=2026, week=3, as_of="t", clv_report=INSUFFICIENT, selections=[],
        screened=214, no_bet_reasons=["nothing cleared"], bands=BANDS, regimes=REGIMES)
    language_guard.assert_clean(page, source="no_bet_page", is_html=True)


def test_rendered_selection_page_passes_the_language_gate():
    page = evidence_view.render_page(
        season=2023, week=10, as_of="t", clv_report=INSUFFICIENT,
        selections=[_payload(), _payload(composite=60.0)], screened=70,
        bands=BANDS, regimes=REGIMES,
        staking={"config": CONFIG, "premortem": PREMORTEM, "killcheck": INSUFFICIENT})
    language_guard.assert_clean(page, source="selection_page", is_html=True)


def test_the_gate_would_actually_fail_a_bad_page():
    """A gate that cannot fail is decoration."""
    page = evidence_view.render_page(
        season=2026, week=3, as_of="t", clv_report=INSUFFICIENT, selections=[],
        screened=1, no_bet_reasons=["This one is a lock with guaranteed profit."],
        bands=BANDS, regimes=REGIMES, verify_language=False)
    with pytest.raises(AssertionError, match="banned-language violation"):
        language_guard.assert_clean(page, source="bad_page", is_html=True)


def test_render_page_fails_closed_on_banned_language():
    with pytest.raises(AssertionError, match="banned-language violation"):
        evidence_view.render_page(
            season=2026, week=3, as_of="t", clv_report=INSUFFICIENT, selections=[],
            screened=1, no_bet_reasons=["Guaranteed profit, a sure thing."],
            bands=BANDS, regimes=REGIMES)


def test_discord_copy_is_scanned_with_the_same_ruleset():
    from nflvalue import notify
    payload = {"season": 2026, "week": 3, "publish": True, "leans": [
        {"name": "A Player", "market": "receiving_yards", "side": "over", "line": 54.5,
         "mean": 60.0, "composite": 40.0, "p_side": 0.58, "reason": "trailing usage up"}]}
    messages = notify.build_messages(payload)
    language_guard.assert_clean(json.dumps(messages), source="discord")
