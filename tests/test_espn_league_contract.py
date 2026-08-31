"""The league contract, imported from the recorded live settings payload.

The fixture was a byte-exact capture of ESPN's ``view=mSettings`` response.
Three identity fields -- the league id, the league name and the team name --
were then substituted, because a public repository is not the place for a
private league's identity.  Nothing else was touched: every scoring item,
band, slot count and rule is the live payload verbatim, and the sha256
asserted below is what stops that changing.  Every expectation in this file is
a HAND CALCULATION against the values ESPN's own settings page displays -- not
a re-run of the implementation, which would only prove the code agrees with
itself.

The league is not ESPN default scoring.  The specifics that make a
reconstructed ruleset wrong:

  * a receiving two-point conversion pays 4.0; passing and rushing pay 2.0
  * a defense returning an interception or fumble for a touchdown scores 20,
    and a blocked kick returned for one scores 30, against 6 for the same
    play by a position player
  * both the points-allowed and yards-allowed ladders are customised, and two
    points-allowed bands plus one yards-allowed band score exactly zero and
    are therefore ABSENT from the payload entirely
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nflvalue.fantasy.config import ScoringRules
from nflvalue.fantasy.espn_contract import (
    CONTRACT_VERSION,
    DST_POSITION_ID,
    FIELD_GOAL_BUCKETS,
    POINTS_ALLOWED_BANDS,
    STAT_REGISTRY,
    YARDS_ALLOWED_BANDS,
    UnsupportedEspnSetting,
    from_settings_payload,
)
from nflvalue.fantasy.scoring import score_components
from nflvalue.fantasy.special_scoring import (
    StatLineError,
    field_goal_bucket,
    points_allowed_key,
    score_dst,
    score_kicker,
    yards_allowed_key,
)

FIXTURE = Path(__file__).parent / "fixtures" / "espn_league_settings_2026_recorded.json"

#: sha256 of the fixture as it stands: the live capture with league identity
#: substituted. The upstream capture hashed to 34449c1e... before that
#: substitution; that value is not re-derivable here without the private
#: identity, which is the point.
FIXTURE_SHA256 = "ec5d68dbae0009cd18df819499f34cf7aa3cda24e03c87b4aa03491710592dd5"

#: The substituted identity. Asserted so a future capture cannot quietly
#: reintroduce the real league's name or id.
ANONYMISED_LEAGUE_ID = 1111111111
ANONYMISED_LEAGUE_NAME = "Test League"


@pytest.fixture(scope="module")
def payload():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def contract(payload):
    return from_settings_payload(payload)


# --------------------------------------------------------------------------- #
# The fixture really is the live payload
# --------------------------------------------------------------------------- #
def test_fixture_is_the_recorded_capture_and_has_not_been_hand_edited():
    raw = FIXTURE.read_bytes().rstrip(b"\n")
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256, (
        "the recorded settings fixture no longer matches the capture it was "
        "taken from; re-capture and re-anonymise it rather than editing it by "
        "hand -- a hand-tuned 'live' payload proves nothing about the parser"
    )


def test_fixture_carries_no_private_league_identity(payload):
    """The substitution is part of the fixture's contract, not a one-off edit."""
    assert payload["id"] == ANONYMISED_LEAGUE_ID
    assert payload["settings"]["name"] == ANONYMISED_LEAGUE_NAME


# --------------------------------------------------------------------------- #
# Registry reconciliation: every live category is represented, exactly
# --------------------------------------------------------------------------- #
def test_every_live_category_is_known_to_the_registry(payload):
    live = {int(i["statId"]) for i in payload["settings"]["scoringSettings"]["scoringItems"]}
    unknown = sorted(live - set(STAT_REGISTRY))
    assert not unknown, f"live statId(s) with no registry entry: {unknown}"


def test_the_registry_carries_no_category_the_league_does_not_score(payload):
    """A registry entry with no live counterpart is a guess about the league."""
    live = {int(i["statId"]) for i in payload["settings"]["scoringSettings"]["scoringItems"]}
    extra = sorted(set(STAT_REGISTRY) - live)
    assert not extra, f"registry invents categories the league does not score: {extra}"


def test_all_forty_eight_categories_import(contract):
    assert len(contract.categories) == 48
    assert contract.contract_version == CONTRACT_VERSION
    assert contract.season == 2026
    assert contract.scoring_type == "H2H_POINTS"


def test_an_unrepresentable_category_is_rejected_not_approximated(payload):
    poisoned = json.loads(json.dumps(payload))
    poisoned["settings"]["scoringSettings"]["scoringItems"].append(
        {"statId": 999999, "points": 3.0, "pointsOverrides": {}}
    )
    with pytest.raises(UnsupportedEspnSetting, match="999999"):
        from_settings_payload(poisoned)


def test_an_unrepresentable_lineup_slot_is_rejected(payload):
    poisoned = json.loads(json.dumps(payload))
    poisoned["settings"]["rosterSettings"]["lineupSlotCounts"]["99"] = 2
    with pytest.raises(UnsupportedEspnSetting, match="99"):
        from_settings_payload(poisoned)


# --------------------------------------------------------------------------- #
# Roster slots
# --------------------------------------------------------------------------- #
def test_starter_slots_match_the_league_exactly(contract):
    assert contract.roster.starters == {
        "QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "D/ST": 1,
    }
    assert contract.roster.total_starters == 9


def test_bench_and_ir_come_from_live_settings(contract):
    assert contract.roster.bench == 7
    assert contract.roster.injured_reserve == 1


def test_position_maximums_are_imported(contract):
    """positionLimits is keyed by PLAYER POSITION id, a different id space from
    lineupSlotCounts. Reading it through the slot map turns "max 4 QBs" into a
    limit on the wrong position entirely."""
    limits = contract.roster.position_limits
    assert limits["QB"] == 4
    assert limits["RB"] == 8
    assert limits["WR"] == 8
    assert limits["TE"] == 3
    assert limits["K"] == 3
    assert limits["D/ST"] == 3


def test_unlimited_positions_are_not_recorded_as_a_limit(contract):
    """-1 means no limit; recording it as a numeric maximum would cap a roster
    at minus one player."""
    assert all(v >= 0 for v in contract.roster.position_limits.values())


# --------------------------------------------------------------------------- #
# Waiver / regular season / matchup / playoff rules
# --------------------------------------------------------------------------- #
def test_league_rules_are_imported_from_the_live_payload(contract):
    r = contract.rules
    assert r.waiver_type == "WAIVERS_TRADITIONAL"
    assert r.waiver_hours == 24
    assert r.waiver_order_reset is True
    assert r.uses_acquisition_budget is False
    assert sorted(r.waiver_process_days) == [
        "FRIDAY", "MONDAY", "SATURDAY", "SUNDAY", "THURSDAY", "WEDNESDAY",
    ]
    assert r.regular_season_matchups == 14
    assert r.matchup_period_length == 1
    assert r.playoff_team_count == 4
    assert r.playoff_matchup_period_length == 2
    assert r.playoff_seeding_rule == "TOTAL_POINTS_SCORED"
    assert r.first_scoring_period == 1 and r.final_scoring_period == 18


# --------------------------------------------------------------------------- #
# Hashes
# --------------------------------------------------------------------------- #
def test_hashes_are_stable_across_rebuilds(payload):
    a = from_settings_payload(json.loads(json.dumps(payload)))
    b = from_settings_payload(json.loads(json.dumps(payload)))
    assert a.scoring_hash == b.scoring_hash
    assert a.roster_slot_hash == b.roster_slot_hash
    assert len(a.scoring_hash) == 64 and len(a.roster_slot_hash) == 64


def test_hashes_do_not_move_when_only_formatting_changes(payload, contract):
    """ESPN returns 6 in one response and 6.0 in another for the same setting.
    A hash that moves when nothing about the league did is worse than none."""
    reformatted = json.loads(json.dumps(payload))
    for item in reformatted["settings"]["scoringSettings"]["scoringItems"]:
        if float(item["points"]) == int(float(item["points"])):
            item["points"] = int(float(item["points"]))
        item["pointsOverrides"] = {
            k: (int(v) if float(v) == int(float(v)) else v)
            for k, v in (item.get("pointsOverrides") or {}).items()
        }
    assert from_settings_payload(reformatted).scoring_hash == contract.scoring_hash


def test_waiver_day_order_does_not_move_the_hash(payload, contract):
    shuffled = json.loads(json.dumps(payload))
    days = shuffled["settings"]["acquisitionSettings"]["waiverProcessDays"]
    shuffled["settings"]["acquisitionSettings"]["waiverProcessDays"] = list(reversed(days))
    assert from_settings_payload(shuffled).rules.canonical() == contract.rules.canonical()


def test_a_changed_point_value_changes_the_scoring_hash(payload, contract):
    changed = json.loads(json.dumps(payload))
    for item in changed["settings"]["scoringSettings"]["scoringItems"]:
        if int(item["statId"]) == 44:          # receiving two-point conversion
            item["points"] = 2.0
    assert from_settings_payload(changed).scoring_hash != contract.scoring_hash


def test_a_changed_slot_changes_the_roster_hash(payload, contract):
    changed = json.loads(json.dumps(payload))
    changed["settings"]["rosterSettings"]["lineupSlotCounts"]["23"] = 2   # FLEX
    assert from_settings_payload(changed).roster_slot_hash != contract.roster_slot_hash
    assert from_settings_payload(changed).scoring_hash == contract.scoring_hash


# --------------------------------------------------------------------------- #
# Offense — hand calculations
# --------------------------------------------------------------------------- #
def test_offensive_values_match_the_live_settings(contract):
    assert contract.points("passing_yards") == 0.04
    assert contract.points("passing_td") == 4.0
    assert contract.points("interception_thrown") == -2.0
    assert contract.points("rushing_yards") == 0.1
    assert contract.points("rushing_td") == 6.0
    assert contract.points("receiving_yards") == 0.1
    assert contract.points("receiving_td") == 6.0
    assert contract.points("reception") == 1.0
    assert contract.points("fumble_lost") == -2.0


def test_the_three_two_point_conversions_are_priced_separately(contract):
    """The headline custom rule: a receiving conversion is worth double."""
    assert contract.points("passing_2pt") == 2.0
    assert contract.points("rushing_2pt") == 2.0
    assert contract.points("receiving_2pt") == 4.0


def test_receiving_two_point_scores_four_not_two(contract):
    """Explicit receiving-two-point test, scored end to end.

    8 receptions, 92 receiving yards, 1 receiving TD, 1 receiving conversion:
      8*1.0 + 92*0.1 + 6.0 + 4.0 = 8 + 9.2 + 6 + 4 = 27.2
    Under a uniform 2.0 conversion this would be 25.2, so the assertion fails
    loudly if the split is ever collapsed back.
    """
    rules = contract.to_scoring_rules()
    line = {
        "receptions": 8, "receiving_yards": 92, "receiving_tds": 1,
        "receiving_2pt_conversions": 1,
    }
    assert float(score_components(line, rules)) == pytest.approx(27.2)


def test_a_quarterback_line_scores_by_hand(contract):
    """287 pass yds, 2 pass TD, 1 INT, 31 rush yds, 1 rush TD, 1 passing 2PT:
    287*0.04 + 2*4 + (-2) + 31*0.1 + 6 + 2 = 11.48 + 8 - 2 + 3.1 + 6 + 2
    """
    rules = contract.to_scoring_rules()
    line = {
        "passing_yards": 287, "passing_tds": 2, "passing_interceptions": 1,
        "rushing_yards": 31, "rushing_tds": 1, "passing_2pt_conversions": 1,
    }
    assert float(score_components(line, rules)) == pytest.approx(28.58)


def test_each_conversion_type_is_independent(contract):
    rules = contract.to_scoring_rules()
    base = {"passing_2pt_conversions": 1}
    assert float(score_components(base, rules)) == pytest.approx(2.0)
    assert float(score_components({"rushing_2pt_conversions": 1}, rules)) == pytest.approx(2.0)
    assert float(score_components({"receiving_2pt_conversions": 1}, rules)) == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# Backwards compatibility of the refactor
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("preset,reception", [("ppr", 1.0), ("half_ppr", 0.5), ("standard", 0.0)])
def test_presets_are_unchanged(preset, reception):
    rules = ScoringRules.preset(preset)
    assert rules.reception == reception
    assert rules.two_point == 2.0
    for kind in ("passing", "rushing", "receiving"):
        assert getattr(rules, f"{kind}_two_point_value") == 2.0


def test_the_legacy_scoring_fingerprint_is_byte_identical():
    """``models.fit_position_models`` compares this serialisation against the
    one stored in every feature frame. If the shape moves, every frame and
    model artifact built before the split is rejected as mismatched."""
    legacy_keys = {
        "reception", "passing_yard", "passing_td", "interception", "rushing_yard",
        "rushing_td", "receiving_yard", "receiving_td", "two_point", "fumble_lost",
        "passing_300_bonus", "rushing_100_bonus", "receiving_100_bonus",
    }
    for preset in ("ppr", "half_ppr", "standard"):
        assert set(ScoringRules.preset(preset).to_dict()) == legacy_keys
    assert set(ScoringRules(two_point=5.0).to_dict()) == legacy_keys


def test_the_split_keys_appear_only_when_they_carry_information():
    split = ScoringRules(receiving_two_point=4.0)
    assert "passing_two_point" in split.to_dict()
    assert split.to_dict()["receiving_two_point"] == 4.0
    assert split.to_dict()["passing_two_point"] == 2.0


def test_uniform_rules_score_exactly_as_before():
    """The old implementation summed the three conversion counts and applied
    one multiplier. For uniform rules the new form must be identical."""
    line = {
        "passing_2pt_conversions": 2, "rushing_2pt_conversions": 1,
        "receiving_2pt_conversions": 3, "receiving_yards": 40, "receptions": 4,
    }
    for value in (2.0, 0.0, 5.0, -1.5):
        rules = ScoringRules(two_point=value)
        expected = (2 + 1 + 3) * value + 40 * 0.1 + 4 * 1.0
        assert float(score_components(line, rules)) == pytest.approx(expected)


def test_legacy_two_point_keyword_still_sets_all_three():
    rules = ScoringRules(two_point=3.0)
    assert rules.passing_two_point_value == 3.0
    assert rules.rushing_two_point_value == 3.0
    assert rules.receiving_two_point_value == 3.0


# --------------------------------------------------------------------------- #
# Kicker — hand calculations and every bucket boundary
# --------------------------------------------------------------------------- #
def test_kicker_bucket_values_match_the_live_settings(contract):
    assert contract.points("fg_made_0_39") == 3.0
    assert contract.points("fg_made_40_49") == 4.0
    assert contract.points("fg_made_50_59") == 5.0
    assert contract.points("fg_made_60_plus") == 6.0
    assert contract.points("fg_missed_total") == -3.0
    assert contract.points("pat_made") == 1.0
    assert contract.points("pat_missed") == -3.0


@pytest.mark.parametrize("distance,expected", [
    (0, "fg_made_0_39"), (39, "fg_made_0_39"),          # upper edge of bucket 1
    (40, "fg_made_40_49"), (49, "fg_made_40_49"),       # both edges of bucket 2
    (50, "fg_made_50_59"), (59, "fg_made_50_59"),       # both edges of bucket 3
    (60, "fg_made_60_plus"), (70, "fg_made_60_plus"),   # lower edge of bucket 4
])
def test_every_field_goal_bucket_boundary(distance, expected):
    assert field_goal_bucket(distance) == expected


def test_the_field_goal_buckets_are_contiguous_and_ordered():
    previous_high = -1
    for low, high, _key in FIELD_GOAL_BUCKETS:
        assert low == previous_high + 1, f"gap or overlap before {low}"
        previous_high = high if high != float("inf") else previous_high
    assert FIELD_GOAL_BUCKETS[-1][1] == float("inf")


def test_a_kicker_line_scores_by_hand(contract):
    """3 PAT made, 1 PAT missed, FGs from 22 / 41 / 55, one miss:
    3*1 + 1*(-3) + 3 + 4 + 5 + 1*(-3) = 3 - 3 + 12 - 3 = 9
    """
    line = {
        "pat_made": 3, "pat_missed": 1,
        "field_goals_made": [22, 41, 55], "field_goals_missed": 1,
    }
    assert score_kicker(line, contract) == pytest.approx(9.0)


def test_a_boundary_kicker_line_scores_by_hand(contract):
    """Every bucket edge in one line: 39 / 40 / 49 / 50 / 59 / 60
    = 3 + 4 + 4 + 5 + 5 + 6 = 27
    """
    line = {"field_goals_made": [39, 40, 49, 50, 59, 60]}
    assert score_kicker(line, contract) == pytest.approx(27.0)


def test_pre_bucketed_counts_are_accepted(contract):
    line = {"fg_made_0_39": 2, "fg_made_50_59": 1, "pat_made": 4}
    assert score_kicker(line, contract) == pytest.approx(2 * 3 + 5 + 4)


def test_mixing_distances_and_bucket_counts_is_refused(contract):
    with pytest.raises(StatLineError, match="twice"):
        score_kicker({"field_goals_made": [30], "fg_made_0_39": 1}, contract)


def test_a_negative_field_goal_distance_is_refused():
    with pytest.raises(StatLineError):
        field_goal_bucket(-1)


# --------------------------------------------------------------------------- #
# Team defense — hand calculations and every tier boundary
# --------------------------------------------------------------------------- #
def test_defensive_event_values_match_the_live_settings(contract):
    assert contract.dst_points("defensive_sack") == 1.0
    assert contract.dst_points("defensive_interception") == 2.0
    assert contract.dst_points("defensive_fumble_recovery") == 2.0
    assert contract.dst_points("defensive_safety") == 10.0
    assert contract.dst_points("blocked_kick") == 2.0
    assert contract.dst_points("two_point_return") == 2.0
    assert contract.dst_points("one_point_safety") == 1.0


def test_return_touchdowns_are_amplified_for_the_defense(contract):
    """The same play is worth 6 to a position player and far more to a D/ST.
    A contract that ignored pointsOverrides would under-score every defense."""
    for key, player, dst in [
        ("interception_return_td", 6.0, 20.0),
        ("fumble_return_td", 6.0, 20.0),
        ("blocked_kick_return_td", 6.0, 30.0),
        ("kickoff_return_td", 6.0, 12.0),
        ("punt_return_td", 6.0, 12.0),
    ]:
        assert contract.points(key) == player, key
        assert contract.dst_points(key) == dst, key


def test_team_win_is_represented_separately(contract):
    assert contract.points("team_win") == 2.0
    assert contract.categories["team_win"].stat_id == 155


@pytest.mark.parametrize("points,expected", [
    (0, 5.0), (1, 4.0), (6, 4.0), (7, 3.0), (13, 3.0), (14, 1.0), (17, 1.0),
    (18, 0.0), (21, 0.0), (22, 0.0), (27, 0.0),
    (28, -1.0), (34, -1.0), (35, -3.0), (45, -3.0), (46, -5.0), (70, -5.0),
])
def test_every_points_allowed_tier_boundary(contract, points, expected):
    key = points_allowed_key(points)
    assert (contract.dst_points(key) if key else 0.0) == pytest.approx(expected)


@pytest.mark.parametrize("yards,expected", [
    (0, 5.0), (99, 5.0), (100, 3.0), (199, 3.0), (200, 2.0), (299, 2.0),
    (300, 0.0), (349, 0.0),
    (350, -1.0), (399, -1.0), (400, -3.0), (449, -3.0), (450, -5.0), (499, -5.0),
    (500, -6.0), (549, -6.0), (550, -7.0), (700, -7.0),
])
def test_every_yards_allowed_tier_boundary(contract, yards, expected):
    key = yards_allowed_key(yards)
    assert (contract.dst_points(key) if key else 0.0) == pytest.approx(expected)


@pytest.mark.parametrize("bands,name", [
    (POINTS_ALLOWED_BANDS, "points allowed"),
    (YARDS_ALLOWED_BANDS, "yards allowed"),
])
def test_tier_ladders_are_exhaustive_and_non_overlapping(bands, name):
    """ESPN omits zero-valued bands from the payload. A ladder built only from
    what the payload contains would have holes, and a defense landing in one
    would score nothing at all rather than the zero the league intends."""
    previous_high = -1
    for low, high, _key in bands:
        assert low == previous_high + 1, f"{name} ladder has a gap or overlap at {low}"
        previous_high = high if high != float("inf") else previous_high
    assert bands[0][0] == 0, f"{name} ladder must start at 0"
    assert bands[-1][1] == float("inf"), f"{name} ladder must be unbounded above"


def test_a_defense_line_scores_by_hand(contract):
    """4 sacks, 2 INT, 1 fumble recovery, 1 safety, 1 INT return TD,
    10 points allowed, 250 yards allowed:
    4*1 + 2*2 + 1*2 + 1*10 + 1*20 + 3 (7-13 PA) + 2 (200-299 YA) = 45
    """
    line = {
        "defensive_sack": 4, "defensive_interception": 2,
        "defensive_fumble_recovery": 1, "defensive_safety": 1,
        "interception_return_td": 1, "points_allowed": 10, "yards_allowed": 250,
    }
    assert score_dst(line, contract) == pytest.approx(45.0)


def test_a_shutout_defense_scores_by_hand(contract):
    """Shutout, 88 yards allowed, 6 sacks, blocked kick returned for a TD:
    6*1 + 30 + 5 (0 PA) + 5 (<100 YA) = 46
    """
    line = {
        "defensive_sack": 6, "blocked_kick_return_td": 1,
        "points_allowed": 0, "yards_allowed": 88,
    }
    assert score_dst(line, contract) == pytest.approx(46.0)


def test_a_blowout_defense_can_score_negative(contract):
    """48 points and 560 yards allowed, one sack: 1 + (-5) + (-7) = -11"""
    line = {"defensive_sack": 1, "points_allowed": 48, "yards_allowed": 560}
    assert score_dst(line, contract) == pytest.approx(-11.0)


def test_a_zero_band_contributes_exactly_nothing(contract):
    """20 points and 320 yards allowed both land in bands ESPN omits."""
    line = {"defensive_sack": 2, "points_allowed": 20, "yards_allowed": 320}
    assert score_dst(line, contract) == pytest.approx(2.0)


@pytest.mark.parametrize("missing", ["points_allowed", "yards_allowed"])
def test_a_defense_without_its_tier_inputs_is_refused(contract, missing):
    line = {"defensive_sack": 3, "points_allowed": 14, "yards_allowed": 300}
    line.pop(missing)
    with pytest.raises(StatLineError, match=missing):
        score_dst(line, contract)


def test_dst_position_id_is_the_one_espn_uses(contract):
    assert DST_POSITION_ID == 16
    assert contract.categories["interception_return_td"].position_overrides == {"16": 20.0}
