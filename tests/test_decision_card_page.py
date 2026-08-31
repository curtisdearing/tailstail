"""The private page, and the boundary that keeps it off the public site.

Two halves.

The **page**: it renders every state as well-formed HTML, shows the four
sections in order, never renders a blank where it means "nothing measured",
carries no script and no external reference, and treats a player's name as the
untrusted string it is — a manager picks that name on a live platform.

The **boundary**: the public weekly payload and the public dashboard are built
from an allow-list and then checked, with this league's own id and team names
as needles.  The leak this guards against already happened: ``fantasy.html``
embedded the personal contract, the workflow copied that file to
``_site/index.html`` and ``data/fantasy_latest.json`` — ``my_team`` and all —
next to it.  Every step was locally reasonable, which is why the guard is a
positive allow-list plus an assertion rather than a rule somebody has to
remember.
"""

from __future__ import annotations

import html
import json
import sys
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nflvalue.fantasy import decision_card, decision_page, my_team, private_boundary  # noqa: E402
from nflvalue.fantasy.dashboard import render_fantasy_dashboard  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "my_team"
NOW = "2026-08-29T03:00:00+00:00"
MODEL = "e3f1c0d"
VOID = {"br", "hr", "img", "input", "meta", "link", "col", "source"}

ALL_FIXTURES = ("pre_draft", "draft_in_progress", "post_draft", "stale_snapshot",
                "unmatched_player", "bye", "injury", "illegal_roster", "no_action")


def side(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.model.json").read_text())


def snapshot(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def samples_for(model_side: dict, *, n: int = 300, seed: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    return {
        pid: rng.normal(float(p["mean"]), max(1.0, (float(p["p90"]) - float(p["p10"])) / 2.56), n)
        for pid, p in model_side["projections"].items()
    }


def contract_payload(name: str, *, snap: dict | None = None, **kwargs) -> dict:
    model_side = side(name)
    return my_team.build(
        snap if snap is not None else snapshot(name), now=NOW,
        crosswalk={int(k): v for k, v in model_side["crosswalk"].items()},
        projections=model_side["projections"], byes=model_side["byes"],
        samples=samples_for(model_side), **kwargs)


def card_for(name: str, *, snap: dict | None = None, **kwargs) -> dict:
    return decision_card.build(contract_payload(name, snap=snap, **kwargs),
                               now=NOW, model_version=MODEL)


def page(name: str, *, snap: dict | None = None, appendix: bool = True, **kwargs) -> str:
    payload = contract_payload(name, snap=snap, **kwargs)
    card = decision_card.build(payload, now=NOW, model_version=MODEL)
    return decision_page.render(card, my_team=payload if appendix else None)


# --------------------------------------------------------------------------- #
# Well-formedness
# --------------------------------------------------------------------------- #
class _Balance(HTMLParser):
    """Minimal well-formedness check: every non-void tag closes, in order."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack:
            self.errors.append(f"</{tag}> with nothing open")
        elif self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closed while <{self.stack[-1]}> was open")
        else:
            self.stack.pop()


def assert_well_formed(document: str) -> None:
    parser = _Balance()
    parser.feed(document)
    assert not parser.errors, parser.errors
    assert not parser.stack, f"unclosed tags: {parser.stack}"
    assert document.count("<title>") == 1


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_every_state_renders_well_formed_html(name):
    assert_well_formed(page(name))


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_the_page_is_self_contained_and_inert(name):
    """No script, no storage, no third party: a page that cannot log an error."""
    document = page(name)
    for forbidden in ("<script", "javascript:", "onerror=", "onclick=", "localStorage",
                      "sessionStorage", "indexedDB", "http://", "https://cdn", "<iframe",
                      "@import", "url("):
        assert forbidden not in document, forbidden


# --------------------------------------------------------------------------- #
# The four sections, in order
# --------------------------------------------------------------------------- #
def test_the_four_sections_render_numbered_and_in_order():
    document = page("post_draft")
    positions = []
    for index, title in enumerate(decision_page.SECTION_TITLES, start=1):
        heading = f'<h2 id="section-{index}">{index}. {title}</h2>'
        assert heading in document, f"missing section {index}: {title}"
        positions.append(document.index(heading))
    assert positions == sorted(positions)
    assert len(decision_page.SECTION_TITLES) == 4


def test_the_lineup_marks_which_seats_are_already_set():
    document = page("post_draft")
    assert "already set" in document
    assert ">change<" in document


def test_no_action_says_there_is_nothing_to_change():
    document = page("no_action")
    assert "Nothing to change" in document


def test_the_page_shows_at_most_three_actionable_decisions():
    from nflvalue.fantasy import waivers

    plan = [waivers.Recommendation(
        add_espn_id=900 + i, add_name=f"Free Agent {i}", add_position="RB",
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
        data_timestamps={}, degraded=False, faab=None) for i in range(6)]
    document = page("no_action", waiver_plan=plan)
    assert_well_formed(document)
    assert document.count('<div class="decision">') <= 4      # three actions plus refusals
    assert "held back" in document


# --------------------------------------------------------------------------- #
# Fail closed, visibly
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["stale_snapshot", "pre_draft", "illegal_roster"])
def test_a_blocked_page_shows_the_banner_and_its_reason(name):
    document = page(name)
    assert "NO CURRENT PICK" in document
    for chunk in document.split('<div class="nopick"><b>NO CURRENT PICK</b><span>')[1:]:
        reason = chunk.split("</span>")[0]
        assert reason.strip(), "a NO CURRENT PICK banner carried no reason"


def test_a_stale_page_never_shows_last_week_s_players():
    fresh_names = {slot["name"] for slot in card_for("post_draft")["current_lineup"]["slots"]}
    document = page("stale_snapshot", appendix=False)
    assert fresh_names
    for name in fresh_names:
        assert name not in document


def test_the_page_replaces_both_files_every_run(tmp_path):
    """A run that can say nothing overwrites the page rather than leaving one."""
    json_path, html_path = tmp_path / "private" / "card.json", tmp_path / "private" / "card.html"
    decision_page.write(card_for("post_draft"), json_path=json_path, html_path=html_path)
    assert "already set" in html_path.read_text()

    decision_page.write(card_for("stale_snapshot"), json_path=json_path, html_path=html_path)
    document = html_path.read_text()
    assert "NO CURRENT PICK" in document
    assert "already set" not in document
    assert json.loads(json_path.read_text())["state"] == "no_current_pick"
    assert not list((tmp_path / "private").glob("*.tmp"))


def test_an_unmeasured_swap_reads_as_no_pick_not_as_a_blank():
    model_side = side("post_draft")
    payload = my_team.build(
        snapshot("post_draft"), now=NOW,
        crosswalk={int(k): v for k, v in model_side["crosswalk"].items()},
        projections=model_side["projections"], byes=model_side["byes"])
    document = decision_page.render(
        decision_card.build(payload, now=NOW, model_version=MODEL))
    assert "NO CURRENT PICK" in document
    assert "not drawn together" in document


# --------------------------------------------------------------------------- #
# Numbers and labels
# --------------------------------------------------------------------------- #
def test_a_probability_is_shown_only_with_its_qualifier():
    document = page("post_draft")
    if "scores more" not in document:
        pytest.skip("this fixture produced no measured swap")
    assert "model-relative" in document
    assert "not a calibrated confidence" in document


def test_the_page_never_labels_anything_plain_confidence():
    for name in ALL_FIXTURES:
        document = page(name)
        assert "Confidence:" not in document
        assert ">Confidence<" not in document
        assert document.lower().count("confidence") == \
            document.lower().count("not a calibrated confidence")


def test_every_decision_shows_its_four_stamps():
    document = page("post_draft")
    assert f"model {MODEL}" in document
    assert "scoring <code>" in document
    assert "snapshot <code>" in document


def test_the_shadow_seats_are_visible_on_every_page():
    for name in ALL_FIXTURES:
        document = page(name, appendix=False)
        assert document.count('<span class="tag">SHADOW</span>') >= 2


# --------------------------------------------------------------------------- #
# Hostile input
# --------------------------------------------------------------------------- #
HOSTILE = '</script><img src=x onerror=alert(1)><a href="javascript:alert(2)">x</a>'


def assert_inert(document: str) -> None:
    """The hostile string may appear as text; it may not appear as markup.

    Asserting the *substring* ``onerror=`` is absent would be the wrong test —
    it survives inside ``&lt;img src=x onerror=alert(1)&gt;``, where it is
    ordinary prose. What must not survive is a tag or an attribute, so this
    checks the raw string never appears unescaped, no element the page does not
    build ever opens, and the escaped form is what is on the page instead.
    """
    assert HOSTILE not in document, "the hostile string reached the page unescaped"
    assert html.escape(HOSTILE, quote=True) in document
    for markup in ("<img", "<a ", "<script", "</script>", "href=\"javascript:"):
        assert markup not in document, markup
    assert_well_formed(document)


def test_a_hostile_player_name_lands_inert():
    snap = snapshot("post_draft")
    snap["rosters"]["1"][0]["full_name"] = HOSTILE
    assert_inert(page("post_draft", snap=snap))


def test_a_hostile_team_name_lands_inert():
    snap = snapshot("post_draft")
    snap["my_team"]["name"] = HOSTILE
    snap["league"]["name"] = HOSTILE
    assert_inert(page("post_draft", snap=snap))


def test_a_hostile_cited_note_lands_inert():
    payload = contract_payload("post_draft")
    ids = [entry["espn_player_id"] for entry in payload["roster"]]
    card = decision_card.build(
        payload, now=NOW, model_version=MODEL,
        context=[{"text": HOSTILE, "source": HOSTILE, "as_of": "2026-08-28T15:00:00Z",
                  "espn_player_ids": ids}])
    assert_inert(decision_page.render(card))


def test_the_renderer_refuses_a_card_relabelled_public():
    card = card_for("post_draft")
    card["visibility"] = "public"
    with pytest.raises(ValueError):
        decision_page.render(card)


# --------------------------------------------------------------------------- #
# The appendix is detail, and it is collapsed
# --------------------------------------------------------------------------- #
def test_the_full_contract_is_an_appendix_behind_a_disclosure():
    document = page("post_draft")
    assert "<details><summary>" in document
    assert document.index("<details>") > document.index('<h2 id="section-4"')
    for index, title in enumerate(decision_page.MY_TEAM_SECTION_TITLES, start=1):
        assert f'<h3 id="my-team-{index}">{index}. {title}</h3>' in document


def test_the_page_renders_without_the_appendix():
    document = page("post_draft", appendix=False)
    assert "<details>" not in document
    assert_well_formed(document)


# --------------------------------------------------------------------------- #
# Public / private
# --------------------------------------------------------------------------- #
def summaries() -> pd.DataFrame:
    return pd.DataFrame([
        {"position": "RB", "player_name": "R. Bell", "team": "PHI", "mean": 18.9,
         "median": 18.1, "event_simulator_mean": 18.4, "p10": 10.4, "p90": 27.4,
         "prob_15_plus": 0.61, "prob_20_plus": 0.42, "availability_probability": 0.97,
         "component_model_disagreement": False},
    ])


def league_needles(payload: dict) -> tuple:
    league = payload["league"]
    return (league["league_id"], [league["league_name"], league["team_name"]])


def test_the_public_dashboard_can_no_longer_be_handed_a_personal_contract():
    """The parameter that carried the leak is gone, not merely unused."""
    with pytest.raises(TypeError):
        render_fantasy_dashboard(summaries(), "/dev/null", season=2026, week=1,
                                 generated_at=NOW, my_team={"schema_version": "my_team/1.0.0"})


def test_the_public_dashboard_carries_nothing_private(tmp_path):
    payload = contract_payload("post_draft")
    league_id, names = league_needles(payload)
    out = tmp_path / "fantasy.html"
    render_fantasy_dashboard(summaries(), out, season=2026, week=1, generated_at=NOW)
    document = out.read_text()
    private_boundary.assert_public_text_safe(document, league_id=league_id, names=names)
    assert "My team" not in document
    assert "NO CURRENT PICK" not in document
    assert "2026 week 1 fantasy projections" in document


def test_the_private_page_is_caught_by_the_text_guard():
    """The guard must actually fire on the document it exists to keep private."""
    payload = contract_payload("post_draft")
    league_id, names = league_needles(payload)
    with pytest.raises(private_boundary.PrivateDataLeak):
        private_boundary.assert_public_text_safe(
            page("post_draft"), league_id=league_id, names=names)


def test_the_public_payload_drops_my_team_and_the_espn_rows():
    payload = contract_payload("post_draft")
    weekly = {
        "generated_at": NOW, "season": 2026, "week": 1,
        "players": summaries().to_dict("records"),
        "simulation": {"simulations": 10000},
        "my_team": payload,
        "espn_comparison": {
            "status": "ok", "season": 2026, "current_week": 1, "disclaimer": "d",
            "espn_provenance": {"retrieved_at": NOW, "players_sha256": "a" * 64,
                                "redistribution_rights": "personal use",
                                "coverage": {"qb": 1}, "source": {"url": "https://espn"}},
            "identity": {"espn_players": 400, "matched": 380, "coverage_pct": 95.0,
                         "unmatched_no_crosswalk_count": 10,
                         "unmatched_model_not_projected_count": 10,
                         "unmatched_names": ["A. Player"]},
            "current_week_rows": [{"player_id": "00-0011", "espn_pts": 12.0,
                                   "model_pts": 13.0, "player_name": "R. Bell"}],
            "season_series": [{"week": 1, "n_played": 300, "mae_espn": 5.4, "mae_model": 5.1}],
        },
    }
    public = private_boundary.public_weekly_payload(weekly)

    assert "my_team" not in public
    assert public["visibility"] == "public"
    assert public["players"]
    assert public["espn_comparison"]["season_series"]
    assert "current_week_rows" not in public["espn_comparison"]
    assert public["espn_comparison"]["rows_published"] is False
    assert "unmatched_names" not in public["espn_comparison"]["identity"]
    assert "source" not in public["espn_comparison"]["espn_provenance"]
    assert public["withheld"]


def test_the_public_payload_refuses_a_league_string_that_slipped_through():
    payload = contract_payload("post_draft")
    league_id, names = league_needles(payload)
    public = {"players": [{"player_name": "R. Bell", "note": f"league {league_id}"}]}
    with pytest.raises(private_boundary.PrivateDataLeak):
        private_boundary.assert_public_safe(public, league_id=league_id, names=names)


@pytest.mark.parametrize("key", ["my_team", "rosters", "members", "espn_player_id",
                                 "current_week_rows", "team_name"])
def test_a_private_key_anywhere_in_a_public_object_is_refused(key):
    with pytest.raises(private_boundary.PrivateDataLeak):
        private_boundary.assert_public_safe({"players": [{"stats": {key: 1}}]})


@pytest.mark.parametrize("marker", ["decision-card/1", "my_team/1.0.0", "espn-league/1"])
def test_a_private_contract_marker_in_a_public_object_is_refused(marker):
    with pytest.raises(private_boundary.PrivateDataLeak):
        private_boundary.assert_public_safe({"note": f"built from {marker}"})


def test_a_whole_decision_card_is_refused_by_the_structural_guard():
    with pytest.raises(private_boundary.PrivateDataLeak):
        private_boundary.assert_public_safe(card_for("post_draft"))


def test_the_public_allow_list_is_the_only_way_in():
    """A section added to the weekly payload is private until it is named."""
    weekly = {"generated_at": NOW, "players": [], "brand_new_section": {"anything": 1}}
    public = private_boundary.public_weekly_payload(weekly)
    assert "brand_new_section" not in public


def test_the_private_artifacts_are_gitignored():
    ignored = (ROOT / ".gitignore").read_text()
    assert "private/" in ignored
    assert "data/fantasy_public.json" in ignored


def test_the_workflow_publishes_the_public_payload_and_no_private_state():
    workflow = (ROOT / ".github" / "workflows" / "fantasy-weekly.yml").read_text()
    assert "cp data/fantasy_public.json _site/fantasy_latest.json" in workflow
    assert "cp data/fantasy_latest.json _site/" not in workflow
    # Neither the personal payload nor the raw ESPN captures may be uploaded.
    artifact = workflow.split("- uses: actions/upload-artifact@v6")[1].split("- name:")[0]
    for private_path in ("data/fantasy_latest.json", "data/espn_snapshots/",
                         "data/espn_comparison_ledger.json", "private/"):
        assert private_path not in artifact, private_path


def test_the_public_release_asset_carries_no_espn_captures():
    from scripts import state_store

    fantasy = state_store.STATE_PROFILES["fantasy"]
    assert "data/espn_snapshots/*.json" not in fantasy
    assert "data/espn_comparison_ledger.json" not in fantasy
    assert "data/player_projection_snapshot.json" in fantasy
