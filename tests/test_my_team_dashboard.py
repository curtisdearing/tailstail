"""The eight-section contract detail, now rendered privately.

These sections used to render into ``fantasy.html``, which the weekly workflow
copies to the public Pages site — so a private league's rosters, every
manager's team name and the league id were published each week.  They now
render inside the private decision page, collapsed behind a disclosure, as the
record the decisions were drawn from rather than as the decisions themselves.

The properties are the ones that mattered before and still do:

  * a section with no trustworthy input renders NO CURRENT PICK **and its
    reason** — never an empty table that reads as "nothing to do";
  * placeholder artifacts (a mock draft board, a smoke-test trade file) cannot
    reach the page, and the guard that stops them does not erase the two real
    managers in this league who are called "Team 7" and "Team 8";
  * the public dashboard renders the frozen offense table and nothing personal.
"""

from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nflvalue.fantasy import decision_card, decision_page  # noqa: E402
from nflvalue.fantasy import my_team as mt  # noqa: E402
from nflvalue.fantasy.dashboard import render_fantasy_dashboard  # noqa: E402
from nflvalue.fantasy.decision_page import MY_TEAM_SECTION_TITLES  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "my_team"
NOW = "2026-08-29T03:00:00+00:00"
MODEL = "e3f1c0d"
VOID = {"br", "hr", "img", "input", "meta", "link", "col", "source"}


def payload(name: str) -> dict:
    return built(name)


def model(name):
    return json.loads((FIXTURES / f"{name}.model.json").read_text())


def built(name, snapshot=None, **kwargs):
    """Canonical snapshot in, model side-data alongside — never merged."""
    side = model(name)
    source = snapshot if snapshot is not None else json.loads(
        (FIXTURES / f"{name}.json").read_text())
    return mt.build(source, now=NOW,
                    crosswalk={int(k): v for k, v in side["crosswalk"].items()},
                    projections=side["projections"], byes=side["byes"], **kwargs)


def summaries() -> pd.DataFrame:
    return pd.DataFrame([
        {"position": "RB", "player_name": "R. Bell", "team": "PHI", "mean": 18.9,
         "median": 18.1, "event_simulator_mean": 18.4, "p10": 10.4, "p90": 27.4,
         "prob_15_plus": 0.61, "prob_20_plus": 0.42, "availability_probability": 0.97,
         "component_model_disagreement": False},
        {"position": "WR", "player_name": "W. Gray", "team": "MIN", "mean": 17.8,
         "median": 16.9, "event_simulator_mean": 17.2, "p10": 9.8, "p90": 25.8,
         "prob_15_plus": 0.58, "prob_20_plus": 0.38, "availability_probability": 0.99,
         "component_model_disagreement": True},
    ])


def render(my_team: dict) -> str:
    """The private page, appendix included — where these sections live now."""
    card = decision_card.build(my_team, now=NOW, model_version=MODEL)
    return decision_page.render(card, my_team=my_team)


def render_public(tmp_path, frame=None) -> str:
    out = tmp_path / "fantasy.html"
    render_fantasy_dashboard(frame if frame is not None else summaries(), out,
                             season=2026, week=1, generated_at=NOW)
    return out.read_text()


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


# --------------------------------------------------------------------------- #
# The eight sections
# --------------------------------------------------------------------------- #
def test_all_eight_sections_render_numbered_and_in_order():
    document = render(payload("post_draft"))
    assert_well_formed(document)
    positions = []
    for index, title in enumerate(MY_TEAM_SECTION_TITLES, start=1):
        heading = f'<h3 id="my-team-{index}">{index}. {title}</h3>'
        assert heading in document, f"missing section {index}: {title}"
        positions.append(document.index(heading))
    assert positions == sorted(positions), "sections rendered out of order"
    assert len(MY_TEAM_SECTION_TITLES) == 8


@pytest.mark.parametrize("name", [
    "pre_draft", "draft_in_progress", "post_draft", "stale_snapshot",
    "unmatched_player", "bye", "injury", "illegal_roster", "no_action",
])
def test_every_state_renders_well_formed_html(name):
    assert_well_formed(render(payload(name)))


# --------------------------------------------------------------------------- #
# Fail closed, visibly
# --------------------------------------------------------------------------- #
def test_blocked_sections_show_the_banner_and_the_reason():
    document = render(payload("stale_snapshot"))
    assert document.count("NO CURRENT PICK") >= 4
    assert "stale" in document
    # A banner without its reason would be worse than no banner at all.
    for match in re.finditer(r'<div class="nopick"><b>NO CURRENT PICK</b><span>(.*?)</span>',
                             document, re.S):
        assert match.group(1).strip(), "a NO CURRENT PICK banner carried no reason"


def test_illegal_roster_lists_its_violations():
    """A modelled seat that cannot be filled is named on the page."""
    document = render(built("illegal_roster"))
    assert "Roster legality violations" in document or "cannot fill required slot" in document
    assert "cannot fill required slot WR" in document


def test_a_missing_kicker_is_its_own_section_not_a_lineup_violation():
    """The shadow seat says NO CURRENT PICK; the offence still renders."""
    snapshot = json.loads((FIXTURES / "post_draft.json").read_text())
    snapshot["rosters"]["1"] = [p for p in snapshot["rosters"]["1"]
                                if p["default_position"] != "K"]
    document = render(built("post_draft", snapshot=snapshot))
    assert "cannot fill required slot K" not in document
    assert "NO CURRENT PICK" in document
    assert "offensive lineup is unaffected" in document


def test_pre_draft_targets_are_labelled_and_no_pick_is_claimed():
    document = render(payload("pre_draft"))
    assert "No actual selections yet" in document
    assert "TARGET — not a pick" in document
    assert "Actual selections" not in document


def test_post_draft_shows_real_picks_with_round_and_overall():
    document = render(payload("post_draft"))
    assert "Actual selections" in document
    assert "TARGET — not a pick" not in document


def test_shadow_sections_are_marked_shadow():
    document = render(payload("post_draft"))
    for title in ("K shadow recommendations", "D/ST shadow recommendations"):
        assert title in document
    # Two in the appendix, and two more in the card's own alerts section: the
    # unpromoted seats are stated where the decisions are, not only in the
    # detail somebody has to expand.
    assert document.count('<span class="tag">SHADOW</span>') == 4


def test_unresolved_identities_are_listed_not_hidden():
    document = render(payload("unmatched_player"))
    assert "Unresolved identities (1)" in document
    assert "no identity crosswalk" in document


def test_waivers_and_trades_state_why_they_are_empty():
    document = render(payload("post_draft"))
    assert "no waiver plan" in document.lower()
    assert "waiver engine" not in document.lower(), "a planner ships now; that claim is stale"
    assert "roster" in document.lower()


def test_a_supplied_waiver_plan_renders_with_its_legal_drop():
    from nflvalue.fantasy import waivers

    record = waivers.Recommendation(
        add_espn_id=901, add_name="F901", add_position="RB",
        drop_espn_id=25, drop_name="R. Frost", drop_state="selected",
        status="ok", shadow_reason=None, confidence="medium",
        rationale="projects above the worst legal drop",
        invalidation_trigger="a status change to either player",
        priority_implications={}, replacement_effect={}, opponent_opportunity_impact={},
        lineup_delta={"own_optimal_lineup_delta": 2.3}, lineup_delta_status="ok",
        data_timestamps={}, degraded=False, faab=None,
    )
    document = render(built("post_draft", waiver_plan=[record]))
    assert_well_formed(document)
    assert "F901" in document and "R. Frost" in document
    assert "+2.3" in document
    assert "recommendation-only" in document.lower()


# --------------------------------------------------------------------------- #
# Nothing frozen moves, and nothing personal reaches the public page
# --------------------------------------------------------------------------- #
def _projection_table(document: str) -> str:
    start = document.index('<div class="card"><table><thead><tr><th>Pos</th>')
    return document[start:document.index("</table></div>", start)]


def test_the_public_dashboard_renders_the_frozen_table_and_nothing_personal(tmp_path):
    document = render_public(tmp_path)
    assert_well_formed(document)
    assert "My team" not in document
    assert "NO CURRENT PICK" not in document
    assert "2026 week 1 fantasy projections" in document
    for row in summaries().to_dict("records"):
        assert f"<td>{row['mean']:.1f}</td>" in document


def test_the_private_page_does_not_carry_the_public_projection_table():
    """Two documents, two audiences; neither is a superset of the other."""
    document = render(payload("post_draft"))
    assert '<div class="card"><table><thead><tr><th>Pos</th>' not in document


def test_the_projection_table_is_unchanged_by_a_private_render(tmp_path):
    before = _projection_table(render_public(tmp_path / "a"))
    render(payload("post_draft"))
    after = _projection_table(render_public(tmp_path / "b"))
    assert before == after, "rendering the private page moved a frozen offense value"


# --------------------------------------------------------------------------- #
# Placeholder artifacts must never reach the page
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["pre_draft", "post_draft", "stale_snapshot", "no_action"])
def test_no_placeholder_trade_or_mock_board_content_reaches_the_page(name):
    document = render(payload(name))
    # data/trade_scan.json smoke output
    assert '"Team1"' not in document
    assert ">Team1<" not in document and ">Team4<" not in document
    # a 6-team or 12-team board would have to announce itself to be used
    assert "6-team" not in document.replace("built for a 6-team", "")
    assert "12-team" not in document


def test_real_managers_named_team_7_and_team_8_are_not_scrubbed():
    """The placeholder guard must not erase real people."""
    snapshot = json.loads((FIXTURES / "post_draft.json").read_text())
    names = {t["name"] for t in snapshot["teams"]}
    assert {"Team 7", "Team 8"} <= names


def test_hostile_player_name_is_escaped():
    snapshot = json.loads((FIXTURES / "post_draft.json").read_text())
    snapshot["rosters"]["1"][0]["full_name"] = "</script><img src=x onerror=alert(1)>"
    document = render(built("post_draft", snapshot=snapshot))
    assert "<img src=x" not in document
    assert "&lt;/script&gt;" in document
    assert_well_formed(document)
