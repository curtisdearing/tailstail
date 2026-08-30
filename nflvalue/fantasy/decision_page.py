"""The private page a ``decision-card/1`` is read from.

Deliberately dull markup: one standalone HTML file, no script, no network
reference, no browser storage.  A page with no JavaScript cannot log a console
error, cannot fetch a third party, and renders the same in five years as it
does today — properties worth more here than any interaction, because the whole
document is four short answers and the reader is in a hurry.

Everything user-supplied is escaped with :func:`html.escape` including quotes.
Player names arrive from a live platform where a manager chooses them, so a
name is untrusted input and is treated as such;
``tests/test_decision_card_page.py`` renders a hostile one and asserts it lands
inert.

Private
-------
This page names a private league, its manager and its roster.  It is written
only to a gitignored location, and :mod:`nflvalue.fantasy.private_boundary`
refuses to let its content into a public payload.  The public Pages site is
built from Tailstail's own projections and the aggregate model audit, and it
never sees this file.

Appendix
--------
The full eight-section ``my_team`` contract used to be rendered into the public
weekly dashboard.  It moved here, collapsed behind a disclosure, because it is
private and because it is not the decision: the four sections above it are.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping

from . import decision_card

SECTION_TITLES = (
    "Lineup to run",
    "Changes from the lineup you have set",
    "Decisions",
    "Alerts",
)

MY_TEAM_SECTION_TITLES = (
    "My optimal lineup",
    "Start/sit decisions",
    "Draft status and picks",
    "Waiver targets and recommended drops",
    "Trade opportunities",
    "K shadow recommendations",
    "D/ST shadow recommendations",
    "Data freshness and unresolved identities",
)

PRIVACY_BANNER = (
    "PRIVATE — this page names a private league, its manager and its rosters. "
    "It is written outside version control and is never published."
)

STYLE = """
:root{--bg:#0b1220;--panel:#111c30;--ink:#ecf3ff;--muted:#92a4bf;--line:#243550;--accent:#67e8b4}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:900px;margin:auto;padding:28px 18px 60px}
h1{margin:0 0 4px;font-size:24px}h2{margin:30px 0 8px;font-size:17px}h3{margin:18px 0 6px;font-size:15px}
p{color:var(--muted)}p.honest{font-size:13px}
ul.why{color:var(--muted);font-size:13px;margin:4px 0 0 18px}
.private{background:#2a2418;border:1px solid #6b5a2b;border-radius:9px;padding:9px 12px;
 color:#f0dfae;font-size:12px;letter-spacing:.04em;margin:0 0 18px}
.nopick{background:#2a1c22;border:1px solid #5a2b38;border-radius:10px;padding:11px 13px;margin:6px 0}
.nopick b{color:#ffb4c0;letter-spacing:.06em;margin-right:9px}.nopick span{color:var(--muted);font-size:13px}
.tag{background:#243550;color:#9fb4d6;border-radius:5px;padding:1px 6px;font-size:11px;letter-spacing:.05em}
.tag.forced{background:#3a2a16;color:#e7c98b}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow-x:auto}
.decision{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin:10px 0}
.decision h3{margin:0 0 6px;color:var(--ink)}
.decision dl{display:grid;grid-template-columns:150px 1fr;gap:4px 14px;margin:10px 0 0;
 font-size:13px;color:var(--muted)}
.decision dt{color:#7f92ad}.decision dd{margin:0}
.stamps{font-size:11px;color:#6f819c;margin-top:10px;word-break:break-all}
table{border-collapse:collapse;width:100%;min-width:620px}
th,td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:right}
th{color:var(--muted);font-size:12px;text-transform:uppercase}
th:nth-child(-n+3),td:nth-child(-n+3){text-align:left}
small{display:block;color:var(--muted)}b{color:var(--accent)}
code{color:#9fb4d6;font-size:12px}
.f-stale,.f-missing{color:#ffb4c0}.f-fresh{color:var(--accent)}.f-aging{color:#e7c98b}
.set{color:var(--muted)}.change{color:#e7c98b}
details{margin-top:34px;border-top:1px solid var(--line);padding-top:14px}
summary{cursor:pointer;color:var(--muted);font-size:13px}
"""


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _no_pick(section: Mapping[str, Any]) -> str:
    """The fail-closed banner. A reason is mandatory; there is no blank state."""
    reason = section.get("reason")
    if isinstance(reason, Mapping):
        reason = reason.get("text")
    return ("<div class=\"nopick\"><b>NO CURRENT PICK</b>"
            f"<span>{_esc(reason or 'input unavailable')}</span></div>")


def _meta_line(section: Mapping[str, Any]) -> str:
    """Why and wrong-if.  The contract's own grade is deliberately not shown:
    an unqualified word like "medium" reads as a calibrated probability, and
    nothing in this project has earned that reading."""
    bits = []
    if section.get("rationale"):
        bits.append(f"Why: {_esc(section['rationale'])}")
    if section.get("invalidation_trigger"):
        bits.append(f"Wrong if: {_esc(section['invalidation_trigger'])}")
    return f"<p class=\"honest\">{' · '.join(bits)}</p>" if bits else ""


def _table(headers, rows) -> str:
    if not rows:
        return ""
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return (f"<div class=\"card\"><table><thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


def _points(value: Any, *, signed: bool = False) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    return f"{value:+.1f}" if signed else f"{value:.1f}"


# --------------------------------------------------------------------------- #
# The four sections
# --------------------------------------------------------------------------- #
def _lineup_section(card: Mapping[str, Any]) -> str:
    lineup = card.get("current_lineup") or {}
    if lineup.get("status") != "ok":
        out = _no_pick(lineup)
        if lineup.get("violations"):
            items = "".join(f"<li>{_esc(v)}</li>" for v in lineup["violations"])
            out += f"<p class=\"honest\">Why it is not legal:</p><ul class=\"why\">{items}</ul>"
        return out
    rows = []
    for seat in lineup.get("slots") or []:
        mark = ("<span class=\"set\">already set</span>" if seat.get("already_set")
                else "<span class=\"change\">change</span>")
        rows.append((
            f"<b>{_esc(seat.get('slot'))}</b>",
            _esc(seat.get("position")),
            _esc(seat.get("name")),
            _points(seat.get("projected_points")),
            mark,
        ))
    total = lineup.get("projected_points_total")
    note = (f"<p class=\"honest\">Legal lineup, {_points(total)} projected points.</p>"
            if total is not None else "")
    shadow = lineup.get("shadow_slots") or []
    if shadow:
        note += (f"<p class=\"honest\">{_esc(', '.join(str(s) for s in shadow))} are not decided "
                 "here — see below.</p>")
    return note + _table(("Slot", "Pos", "Player", "Proj", "Status"), rows)


def _changes_section(card: Mapping[str, Any]) -> str:
    changes = card.get("lineup_changes") or []
    if not changes:
        if (card.get("current_lineup") or {}).get("status") == "ok":
            return ("<p class=\"honest\">Nothing to change — every seat is already filled by the "
                    "player the model would choose.</p>")
        return _no_pick(card.get("current_lineup") or {})
    rows = [(
        f"<b>{_esc(row.get('slot'))}</b>",
        f"{_esc((row.get('start') or {}).get('name'))}"
        f"<small>{_esc((row.get('start') or {}).get('position'))}</small>",
        f"{_esc((row.get('sit') or {}).get('name') or 'nobody set')}"
        f"<small>{_esc((row.get('sit') or {}).get('position') or '')}</small>",
        _points(row.get("mean_delta"), signed=True),
        _points(row.get("median_delta"), signed=True),
    ) for row in changes]
    return _table(("Slot", "In", "Out", "Mean Δ", "Median Δ"), rows)


def _decision_block(decision: Mapping[str, Any]) -> str:
    status = str(decision.get("status") or "")
    if status == "no_current_pick":
        return ("<div class=\"decision\">"
                f"<h3>{_esc(decision.get('headline'))}</h3>"
                + _no_pick(decision) + "</div>")

    interval = decision.get("interval") or {}
    probability = decision.get("model_relative_probability")
    tag = ("<span class=\"tag forced\">FORCED</span> " if decision.get("forced")
           else f"<span class=\"tag\">{_esc(status.upper())}</span> ")
    rows = [("Why", _esc((decision.get("reason") or {}).get("text")))]

    if interval.get("status") == "ok":
        rows.append(("Points gained",
                     f"{_points(decision.get('mean_delta'), signed=True)} on average, "
                     f"{_points(decision.get('median_delta'), signed=True)} in the middle week"))
        rows.append(("Range over " + _esc(interval.get("simulations")) + " paired weeks",
                     f"{_points(interval.get('p10'), signed=True)} to "
                     f"{_points(interval.get('p90'), signed=True)}"))
    else:
        reason = interval.get("reason") or {}
        rows.append(("Range", _esc(reason.get("text") or "not measured")))

    if probability:
        rows.append((
            _esc(probability.get("label")),
            f"{float(probability['value']):.0%} — {_esc(probability.get('qualifier'))}",
        ))

    drivers = decision.get("drivers") or []
    if drivers:
        items = "".join(
            f"<li>{_esc(d.get('text'))} <small>{_esc(d.get('source'))}, as of "
            f"{_esc(d.get('as_of'))}</small></li>" for d in drivers)
        rows.append(("What changed", f"<ul class=\"why\">{items}</ul>"))

    risk = decision.get("risk") or {}
    if risk.get("text"):
        source = (f" <small>{_esc(risk.get('source'))}"
                  + (f", as of {_esc(risk['as_of'])}" if risk.get("as_of") else "")
                  + "</small>")
        rows.append(("What could go wrong", _esc(risk["text"]) + source))
    if decision.get("invalidation_trigger"):
        rows.append(("Wrong if", _esc(decision["invalidation_trigger"])))

    stamps = decision.get("provenance") or {}
    body = "".join(f"<dt>{label}</dt><dd>{value}</dd>" for label, value in rows)
    return (
        "<div class=\"decision\">"
        f"<h3>{tag}{_esc(decision.get('headline'))}</h3>"
        f"<dl>{body}</dl>"
        "<p class=\"stamps\">"
        f"model {_esc(stamps.get('model_version'))} · "
        f"scoring <code>{_esc(str(stamps.get('scoring_hash') or '')[:12])}</code> · "
        f"snapshot <code>{_esc(str(stamps.get('snapshot_hash') or '')[:12])}</code> · "
        f"data <span class=\"f-{_esc(stamps.get('freshness_state'))}\">"
        f"{_esc(stamps.get('freshness_state'))}</span></p>"
        "</div>"
    )


def _decisions_section(card: Mapping[str, Any]) -> str:
    decisions = card.get("decisions") or []
    if not decisions:
        return _no_pick({"reason": card.get("reason") or {
            "text": "Nothing on this page's inputs supports a recommendation right now."}})
    out = "".join(_decision_block(d) for d in decisions)
    withheld = card.get("withheld") or []
    if withheld:
        items = "".join(f"<li>{_esc(w.get('headline'))}</li>" for w in withheld)
        out += (f"<p class=\"honest\">{len(withheld)} further action(s) were held back to keep "
                f"this page to {decision_card.MAX_ACTIONABLE_DECISIONS}:</p>"
                f"<ul class=\"why\">{items}</ul>")
    return out


def _alerts_section(card: Mapping[str, Any]) -> str:
    alerts = card.get("alerts") or []
    shadow = "".join(
        "<div class=\"nopick\"><b>NO CURRENT PICK</b><span>"
        f"<span class=\"tag\">SHADOW</span> {_esc(seat.get('reason', {}).get('text'))}"
        "</span></div>"
        for seat in card.get("shadow_seats") or [])
    if not alerts:
        return ("<p class=\"honest\">No freshness or identity alerts: the capture is current and "
                "every rostered player was tied to a projection.</p>" + shadow)
    items = []
    for alert in alerts:
        extra = ""
        if alert.get("players"):
            extra = " (" + _esc(", ".join(str(p) for p in alert["players"])) + ")"
        elif alert.get("violations"):
            extra = " (" + _esc("; ".join(str(v) for v in alert["violations"])) + ")"
        items.append(f"<li><b>{_esc(alert.get('kind'))}</b> · {_esc(alert.get('severity'))} — "
                     f"{_esc(alert.get('text'))}{extra}</li>")
    return f"<ul class=\"why\">{''.join(items)}</ul>" + shadow


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def render(card: Mapping[str, Any], *, my_team: Mapping[str, Any] | None = None) -> str:
    """One self-contained HTML document for one card.

    Refuses a card that is not private, so a public renderer cannot be handed
    this function's output by mistake.
    """
    if card.get("visibility") != decision_card.VISIBILITY:
        raise ValueError("decision_page renders private cards only")
    league = card.get("league") or {}
    prov = card.get("provenance") or {}
    title = (f"Decision card · {league.get('team_name')} · {league.get('season')} "
             f"week {league.get('scoring_period')}")
    blocked = (_no_pick(card) if card.get("state") != "ok" else "")
    sections = (
        _lineup_section(card),
        _changes_section(card),
        _decisions_section(card),
        _alerts_section(card),
    )
    numbered = "".join(
        f"<h2 id=\"section-{index}\">{index}. {_esc(title_)}</h2>{body}"
        for index, (title_, body) in enumerate(zip(SECTION_TITLES, sections), start=1))
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<meta name=\"robots\" content=\"noindex,nofollow\">"
        f"<title>{_esc(title)}</title><style>{STYLE}</style></head><body><main>"
        f"<p class=\"private\">{_esc(PRIVACY_BANNER)}</p>"
        f"<h1>{_esc(title)}</h1>"
        f"<p class=\"honest\">{_esc(league.get('league_name'))} · team "
        f"{_esc(league.get('team_id'))} · card {_esc(card.get('schema_version'))} from "
        f"{_esc(card.get('source_contract'))} · generated {_esc(card.get('generated_at'))} · "
        f"capture <span class=\"f-{_esc(prov.get('freshness_state'))}\">"
        f"{_esc(prov.get('freshness_state'))}</span> "
        f"({_esc(prov.get('snapshot_retrieved_at'))}).</p>"
        f"<p class=\"honest\">{_esc(card.get('espn_use'))}</p>"
        f"{blocked}{numbered}"
        f"{_appendix(my_team)}"
        "</main></body></html>"
    )


def _appendix(my_team: Mapping[str, Any] | None) -> str:
    if not my_team:
        return ""
    return ("<details><summary>Full contract detail — the record behind the decisions, not the "
            "decisions themselves</summary>"
            + _my_team_section(my_team) + "</details>")


def write(card: Mapping[str, Any], *, json_path: str | Path, html_path: str | Path,
          my_team: Mapping[str, Any] | None = None) -> tuple[Path, Path]:
    """Write both private artifacts, replacing whatever was there.

    Both files are written every run, so a run that can say nothing replaces
    the page rather than leaving last week's answer on disk looking current.
    Each write goes to a temporary sibling and is renamed into place, so a
    crash mid-write cannot leave a half-page that still parses.
    """
    decision_card.validate(card)
    written = []
    for path, text in ((Path(json_path), json.dumps(card, indent=2, sort_keys=True,
                                                    default=str) + "\n"),
                       (Path(html_path), render(card, my_team=my_team))):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        tmp.replace(path)
        written.append(path)
    return written[0], written[1]


# --------------------------------------------------------------------------- #
# Appendix: the full my_team contract, moved out of the public dashboard
# --------------------------------------------------------------------------- #
# These eight sections used to render into `fantasy.html`, which the weekly
# workflow copies to the public Pages site -- so every roster, every manager's
# team name and the league id were published each week. They render here, on a
# private page, behind a disclosure, because they are the record the decisions
# were drawn from rather than the decisions themselves.

def _lineup_block(section: Mapping[str, Any]) -> str:
    if section.get("status") != "ok":
        out = _no_pick(section)
        if section.get("violations"):
            items = "".join(f"<li>{_esc(v)}</li>" for v in section["violations"])
            out += f"<p class=\"honest\">Roster legality violations:</p><ul class=\"why\">{items}</ul>"
        return out
    rows = [
        (f"<b>{_esc(e['slot'])}</b>", _esc(e["position"]), _esc(e["name"]),
         f"{e['projected_mean']:.1f}", f"{e['projected_p10']:.1f}–{e['projected_p90']:.1f}")
        for e in section["starters"]
    ]
    total = section.get("projected_total")
    note = (f"<p class=\"honest\">Legal lineup, projected total "
            f"{total:.1f} points.</p>" if isinstance(total, (int, float)) else "")
    excluded = section.get("excluded") or []
    excluded_html = ""
    if excluded:
        items = "".join(
            f"<li>{_esc(e.get('name'))} ({_esc(e.get('position'))}) — {_esc(e.get('reason'))}</li>"
            for e in excluded)
        excluded_html = (f"<p class=\"honest\">Excluded from consideration "
                         f"({len(excluded)}):</p><ul class=\"why\">{items}</ul>")
    return note + _table(("Slot", "Pos", "Player", "Proj", "P10–P90"), rows) + excluded_html


def _start_sit_block(section: Mapping[str, Any]) -> str:
    if section.get("status") != "ok":
        return _no_pick(section)
    rows = []
    for d in section["decisions"]:
        sit = d.get("sit") or {}
        unc = d.get("uncertainty") or {}
        if unc.get("status") == "ok":
            spread = f"{unc['p10_delta']:+.1f} … {unc['p90_delta']:+.1f}"
            # Named for what it is. It is the share of the model's own simulated
            # weeks in which the start wins, so it carries every error in the
            # model that drew them and has not been checked against outcomes.
            odds = f"{unc['model_relative_prob_start_scores_more']:.0%} of sims"
        else:
            spread = "—"
            odds = "unavailable"
        rows.append((
            _esc(d["slot"]),
            f"<b>{_esc(d['start']['name'])}</b><small>{_esc(d['start']['position'])}</small>",
            f"{_esc(sit.get('name') or '—')}<small>{_esc(sit.get('position') or '')}</small>",
            f"{d['projected_delta']:+.1f}",
            _esc(spread),
            _esc(odds),
        ))
    return _table(
        ("Slot", "Start", "Sit", "Δ proj", "Δ P10…P90 (paired sims)",
         "Start scores more (model-relative, not calibrated)"), rows)


def _draft_block(section: Mapping[str, Any]) -> str:
    state = _esc(section.get("state"))
    note = _esc(section.get("note") or "")
    head = (f"<p class=\"honest\">Draft state: <b>{state}</b>"
            + (f" · scheduled {_esc(section.get('scheduled_at_utc'))}"
               if section.get("scheduled_at_utc") else "")
            + (f" · slot {_esc(section.get('my_draft_slot'))} of "
               f"{_esc(section.get('rounds'))} rounds" if section.get("my_draft_slot") else "")
            + f". {note}</p>")
    picks = section.get("selections") or []
    if picks:
        rows = [(_esc(p.get("round")), _esc(p.get("overall_pick")), _esc(p.get("team_id")),
                 f"<b>{_esc(p.get('player_name'))}</b><small>{_esc(p.get('position'))}</small>",
                 _esc(p.get("espn_player_id")), _esc(p.get("selected_at") or "—"))
                for p in picks]
        body = ("<h3>Actual selections</h3>"
                + _table(("Rd", "Overall", "Team", "Player", "ESPN id", "Selected"), rows))
    else:
        body = ("<p class=\"honest\">No actual selections yet — every pick slot is still "
                "empty. Nothing below is a pick.</p>")
    targets = section.get("targets") or {}
    if targets.get("status") == "ok" and targets.get("entries"):
        rows = [(_esc(t.get("board_rank")), _esc(t.get("position")),
                 f"<b>{_esc(t.get('name'))}</b>", _esc(t.get("tier")),
                 "<span class=\"tag\">TARGET — not a pick</span>")
                for t in targets["entries"]]
        body += ("<h3>Pre-draft targets</h3>"
                 + _table(("Board", "Pos", "Player", "Tier", "Label"), rows))
    elif targets:
        body += "<h3>Pre-draft targets</h3>" + _no_pick(targets)
    return head + body


def _waiver_block(section: Mapping[str, Any]) -> str:
    if section.get("status") != "ok":
        return _no_pick(section)
    rows = []
    for t in section.get("targets", []):
        add = t.get("add") or {}
        drop = t.get("drop") or {}
        delta = (t.get("lineup_delta") or {}).get("own_optimal_lineup_delta")
        shadow = ('<span class="tag">SHADOW</span> ' if t.get("status") == "shadow" else "")
        rows.append((
            f"{shadow}<b>{_esc(add.get('name'))}</b><small>{_esc(add.get('position'))}</small>",
            f"{_esc(drop.get('name') or '—')}<small>{_esc(t.get('drop_state'))}</small>",
            (f"{delta:+.1f}" if isinstance(delta, (int, float))
             else _esc(t.get("lineup_delta_status") or "unavailable")),
            _esc(t.get("status")),
            _esc(t.get("rationale")),
        ))
    # The planner's own one-word grade is shown as its status, not as
    # "confidence": an unqualified grade is read as a calibrated probability,
    # and nothing behind this column has been graded against outcomes.
    return (_table(("Add", "Legal drop", "Lineup Δ", "Planner status", "Why"), rows)
            + "<p class=\"honest\">Every row is recommendation-only — no claim is "
              "submitted to ESPN from this page or the pipeline behind it.</p>")


def _trade_block(section: Mapping[str, Any]) -> str:
    if section.get("status") != "ok":
        return _no_pick(section)
    rows = [(_esc(o.get("counterparty_team_id")), _esc(o.get("send")), _esc(o.get("receive")),
             _esc(o.get("rationale"))) for o in section.get("opportunities", [])]
    return _table(("Team", "Send", "Receive", "Why"), rows)


def _shadow_block(section: Mapping[str, Any]) -> str:
    label = (f"<p class=\"honest\"><span class=\"tag\">SHADOW</span> "
             f"{_esc(section.get('label'))}</p>")
    if section.get("status") != "ok":
        return label + _no_pick(section)
    rows = [(_esc(e["slot"]), f"<b>{_esc(e['name'])}</b>", f"{e['projected_mean']:.1f}",
             f"{e['projected_p10']:.1f}–{e['projected_p90']:.1f}")
            for e in section.get("starters", [])]
    return label + _table(("Slot", "Player", "Proj", "P10–P90"), rows)


def _freshness_block(my_team: Mapping[str, Any]) -> str:
    fresh = my_team.get("freshness") or {}
    state = str(fresh.get("state", "unknown"))
    sources = my_team.get("sources") or []
    rows = [(_esc(s.get("name")), _esc(s.get("access")), _esc(s.get("retrieved_at")),
             f"<code>{_esc((s.get('raw_sha256') or '')[:12])}</code>") for s in sources]
    unresolved = my_team.get("unresolved_identities") or {}
    entries = unresolved.get("entries") or []
    if entries:
        items = "".join(
            f"<li>ESPN id {_esc(e.get('espn_player_id'))} — {_esc(e.get('name'))} "
            f"({_esc(e.get('position'))}): {_esc(e.get('reason'))}</li>" for e in entries)
        unresolved_html = (f"<p class=\"honest\">Unresolved identities "
                           f"({unresolved.get('count', 0)}) — reported, never scored as zero:"
                           f"</p><ul class=\"why\">{items}</ul>")
    else:
        unresolved_html = ("<p class=\"honest\">Unresolved identities: none — every rostered "
                           "player mapped to a model identity.</p>")
    return (
        f"<p class=\"honest\">Snapshot freshness: <b class=\"f-{_esc(state)}\">{_esc(state)}</b>"
        f" · captured {_esc(fresh.get('captured_at'))} · age "
        f"{_esc(fresh.get('age_hours'))}h · evaluated {_esc(fresh.get('evaluated_at'))}. "
        f"Scoring identity: {_hash_label(my_team)}.</p>"
        + _table(("Source", "Access", "Retrieved", "SHA-256"), rows)
        + unresolved_html
    )


def _hash_label(my_team: Mapping[str, Any]) -> str:
    """Null hashes are stated as unestablished, never rendered as blanks."""
    scoring = my_team.get("scoring_hash")
    roster = my_team.get("roster_slot_hash")
    if not scoring and not roster:
        return ("<b class=\"f-stale\">not established</b> — "
                + _esc(my_team.get("hash_reason") or "no league contract supplied"))
    return (f"scoring <code>{_esc(str(scoring)[:12])}</code>, "
            f"roster-slot <code>{_esc(str(roster)[:12])}</code> "
            f"(source: {_esc(my_team.get('hash_source') or 'unknown')})")


def _my_team_section(my_team: dict[str, Any] | None) -> str:
    """Eight labelled sections. Absent input renders nothing at all."""
    if not my_team:
        return ""
    league = my_team.get("league") or {}
    header = (
        "<h2>My team — Monitor</h2>"
        f"<p class=\"honest\">{_esc(league.get('team_name'))} · "
        f"{_esc(league.get('platform'))} league {_esc(league.get('league_id'))} · "
        f"{_esc(league.get('season'))} scoring period {_esc(league.get('scoring_period'))} · "
        f"team {_esc(league.get('team_id'))} of {_esc(league.get('size'))} · "
        f"contract {_esc(my_team.get('schema_version'))}. "
        "Fantasy only — separate from Fablesfable betting selections. "
        f"{_esc(my_team.get('espn_use'))}.</p>"
    )
    blocks = (
        (MY_TEAM_SECTION_TITLES[0], _lineup_block(my_team.get("optimal_lineup") or {}),
         my_team.get("optimal_lineup") or {}),
        (MY_TEAM_SECTION_TITLES[1], _start_sit_block(my_team.get("start_sit") or {}),
         my_team.get("start_sit") or {}),
        (MY_TEAM_SECTION_TITLES[2], _draft_block(my_team.get("draft") or {}),
         my_team.get("draft") or {}),
        (MY_TEAM_SECTION_TITLES[3], _waiver_block(my_team.get("waivers") or {}),
         my_team.get("waivers") or {}),
        (MY_TEAM_SECTION_TITLES[4], _trade_block(my_team.get("trades") or {}),
         my_team.get("trades") or {}),
        (MY_TEAM_SECTION_TITLES[5], _shadow_block(my_team.get("kicker_shadow") or {}),
         my_team.get("kicker_shadow") or {}),
        (MY_TEAM_SECTION_TITLES[6], _shadow_block(my_team.get("dst_shadow") or {}),
         my_team.get("dst_shadow") or {}),
        (MY_TEAM_SECTION_TITLES[7], _freshness_block(my_team), {}),
    )
    parts = [header]
    for index, (title, body, section) in enumerate(blocks, start=1):
        parts.append(f"<h3 id=\"my-team-{index}\">{index}. {_esc(title)}</h3>")
        parts.append(body)
        parts.append(_meta_line(section))
    return "".join(parts)
