"""Small dependency-free HTML renderer for weekly fantasy distributions."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def _espn_section(espn_comparison: dict[str, Any] | None) -> str:
    """ESPN-vs-model section: honest labels, coverage, no edge language.

    The existing projection table is untouched; this renders (or explains the
    absence of) the external-challenger comparison below it.
    """
    if not espn_comparison:
        return ""
    status = espn_comparison.get("status", "ok")
    disclaimer = html.escape(str(espn_comparison.get("disclaimer", "")))
    if status != "ok":
        reason = html.escape(str(espn_comparison.get("error", "unavailable")))
        return (
            "<h2>ESPN vs model</h2>"
            f"<p class=\"honest\">ESPN pull unavailable this run: {reason}. "
            "No comparison is fabricated; prior graded weeks (if any) remain below.</p>"
            + _espn_series_table(espn_comparison)
            + f"<p class=\"honest\">{disclaimer}</p>"
        )
    provenance = espn_comparison.get("espn_provenance") or {}
    identity = espn_comparison.get("identity") or {}
    retrieved = html.escape(str(provenance.get("retrieved_at", "unknown")))
    source = html.escape(str((provenance.get("source") or {}).get("name", "ESPN")))
    scoring = provenance.get("scoring") or {}
    basis_delta = scoring.get("rescored_vs_applied_mean_abs_delta")
    coverage = (
        f"{identity.get('matched', 0)}/{identity.get('espn_players', 0)} ESPN players matched "
        f"({identity.get('coverage_pct', 0)}%); "
        f"{identity.get('unmatched_no_crosswalk_count', 0)} without identity crosswalk and "
        f"{identity.get('unmatched_model_not_projected_count', 0)} not projected by the model "
        "are reported in fantasy_latest.json, not dropped silently"
    )
    rows_html = []
    for row in espn_comparison.get("current_week_rows", []):
        rows_html.append(
            "<tr>"
            f"<td>{row['abs_delta_rank']}</td>"
            f"<td>{html.escape(str(row['position']))}</td>"
            f"<td><b>{html.escape(str(row['player_name']))}</b>"
            f"<small>{html.escape(str(row['team']))}</small></td>"
            f"<td>{row['espn_pts']:.1f}</td>"
            f"<td>{row['model_pts']:.1f}</td>"
            f"<td>{row['delta']:+.1f}</td>"
            "</tr>"
        )
    table = (
        "<div class=\"card\"><table><thead><tr>"
        "<th>Δ rank</th><th>Pos</th><th>Player</th>"
        "<th>ESPN (PPR)</th><th>Model (PPR)</th><th>Model − ESPN</th>"
        f"</tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>"
        if rows_html
        else "<p class=\"honest\">No pre-kickoff comparison rows for this week yet.</p>"
    )
    basis_note = (
        " ESPN raw stat projections re-scored with the model's own full-PPR scorer"
        + (
            f" (mean |rescored − ESPN applied| = {basis_delta:.2f} pts, recorded per snapshot)."
            if isinstance(basis_delta, (int, float))
            else " (applied totals used where raw stats were absent; basis recorded per player)."
        )
    )
    return (
        "<h2>ESPN vs model</h2>"
        f"<p class=\"honest\">{source} projections retrieved {retrieved} (pre-kickoff snapshot, "
        f"content-hashed).{basis_note} Coverage: {coverage}.</p>"
        + table
        + _espn_series_table(espn_comparison)
        + f"<p class=\"honest\">{disclaimer}</p>"
    )


def _espn_series_table(espn_comparison: dict[str, Any]) -> str:
    series = espn_comparison.get("season_series") or []
    if not series:
        return (
            "<p class=\"honest\">No graded weeks yet — grading appears here after "
            "each week's games, using only pre-kickoff snapshots (never backfilled).</p>"
        )
    rows = []
    for week in series:
        mae_espn = week.get("mae_espn")
        mae_model = week.get("mae_model")
        verdict = "—"
        if isinstance(mae_espn, (int, float)) and isinstance(mae_model, (int, float)):
            if mae_model < mae_espn:
                verdict = "model"
            elif mae_espn < mae_model:
                verdict = "ESPN"
            else:
                verdict = "tie"
        rows.append(
            "<tr>"
            f"<td>{week['week']}</td><td>{week['n_played']}</td>"
            f"<td>{'' if mae_espn is None else format(mae_espn, '.2f')}</td>"
            f"<td>{'' if mae_model is None else format(mae_model, '.2f')}</td>"
            f"<td>{week.get('espn_closer', 0)}</td><td>{week.get('model_closer', 0)}</td>"
            f"<td>{verdict}</td>"
            "</tr>"
        )
    return (
        "<h3>Season grading — lower MAE was closer</h3>"
        "<div class=\"card\"><table><thead><tr>"
        "<th>Week</th><th>n</th><th>ESPN MAE</th><th>Model MAE</th>"
        "<th>ESPN closer</th><th>Model closer</th><th>Week winner</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


# --------------------------------------------------------------------------- #
# My team — the Monitor surface
# --------------------------------------------------------------------------- #
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


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _no_pick(section: Mapping[str, Any]) -> str:
    """The fail-closed banner. A reason is mandatory; there is no blank state."""
    reason = _esc(section.get("reason") or "input unavailable")
    return (
        "<div class=\"nopick\"><b>NO CURRENT PICK</b>"
        f"<span>{reason}</span></div>"
    )


def _meta_line(section: Mapping[str, Any]) -> str:
    bits = []
    if section.get("rationale"):
        bits.append(f"Why: {_esc(section['rationale'])}")
    if section.get("confidence"):
        bits.append(f"Confidence: {_esc(section['confidence'])}")
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
        rows.append((
            _esc(d["slot"]),
            f"<b>{_esc(d['start']['name'])}</b><small>{_esc(d['start']['position'])}</small>",
            f"{_esc(sit.get('name') or '—')}<small>{_esc(sit.get('position') or '')}</small>",
            f"{d['projected_delta']:+.1f}",
            f"{unc.get('p10_delta', 0):+.1f} … {unc.get('p90_delta', 0):+.1f}",
            _esc(d.get("confidence")),
        ))
    return _table(("Slot", "Start", "Sit", "Δ proj", "Δ P10…P90", "Confidence"), rows)


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
            _esc(t.get("confidence")),
            _esc(t.get("rationale")),
        ))
    return (_table(("Add", "Legal drop", "Lineup Δ", "Confidence", "Why"), rows)
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
        f"contract {_esc(my_team.get('schema_version'))} · "
        f"overall confidence {_esc(my_team.get('confidence'))}. "
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


def render_fantasy_dashboard(
    summaries: pd.DataFrame,
    path: str | Path,
    *,
    season: int,
    week: int,
    generated_at: str,
    espn_comparison: dict[str, Any] | None = None,
    my_team: dict[str, Any] | None = None,
) -> None:
    rows = []
    for row in summaries.sort_values(["position", "mean"], ascending=[True, False]).to_dict("records"):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['position']))}</td>"
            f"<td><b>{html.escape(str(row['player_name']))}</b><small>{html.escape(str(row['team']))}</small></td>"
            f"<td>{row['mean']:.1f}</td><td>{row['median']:.1f}</td>"
            f"<td>{row['event_simulator_mean']:.1f}</td>"
            f"<td>{row['p10']:.1f}</td><td>{row['p90']:.1f}</td>"
            f"<td>{100 * row['prob_15_plus']:.0f}%</td>"
            f"<td>{100 * row['prob_20_plus']:.0f}%</td>"
            f"<td>{100 * row['availability_probability']:.0f}%</td>"
            f"<td>{'review' if row.get('component_model_disagreement') else ''}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fantasy projections · {season} week {week}</title>
<style>
:root{{--bg:#0b1220;--panel:#111c30;--ink:#ecf3ff;--muted:#92a4bf;--line:#243550;--accent:#67e8b4}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:30px 18px}}h1{{margin:0}}h2{{margin:34px 0 6px}}h3{{margin:22px 0 6px}}p{{color:var(--muted)}}
p.honest{{font-size:13px}}ul.why{{color:var(--muted);font-size:13px;margin:4px 0 0 18px}}
.nopick{{background:#2a1c22;border:1px solid #5a2b38;border-radius:10px;padding:11px 13px;margin:6px 0}}
.nopick b{{color:#ffb4c0;letter-spacing:.06em;margin-right:9px}}.nopick span{{color:var(--muted);font-size:13px}}
.tag{{background:#243550;color:#9fb4d6;border-radius:5px;padding:1px 6px;font-size:11px;letter-spacing:.05em}}
code{{color:#9fb4d6;font-size:12px}}.f-stale,.f-missing{{color:#ffb4c0}}.f-fresh{{color:var(--accent)}}.f-aging{{color:#e7c98b}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:auto}}
table{{border-collapse:collapse;width:100%;min-width:820px}}th,td{{padding:11px 13px;border-bottom:1px solid var(--line);text-align:right}}
th{{color:var(--muted);font-size:12px;text-transform:uppercase;position:sticky;top:0;background:var(--panel)}}
th:nth-child(-n+2),td:nth-child(-n+2){{text-align:left}}small{{display:block;color:var(--muted)}}b{{color:var(--accent)}}
</style></head><body><main>
<h1>{season} week {week} fantasy projections</h1>
<p>Correlated football-event Monte Carlo centered on a season-forward Bayesian/boosting/forest ensemble. Generated {html.escape(generated_at)}. Ranges are outcomes, not guarantees.</p>
<div class="card"><table><thead><tr><th>Pos</th><th>Player</th><th>Mean</th><th>Median</th><th>Raw events</th><th>P10</th><th>P90</th><th>15+</th><th>20+</th><th>Active</th><th>Check</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
{_my_team_section(my_team)}
{_espn_section(espn_comparison)}
</main></body></html>"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document)
