"""Small dependency-free HTML renderer for weekly fantasy distributions."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

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
    # The per-player comparison table used to be rendered here, and this file is
    # copied verbatim to the public Pages site every week. Those rows are ESPN's
    # own projections, fetched under terms recorded on every snapshot as granting
    # no redistribution right, so publishing them was the leak -- and it survived
    # the move of `data/fantasy_latest.json` out of `_site`, because it travelled
    # inside the page rather than beside it. The withheld count is stated so the
    # absence reads as a decision rather than an empty week.
    withheld = int(identity.get("matched") or 0)
    table = (
        "<p class=\"honest\">Per-player rows are not published: ESPN's projections are "
        f"used here as an external challenger under terms that grant no redistribution "
        f"right, so the {withheld} matched player comparisons behind this week's grading "
        "stay local. The week-by-week scoreboard below is the published result.</p>"
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


#: Row-level fields that may never reach the published page.
ROW_LEVEL_ESPN_FIELDS = ("current_week_rows", "espn_pts", "model_pts", "player_name")


def render_fantasy_dashboard(
    summaries: pd.DataFrame,
    path: str | Path,
    *,
    season: int,
    week: int,
    generated_at: str,
    espn_comparison: dict[str, Any] | None = None,
) -> None:
    """The public weekly page: Tailstail's own projections and the ESPN grading.

    It carries no personalised content.  The eight-section ``my_team`` contract
    that used to render below the table moved to
    :mod:`nflvalue.fantasy.decision_page`, which writes to a gitignored path,
    because this file is copied verbatim to a public site every week and a
    private league's rosters, team names and league id went with it.
    """
    # Fail closed at the renderer rather than at the caller. This page is copied
    # to a public site by the weekly workflow, so "the pipeline passes the
    # redacted object" has to be enforced where the markup is written -- one
    # caller passing the raw payload is all it took last time.
    if espn_comparison is not None:
        from .private_boundary import PrivateDataLeak

        for field in ROW_LEVEL_ESPN_FIELDS:
            if field in espn_comparison:
                raise PrivateDataLeak(
                    f"the public dashboard was handed row-level ESPN data ({field!r}); "
                    "pass private_boundary.public_espn_comparison(...) instead")

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
{_espn_section(espn_comparison)}
</main></body></html>"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document)
