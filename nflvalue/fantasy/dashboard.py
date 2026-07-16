"""Small dependency-free HTML renderer for weekly fantasy distributions."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd


def render_fantasy_dashboard(
    summaries: pd.DataFrame,
    path: str | Path,
    *,
    season: int,
    week: int,
    generated_at: str,
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
main{{max-width:1180px;margin:auto;padding:30px 18px}}h1{{margin:0}}p{{color:var(--muted)}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:auto}}
table{{border-collapse:collapse;width:100%;min-width:820px}}th,td{{padding:11px 13px;border-bottom:1px solid var(--line);text-align:right}}
th{{color:var(--muted);font-size:12px;text-transform:uppercase;position:sticky;top:0;background:var(--panel)}}
th:nth-child(-n+2),td:nth-child(-n+2){{text-align:left}}small{{display:block;color:var(--muted)}}b{{color:var(--accent)}}
</style></head><body><main>
<h1>{season} week {week} fantasy projections</h1>
<p>Correlated football-event Monte Carlo centered on a season-forward Bayesian/boosting/forest ensemble. Generated {html.escape(generated_at)}. Ranges are outcomes, not guarantees.</p>
<div class="card"><table><thead><tr><th>Pos</th><th>Player</th><th>Mean</th><th>Median</th><th>Raw events</th><th>P10</th><th>P90</th><th>15+</th><th>20+</th><th>Active</th><th>Check</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
</main></body></html>"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document)
