"""Phase D: render the Phase C evidence as an interface that can say no.

The design constraint that shapes everything here: a week with nothing worth
publishing is a SUCCESSFUL week, and the interface has to look like it means
that. So the no-selection state is a real screen with a real explanation, not an
empty table, and never a fallback to the least-bad candidate.

Three rules the markup enforces structurally rather than by convention:

* Counter-evidence is never inside a <details>. If a reader has to click to see
  what argues against a call, the default view is an advertisement.
* The CLV banner is not dismissible while CLV is unproven. There is no close
  button and no JavaScript that could remove it -- the page ships without any
  script at all, so "dismissible" is not expressible.
* No stake sizing renders without its loss distribution in the same panel.
  ``staking_panel`` raises rather than emitting a number alone.

Charts are inline SVG built here from measured values. No CDN, no chart library,
no JavaScript: the page is a static artifact that renders identically offline
and can be diffed in review.
"""

from __future__ import annotations

import html
import math
import os
from typing import Dict, Optional, Sequence

from . import evidence as evidence_mod
from . import language_guard

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
REPORTS_DIR = os.path.join(ROOT, "reports")

BREAKEVEN = 0.5238


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def _fmt(value, digits: int = 4, dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return esc(value)


# --------------------------------------------------------------------------- #
# D.3 -- CLV panel and its non-dismissible banner
# --------------------------------------------------------------------------- #
def clv_banner(clv_report: Dict) -> str:
    """Header stating what is and is not established. Not dismissible."""
    verdict = (clv_report or {}).get("verdict", "INSUFFICIENT_SAMPLE")
    if verdict == "INSUFFICIENT_SAMPLE":
        resolved = clv_report.get("n_resolved", 0)
        minimum = clv_report.get("min_sample", 150)
        unresolved = clv_report.get("n_unresolved", 0)
        return f"""
<div class="banner banner-warn" role="alert" aria-live="polite">
  <div class="banner-title">Directional skill only — no closing-line edge is established.</div>
  <p>Closing-line value is unproven. {esc(resolved)} of the {esc(minimum)} precommitted
     decision/close pairs have resolved ({esc(unresolved)} unresolved), so this tool
     reports <strong>INSUFFICIENT_SAMPLE</strong> and makes no claim about beating a
     posted price. Grading below is directional at synthetic lines and is not a
     profit, ROI, market edge, or closing-line value claim.</p>
  <p class="banner-foot">This notice cannot be dismissed while the sample is below
     {esc(minimum)} resolved pairs. Precommitment
     <code>{esc(clv_report.get('precommitment_id', 'n/a'))}</code>.</p>
</div>"""
    tone = "banner-ok" if verdict == "GO" else "banner-stop"
    return f"""
<div class="banner {tone}" role="alert">
  <div class="banner-title">Closing-line check: {esc(verdict)}</div>
  <p>{esc(clv_report.get('detail', ''))}</p>
</div>"""


def clv_panel(clv_report: Dict, history: Optional[Sequence[Dict]] = None) -> str:
    verdict = (clv_report or {}).get("verdict", "INSUFFICIENT_SAMPLE")
    rows = [
        ("Verdict", esc(verdict)),
        ("Resolved decision/close pairs", esc(clv_report.get("n_resolved", 0))),
        ("Unresolved", esc(clv_report.get("n_unresolved", 0))),
        ("Precommitted minimum", esc(clv_report.get("min_sample", 150))),
    ]
    if verdict != "INSUFFICIENT_SAMPLE":
        ci = clv_report.get("ci95") or []
        rows += [
            ("Mean probability CLV", _fmt(clv_report.get("mean_clv_prob"), 5)),
            ("95% interval (week-block bootstrap)",
             f"{_fmt(ci[0], 5)} to {_fmt(ci[1], 5)}" if len(ci) == 2 else "—"),
            ("Share beating the close", _fmt(clv_report.get("beat_close_rate"), 4)),
        ]
    body = "".join(f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in rows)
    chart = (svg_clv_over_time(history) if history else
             '<p class="muted">No resolved pairs yet, so there is no CLV series to plot. '
             'An empty chart is shown as empty rather than as a flat line at zero.</p>')
    return f"""
<section class="panel" id="clv">
  <h2>Closing-line value</h2>
  <table class="kv">{body}</table>
  <h3>CLV over time</h3>
  {chart}
</section>"""


# --------------------------------------------------------------------------- #
# D.1 -- the no-bet state, as a first-class screen
# --------------------------------------------------------------------------- #
def no_bet_screen(*, season: int, week: int, screened: int, reasons: Sequence[str],
                  nearest: Optional[Dict] = None) -> str:
    """The primary view when nothing clears the gates."""
    reason_items = "".join(f"<li>{esc(reason)}</li>" for reason in reasons) or \
        "<li>No gate reason was recorded, which is itself a defect worth reporting.</li>"
    nearest_block = ""
    if nearest:
        nearest_block = f"""
  <div class="nearest">
    <h3>Closest candidate, shown for audit — not a selection</h3>
    <p>{esc(nearest.get('name'))} {esc(nearest.get('market'))} scored
       {esc(nearest.get('composite'))} and still failed:
       {esc(nearest.get('failed_because'))}. It is listed so the screen can be
       checked, and it is <strong>not</strong> offered as a fallback call.</p>
  </div>"""
    return f"""
<section class="panel no-bet" id="selections">
  <div class="no-bet-mark" aria-hidden="true">∅</div>
  <h2>No qualifying selections this week.</h2>
  <p class="lede">{esc(screened)} candidates were screened for {esc(season)} week
     {esc(week)}. None cleared the gates, so none is published.</p>
  <h3>Why</h3>
  <ul class="reasons">{reason_items}</ul>
  <p class="note">This is the tool working. A week with nothing to publish is a
     successful week; the alternative — promoting the least-bad candidate to fill
     the page — is how a screen becomes a recommendation engine.</p>
  {nearest_block}
</section>"""


# --------------------------------------------------------------------------- #
# D.2 -- selection card
# --------------------------------------------------------------------------- #
def _band_summary(band: Dict) -> str:
    if band.get("status") != "MEASURED":
        return (f'<span class="chip chip-unmeasured">UNMEASURED_BUCKET</span> '
                f'<span class="muted">{esc(band.get("reason", ""))}</span>')
    ci = band.get("ci95") or []
    interval = f"95% CI {_fmt(ci[0])}–{_fmt(ci[1])}" if len(ci) == 2 else "no interval"
    spans = bool(ci) and ci[0] <= BREAKEVEN
    tone = "chip-caution" if spans else "chip-measured"
    return (f'<span class="chip {tone}">band {esc(band.get("band"))}: '
            f'{_fmt(band.get("hit_rate"))} over n={esc(band.get("n"))}, {esc(interval)}</span>')


def _driver_row(driver: Dict) -> str:
    ev = driver.get("evidence") or {}
    if ev.get("status") == "REGISTERED":
        n_raw = ev.get("n_raw") or {}
        posterior = ev.get("posterior") or {}
        ci = posterior.get("ci95") or []
        sample = (f'n={esc(n_raw.get("exposed"))} exposed / {esc(n_raw.get("control"))} control, '
                  f'effective n {esc(ev.get("n_effective"))}')
        effect = (f'posterior {_fmt(posterior.get("mean"))} '
                  f'(95% CI {_fmt(ci[0])}–{_fmt(ci[1])})' if len(ci) == 2 else "—")
        label = ev.get("strength_label", "")
        registry = f'<code>{esc(ev.get("registry_id"))}</code> <span class="tag tag-{esc(label)}">{esc(label)}</span>'
    else:
        n_raw = ev.get("n_raw") or {}
        games = n_raw.get("player_games")
        sample = (f'player history {esc(games)} games, effective n {esc(ev.get("n_effective"))} '
                  f'after shrinkage' if games is not None else "no usable sample")
        effect = "—"
        label = ev.get("strength_label", "unmeasured")
        registry = f'<span class="tag tag-{esc(label)}">{esc(label)}</span> no registered study'
    notes = "".join(f'<div class="note-line">⚠ {esc(note)}</div>' for note in driver.get("notes") or [])
    return f"""
  <tr class="dir-{esc(driver.get('direction'))}">
    <td class="rank">{esc(driver.get('rank'))}</td>
    <td><strong>{esc(driver.get('label'))}</strong><div class="stmt">{esc(driver.get('statement'))}</div>{notes}</td>
    <td class="dir">{esc(driver.get('direction'))}</td>
    <td class="num">{_fmt(driver.get('log_contribution'), 5)}</td>
    <td>{sample}</td>
    <td>{effect}</td>
    <td>{registry}</td>
  </tr>"""


def selection_card(payload: Dict, *, screened: Optional[int] = None,
                   as_of: Optional[str] = None) -> str:
    selection = payload["selection"]
    band = payload["calibration_band"]
    drivers = "".join(_driver_row(d) for d in payload["decomposition"]["drivers"])
    counters = "".join(
        f'<li><span class="kind">{esc(item.get("kind"))}</span> {esc(item.get("detail"))}</li>'
        for item in payload["counter_evidence"]) or \
        '<li class="muted">Nothing was found by the checks that ran. That is a statement about the checks, not a clean bill of health.</li>'
    falsifiers = "".join(
        f'<li>{esc(item.get("statement"))} '
        f'<span class="muted">(observable by: {esc(item.get("observable_by"))})</span>'
        + (f'<div class="note-line">gap: {esc(item.get("gap"))}</div>' if item.get("gap") else "")
        + "</li>"
        for item in payload["falsifiers"])
    denominator = ("" if screened is None else
                   f'<span class="chip chip-plain">{esc(screened)} screened</span>')
    return f"""
<article class="card">
  <header class="card-head">
    <h3>{esc(selection.get('name'))} · {esc(str(selection.get('market')).replace('_', ' '))}
        {esc(selection.get('side'))} {esc(selection.get('line'))}</h3>
    <div class="chips">{_band_summary(band)} {denominator}
      <span class="chip chip-plain">projected {_fmt(selection.get('projected_mean'), 3)}</span>
      <span class="chip chip-plain">as of {esc(as_of or selection.get('as_of'))}</span>
    </div>
    <div class="muted">{esc(selection.get('team'))} vs {esc(selection.get('opponent'))} ·
      {esc(selection.get('season'))} week {esc(selection.get('week'))} ·
      line source {esc(selection.get('line_source'))}</div>
  </header>

  <h4>Drivers, ranked by contribution</h4>
  <table class="drivers">
    <thead><tr><th>#</th><th>Driver</th><th>Direction</th><th>log contribution</th>
      <th>Sample</th><th>Measured effect</th><th>Evidence</th></tr></thead>
    <tbody>{drivers}</tbody>
  </table>

  <section class="counter">
    <h4>What argues against this</h4>
    <ul>{counters}</ul>
  </section>

  <section class="falsifiers">
    <h4>What would flip it</h4>
    <ul>{falsifiers}</ul>
  </section>
</article>"""


# --------------------------------------------------------------------------- #
# D.4 -- trend views, including where the model is worst
# --------------------------------------------------------------------------- #
def _svg_open(width: int, height: int, title: str) -> str:
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}" '
            f'class="chart">')


def svg_calibration_curve(bins: Sequence[Dict], width: int = 420, height: int = 300) -> str:
    """Predicted vs realized. The diagonal is drawn so miscalibration is visible."""
    if not bins:
        return '<p class="muted">No calibration bins recorded.</p>'
    pad = 42
    inner_w, inner_h = width - pad * 2, height - pad * 2

    def point(px, py):
        return (pad + px * inner_w, height - pad - py * inner_h)

    parts = [_svg_open(width, height, "Calibration: predicted vs realized")]
    parts.append(f'<rect x="{pad}" y="{pad}" width="{inner_w}" height="{inner_h}" class="plot"/>')
    x0, y0 = point(0, 0)
    x1, y1 = point(1, 1)
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" class="ideal"/>')
    path = []
    for entry in bins:
        predicted = float(entry.get("predicted") or 0)
        actual = float(entry.get("actual") or 0)
        cx, cy = point(predicted, actual)
        path.append(f"{cx:.1f},{cy:.1f}")
        radius = 3 + min(6.0, math.sqrt(float(entry.get("n") or 1)) / 4.0)
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" class="pt">'
                     f'<title>predicted {predicted:.4f}, realized {actual:.4f}, '
                     f'n={entry.get("n")}</title></circle>')
    if len(path) > 1:
        parts.append(f'<polyline points="{" ".join(path)}" class="series"/>')
    parts.append(f'<text x="{width/2:.0f}" y="{height-8}" class="axis">predicted probability</text>')
    parts.append(f'<text x="12" y="{height/2:.0f}" class="axis" '
                 f'transform="rotate(-90 12 {height/2:.0f})">realized rate</text>')
    parts.append('</svg>')
    return "".join(parts)


def svg_bucket_hit_rates(bands: Dict, width: int = 460, height: int = 300) -> str:
    """Hit rate by bucket WITH intervals, and the breakeven line drawn in."""
    entries = [(name, data) for name, data in (bands or {}).items() if data.get("n")]
    if not entries:
        return '<p class="muted">No graded buckets.</p>'
    pad_l, pad_b, pad_t = 52, 46, 16
    inner_w = width - pad_l - 16
    inner_h = height - pad_b - pad_t
    lo, hi = 0.35, 0.72

    def ypos(value):
        return pad_t + inner_h * (1 - (value - lo) / (hi - lo))

    step = inner_w / len(entries)
    parts = [_svg_open(width, height, "Hit rate by composite bucket with 95% intervals")]
    parts.append(f'<rect x="{pad_l}" y="{pad_t}" width="{inner_w}" height="{inner_h}" class="plot"/>')
    be = ypos(BREAKEVEN)
    parts.append(f'<line x1="{pad_l}" y1="{be:.1f}" x2="{pad_l+inner_w}" y2="{be:.1f}" class="breakeven"/>')
    parts.append(f'<text x="{pad_l+inner_w-4}" y="{be-6:.1f}" class="axis end">breakeven {BREAKEVEN}</text>')
    for index, (name, data) in enumerate(entries):
        n = int(data["n"])
        rate = float(data["hit_rate"])
        hits = int(round(rate * n))
        ci = evidence_mod.jeffreys_interval(hits, n) or [rate, rate]
        cx = pad_l + step * (index + 0.5)
        unmeasured = n < evidence_mod.MIN_BUCKET_N
        klass = "bar-unmeasured" if unmeasured else "bar"
        parts.append(f'<line x1="{cx:.1f}" y1="{ypos(ci[0]):.1f}" x2="{cx:.1f}" '
                     f'y2="{ypos(ci[1]):.1f}" class="{klass}-ci"/>')
        parts.append(f'<circle cx="{cx:.1f}" cy="{ypos(rate):.1f}" r="4" class="{klass}">'
                     f'<title>{name}: {rate:.4f} over n={n}, 95% CI {ci[0]:.4f}–{ci[1]:.4f}</title></circle>')
        parts.append(f'<text x="{cx:.1f}" y="{height-24}" class="axis mid">{esc(name)}</text>')
        label = f"n={n}" + (" · unmeasured" if unmeasured else "")
        parts.append(f'<text x="{cx:.1f}" y="{height-10}" class="axis mid small">{esc(label)}</text>')
    for tick in (0.40, 0.50, 0.60, 0.70):
        parts.append(f'<text x="{pad_l-8}" y="{ypos(tick)+4:.1f}" class="axis end">{tick:.2f}</text>')
    parts.append('</svg>')
    return "".join(parts)


def svg_clv_over_time(series: Sequence[Dict], width: int = 460, height: int = 240) -> str:
    points = [(entry.get("label"), entry.get("mean_clv_prob")) for entry in series or []]
    points = [(label, float(value)) for label, value in points if value is not None]
    if not points:
        return ('<p class="muted">No resolved decision/close pairs, so no CLV series exists. '
                'Nothing is plotted rather than plotting zero.</p>')
    pad = 42
    inner_w, inner_h = width - pad * 2, height - pad - 28
    values = [value for _label, value in points]
    lo, hi = min(values + [0.0]), max(values + [0.0])
    span = (hi - lo) or 1.0

    def ypos(value):
        return 20 + inner_h * (1 - (value - lo) / span)

    step = inner_w / max(1, len(points) - 1)
    coords = [f"{pad + step*i:.1f},{ypos(v):.1f}" for i, (_l, v) in enumerate(points)]
    parts = [_svg_open(width, height, "Mean probability CLV over time")]
    parts.append(f'<line x1="{pad}" y1="{ypos(0):.1f}" x2="{pad+inner_w}" y2="{ypos(0):.1f}" class="breakeven"/>')
    parts.append(f'<polyline points="{" ".join(coords)}" class="series"/>')
    parts.append('</svg>')
    return "".join(parts)


def svg_regime_errors(regimes: Dict, width: int = 470, height: int = 320) -> str:
    """Interval coverage by regime, worst first. The point of this chart is the bad end."""
    rows = []
    for name, methods in (regimes or {}).items():
        metrics = (methods or {}).get("calibrated_monte_carlo") or \
            next(iter((methods or {}).values()), None)
        if not metrics or metrics.get("coverage80") is None:
            continue
        rows.append((name, float(metrics["coverage80"]), int(metrics.get("n") or 0),
                     float(metrics.get("mae") or 0.0)))
    if not rows:
        return '<p class="muted">No per-regime error rates recorded.</p>'
    rows.sort(key=lambda r: abs(r[1] - 0.80), reverse=True)
    pad_l, pad_t = 190, 18
    row_h = min(26, (height - pad_t - 26) / max(1, len(rows)))
    inner_w = width - pad_l - 20
    lo, hi = 0.60, 1.0

    def xpos(value):
        return pad_l + inner_w * (value - lo) / (hi - lo)

    parts = [_svg_open(width, height, "80% interval coverage by regime, worst first")]
    nominal = xpos(0.80)
    parts.append(f'<line x1="{nominal:.1f}" y1="{pad_t}" x2="{nominal:.1f}" '
                 f'y2="{pad_t + row_h*len(rows):.1f}" class="breakeven"/>')
    parts.append(f'<text x="{nominal:.1f}" y="{pad_t-6}" class="axis mid">nominal 0.8</text>')
    for index, (name, coverage, n, mae) in enumerate(rows):
        y = pad_t + row_h * index + row_h / 2
        worst = index == 0
        parts.append(f'<text x="{pad_l-10}" y="{y+4:.1f}" class="axis end">{esc(name)}</text>')
        parts.append(f'<line x1="{nominal:.1f}" y1="{y:.1f}" x2="{xpos(coverage):.1f}" '
                     f'y2="{y:.1f}" class="{"regime-worst" if worst else "regime"}"/>')
        parts.append(f'<circle cx="{xpos(coverage):.1f}" cy="{y:.1f}" r="4" '
                     f'class="{"regime-worst" if worst else "regime"}">'
                     f'<title>{esc(name)}: coverage {coverage:.4f}, n={n}, MAE {mae:.4f}</title></circle>')
        parts.append(f'<text x="{xpos(coverage)+9:.1f}" y="{y+4:.1f}" class="axis small">'
                     f'{coverage:.4f} (n={n})</text>')
    parts.append('</svg>')
    return "".join(parts)


def trends_panel(*, calibration_bins: Sequence[Dict], bands: Dict,
                 regimes: Dict, clv_history: Sequence[Dict]) -> str:
    worst = None
    for name, methods in (regimes or {}).items():
        metrics = (methods or {}).get("calibrated_monte_carlo") or \
            next(iter((methods or {}).values()), None)
        if metrics and metrics.get("coverage80") is not None:
            deviation = abs(float(metrics["coverage80"]) - 0.80)
            if worst is None or deviation > worst[0]:
                worst = (deviation, name, metrics)
    worst_note = ""
    if worst:
        _dev, name, metrics = worst
        worst_note = (f'<p class="note">Worst measured regime: <strong>{esc(name)}</strong> — '
                      f'80% interval coverage {_fmt(metrics.get("coverage80"))} against a nominal '
                      f'0.8 on n={esc(metrics.get("n"))}, MAE {_fmt(metrics.get("mae"))}. '
                      f'It is listed first because a dashboard that sorts its weakest regime to '
                      f'the bottom is choosing not to show it.</p>')
    return f"""
<section class="panel" id="trends">
  <h2>Trends</h2>
  <div class="grid2">
    <div><h3>Calibration: predicted vs realized</h3>{svg_calibration_curve(calibration_bins)}</div>
    <div><h3>Hit rate by bucket, with 95% intervals</h3>{svg_bucket_hit_rates(bands)}
      <p class="note">Buckets below n={evidence_mod.MIN_BUCKET_N} are drawn hollow and carry no
         claim. Intervals are Jeffreys; the horizontal line is the {BREAKEVEN} directional
         reference at a -110 price.</p></div>
  </div>
  <div class="grid2">
    <div><h3>Mean probability CLV over time</h3>{svg_clv_over_time(clv_history)}</div>
    <div><h3>Interval coverage by regime, worst first</h3>{svg_regime_errors(regimes)}{worst_note}</div>
  </div>
</section>"""


# --------------------------------------------------------------------------- #
# D.5 -- staking, only ever beside its loss distribution
# --------------------------------------------------------------------------- #
class StakingRefused(RuntimeError):
    """Stake sizing was requested without the evidence required to show it."""


def staking_panel(*, config: Dict, premortem: Dict, killcheck: Dict,
                  edge: Optional[float] = None) -> str:
    """Render stake sizing bounded by the precommitted fraction, with the fan.

    Raises if the drawdown distribution or the kill-check status is missing. A
    recommended stake with no loss distribution beside it is the single most
    dangerous thing this interface could render (PREMORTEM.md F8), so it is not
    reachable by omission.
    """
    if not premortem:
        raise StakingRefused(
            "no premortem drawdown distribution available; a stake size will not be "
            "rendered without the loss distribution beside it")
    if not killcheck:
        raise StakingRefused("no kill-check status available; staking display refused")

    kelly = float(config.get("kelly_multiplier", 0.0))
    cap = float(config.get("max_stake_pct", 0.0))
    bankroll = float(config.get("bankroll_units", 100.0))
    verdict = killcheck.get("verdict", "INSUFFICIENT_SAMPLE")

    raw = None if edge is None else max(0.0, float(edge))
    fraction = None if raw is None else min(raw * kelly, cap)
    if verdict != "GO":
        sizing = (f'<p class="stop"><strong>No stake is sized.</strong> The precommitted '
                  f'kill check reads <code>{esc(verdict)}</code>, so this panel shows the '
                  f'bounds that WOULD apply and the loss distribution, not a recommendation.</p>')
    else:
        sizing = (f'<p>Precommitted bound: {esc(kelly)}x Kelly, capped at {esc(cap)} of a '
                  f'{esc(bankroll)}-unit bankroll → '
                  f'{_fmt(fraction, 4)} ({_fmt((fraction or 0) * bankroll, 2)} units).</p>')

    rows = []
    for scenario in premortem.get("A_flat", [])[:6]:
        rows.append(
            f"<tr><td>{_fmt(scenario.get('p'), 4)}</td><td>{esc(scenario.get('nbets'))}</td>"
            f"<td>{_fmt(scenario.get('unit_frac'), 3)}</td>"
            f"<td>{_fmt(scenario.get('median'), 1)}</td>"
            f"<td>{_fmt(scenario.get('p5'), 1)}</td><td>{_fmt(scenario.get('p95'), 1)}</td>"
            f"<td>{_fmt(scenario.get('prob_down_50pct'), 4)}</td>"
            f"<td>{_fmt(scenario.get('prob_ruin'), 4)}</td></tr>")
    return f"""
<section class="panel" id="staking">
  <h2>Staking</h2>
  {sizing}
  <h3>Drawdown fan (premortem Monte Carlo, {esc(premortem.get('constants', {}).get('trials'))} trials)</h3>
  <table class="fan">
    <thead><tr><th>true p</th><th>bets</th><th>unit frac</th><th>median</th>
      <th>5th pct</th><th>95th pct</th><th>P(down 50%)</th><th>P(ruin)</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <p class="note">These are outcomes of the precommitted sizing under assumed true win
     rates, not predictions. The 5th percentile column is the one to read first.</p>
  <p class="note">Kill check: <code>{esc(verdict)}</code> —
     {esc(killcheck.get('detail', ''))}</p>
</section>"""


# --------------------------------------------------------------------------- #
# Page assembly
# --------------------------------------------------------------------------- #
CSS = """
:root{--bg:#0e1116;--panel:#161b22;--line:#2b323c;--ink:#e6edf3;--muted:#8b949e;
--warn:#d9a441;--stop:#e5534b;--ok:#3fb950;--accent:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:24px}
h1{font-size:21px;margin:0 0 4px} h2{font-size:17px;margin:0 0 12px}
h3{font-size:14px;margin:16px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
h4{font-size:13px;margin:18px 0 6px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px;margin:16px 0}
.muted{color:var(--muted)} .note{color:var(--muted);font-size:13px;margin:8px 0 0}
.note-line{color:var(--warn);font-size:12.5px;margin-top:4px}
code{background:#0b0f14;padding:1px 5px;border-radius:4px;font-size:12.5px}
.banner{border-radius:10px;padding:14px 16px;margin:16px 0;border:1px solid}
.banner-warn{background:#2a2113;border-color:var(--warn)}
.banner-stop{background:#2c1618;border-color:var(--stop)}
.banner-ok{background:#12261a;border-color:var(--ok)}
.banner-title{font-weight:600;margin-bottom:6px}
.banner p{margin:6px 0} .banner-foot{font-size:12.5px;color:var(--muted)}
.no-bet{text-align:center;padding:38px 22px}
.no-bet-mark{font-size:52px;color:var(--muted);line-height:1}
.no-bet h2{font-size:22px;margin:10px 0 6px}
.no-bet .lede{color:var(--muted);margin:0 0 18px}
.reasons{display:inline-block;text-align:left;margin:0 auto;padding-left:20px}
.reasons li{margin:5px 0}
.nearest{margin-top:22px;border-top:1px dashed var(--line);padding-top:16px;text-align:left}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px;margin:16px 0}
.card-head h3{font-size:17px;color:var(--ink);text-transform:none;letter-spacing:0;margin:0 0 8px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px}
.chip{font-size:12.5px;padding:3px 9px;border-radius:999px;border:1px solid var(--line);background:#0b0f14}
.chip-measured{border-color:var(--ok);color:#8fe0a4}
.chip-caution{border-color:var(--warn);color:#f0c674}
.chip-unmeasured{border-color:var(--stop);color:#ff9c96;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}
thead th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase}
.kv th{width:280px;color:var(--muted);font-weight:500}
.drivers .rank{width:26px;color:var(--muted)} .drivers .num{font-variant-numeric:tabular-nums}
.stmt{color:var(--muted);margin-top:3px}
.dir{text-transform:uppercase;font-size:11.5px;letter-spacing:.04em}
tr.dir-supports .dir{color:var(--ok)} tr.dir-opposes .dir{color:var(--stop)}
tr.dir-not_ranked .dir{color:var(--muted)}
.tag{font-size:11px;padding:1px 7px;border-radius:999px;border:1px solid var(--line)}
.tag-thin,.tag-weak,.tag-unsupported,.tag-unmeasured,.tag-rejected-by-gate{color:#f0c674;border-color:var(--warn)}
.tag-strong,.tag-adequate{color:#8fe0a4;border-color:var(--ok)}
.counter{border-left:3px solid var(--stop);padding-left:14px;margin-top:18px;background:#1b1416;
border-radius:0 8px 8px 0;padding-top:2px;padding-bottom:8px}
.counter ul,.falsifiers ul{margin:6px 0;padding-left:20px}
.counter li,.falsifiers li{margin:6px 0}
.kind{display:inline-block;font-size:11px;color:var(--muted);text-transform:uppercase;
letter-spacing:.04em;margin-right:6px}
.falsifiers{border-left:3px solid var(--accent);padding-left:14px;margin-top:18px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:22px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
.chart{width:100%;height:auto}
.plot{fill:#0b0f14;stroke:var(--line)}
.ideal{stroke:var(--muted);stroke-dasharray:4 4}
.breakeven{stroke:var(--warn);stroke-dasharray:5 4}
.series{fill:none;stroke:var(--accent);stroke-width:2}
.pt{fill:var(--accent);opacity:.85}
.bar{fill:var(--accent)} .bar-ci{stroke:var(--accent);stroke-width:2}
.bar-unmeasured{fill:none;stroke:var(--stop);stroke-width:1.5}
.bar-unmeasured-ci{stroke:var(--stop);stroke-width:2;stroke-dasharray:3 3}
.regime{stroke:var(--accent);stroke-width:2;fill:var(--accent)}
.regime-worst{stroke:var(--stop);stroke-width:3;fill:var(--stop)}
.axis{fill:var(--muted);font-size:10.5px} .axis.mid{text-anchor:middle}
.axis.end{text-anchor:end} .axis.small{font-size:9.5px}
.stop{color:#ff9c96} .fan td{font-variant-numeric:tabular-nums}
footer{color:var(--muted);font-size:12.5px;margin:26px 0 8px}
"""


def render_page(*, season: int, week: int, as_of: str, clv_report: Dict,
                selections: Sequence[Dict] = (), screened: int = 0,
                no_bet_reasons: Sequence[str] = (), nearest: Optional[Dict] = None,
                calibration_bins: Sequence[Dict] = (), bands: Optional[Dict] = None,
                regimes: Optional[Dict] = None, clv_history: Sequence[Dict] = (),
                staking: Optional[Dict] = None, verify_language: bool = True) -> str:
    """Assemble the page. Fails closed on banned language."""
    if selections:
        body = "".join(selection_card(p, screened=screened, as_of=as_of) for p in selections)
        body = f'<section class="panel" id="selections"><h2>Selections</h2>' \
               f'<p class="muted">{len(selections)} published from {screened} screened.</p></section>{body}'
    else:
        body = no_bet_screen(season=season, week=week, screened=screened,
                             reasons=no_bet_reasons, nearest=nearest)

    staking_html = ""
    if staking:
        staking_html = staking_panel(config=staking["config"], premortem=staking["premortem"],
                                     killcheck=staking.get("killcheck") or clv_report,
                                     edge=staking.get("edge"))

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prop screen — {esc(season)} week {esc(week)}</title>
<style>{CSS}</style></head>
<body><div class="wrap">
<h1>Prop screen — {esc(season)} week {esc(week)}</h1>
<div class="muted">Generated {esc(as_of)} · every figure below is a stored deterministic
value with its own sample size, interval, and as-of stamp.</div>
{clv_banner(clv_report)}
{body}
{clv_panel(clv_report, clv_history)}
{trends_panel(calibration_bins=calibration_bins, bands=bands or {},
              regimes=regimes or {}, clv_history=clv_history)}
{staking_html}
<footer>Directional grading at synthetic trailing-mean lines. This page makes no
profit, ROI, market edge, or closing-line value claim; see the accuracy protocol
for the pre-registered limits on what these numbers can support.</footer>
</div></body></html>"""

    if verify_language:
        language_guard.assert_clean(page, source="evidence_view", is_html=True)
    return page
