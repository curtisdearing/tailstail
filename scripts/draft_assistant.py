"""Generate the self-contained live draft-day assistant page.

Usage:
    python scripts/draft_assistant.py [--board data/draft_board_2026_12team.csv]
        [--teams 12] [--rounds 15] [--output reports/draft_assistant_2026.html]

The page embeds the full board and runs entirely client-side on draft night:
mark picks as they happen, and it recomputes best available (progressive
ceiling weighting for YOUR next round), tier-break alerts, positional runs,
roster needs, and survival odds to your next pick.  State can be exported /
restored as a compact string (no browser storage is used).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>tailstail · Draft Assistant 2026</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--line:#21262d;--text:#e6edf3;--dim:#8b949e;
--accent:#4f9cf9;--good:#3fb950;--warn:#d29922;--bad:#f85149;
--qb:#4f9cf9;--rb:#f0883e;--wr:#3fb950;--te:#d29922}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--text);font:14px/1.45 -apple-system,'Segoe UI',Roboto,sans-serif;padding:14px}
h1{font-size:19px;margin-bottom:2px} .sub{color:var(--dim);font-size:12px;margin-bottom:12px}
.grid{display:grid;grid-template-columns:minmax(430px,3fr) minmax(280px,2fr);gap:12px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:12px}
.panel h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);margin-bottom:8px}
#clock{font-size:16px;font-weight:700} #clock .you{color:var(--accent)}
.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:10px}
select,input,button{background:#0d1117;border:1px solid var(--line);color:var(--text);border-radius:7px;padding:6px 9px;font-size:13px}
button{cursor:pointer} button:hover{border-color:var(--accent)}
button.primary{background:var(--accent);color:#04121f;font-weight:700;border-color:var(--accent)}
button.mini{padding:2px 8px;font-size:11px}
table{width:100%;border-collapse:collapse} td,th{padding:4px 7px;text-align:left;border-bottom:1px solid var(--line);font-size:13px}
th{color:var(--dim);font-size:11px;text-transform:uppercase}
.pos{display:inline-block;min-width:26px;text-align:center;border-radius:5px;font-size:11px;font-weight:700;padding:1px 4px;color:#0d1117}
.pos.QB{background:var(--qb)} .pos.RB{background:var(--rb)} .pos.WR{background:var(--wr)} .pos.TE{background:var(--te)}
.dim{color:var(--dim)} .alert{color:var(--warn);font-weight:600} .crit{color:var(--bad);font-weight:700}
.tag{font-size:11px;border:1px solid var(--line);border-radius:5px;padding:1px 6px;color:var(--dim)}
.runbar{display:flex;gap:3px;margin-top:4px} .runbar span{width:22px;text-align:center;border-radius:4px;font-size:11px;font-weight:700;color:#0d1117;padding:2px 0}
#log{max-height:180px;overflow:auto} textarea{width:100%;background:#0d1117;color:var(--text);border:1px solid var(--line);border-radius:7px;padding:6px;font-size:11px;height:52px}
.filters button.on{border-color:var(--accent);color:var(--accent)}
.need{color:var(--bad);font-weight:700}.ok{color:var(--good)}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style></head><body>
<h1>tailstail · Live Draft Assistant <span class="dim">2026</span></h1>
<div class="sub">12-team full PPR snake · progressive ceiling weighting · board generated __DATE__ · state lives in this tab only — use Export before refreshing</div>

<div class="panel"><div class="controls">
  My slot: <select id="slot"></select>
  <span id="clock"></span>
  <span style="flex:1"></span>
  <button id="undo" class="mini">↩ Undo</button>
  <button id="exportBtn" class="mini">Export state</button>
  <button id="importBtn" class="mini">Import</button>
</div>
<textarea id="stateBox" placeholder="Export writes a state string here; paste one and hit Import to restore." spellcheck="false"></textarea>
</div>

<div class="grid">
<div>
  <div class="panel">
    <h2>Best available <span class="dim" id="scorenote"></span></h2>
    <div class="controls filters" id="posf">
      <button data-p="ALL" class="on">All</button><button data-p="QB">QB</button><button data-p="RB">RB</button>
      <button data-p="WR">WR</button><button data-p="TE">TE</button>
      <input id="q" placeholder="Search…" style="flex:1">
    </div>
    <table><thead><tr><th></th><th>Player</th><th>Pos</th><th>Tier</th><th>Score</th><th>P90</th><th>ADP</th><th>Lasts to my next</th><th></th></tr></thead>
    <tbody id="bestrows"></tbody></table>
  </div>
</div>
<div>
  <div class="panel"><h2>Alerts</h2><div id="alerts" class="dim">No alerts yet.</div>
    <div style="margin-top:8px"><span class="dim" style="font-size:11px">LAST 10 PICKS</span><div class="runbar" id="runbar"></div></div></div>
  <div class="panel"><h2>My roster <span id="needsum" class="dim"></span></h2><table><tbody id="roster"></tbody></table></div>
  <div class="panel"><h2>Tiers remaining <span class="dim">— current, then next</span></h2><table><tbody id="tiers"></tbody></table></div>
  <div class="panel"><h2>Draft log</h2><div id="log" class="dim">—</div></div>
</div>
</div>

<script>
const DATA = __DATA__;
const TEAMS = __TEAMS__, ROUNDS = __ROUNDS__;
const STARTERS = {QB:1, RB:2, WR:2, TE:1}, FLEX = 1, FLEXPOS = ["RB","WR","TE"];
const P = DATA.players; // sorted by overall_rank
let mySlot = 8, picks = [], posFilter = "ALL", query = "";

const slotSel = document.getElementById('slot');
for (let s=1;s<=TEAMS;s++){const o=document.createElement('option');o.value=s;o.textContent='Slot '+s;slotSel.appendChild(o);}
slotSel.value = mySlot;
slotSel.onchange = () => { mySlot = +slotSel.value; render(); };

function slotForPick(overall){
  const r = Math.floor((overall-1)/TEAMS)+1, k = (overall-1)%TEAMS+1;
  return r%2===1 ? k : TEAMS-k+1;
}
function myPicks(){ const out=[]; for(let o=1;o<=TEAMS*ROUNDS;o++) if(slotForPick(o)===mySlot) out.push(o); return out; }
function ceilW(r){ return Math.min(0.30+0.08*(r-1),0.80); }
function erf(x){ // Abramowitz-Stegun 7.1.26
  const s = x<0?-1:1; x=Math.abs(x);
  const t=1/(1+0.3275911*x);
  const y=1-((((1.061405429*t-1.453152027)*t+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x);
  return s*y;
}
function survival(p, targetPick, current){
  if(p.adp==null) return 1.0;
  const sd = Math.max(p.sd||6, 2);
  return 0.5*(1+erf(((p.adp - targetPick)/sd)/Math.SQRT2));
}
const drafted = () => new Set(picks.map(p=>p.i));
function myNextOverall(){ const cur=picks.length+1; return myPicks().find(o=>o>=cur) ?? null; }

function markPick(i, mine){
  picks.push({i, mine: !!mine});
  render();
}
document.getElementById('undo').onclick = ()=>{ picks.pop(); render(); };
document.getElementById('exportBtn').onclick = ()=>{
  document.getElementById('stateBox').value = JSON.stringify({slot:mySlot, picks});
};
document.getElementById('importBtn').onclick = ()=>{
  try{ const s = JSON.parse(document.getElementById('stateBox').value);
    mySlot = s.slot||mySlot; slotSel.value=mySlot; picks = s.picks||[]; render();
  }catch(e){ alert_('Bad state string'); }
};
function alert_(m){ const a=document.getElementById('alerts'); a.innerHTML='<span class="crit">'+m+'</span>'; }
document.querySelectorAll('#posf button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('#posf button').forEach(x=>x.classList.remove('on'));
  b.classList.add('on'); posFilter=b.dataset.p; render();
});
document.getElementById('q').oninput = e=>{ query=e.target.value.toLowerCase(); render(); };

function render(){
  const cur = picks.length+1, done = drafted();
  const round = Math.floor((cur-1)/TEAMS)+1;
  const onClock = cur<=TEAMS*ROUNDS ? slotForPick(cur) : null;
  const mine = onClock===mySlot;
  const nextO = myNextOverall();
  const myRound = nextO ? Math.floor((nextO-1)/TEAMS)+1 : null;
  document.getElementById('clock').innerHTML = onClock==null ? 'Draft complete' :
    `Pick ${cur} · Round ${round} · ` + (mine
      ? '<span class="you">YOU are on the clock</span>'
      : `Slot ${onClock} on the clock — <span class="you">your pick at #${nextO??'—'}</span> (${nextO?nextO-cur:'—'} away)`);
  const w = ceilW(myRound||round);
  document.getElementById('scorenote').textContent = `— scored for your round ${myRound||round} (ceiling wt ${w.toFixed(2)})`;

  // best available
  const rows = [];
  for (let idx=0; idx<P.length && rows.length<40; idx++){
    const p = P[idx];
    if (done.has(idx)) continue;
    if (posFilter!=='ALL' && p.pos!==posFilter) continue;
    if (query && !(p.name.toLowerCase().includes(query)||p.team.toLowerCase().includes(query))) continue;
    const score = (1-w)*p.vorM + w*p.vorP;
    const surv = nextO && !mine ? survival(p, nextO, cur) : null;
    rows.push({idx,p,score,surv});
  }
  rows.sort((a,b)=>b.score-a.score);
  document.getElementById('bestrows').innerHTML = rows.map(({idx,p,score,surv})=>{
    const sc = surv==null?'—':(surv>0.85?'<span class="ok">safe</span>':surv>0.5?(100*surv).toFixed(0)+'%':'<span class="crit">'+(100*surv).toFixed(0)+'%</span>');
    return `<tr><td class="dim">${p.rank}</td><td><b>${p.name}</b> <span class="dim">${p.team}</span></td>
    <td><span class="pos ${p.pos}">${p.pos}</span></td><td class="dim">${p.tier}</td>
    <td><b>${score.toFixed(1)}</b></td><td class="dim">${p.p90}</td><td class="dim">${p.adp??'—'}</td><td>${sc}</td>
    <td><button class="mini" onclick="markPick(${idx},false)">Gone</button>
    <button class="mini primary" onclick="markPick(${idx},true)">Mine</button></td></tr>`;
  }).join('');

  // alerts: tier breaks at positions of need + positional runs
  const alerts = [];
  const remByPosTier = {};
  P.forEach((p,idx)=>{ if(done.has(idx)) return;
    (remByPosTier[p.pos] ??= {})[p.tier] = ((remByPosTier[p.pos]??={})[p.tier]||0)+1; });
  const counts = {QB:0,RB:0,WR:0,TE:0};
  picks.filter(x=>x.mine).forEach(x=>counts[P[x.i].pos]++);
  for (const pos of ["RB","WR","TE","QB"]){
    const tiers = Object.keys(remByPosTier[pos]||{}).map(Number).sort((a,b)=>a-b);
    if (!tiers.length) continue;
    const top = tiers[0], n = remByPosTier[pos][top];
    if (n<=2 && top<=8 && counts[pos] < (STARTERS[pos]||1)+1)
      alerts.push(`<span class="${n===1?'crit':'alert'}">Tier ${top} ${pos} almost gone — ${n} left</span>`);
  }
  const last10 = picks.slice(-10).map(x=>P[x.i].pos);
  const runCount = {};
  last10.forEach(p=>runCount[p]=(runCount[p]||0)+1);
  for (const pos in runCount) if (runCount[pos]>=5) alerts.push(`<span class="alert">${pos} run in progress (${runCount[pos]} of last ${last10.length})</span>`);
  document.getElementById('alerts').innerHTML = alerts.length?alerts.join('<br>'):'<span class="dim">No alerts.</span>';
  document.getElementById('runbar').innerHTML = last10.map(p=>`<span class="pos ${p}">${p[0]}</span>`).join('');

  // roster + needs
  const mineList = picks.filter(x=>x.mine).map(x=>P[x.i]);
  const needs = [];
  for (const pos in STARTERS) if (counts[pos]<STARTERS[pos]) needs.push(`${pos}×${STARTERS[pos]-counts[pos]}`);
  const flexFilled = mineList.filter(p=>FLEXPOS.includes(p.pos)).length > (STARTERS.RB+STARTERS.WR+STARTERS.TE);
  document.getElementById('needsum').innerHTML = needs.length?`— still need <span class="need">${needs.join(' ')}</span>`+(flexFilled?'':' + FLEX'):'— starters filled ✓';
  document.getElementById('roster').innerHTML = mineList.map((p,j)=>
    `<tr><td class="dim">R${(()=>{const mp=myPicks();return Math.floor((mp[j]-1)/TEAMS)+1})()}</td>
     <td><b>${p.name}</b></td><td><span class="pos ${p.pos}">${p.pos}</span></td><td class="dim">P90 ${p.p90}</td></tr>`).join('')
    || '<tr><td class="dim">No picks yet.</td></tr>';

  // tier table: the position's current (lowest remaining) tier and the two after it
  document.getElementById('tiers').innerHTML = ["QB","RB","WR","TE"].map(pos=>{
    const t = remByPosTier[pos]||{};
    const tiersLeft = Object.keys(t).map(Number).sort((a,b)=>a-b).slice(0,3);
    const cells = tiersLeft.map((k,j)=>{
      const cls = j===0 && t[k]<=2 ? (t[k]===1?'crit':'alert') : (j===0?'':'dim');
      return `<span class="${cls}">T${k}&times;${t[k]}</span>`;
    }).join(' <span class="dim">→</span> ') || '<span class="dim">exhausted</span>';
    return `<tr><td><span class="pos ${pos}">${pos}</span></td><td>${cells}</td></tr>`;
  }).join('');

  // log
  document.getElementById('log').innerHTML = picks.map((x,j)=>{
    const p=P[x.i]; return `<div>#${j+1} <span class="dim">S${slotForPick(j+1)}</span> ${x.mine?'⭐ ':''}${p.name} <span class="pos ${p.pos}">${p.pos}</span></div>`;
  }).reverse().join('') || '—';
}
render();
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", default="data/draft_board_2026_12team.csv")
    parser.add_argument("--teams", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--output", default="reports/draft_assistant_2026.html")
    args = parser.parse_args()

    board = pd.read_csv(args.board).sort_values("overall_rank")
    players = [
        {
            "name": r.player_name, "pos": r.position, "team": r.team,
            "tier": int(r.tier), "rank": int(r.overall_rank),
            "vorM": round(float(r.vor_mean), 1), "vorP": round(float(r.vor_p90), 1),
            "p90": int(round(r.season_p90)),
            "adp": (None if pd.isna(r.adp) else round(float(r.adp), 1)),
            "sd": (None if pd.isna(r.adp_sd) else round(float(r.adp_sd), 1)),
        }
        for r in board.itertuples()
    ]
    import datetime as _dt

    html = (
        TEMPLATE
        .replace("__DATA__", json.dumps({"players": players}))
        .replace("__TEAMS__", str(args.teams))
        .replace("__ROUNDS__", str(args.rounds))
        .replace("__DATE__", _dt.date.today().isoformat())
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({len(players)} players)")


if __name__ == "__main__":
    main()
