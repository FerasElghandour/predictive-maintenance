"""
BOX 5 — ACTION LAYER (the interface)

Generates a single self-contained HTML file: docs/index.html

WHY STATIC, NOT A LIVE SERVER
-----------------------------
The predictions are computed in a batch. Nothing about this dashboard needs a
Python process running to answer a request. A static file therefore:
  - loads instantly, never sleeps, never cold-starts
  - costs nothing and cannot fall over
  - deploys to GitHub Pages with zero infrastructure

If the fleet were streaming live sensor data you would need a server. It isn't,
so we don't. Matching the architecture to the actual requirement — rather than to
what looks impressive — is the point.

The page leads with the SYSTEM's honest numbers (did the alarm fire in time?),
not the model's flattering ones (MAE). A maintenance chief does not care about
mean absolute error.
"""

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

AUTHOR = "Feras Elghandour"
AUTHOR_LINK = "https://github.com/FERAS-GITHUB-HANDLE"   # <-- replace with your profile URL

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
background:#0f1216;color:#e8ecf1;-webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto;padding:32px 20px 80px}
header{border-bottom:1px solid #232a33;padding-bottom:24px;margin-bottom:28px}
h1{font-size:26px;font-weight:650;letter-spacing:-.01em}
.sub{color:#8b97a6;margin-top:8px;font-size:15px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.09em;color:#8b97a6;
margin:38px 0 14px;font-weight:600}
.hero{background:#151a21;border:1px solid #232a33;border-radius:10px;padding:22px 24px}
.hero p{color:#c3ccd7;font-size:15px}
.hero strong{color:#fff}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px}
.kpi{background:#0f1419;border:1px solid #232a33;border-radius:8px;padding:14px}
.kpi .v{font-size:26px;font-weight:680;letter-spacing:-.02em}
.kpi .l{font-size:12px;color:#8b97a6;margin-top:3px;line-height:1.35}
.good{color:#3fb950}.warn{color:#d29922}.bad{color:#f85149}
.fleet{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.card{background:#151a21;border:1px solid #232a33;border-radius:10px;padding:18px;
border-left:3px solid #2d3540}
.card.red{border-left-color:#f85149}.card.amber{border-left-color:#d29922}
.card.green{border-left-color:#3fb950}
.card .n{font-size:30px;font-weight:680;letter-spacing:-.02em}
.card .t{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:#8b97a6;margin-top:2px}
.card .d{font-size:13px;color:#6e7a89;margin-top:8px}
.filters{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.filters button{background:#151a21;border:1px solid #232a33;color:#8b97a6;padding:7px 14px;
border-radius:999px;font-size:13px;cursor:pointer;font-family:inherit}
.filters button:hover{border-color:#3a4453;color:#e8ecf1}
.filters button.on{background:#e8ecf1;color:#0f1216;border-color:#e8ecf1;font-weight:600}
.tbl{width:100%;border-collapse:collapse;font-size:14px}
.tbl th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.07em;
color:#6e7a89;padding:10px 12px;border-bottom:1px solid #232a33;font-weight:600}
.tbl td{padding:11px 12px;border-bottom:1px solid #1a2028}
.tbl tr:hover td{background:#151a21}
.eng{font-weight:600}
.bar{width:74px;height:5px;background:#232a33;border-radius:3px;overflow:hidden;display:inline-block;
vertical-align:middle;margin-right:9px}
.bar i{display:block;height:100%}
.pill{font-size:11px;font-weight:650;padding:3px 9px;border-radius:999px;white-space:nowrap}
.pill.red{background:rgba(248,81,73,.14);color:#f85149}
.pill.amber{background:rgba(210,153,34,.14);color:#d29922}
.pill.green{background:rgba(63,185,80,.13);color:#3fb950}
.note{color:#6e7a89;font-size:13px;margin-top:10px}
footer{margin-top:44px;padding-top:20px;border-top:1px solid #232a33;color:#6e7a89;font-size:13px}
footer a{color:#8b97a6}
.mono{font-variant-numeric:tabular-nums}
@media(max-width:760px){
 .kpis{grid-template-columns:repeat(2,1fr)}
 .fleet{grid-template-columns:1fr}
 .wrap{padding:22px 14px 60px}
 h1{font-size:21px}
 .hide-sm{display:none}
 .tbl td,.tbl th{padding:9px 8px}
}
"""

JS = """
const btns=document.querySelectorAll('.filters button');
btns.forEach(b=>b.onclick=()=>{
  btns.forEach(x=>x.classList.remove('on'));b.classList.add('on');
  const f=b.dataset.f;
  document.querySelectorAll('.tbl tbody tr').forEach(r=>{
    r.style.display=(f==='all'||r.dataset.s===f)?'':'none';
  });
});
"""


def build():
    fleet = pd.read_csv(ART / "fleet_decisions.csv")
    pol = json.load(open(ART / "policy_metrics.json"))
    met = json.load(open(ART / "metrics.json"))
    c = pol["chosen"]

    counts = fleet["status"].value_counts()
    n_maint = int(counts.get("MAINTENANCE REQUIRED", 0))
    n_mon = int(counts.get("MONITOR", 0))
    n_ok = int(counts.get("HEALTHY", 0))

    best = min(met["results"], key=lambda r: r["MAE"])

    rows = []
    for _, r in fleet.iterrows():
        st = r["status"]
        cls = {"MAINTENANCE REQUIRED": "red", "MONITOR": "amber", "HEALTHY": "green"}[st]
        label = {"MAINTENANCE REQUIRED": "MAINTENANCE", "MONITOR": "MONITOR", "HEALTHY": "HEALTHY"}[st]
        col = {"red": "#f85149", "amber": "#d29922", "green": "#3fb950"}[cls]
        h = float(r["health_pct"])
        rows.append(f"""<tr data-s="{cls}">
<td class="eng">Engine {int(r['unit'])}</td>
<td class="mono">{r['pred_rul']:.0f}</td>
<td class="mono hide-sm">{int(r['true_rul'])}</td>
<td><span class="bar"><i style="width:{min(h,100):.0f}%;background:{col}"></i></span><span class="mono">{h:.0f}%</span></td>
<td><span class="pill {cls}">{label}</span></td>
</tr>""")

    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Turbofan Fleet — Predictive Maintenance</title>
<meta name="description" content="Predicts jet engine failure ~40 cycles before it happens. 0% missed, 0% too late to act, on 100 unseen engines.">
<style>{CSS}</style>
</head><body><div class="wrap">

<header>
<h1>Turbofan Fleet — Predictive Maintenance</h1>
<div class="sub">Which of these 100 jet engines is about to fail, and which do I fix first?</div>
</header>

<div class="hero">
<p><strong>The problem.</strong> Jet engines don't fail on a schedule. In this fleet they last anywhere
from 128 to 362 cycles. Replace them all early and you burn money; wait too long and one fails in flight.</p>
<p style="margin-top:10px"><strong>The result.</strong> This system reads 12 sensors, predicts how many
cycles each engine has left, and turns that into a maintenance decision. Tested by replaying
<strong>100 engines it had never seen</strong> against the true failure times — it caught
<strong>every engine that was going to fail</strong>, and gave the crew a median of
<strong>{c['median_runway_cycles']} cycles</strong> of warning.</p>

<div class="kpis">
<div class="kpi"><div class="v good">{c['pct_too_late']:.0f}%</div><div class="l">Flagged too late to act</div></div>
<div class="kpi"><div class="v good">{c['pct_never_warned']:.0f}%</div><div class="l">Failures we missed entirely</div></div>
<div class="kpi"><div class="v warn">{c['pct_false_alarm']:.0f}%</div><div class="l">False alarms (wasted ~{c['median_life_wasted_cycles']} cycles each)</div></div>
<div class="kpi"><div class="v">{c['median_runway_cycles']}</div><div class="l">Median cycles of warning given</div></div>
</div>
<div class="note">Policy: flag an engine when predicted health drops below {c['threshold_pct']}%,
assuming the crew needs {c['lead_time_cycles']}+ cycles of notice. Model:
{best['model'].split('. ')[-1]}, average error {best['MAE']} cycles.</div>
</div>

<h2>Fleet status today</h2>
<div class="fleet">
<div class="card red"><div class="n">{n_maint}</div><div class="t">Maintenance required</div>
<div class="d">Below {c['threshold_pct']}% health. Schedule now.</div></div>
<div class="card amber"><div class="n">{n_mon}</div><div class="t">Monitor</div>
<div class="d">Degrading. Watch the trend.</div></div>
<div class="card green"><div class="n">{n_ok}</div><div class="t">Healthy</div>
<div class="d">No action needed.</div></div>
</div>

<h2>Every engine, most urgent first</h2>
<div class="filters">
<button class="on" data-f="all">All 100</button>
<button data-f="red">Maintenance ({n_maint})</button>
<button data-f="amber">Monitor ({n_mon})</button>
<button data-f="green">Healthy ({n_ok})</button>
</div>
<table class="tbl">
<thead><tr>
<th>Engine</th><th>Predicted cycles left</th><th class="hide-sm">Actual (answer key)</th>
<th>Health</th><th>Status</th>
</tr></thead>
<tbody>
{''.join(rows)}
</tbody></table>
<div class="note">The "actual" column is NASA's sealed answer key. The model never saw it —
it's shown here so you can check the predictions yourself.</div>

<footer>
Built by <a href="{AUTHOR_LINK}">{AUTHOR}</a> — part of a series on building intelligent systems
that solve real-world problems.<br>
Data: NASA C-MAPSS Turbofan Engine Degradation benchmark (FD001). A demonstration of end-to-end
system architecture — the same pipeline transfers to business problems like customer churn.
</footer>

</div><script>{JS}</script></body></html>"""

    out = DOCS / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({len(html)/1024:.0f} KB, self-contained)")
    print(f"fleet: {n_maint} maintenance / {n_mon} monitor / {n_ok} healthy")
    return out


if __name__ == "__main__":
    build()
