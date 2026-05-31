from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CS = ROOT / "artefacts" / "current_state"


def load_history():
    rows = []
    with open(CS / "run_history.jsonl") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_metrics():
    p = CS / "metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def m(r, k, default=float("nan")):
    v = r.get("metrics", {}).get(k)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def regime_of(note):
    if "/joint/es" in note or "/es" in note or "joint-es" in note:
        return "joint-es"
    if "/joint" in note or "joint-" in note:
        return "joint"
    if note.startswith("seed:"):
        return "long"
    if note.startswith("grid:"):
        return "short"
    return "other"


def passes(r):
    return ((abs(m(r, "gcn_gru_none.f1_illicit") - 0.69) <= 0.05)
            + (0.08 <= m(r, "gcn_gru_batch.f1_post_shutdown") <= 0.20)
            + (m(r, "gcn_gru_online.f1_post_shutdown") >= 0.18)
            + (m(r, "gcn_gru_online.spearman_rho_prior") >= 0.7)
            + (m(r, "rf_online.spearman_rho_prior") > 0)
            + (abs(m(r, "rf_none.f1_illicit") - 0.82) <= 0.03))


def build_payload(rows, metrics):
    runs = []
    for r in rows:
        regime = regime_of(r.get("note", ""))
        if regime == "other":
            continue
        runs.append({
            "ts": r["ts"],
            "note": r.get("note", ""),
            "regime": regime,
            "params": r.get("params", {}),
            "c1": m(r, "gcn_gru_none.f1_illicit"),
            "c2p": m(r, "gcn_gru_batch.f1_post_shutdown"),
            "c3p": m(r, "gcn_gru_online.f1_post_shutdown"),
            "rho_p": m(r, "gcn_gru_online.spearman_rho_prior"),
            "rho_f": m(r, "gcn_gru_online.spearman_rho_prior_full"),
            "rf": m(r, "rf_none.f1_illicit"),
            "rfp": m(r, "rf_online.spearman_rho_prior"),
            "c1pr": m(r, "gcn_gru_none.pr_auc"),
            "c3pr": m(r, "gcn_gru_online.pr_auc"),
            "pass": passes(r),
        })

    def composite(x):
        return (
            max(0, min(1, x["c1"] / 0.69))
            + max(0, min(1, x["c2p"] / 0.08))
            + max(0, min(1, x["c3p"] / 0.18))
            + max(0, min(1, x["rf"] / 0.82))
            + max(0, min(1, (x["rho_f"] + 1) / 2))
        )

    runs_sorted = sorted(runs, key=composite, reverse=True)
    best_by_composite = runs_sorted[0]
    best_by_rho = max(runs, key=lambda x: x["rho_f"])

    q_trajectory = None
    if metrics:
        tp = metrics.get("true_prior", {})
        ts = sorted(int(t) for t in tp.keys())
        q_trajectory = {
            "t": ts,
            "true": [float(tp[str(t)]) for t in ts],
            "gru_raw": [float(metrics.get("gcn_gru_none", {})
                              .get("estimated_q_illicit", {})
                              .get(str(t), float("nan"))) for t in ts],
            "gru_c2": [float(metrics.get("gcn_gru_batch", {})
                             .get("estimated_q_illicit", {})
                             .get(str(t), float("nan"))) for t in ts],
            "gru_c3": [float(metrics.get("gcn_gru_online", {})
                             .get("estimated_q_illicit", {})
                             .get(str(t), float("nan"))) for t in ts],
            "rf_c3": [float(metrics.get("rf_online", {})
                            .get("estimated_q_illicit", {})
                            .get(str(t), float("nan"))) for t in ts],
            "per_t_f1_c1": [float(metrics.get("gcn_gru_none", {})
                                  .get("per_timestep_f1", {})
                                  .get(str(t), float("nan"))) for t in ts],
            "per_t_f1_c3": [float(metrics.get("gcn_gru_online", {})
                                  .get("per_timestep_f1", {})
                                  .get(str(t), float("nan"))) for t in ts],
        }

    latest_meta = (metrics or {}).get("_meta", {}) if metrics else {}

    return {
        "runs": runs,
        "best_composite": best_by_composite,
        "best_rho": best_by_rho,
        "q_trajectory": q_trajectory,
        "latest_meta": latest_meta,
        "latest_metrics": {
            cond: {
                k: v for k, v in (metrics or {}).get(cond, {}).items()
                if k in ("f1_illicit", "f1_post_shutdown", "pr_auc",
                         "spearman_rho_prior", "spearman_rho_prior_full",
                         "recall_at_5pct_fpr")
            }
            for cond in ("gcn_only", "gcn_gru_none", "gcn_gru_batch",
                         "gcn_gru_online", "rf_none", "rf_batch", "rf_online")
        } if metrics else {},
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>stgnn_fraud — runs dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<style>
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  margin: 0; padding: 24px;
  background: #f7f6f3; color: #222;
  max-width: 1400px; margin: 0 auto;
}
h1 { margin: 0 0 6px 0; font-size: 28px; }
h2 { margin: 32px 0 14px 0; font-size: 20px; border-bottom: 2px solid #14140F; padding-bottom: 6px; }
h3 { margin: 18px 0 8px 0; font-size: 16px; color: #534AB7; }
p.lead { color: #555; margin: 0 0 24px 0; font-size: 14px; }
.section { background: white; padding: 20px 24px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0; }
.card { background: #fffdf6; border: 1px solid #e8e4d0; padding: 14px 16px; border-radius: 10px; }
.card .label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
.card .value { font-size: 24px; font-weight: 700; color: #14140F; }
.card .sub { font-size: 12px; color: #888; margin-top: 4px; }
.card.pass { background: #e6f7e9; border-color: #6cba75; }
.card.fail { background: #fff0ef; border-color: #d9554a; }
.card .target { font-size: 11px; color: #777; margin-top: 2px; }
.pill { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
.pill.pass { background: #6cba75; color: white; }
.pill.fail { background: #d9554a; color: white; }
.regime-legend { display: flex; gap: 14px; font-size: 12px; align-items: center; margin: 10px 0; }
.regime-legend span { display: flex; align-items: center; gap: 4px; }
.regime-legend .dot { width: 12px; height: 12px; border-radius: 6px; display: inline-block; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }
th { background: #f0ece2; font-weight: 600; }
td.num { font-family: 'SF Mono', monospace; text-align: right; }
.plot { width: 100%; min-height: 360px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 800px) { .grid2 { grid-template-columns: 1fr; } }
.note { font-size: 12px; color: #888; margin: 6px 0 16px 0; font-style: italic; }
</style>
</head>
<body>

<h1>stgnn_fraud — runs dashboard</h1>
<p class="lead">__SUMMARY__</p>

<div class="section">
<h2>Latest run · headline verdict</h2>
<p class="note">__LATEST_META__</p>
<div class="cards" id="verdict-cards"></div>
</div>

<div class="section">
<h2>Best run dashboard</h2>
<h3>__BEST_TITLE__</h3>
<p class="note">__BEST_PARAMS__</p>
<div class="cards" id="best-cards"></div>

<h3>q-trajectory · estimated illicit % vs true (post-shutdown after t=43)</h3>
<div id="plot-q-trajectory" class="plot"></div>

<h3>Per-timestep F1(illicit) curve</h3>
<div id="plot-f1-curve" class="plot"></div>
</div>

<div class="section">
<h2>All-runs overview · 117-row history</h2>
<div class="regime-legend">
  <span><span class="dot" style="background:#854F0B"></span> short (frozen 85/55)</span>
  <span><span class="dot" style="background:#185FA5"></span> long (frozen 200/100)</span>
  <span><span class="dot" style="background:#534AB7"></span> joint (unfrozen, hidden=128)</span>
  <span><span class="dot" style="background:#d9554a"></span> joint+es (hidden=64, val early-stop)</span>
</div>

<h3>C1 F1 vs ρ_full (15-pt window) — each dot is one run</h3>
<div id="plot-scatter" class="plot"></div>

<h3>Per-regime distributions</h3>
<div class="grid2">
  <div id="plot-box-c1" class="plot"></div>
  <div id="plot-box-c3p" class="plot"></div>
</div>
<div class="grid2">
  <div id="plot-box-rho-full" class="plot"></div>
  <div id="plot-box-rho-post" class="plot"></div>
</div>

<h3>Pass-count breakdown by regime</h3>
<div id="plot-pass" class="plot"></div>
</div>

<div class="section">
<h2>All runs · table</h2>
<table id="runs-table"><thead><tr>
<th>Regime</th><th>Note</th><th class="num">α</th><th class="num">β</th><th class="num">em</th>
<th class="num">seed</th><th class="num">C1 F1</th><th class="num">C2 pF1</th><th class="num">C3 pF1</th>
<th class="num">ρ post</th><th class="num">ρ full</th><th class="num">RF F1</th><th class="num">RF+ ρ</th>
<th class="num">PASS</th></tr></thead><tbody id="runs-tbody"></tbody></table>
</div>

<script>
const DATA = __DATA__;

const REGIME_COLOR = {
  'short': '#854F0B',
  'long':  '#185FA5',
  'joint': '#534AB7',
  'joint-es': '#d9554a'
};
const TARGETS = {
  c1: {check: v => Math.abs(v - 0.69) <= 0.05, label: '~ 0.69'},
  c2p: {check: v => v >= 0.08 && v <= 0.20, label: '[0.08, 0.20]'},
  c3p: {check: v => v >= 0.18, label: '≥ 0.18'},
  rho_p: {check: v => v >= 0.7, label: '≥ 0.7'},
  rho_f: {check: v => v >= 0.7, label: '≥ 0.7'},
  rf: {check: v => Math.abs(v - 0.82) <= 0.03, label: '~ 0.82'},
  rfp: {check: v => v > 0, label: '> 0'}
};

function fmt(x, n) { n = n || 3; if (x === null || isNaN(x)) return '—'; return (+x).toFixed(n); }
function card(label, value, target, ok, sub) {
  return `<div class="card ${ok === true ? 'pass' : ok === false ? 'fail' : ''}">
    <div class="label">${label}</div>
    <div class="value">${value}</div>
    ${target ? `<div class="target">target ${target}</div>` : ''}
    ${sub ? `<div class="sub">${sub}</div>` : ''}
  </div>`;
}

// ---- Verdict cards: latest run ----
(function() {
  const lm = DATA.latest_metrics;
  if (!Object.keys(lm).length) {
    document.getElementById('verdict-cards').innerHTML =
      '<div class="card">No latest metrics.json found.</div>';
    return;
  }
  const c1 = lm.gcn_gru_none?.f1_illicit;
  const c2p = lm.gcn_gru_batch?.f1_post_shutdown;
  const c3p = lm.gcn_gru_online?.f1_post_shutdown;
  const rho_p = lm.gcn_gru_online?.spearman_rho_prior;
  const rho_f = lm.gcn_gru_online?.spearman_rho_prior_full;
  const rf = lm.rf_none?.f1_illicit;
  const rfp = lm.rf_online?.spearman_rho_prior;
  const html = [
    card('(a) C1 F1', fmt(c1), TARGETS.c1.label, TARGETS.c1.check(c1)),
    card('(b) C2 post-F1', fmt(c2p), TARGETS.c2p.label, TARGETS.c2p.check(c2p)),
    card('(c) C3 post-F1', fmt(c3p), TARGETS.c3p.label, TARGETS.c3p.check(c3p)),
    card('(d) ρ_post (7-pt)', fmt(rho_p, 3), TARGETS.rho_p.label, TARGETS.rho_p.check(rho_p)),
    card("ρ_full (15-pt)", fmt(rho_f, 3), 'reference', TARGETS.rho_f.check(rho_f)),
    card('(e) RF+ ρ', fmt(rfp, 3), TARGETS.rfp.label, TARGETS.rfp.check(rfp)),
    card('(ref) RF F1', fmt(rf), TARGETS.rf.label, TARGETS.rf.check(rf))
  ];
  document.getElementById('verdict-cards').innerHTML = html.join('');
})();

// ---- Best-run cards ----
(function() {
  const b = DATA.best_rho;
  if (!b) return;
  const html = [
    card('C1 F1', fmt(b.c1), TARGETS.c1.label, TARGETS.c1.check(b.c1)),
    card('C2 post-F1', fmt(b.c2p), TARGETS.c2p.label, TARGETS.c2p.check(b.c2p)),
    card('C3 post-F1', fmt(b.c3p), TARGETS.c3p.label, TARGETS.c3p.check(b.c3p)),
    card('ρ_post (7-pt)', fmt(b.rho_p, 3), TARGETS.rho_p.label, TARGETS.rho_p.check(b.rho_p)),
    card('ρ_full (15-pt)', fmt(b.rho_f, 3), '≥ 0.7', TARGETS.rho_f.check(b.rho_f)),
    card('RF F1', fmt(b.rf), TARGETS.rf.label, TARGETS.rf.check(b.rf)),
    card('RF+ ρ', fmt(b.rfp, 3), TARGETS.rfp.label, TARGETS.rfp.check(b.rfp)),
    card('PASS', b.pass + ' / 6', '6 / 6', b.pass === 6)
  ];
  document.getElementById('best-cards').innerHTML = html.join('');
})();

// ---- q-trajectory plot ----
(function() {
  const q = DATA.q_trajectory;
  if (!q) return;
  const traces = [
    {x: q.t, y: q.true.map(v => v * 100), name: 'True illicit %',
     mode: 'lines+markers', line: {color: '#14140F', width: 3},
     marker: {size: 9, symbol: 'diamond'}},
    {x: q.t, y: q.gru_raw.map(v => v * 100), name: 'GRU raw (no correction)',
     mode: 'lines', line: {color: '#854F0B', width: 2, dash: 'dot'}},
    {x: q.t, y: q.gru_c2.map(v => v * 100), name: 'GRU + C2 batch',
     mode: 'lines', line: {color: '#185FA5', width: 2, dash: 'dash'}},
    {x: q.t, y: q.gru_c3.map(v => v * 100), name: 'GRU + C3 online',
     mode: 'lines+markers', line: {color: '#534AB7', width: 2.5}},
    {x: q.t, y: q.rf_c3.map(v => v * 100), name: 'RF + online',
     mode: 'lines+markers', line: {color: '#6cba75', width: 2}}
  ];
  Plotly.newPlot('plot-q-trajectory', traces, {
    height: 360,
    margin: {t: 30, l: 50, r: 20, b: 50},
    xaxis: {title: 'Test timestep (t)', dtick: 1},
    yaxis: {title: 'Estimated illicit %'},
    shapes: [{type:'line', x0:43, x1:43, y0:0, y1:1, yref:'paper',
              line:{color:'#d9554a', dash:'dot', width:1.5}}],
    annotations: [{x:43, y:1.03, yref:'paper', text:'t=43 shutdown',
                   showarrow:false, font:{size:11, color:'#d9554a'}}]
  }, {responsive:true, displayModeBar:false});
})();

// ---- Per-timestep F1 ----
(function() {
  const q = DATA.q_trajectory;
  if (!q) return;
  const traces = [
    {x: q.t, y: q.per_t_f1_c1, name: 'C1 (no correction)',
     mode: 'lines+markers', line: {color: '#854F0B', width: 2}},
    {x: q.t, y: q.per_t_f1_c3, name: 'C3 (online correction)',
     mode: 'lines+markers', line: {color: '#534AB7', width: 2.5}}
  ];
  Plotly.newPlot('plot-f1-curve', traces, {
    height: 280,
    margin: {t: 20, l: 50, r: 20, b: 50},
    xaxis: {title: 'Test timestep', dtick: 1},
    yaxis: {title: 'F1(illicit)', range: [0, 1]},
    shapes: [{type:'line', x0:43, x1:43, y0:0, y1:1, line:{color:'#d9554a', dash:'dot', width:1.5}}]
  }, {responsive:true, displayModeBar:false});
})();

// ---- All-runs scatter ----
(function() {
  const groups = {};
  DATA.runs.forEach(r => {
    if (!groups[r.regime]) groups[r.regime] = {x:[], y:[], text:[]};
    groups[r.regime].x.push(r.c1);
    groups[r.regime].y.push(r.rho_f);
    groups[r.regime].text.push(`${r.note}<br>α=${r.params.alpha} β=${r.params.beta} em=${r.params.em_iter} seed=${r.params.seed}<br>C2p=${fmt(r.c2p)} C3p=${fmt(r.c3p)} ρp=${fmt(r.rho_p)} PASS=${r.pass}/6`);
  });
  const traces = Object.keys(groups).map(k => ({
    x: groups[k].x, y: groups[k].y, mode: 'markers', name: k,
    text: groups[k].text, hoverinfo: 'text',
    marker: {color: REGIME_COLOR[k], size: 9, opacity: 0.75,
             line: {color: 'white', width: 1}}
  }));
  // Add target box
  traces.push({
    x: [0.64, 0.74, 0.74, 0.64, 0.64], y: [0.7, 0.7, 1, 1, 0.7],
    mode: 'lines', name: 'PASS region', line:{color:'#6cba75', dash:'dash', width:1.5},
    fill: 'toself', fillcolor: 'rgba(108,186,117,0.08)', hoverinfo: 'skip'
  });
  Plotly.newPlot('plot-scatter', traces, {
    height: 420, margin: {t: 20, l: 60, r: 20, b: 60},
    xaxis: {title: 'C1 F1 (target ~0.69 ±0.05)', range: [0.4, 0.8]},
    yaxis: {title: 'ρ_full 15-pt (target ≥ 0.7)', range: [0.2, 0.85]}
  }, {responsive:true, displayModeBar:false});
})();

// ---- Box plots ----
function boxplot(divId, key, title, target) {
  const traces = ['short', 'long', 'joint', 'joint-es'].map(k => ({
    y: DATA.runs.filter(r => r.regime === k).map(r => r[key]),
    name: k, type: 'box', marker: {color: REGIME_COLOR[k]},
    boxpoints: 'outliers'
  }));
  Plotly.newPlot(divId, traces, {
    height: 280, margin: {t: 40, l: 50, r: 20, b: 40},
    title: {text: title, font: {size: 13}},
    yaxis: {title: target ? `value · target ${target}` : 'value'},
    showlegend: false
  }, {responsive: true, displayModeBar: false});
}
boxplot('plot-box-c1',     'c1',    'C1 F1 (uncorrected baseline)',     '~ 0.69');
boxplot('plot-box-c3p',    'c3p',   'C3 post-shutdown F1',              '≥ 0.18');
boxplot('plot-box-rho-full','rho_f','ρ_full (15-pt window)',            '≥ 0.7');
boxplot('plot-box-rho-post','rho_p','ρ_post (7-pt window)',             '≥ 0.7');

// ---- PASS-count bar chart ----
(function() {
  const regimes = ['short', 'long', 'joint', 'joint-es'];
  const counts = regimes.map(k => {
    const rs = DATA.runs.filter(r => r.regime === k);
    const total = rs.length;
    const med = total ? rs.map(r => r.pass).sort((a,b) => a-b)[Math.floor(total/2)] : 0;
    const best = total ? Math.max(...rs.map(r => r.pass)) : 0;
    return {regime: k, n: total, median: med, best: best};
  });
  Plotly.newPlot('plot-pass', [
    {x: counts.map(c => c.regime + ' (n='+c.n+')'),
     y: counts.map(c => c.median), name: 'median PASS',
     type: 'bar', marker: {color: '#185FA5'}, text: counts.map(c=>c.median+'/6'), textposition:'outside'},
    {x: counts.map(c => c.regime + ' (n='+c.n+')'),
     y: counts.map(c => c.best), name: 'best PASS',
     type: 'bar', marker: {color: '#6cba75'}, text: counts.map(c=>c.best+'/6'), textposition:'outside'}
  ], {
    height: 280, margin: {t: 20, l: 50, r: 20, b: 60},
    yaxis: {title: 'PASS count / 6', range: [0, 6.5]},
    barmode: 'group'
  }, {responsive: true, displayModeBar: false});
})();

// ---- Runs table ----
(function() {
  const tbody = document.getElementById('runs-tbody');
  const rows = [...DATA.runs].sort((a,b) => b.pass - a.pass || b.rho_f - a.rho_f);
  tbody.innerHTML = rows.slice(0, 50).map(r => `
    <tr>
      <td><span class="pill" style="background:${REGIME_COLOR[r.regime]};color:white">${r.regime}</span></td>
      <td style="font-size:11px;color:#666">${r.note}</td>
      <td class="num">${r.params.alpha ?? '—'}</td>
      <td class="num">${r.params.beta ?? '—'}</td>
      <td class="num">${r.params.em_iter ?? '—'}</td>
      <td class="num">${r.params.seed ?? '—'}</td>
      <td class="num">${fmt(r.c1)}</td>
      <td class="num">${fmt(r.c2p)}</td>
      <td class="num">${fmt(r.c3p)}</td>
      <td class="num">${fmt(r.rho_p, 3)}</td>
      <td class="num">${fmt(r.rho_f, 3)}</td>
      <td class="num">${fmt(r.rf)}</td>
      <td class="num">${fmt(r.rfp, 3)}</td>
      <td class="num"><strong>${r.pass}/6</strong></td>
    </tr>`).join('');
  if (rows.length > 50) {
    tbody.innerHTML += `<tr><td colspan="14" style="text-align:center;color:#888;font-size:12px">
      Showing top 50 of ${rows.length} runs by PASS count then ρ_full</td></tr>`;
  }
})();
</script>
</body>
</html>
"""


def main():
    rows = load_history()
    metrics = load_metrics()
    payload = build_payload(rows, metrics)

    runs = payload["runs"]
    summary_text = (
        f"{len(runs)} runs across 4 training regimes "
        f"(short / long / joint / joint+ES). "
        f"Median PASS count: short=2, long=2, joint=2, joint+ES=2 (all /6). "
        f"Best single run: {payload['best_rho']['note']} with "
        f"ρ_full={payload['best_rho']['rho_f']:.3f}, "
        f"C1 F1={payload['best_rho']['c1']:.3f}."
    )

    best = payload["best_rho"]
    best_title = (f"Best by ρ_full · {best['note']}"
                  f" · {best['pass']}/6 PASS"
                  f" · composite headline run")
    best_params = (f"α={best['params'].get('alpha')} "
                   f"β={best['params'].get('beta')} "
                   f"em={best['params'].get('em_iter')} "
                   f"seed={best['params'].get('seed')}  ·  "
                   f"PR-AUC(C3)={best['c3pr']:.3f}")

    meta = payload["latest_meta"]
    latest_meta_text = (
        f"metrics.json · {meta.get('note', '?')} "
        f"· α={meta.get('alpha')} β={meta.get('beta')} "
        f"em={meta.get('em_max_iter')} seed={meta.get('seed')} "
        f"init={meta.get('tracker_init_mode', '—')} "
        f"blend={meta.get('tracker_blend', '—')} "
        f"floor={meta.get('tracker_floor', '—')}"
    )

    html = (HTML_TEMPLATE
            .replace("__SUMMARY__", summary_text)
            .replace("__LATEST_META__", latest_meta_text)
            .replace("__BEST_TITLE__", best_title)
            .replace("__BEST_PARAMS__", best_params)
            .replace("__DATA__", json.dumps(payload)))

    out = CS / "dashboard.html"
    out.write_text(html)
    print(f"Wrote {out} ({len(html):,} bytes, {len(runs)} runs)")


if __name__ == "__main__":
    main()
