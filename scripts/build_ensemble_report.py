"""Render the corrected ensemble study + audit as one self-contained HTML
page: paper vs original notebook vs corrected run, then the deeper analysis.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# validated light-mode palette (dataviz reference instance; light only --
# a dark mode would need its own validated steps, not an automatic flip)
P = {
    "surface": "#fcfcfb", "page": "#f9f9f7",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7",
    "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#4a3aa7",
    "good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a",
    "critical": "#d03b3b",
    "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
            "#184f95", "#0d366b"],
}
VERDICT = {
    "consistent": ("good", "✔", "paper value inside our 95% interval"),
    "close": ("good", "✔", "within 5%"),
    "shifted": ("warning", "≈", "5–20% away"),
    "diverges": ("critical", "✕", "more than 20% away"),
    "n/a": ("muted", "–", "not reported there"),
}
BASE_LABEL = {"RF": "Random Forest", "XGB": "XGBoost", "LGBM": "LightGBM",
              "LR": "Logistic Regression", "GNN": "GCN-GRU"}


def f(v, d=3):
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if x != x else f"{x:.{d}f}"


def chip(v: str) -> str:
    key, icon, _ = VERDICT.get(v, ("muted", "–", ""))
    return (f'<span class="chip" style="--c:{P[key]}">'
            f'<span class="ico">{icon}</span>{v}</span>')


def md_to_html(text: str) -> str:
    """Minimal markdown -> HTML for the audit section."""
    out, in_table, in_code, in_list = [], False, False, False
    for line in text.splitlines():
        if line.startswith("```"):
            out.append("</code></pre>" if in_code else "<pre><code>")
            in_code = not in_code
            continue
        if in_code:
            out.append(line.replace("&", "&amp;").replace("<", "&lt;"))
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            tag = "th" if not in_table else "td"
            if not in_table:
                out.append('<div class="tablewrap"><table><thead><tr>')
            elif tag == "td" and not out[-1].startswith("<tr>"):
                pass
            row = "".join(f"<{tag}>{inline(c)}</{tag}>" for c in cells)
            if not in_table:
                out.append(row + "</tr></thead><tbody>")
                in_table = True
            else:
                out.append(f"<tr>{row}</tr>")
            continue
        if in_table:
            out.append("</tbody></table></div>")
            in_table = False
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(line[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if line.startswith("#### "):
            out.append(f"<h4>{inline(line[5:])}</h4>")
        elif line.startswith("### "):
            out.append(f"<h3>{inline(line[4:])}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("# "):
            continue
        elif line.strip() == "---":
            out.append("<hr>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    if in_table:
        out.append("</tbody></table></div>")
    if in_list:
        out.append("</ul>")
    html = "\n".join(out)
    return re.sub(r"</p>\n<p>", " ", html)


def inline(s: str) -> str:
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", s)
    s = re.sub(r"\[(verified|code|arith)\]",
               r'<span class="tag tag-\1">\1</span>', s)
    return s



def _plotly_bundle() -> str:
    """Inline the plotly.js bundle so the report is a single self-contained
    file -- it has to open from disk, offline, years from now."""
    import plotly
    js = (Path(plotly.__file__).parent / "package_data" / "plotly.min.js")
    if js.exists():
        return js.read_text(encoding="utf-8")
    raise FileNotFoundError(
        "plotly.min.js not found; install plotly or vendor the bundle")


BASE_LAYOUT = {
    "paper_bgcolor": P["surface"], "plot_bgcolor": P["surface"],
    "font": {"family": 'system-ui, -apple-system, "Segoe UI", sans-serif',
             "color": P["ink"], "size": 13},
    "xaxis": {"gridcolor": P["grid"], "linecolor": P["axis"],
              "zerolinecolor": P["grid"], "ticks": "outside",
              "tickcolor": P["axis"], "automargin": True,
              "tickfont": {"color": P["muted"]},
              "title": {"font": {"size": 12, "color": P["muted"]}}},
    "yaxis": {"gridcolor": P["grid"], "linecolor": P["axis"],
              "zerolinecolor": P["grid"], "ticks": "outside",
              "tickcolor": P["axis"], "automargin": True,
              "tickfont": {"color": P["muted"]},
              "title": {"font": {"size": 12, "color": P["muted"]}}},
    "legend": {"orientation": "h", "y": 1.12, "x": 0,
               "font": {"size": 12, "color": P["ink2"]}},
    "hoverlabel": {"bgcolor": "rgba(11,11,11,0.92)",
                   "bordercolor": "rgba(11,11,11,0.92)",
                   "font": {"color": "#fff", "size": 12}},
}


def build(A: dict, audit_md: str, generated: str) -> str:
    S, raw = A["summary"], A["raw"]
    recon = A["reconciliation"]
    paired = A["paired_seed_tests"]
    meta = A.get("meta", {})
    n = A["n_seeds"]

    def s(k, field="median"):
        return S.get(k, {}).get(field)

    counts: dict[str, int] = {}
    for r in recon:
        if r["paper"] is not None:
            counts[r["vs_paper"]] = counts.get(r["vs_paper"], 0) + 1
    ok = counts.get("consistent", 0) + counts.get("close", 0)
    n_paper = sum(1 for r in recon if r["paper"] is not None)

    a = S.get("track.stack_RF.f1_post", {})
    b = S.get("track.rf_alone.f1_post", {})
    h2h = paired.get("stackRF_vs_rfalone__post_f1", {})
    rho_h = paired.get("stackRF_vs_rfalone__rho", {})
    mae_h = paired.get("stackRF_vs_rfalone__rate_mae", {})
    overlap = a.get("lo", 0) <= b.get("hi", 0)

    # ---------------- figures ----------------
    def rng_row(keys_labels, title, xtitle):
        rows = [(k, lbl) for k, lbl in keys_labels if k in S]
        return {
            "data": [
                {"type": "scatter", "mode": "markers", "name": "paper",
                 "x": [next((r["paper"] for r in recon if r["key"] == k), None)
                       for k, _ in rows],
                 "y": [lbl for _, lbl in rows],
                 "marker": {"symbol": "line-ns-open", "size": 14,
                            "line": {"width": 2.4, "color": P["muted"]}},
                 "hovertemplate": "paper %{x:.3f}<extra></extra>"},
                {"type": "scatter", "mode": "markers", "name": "original run",
                 "x": [next((r["original"] for r in recon if r["key"] == k),
                            None) for k, _ in rows],
                 "y": [lbl for _, lbl in rows],
                 "marker": {"symbol": "diamond-open", "size": 10,
                            "line": {"width": 2, "color": P["s2"]},
                            "color": P["s2"]},
                 "hovertemplate": "original %{x:.3f}<extra></extra>"},
                {"type": "scatter", "mode": "markers", "name": "corrected run",
                 "x": [S[k]["median"] for k, _ in rows],
                 "y": [lbl for _, lbl in rows],
                 "error_x": {"type": "data", "symmetric": False,
                             "array": [max(S[k]["hi"] - S[k]["median"], 0)
                                       for k, _ in rows],
                             "arrayminus": [max(S[k]["median"] - S[k]["lo"], 0)
                                            for k, _ in rows],
                             "color": P["axis"], "thickness": 1.4, "width": 4},
                 "marker": {"size": 10, "color": P["s1"],
                            "line": {"width": 2, "color": P["surface"]}},
                 "hovertemplate": "%{y}<br>corrected %{x:.3f}<extra></extra>"},
            ],
            "layout": {"height": 60 + 30 * len(rows),
                       "margin": {"l": 300, "r": 30, "t": 34, "b": 48},
                       "xaxis": {"title": xtitle},
                       "yaxis": {"autorange": "reversed"},
                       "hovermode": "closest"},
        }

    fig_gate = rng_row(
        [("gate.rf_f1", "RF F1 · legacy"),
         ("gate.xgb_f1", "XGBoost F1 · legacy"),
         ("gate.lgbm_f1", "LightGBM F1 · legacy"),
         ("gate.rf_pr_auc", "RF PR-AUC · legacy"),
         ("gate.rf_recall5", "RF recall @ 5% FPR"),
         ("la.rf_f1", "RF F1 · three-way"),
         ("la.rf_pr_auc", "RF PR-AUC · three-way")],
        "reference gate", "value")

    fig_meta = rng_row(
        [("meta.DT.pr_auc", "Decision tree"),
         ("meta.LR.pr_auc", "Logistic regression"),
         ("meta.RF.pr_auc", "Random Forest"),
         ("meta.XGBs.pr_auc", "XGBoost"),
         ("la.lgbm_pr_auc", "best single base (LightGBM)")],
        "meta ablation", "PR-AUC on the leak-aware test window")

    hl = [("track.stack_RF.f1_post", "Stack (RF meta)<br>+ tracking", P["s4"]),
          ("track.stack_XGBs.f1_post", "Stack (XGB meta)<br>+ tracking",
           P["s1"]),
          ("track.rf_alone.f1_post", "Random Forest<br>+ tracking", P["s2"]),
          ("track.gnn_alone.f1_post", "GCN-GRU<br>+ tracking", P["s3"])]
    hl = [(k, lb, c) for k, lb, c in hl if k in S]
    fig_head = {
        "data": [{
            "type": "bar", "name": "median across seeds",
            "x": [lb for _, lb, _ in hl],
            "y": [S[k]["median"] for k, _, _ in hl],
            "marker": {"color": [c for _, _, c in hl], "cornerradius": 4},
            "width": 0.5,
            "text": [f(S[k]["median"]) for k, _, _ in hl],
            "textposition": "outside",
            "textfont": {"color": P["ink2"], "size": 12},
            "error_y": {"type": "data", "symmetric": False,
                        "array": [S[k]["hi"] - S[k]["median"] for k, _, _ in hl],
                        "arrayminus": [S[k]["median"] - S[k]["lo"]
                                       for k, _, _ in hl],
                        "color": P["ink"], "thickness": 1.4, "width": 7},
            "hovertemplate": "%{x}<br>median %{y:.3f}<extra></extra>"}]
        + [{"type": "scatter", "mode": "markers", "showlegend": i == 0,
            "name": "individual seed",
            "x": [lb] * len(raw.get(k, [])),
            "y": [v for v in raw.get(k, []) if v == v],
            "marker": {"size": 7, "color": "rgba(11,11,11,0.30)",
                       "line": {"width": 1.5, "color": P["surface"]}},
            "hovertemplate": "seed %{y:.3f}<extra></extra>"}
           for i, (k, lb, _) in enumerate(hl)],
        "layout": {"height": 400, "hovermode": "closest",
                   "yaxis": {"title": "post-shutdown illicit F1 "
                                      "(post-subset threshold)"},
                   "margin": {"l": 70, "r": 24, "t": 30, "b": 80}},
    }

    conv_systems = [("track.stack_RF", "Stack (RF meta) + tracking"),
                    ("track.rf_alone", "Random Forest + tracking")]
    conv_metrics = [("f1_post_wholewin", "whole-window threshold", P["s2"]),
                    ("f1_post", "post-subset threshold (oracle)", P["s4"]),
                    ("f1_post_deployable", "validation-fitted threshold",
                     P["s1"]),
                    ("f1_post_prior_matched", "prior-matched (tracked rate)",
                     P["s3"])]
    fig_conv = {
        "data": [{
            "type": "bar", "name": lbl,
            "x": [sl for _, sl in conv_systems],
            "y": [s(f"{sk}.{mk}") for sk, _ in conv_systems],
            "marker": {"color": c, "cornerradius": 4},
            "text": [f(s(f"{sk}.{mk}")) for sk, _ in conv_systems],
            "textposition": "outside",
            "textfont": {"color": P["ink2"], "size": 11},
            "hovertemplate": lbl + ": %{y:.3f}<extra></extra>"}
            for mk, lbl, c in conv_metrics],
        "layout": {"height": 400, "barmode": "group", "bargap": 0.42,
                   "bargroupgap": 0.06,
                   "yaxis": {"title": "post-shutdown illicit F1"},
                   "margin": {"l": 70, "r": 24, "t": 40, "b": 60}},
    }

    ec = A.get("error_correlation", {})
    labels = [BASE_LABEL.get(x, x) for x in ec.get("names", [])]
    fig_corr = {
        "data": [{"type": "heatmap", "z": ec.get("matrix", []),
                  "x": labels, "y": labels, "zmin": 0, "zmax": 1,
                  "xgap": 2, "ygap": 2,
                  "colorscale": [[i / (len(P["seq"]) - 1), c]
                                 for i, c in enumerate(P["seq"])],
                  "text": [[f"{v:.2f}" for v in row]
                           for row in ec.get("matrix", [])],
                  "texttemplate": "%{text}", "textfont": {"size": 12},
                  "hovertemplate": "%{y} vs %{x}<br>error corr %{z:.3f}"
                                   "<extra></extra>",
                  "colorbar": {"title": "corr", "thickness": 10, "len": 0.8}}],
        "layout": {"height": 430, "hovermode": "closest",
                   "margin": {"l": 150, "r": 78, "t": 20, "b": 112},
                   "yaxis": {"autorange": "reversed"}},
    }

    traj = A.get("trajectories", {})
    true_rate = meta.get("true_rate", {})
    tr_series = [("true", "true illicit rate", P["ink"], "solid"),
                 ("stack_RF", "tracked — stack (RF meta)", P["s4"], "dash"),
                 ("rf_alone", "tracked — Random Forest", P["s2"], "dash"),
                 ("gnn_alone", "tracked — GCN-GRU", P["s3"], "dot")]
    tdata = []
    for key, lbl, c, dash in tr_series:
        src = ({k: v for k, v in true_rate.items() if int(k) >= 42}
               if key == "true" else traj.get(key, {}))
        xs = sorted(int(t) for t in src)
        if not xs:
            continue
        tdata.append({"type": "scatter", "mode": "lines+markers", "name": lbl,
                      "x": xs, "y": [src[str(t)] for t in xs],
                      "line": {"color": c, "width": 2, "dash": dash},
                      "marker": {"size": 8, "line": {"width": 2,
                                                     "color": P["surface"]}},
                      "hovertemplate": lbl + ": %{y:.4f}<extra></extra>"})
    fig_traj = {"data": tdata,
                "layout": {"height": 380, "hovermode": "x unified",
                           "xaxis": {"title": "timestep (week)", "type": "linear",
                             "dtick": 1},
                           "yaxis": {"title": "illicit rate"},
                           "shapes": [{"type": "line", "x0": 42.5, "x1": 42.5,
                                       "y0": 0, "y1": 1, "yref": "paper",
                                       "line": {"color": P["critical"],
                                                "width": 1, "dash": "dot"}}],
                           "margin": {"l": 66, "r": 24, "t": 34, "b": 52}}}

    pw = A.get("per_week_median", [])
    fig_week = {
        "data": [{"type": "scatter", "mode": "lines+markers", "name": lbl,
                  "x": [r["t"] for r in pw if r.get(k) is not None],
                  "y": [r[k] for r in pw if r.get(k) is not None],
                  "line": {"color": c, "width": 2},
                  "marker": {"size": 8, "line": {"width": 2,
                                                 "color": P["surface"]}},
                  "hovertemplate": lbl + ": %{y:.3f}<extra></extra>"}
                 for k, lbl, c in (("rf", "Random Forest", P["s2"]),
                                   ("gnn", "GCN-GRU", P["s3"]),
                                   ("stack", "stack (RF meta)", P["s1"]),
                                   ("stack_tracking", "stack + tracking",
                                    P["s4"]))],
        "layout": {"height": 380, "hovermode": "x unified",
                   "xaxis": {"title": "timestep (week)", "type": "linear",
                             "dtick": 1},
                   "yaxis": {"title": "F1 (illicit)"},
                   "shapes": [{"type": "line", "x0": 42.5, "x1": 42.5,
                               "y0": 0, "y1": 1, "yref": "paper",
                               "line": {"color": P["critical"], "width": 1,
                                        "dash": "dot"}}],
                   "margin": {"l": 66, "r": 24, "t": 30, "b": 52}},
    }

    cal = A.get("calibration", {})
    fig_cal = {
        "data": [{"type": "scatter", "mode": "lines", "name": "perfect",
                  "x": [0, 1], "y": [0, 1], "hoverinfo": "skip",
                  "line": {"color": P["axis"], "width": 1.5, "dash": "dot"}}]
        + [{"type": "scatter", "mode": "lines+markers",
            "name": f"{lbl} · ECE {f(cal[k].get('ece'))}",
            "x": cal[k].get("bin_conf", []), "y": cal[k].get("bin_acc", []),
            "line": {"color": c, "width": 2},
            "marker": {"size": 8, "line": {"width": 2, "color": P["surface"]}},
            "hovertemplate": lbl + "<br>predicted %{x:.2f} → observed "
                                   "%{y:.3f}<extra></extra>"}
           for k, lbl, c in (("rf", "Random Forest", P["s2"]),
                             ("gnn", "GCN-GRU", P["s3"]),
                             ("stack_tracking", "stack + tracking", P["s4"]))
           if cal.get(k)],
        "layout": {"height": 400, "hovermode": "closest",
                   "xaxis": {"title": "predicted illicit probability"},
                   "yaxis": {"title": "observed illicit fraction"},
                   "margin": {"l": 70, "r": 24, "t": 34, "b": 52}},
    }

    wkeys = [(f"win.{w}", f"W = {w}") for w in (1, 3, 5, 8)]
    fig_win = {
        "data": [
            {"type": "bar", "name": "corrected (population-context window)",
             "x": [lbl for _, lbl in wkeys],
             "y": [s(k) for k, _ in wkeys],
             "marker": {"color": P["s1"], "cornerradius": 4}, "width": 0.34,
             "text": [f(s(k)) for k, _ in wkeys], "textposition": "outside",
             "textfont": {"color": P["ink2"], "size": 11},
             "hovertemplate": "%{x}: %{y:.3f}<extra></extra>"},
            {"type": "bar", "name": "original (zero-padded window)",
             "x": [lbl for _, lbl in wkeys],
             "y": [next((r["original"] for r in recon if r["key"] == k), None)
                   for k, _ in wkeys],
             "marker": {"color": P["muted"], "cornerradius": 4}, "width": 0.34,
             "text": [f(next((r["original"] for r in recon if r["key"] == k),
                             None)) for k, _ in wkeys],
             "textposition": "outside",
             "textfont": {"color": P["muted"], "size": 11},
             "hovertemplate": "%{x}: %{y:.3f}<extra></extra>"}],
        "layout": {"height": 340, "barmode": "group", "bargap": 0.4,
                   "yaxis": {"title": "three-way F1 (illicit)"},
                   "margin": {"l": 70, "r": 24, "t": 40, "b": 52}},
    }

    # ---------------- tables ----------------
    recon_rows = "\n".join(
        f"<tr><td>{r['label']}</td>"
        f"<td class='n'>{f(r['paper'])}</td>"
        f"<td class='n'>{f(r['original'])}</td>"
        f"<td class='n'><strong>{f(r['median'])}</strong></td>"
        f"<td class='n dim'>[{f(r['lo'])}, {f(r['hi'])}]</td>"
        f"<td>{chip(r['vs_paper'])}</td><td>{chip(r['vs_original'])}</td>"
        f"</tr>" for r in recon)

    paired_rows = "\n".join(
        f"<tr><td>{k.replace('__', ' · ').replace('_', ' ')}</td>"
        f"<td class='n'>{f(v.get('median_diff'), 4)}</td>"
        f"<td class='n'>{v.get('wins', 0)}/{v.get('n', 0)}</td>"
        f"<td class='n'>{f(v.get('sign_p'), 4)}</td></tr>"
        for k, v in paired.items())

    sens_rows = "\n".join(
        f"<tr><td>{k.split('|')[0].replace('_', ' ')}</td>"
        f"<td>{k.split('|')[1].replace('_', ' ')}</td>"
        f"<td>{k.split('|')[2].replace('_', ' ')}</td>"
        f"<td class='n'>{f(v['f1_post'])}</td>"
        f"<td class='n'>{f(v['rho'])}</td>"
        f"<td class='n'>{f(v['rate_mae'], 4)}</td></tr>"
        for k, v in A.get("tracker_sensitivity", {}).items()
        if v.get("f1_post") is not None)

    lift_rows = "\n".join(
        f"<tr><td>{BASE_LABEL.get(nm, nm)}</td>"
        f"<td class='n'>{f(s(f'base.raw166.{nm}.f1'), 4)}</td>"
        f"<td class='n'>{f(s(f'base.enriched.{nm}.f1'), 4)}</td>"
        f"<td class='n'>{f(s(f'lift.{nm}.f1'), 4)}</td>"
        f"<td class='n'>{f(s(f'base.raw166.{nm}.pr_auc'), 4)}</td>"
        f"<td class='n'>{f(s(f'base.enriched.{nm}.pr_auc'), 4)}</td></tr>"
        for nm in ("RF", "XGB", "LGBM", "LR", "GNN"))

    cost = A.get("cost_curves", {})
    cost_rows = "\n".join(
        f"<tr><td class='n'>{k} : 1</td>"
        f"<td class='n'>{f(v['best_threshold'], 2)}</td>"
        f"<td class='n'>{f(v['best_cost'], 4)}</td>"
        f"<td class='n dim'>{f(v['cost_at_0p5'], 4)}</td></tr>"
        for k, v in sorted(cost.items(), key=lambda kv: int(kv[0])))

    err_rows = "\n".join(
        f"<tr><td>{BASE_LABEL.get(k, k)}</td><td class='n'>{f(v, 4)}</td></tr>"
        for k, v in ec.get("error_rate", {}).items())

    rho_note = (
        f"n = {int(s('track.stack_RF.rho') is not None) and 8} weekly points; "
        f"median permutation p = "
        f"{f(s('track.stack_RF.rho_p'), 3)} for the stack and "
        f"{f(s('track.rf_alone.rho_p'), 3)} for the Random Forest")

    figs = {"fig-gate": fig_gate, "fig-meta": fig_meta, "fig-head": fig_head,
            "fig-conv": fig_conv, "fig-corr": fig_corr, "fig-traj": fig_traj,
            "fig-week": fig_week, "fig-cal": fig_cal, "fig-win": fig_win}

    blocks = meta.get("feature_blocks", {})
    return TEMPLATE.format(
        n_seeds=n, ok=ok, n_paper=n_paper,
        n_consistent=counts.get("consistent", 0),
        n_close=counts.get("close", 0), n_shifted=counts.get("shifted", 0),
        n_diverges=counts.get("diverges", 0),
        generated=generated,
        n_nodes=f"{meta.get('n_nodes', 0):,}",
        n_edges=f"{meta.get('n_edges', 0):,}",
        n_illicit=f"{meta.get('n_illicit', 0):,}",
        n_features=meta.get("n_features_total", "—"),
        blocks=", ".join(f"{k} {v}" for k, v in blocks.items()),
        causal="causal" if meta.get("causal_wavelet") else "non-causal",
        multi_nodes=meta.get("multi_timestep_nodes", "—"),
        stack_f1=f(a.get("median")),
        stack_ci=f"[{f(a.get('lo'))}, {f(a.get('hi'))}]",
        rf_f1=f(b.get("median")),
        rf_ci=f"[{f(b.get('lo'))}, {f(b.get('hi'))}]",
        gain=f(h2h.get("median_diff"), 4),
        gain_wins=f"{h2h.get('wins', 0)}/{h2h.get('n', 0)}",
        gain_p=f(h2h.get("sign_p"), 4),
        rho_diff=f(rho_h.get("median_diff"), 4),
        rho_wins=f"{rho_h.get('wins', 0)}/{rho_h.get('n', 0)}",
        mae_diff=f(mae_h.get("median_diff"), 4),
        mae_wins=f"{mae_h.get('wins', 0)}/{mae_h.get('n', 0)}",
        overlap_word="overlap" if overlap else "do not overlap",
        resolvable="not resolvable" if overlap else "resolvable",
        rho_note=rho_note,
        disagree=f(ec.get("disagreement_fraction")),
        rf_gate=f(s("gate.rf_f1")),
        rf_gate_ci=f"[{f(s('gate.rf_f1', 'lo'))}, {f(s('gate.rf_f1', 'hi'))}]",
        la_rf=f(s("la.rf_f1")),
        recon_rows=recon_rows, paired_rows=paired_rows, sens_rows=sens_rows,
        lift_rows=lift_rows, cost_rows=cost_rows, err_rows=err_rows,
        audit=md_to_html(audit_md),
        plotly_js=_plotly_bundle(),
        figs_json=json.dumps(figs), base_layout=json.dumps(BASE_LAYOUT),
        **{f"c_{k}": v for k, v in P.items() if isinstance(v, str)})


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stacking ensemble — corrected study and audit</title>
<script>{plotly_js}</script>
<style>
  :root {{
    color-scheme: light;
    --surface: {c_surface}; --page: {c_page};
    --ink: {c_ink}; --ink2: {c_ink2}; --muted: {c_muted};
    --grid: {c_grid}; --axis: {c_axis};
    --s1: {c_s1}; --s2: {c_s2}; --s3: {c_s3}; --s4: {c_s4};
    --good: {c_good}; --warning: {c_warning}; --critical: {c_critical};
    --border: rgba(11,11,11,0.10);
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--page); color:var(--ink);
    font:15px/1.62 system-ui,-apple-system,"Segoe UI",sans-serif; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:40px 26px 90px; }}
  header {{ border-bottom:1px solid var(--border); padding-bottom:22px; margin-bottom:30px; }}
  h1 {{ font-size:30px; line-height:1.2; margin:0 0 8px; letter-spacing:-0.02em; }}
  h2 {{ font-size:21px; margin:48px 0 10px; letter-spacing:-0.01em; }}
  h3 {{ font-size:16px; margin:30px 0 8px; }}
  h4 {{ font-size:14px; margin:22px 0 6px; color:var(--ink2); }}
  p {{ max-width:76ch; }}
  .sub {{ color:var(--ink2); font-size:15px; margin:0; max-width:70ch; }}
  .prov {{ color:var(--muted); font-size:12.5px; margin-top:14px;
    font-variant-numeric:tabular-nums; }}
  .toc {{ display:flex; flex-wrap:wrap; gap:8px; margin:22px 0 0; }}
  .toc a {{ font-size:13px; color:var(--ink2); text-decoration:none;
    border:1px solid var(--border); border-radius:999px; padding:4px 12px;
    background:var(--surface); }}
  .toc a:hover {{ border-color:var(--s1); color:var(--s1); }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(178px,1fr));
    gap:12px; margin:22px 0 6px; }}
  .tile {{ background:var(--surface); border:1px solid var(--border);
    border-radius:12px; padding:15px 16px; }}
  .tile .lab {{ font-size:11.5px; color:var(--muted); text-transform:uppercase;
    letter-spacing:0.07em; }}
  .tile .val {{ font-size:27px; font-weight:620; margin-top:4px; letter-spacing:-0.02em; }}
  .tile .ci {{ font-size:12px; color:var(--ink2); margin-top:2px;
    font-variant-numeric:tabular-nums; }}
  .card {{ background:var(--surface); border:1px solid var(--border);
    border-radius:14px; padding:16px 16px 8px; margin:18px 0; }}
  .cap {{ color:var(--muted); font-size:12.5px; margin:2px 0 14px; max-width:80ch; }}
  .tablewrap {{ overflow-x:auto; background:var(--surface);
    border:1px solid var(--border); border-radius:14px; margin:16px 0; }}
  table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
  th,td {{ text-align:left; padding:9px 14px; border-bottom:1px solid var(--grid);
    vertical-align:top; }}
  th {{ font-size:11.5px; text-transform:uppercase; letter-spacing:0.06em;
    color:var(--muted); font-weight:600; background:var(--page); }}
  tbody tr:last-child td {{ border-bottom:0; }}
  td.n, th.n {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td.dim {{ color:var(--muted); }}
  .chip {{ display:inline-flex; align-items:center; gap:5px; font-size:12px;
    font-weight:560; color:var(--c);
    border:1px solid color-mix(in srgb, var(--c) 34%, transparent);
    background:color-mix(in srgb, var(--c) 9%, transparent);
    border-radius:999px; padding:2px 9px; white-space:nowrap; }}
  .verdict {{ border-left:3px solid var(--s4); padding:4px 0 4px 16px; margin:20px 0; }}
  .note {{ background:var(--surface); border:1px solid var(--border);
    border-left:3px solid var(--warning); border-radius:0 12px 12px 0;
    padding:14px 18px; margin:20px 0; font-size:14px; }}
  code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px;
    background:var(--page); border:1px solid var(--grid); border-radius:5px; padding:1px 5px; }}
  pre {{ background:var(--surface); border:1px solid var(--border);
    border-radius:12px; padding:14px 16px; overflow-x:auto; }}
  pre code {{ border:0; background:none; padding:0; font-size:12.5px; line-height:1.5; }}
  hr {{ border:0; border-top:1px solid var(--border); margin:34px 0; }}
  .tag {{ font-size:10.5px; font-weight:600; text-transform:uppercase;
    letter-spacing:0.05em; border-radius:4px; padding:1px 5px; margin-left:4px; }}
  .tag-verified {{ background:color-mix(in srgb, var(--good) 12%, transparent);
    color:var(--good); }}
  .tag-code {{ background:color-mix(in srgb, var(--s1) 12%, transparent); color:var(--s1); }}
  .tag-arith {{ background:color-mix(in srgb, var(--s4) 12%, transparent); color:var(--s4); }}
  #audit ul {{ max-width:76ch; }}
  footer {{ margin-top:60px; padding-top:20px; border-top:1px solid var(--border);
    color:var(--muted); font-size:12.5px; }}
</style></head><body><div class="wrap">

<header>
  <h1>Stacking ensemble: corrected study and audit</h1>
  <p class="sub">The ensemble extension rebuilt with six methodology fixes,
  measured over {n_seeds} seeds on the real Elliptic graph, and reconciled
  three ways — against the paper, against the original notebook's own
  outputs, and against itself across seeds.</p>
  <div class="prov">{n_nodes} nodes · {n_edges} edges · {n_illicit} illicit ·
  {n_features} features ({blocks}) · {causal} wavelet ·
  {multi_nodes} nodes appear in more than one timestep · generated {generated}</div>
  <nav class="toc">
    <a href="#gate">1 Reference gate</a><a href="#claim">2 The claim</a>
    <a href="#conv">3 Thresholds</a><a href="#meta">4 Meta-learners</a>
    <a href="#decor">5 Decorrelation</a><a href="#track">6 Tracker</a>
    <a href="#deep">7 Deeper analysis</a><a href="#recon">8 Reconciliation</a>
    <a href="#audit">9 Audit</a>
  </nav>
</header>

<section id="gate">
  <h2>1 · The reference gate still passes</h2>
  <p>Nothing below is interpretable unless the pipeline reproduces the
  benchmark it claims to. It does: the Random Forest reaches
  <strong>{rf_gate}</strong> {rf_gate_ci} on the legacy split against the
  paper's 0.830 and the original notebook's 0.8304, and drops to {la_rf} once
  evaluation crosses week 43.</p>
  <div class="card">
    <div id="fig-gate"></div>
    <p class="cap">Grey tick: the paper. Orange diamond: the original
    notebook's run. Blue dot with interval: this corrected run, median across
    {n_seeds} seeds. The full numbers are in the reconciliation table.</p>
  </div>
</section>

<section id="claim">
  <h2>2 · The claim the ensemble makes</h2>
  <p>Not that the stack wins on aggregate F1 — that feeding the online
  prior-tracking head a decorrelated score, instead of one model's score,
  improves the rare-class decision after the shutdown. A paired comparison on
  identical test nodes: base learners trained on weeks 1–34, meta-learner
  fitted on 35–41, everything reported on 42–49.</p>
  <div class="tiles">
    <div class="tile"><div class="lab">Stack (RF meta) + tracking</div>
      <div class="val" style="color:var(--s4)">{stack_f1}</div>
      <div class="ci">post-43 F1 · {stack_ci}</div></div>
    <div class="tile"><div class="lab">Random Forest + tracking</div>
      <div class="val" style="color:var(--s2)">{rf_f1}</div>
      <div class="ci">post-43 F1 · {rf_ci}</div></div>
    <div class="tile"><div class="lab">Median paired gain</div>
      <div class="val">{gain}</div>
      <div class="ci">wins {gain_wins} · sign p = {gain_p}</div></div>
    <div class="tile"><div class="lab">Paired change in tracking ρ</div>
      <div class="val">{rho_diff}</div>
      <div class="ci">wins {rho_wins}</div></div>
  </div>
  <div class="verdict"><p style="margin:0"><strong>Verdict.</strong> The
  intervals {overlap_word}, so the post-shutdown F1 gain from a decorrelated
  input is <strong>{resolvable}</strong> at this seed count. The tracking
  correlation is a separate quantity and does not follow it: paired median
  change {rho_diff} ({rho_wins} seeds), and on the rate-error metric that
  does not degrade at n = 8, the stack improves the mean absolute error by
  {mae_diff} ({mae_wins} seeds).</p></div>
  <div class="card">
    <div id="fig-head"></div>
    <p class="cap">Medians with 95% across-seed intervals; grey dots are the
    individual seeds, so the spread is visible rather than implied. All four
    systems are scored at the post-subset oracle threshold — the same
    convention on every bar.</p>
  </div>
</section>

<section id="conv">
  <h2>3 · One thresholding convention at a time</h2>
  <p>The original notebook prints <code>F1 post-43(oracle)</code> — an oracle
  threshold fitted on the <em>whole</em> test window — beside
  <code>F1 post-43 +track</code>, fitted on the <em>post-43 subset</em>. Read
  across the row, the tracking head appears to quadruple F1; part of that is
  the head, part is the change of convention. Below, all four conventions for
  both headline systems, so the comparison can be made down a column.</p>
  <div class="card">
    <div id="fig-conv"></div>
    <p class="cap">Left to right within each group: whole-window oracle
    threshold, post-subset oracle threshold, threshold fitted on the
    validation window (deployable), and the prior-matched rule that flags the
    top q<sub>t</sub> of each week from the tracked rate alone. Only the last
    two use no test labels.</p>
  </div>
</section>

<section id="meta">
  <h2>4 · Which combiner earns its place</h2>
  <p>The meta-learner is fitted on the validation window only, so it cannot
  inherit base-model overfitting. PR-AUC is the right axis because it measures
  ranking quality independently of where the threshold lands — and ranking
  quality is what the tracking head consumes downstream.</p>
  <div class="card">
    <div id="fig-meta"></div>
    <p class="cap">A single decision tree is the weak combiner the stacking
    literature warns about. The forest and boosted stackers clear the best
    single base learner.</p>
  </div>
</section>

<section id="decor">
  <h2>5 · Where the decorrelation comes from</h2>
  <p>Stacking pays only when the base learners make <em>different</em>
  mistakes. Pairwise correlation of per-node 0/1 error vectors on the test
  window, seed-averaged — and, unlike the original, with the GCN-GRU
  retrained per seed so its column is not a single frozen draw.</p>
  <div class="card">
    <div id="fig-corr"></div>
    <p class="cap">Disagreement fraction <strong>{disagree}</strong> — the
    share of test nodes where at least one, but not every, base learner is
    wrong.</p>
  </div>
  <div class="tablewrap"><table>
    <thead><tr><th>Base learner</th><th class="n">Error rate on test</th></tr></thead>
    <tbody>{err_rows}</tbody></table></div>
</section>

<section id="track">
  <h2>6 · What the tracking head is actually sensitive to</h2>
  <p>Two choices were implicit in the original. The paper's Methods states
  Beta(5, 10); the code passes no α/β and runs at the Beta(0.2, 1.8)
  defaults. And the EM estimates the deployment prior from labelled test rows
  only, though Elliptic labels 23% of nodes and the labelled subset is not a
  random sample of the stream. Both are varied here.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>System</th><th>Regulariser</th><th>EM population</th>
    <th class="n">post-43 F1</th><th class="n">tracking ρ</th>
    <th class="n">rate MAE</th></tr></thead>
    <tbody>{sens_rows}</tbody></table></div>
  <div class="note"><strong>On ρ.</strong> The leak-aware test window is eight
  weeks, so Spearman ρ lives on a lattice of 84 steps and the exact two-sided
  permutation p-value for |ρ| ≥ 0.333 at n = 8 is <strong>0.428</strong>.
  Here: {rho_note}. Rank correlation cannot separate these systems; the mean
  absolute error between the tracked and the true weekly rate uses the
  magnitudes and does not degrade at this n, so it is reported alongside.</div>
  <div class="card">
    <div id="fig-traj"></div>
    <p class="cap">The tracked rate follows the true rate into the
    post-shutdown trough and is then held low through the recovery — the
    trapped-low behaviour the paper describes, reproduced here.</p>
  </div>
</section>

<section id="deep">
  <h2>7 · Deeper analysis</h2>
  <h3>7.1 Feature enrichment</h3>
  <p>Topological, node2vec, delta and <em>causal</em> wavelet blocks against
  the raw 166 columns, paired across seeds.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Learner</th><th class="n">raw 166 F1</th>
    <th class="n">enriched F1</th><th class="n">lift</th>
    <th class="n">raw PR-AUC</th><th class="n">enriched PR-AUC</th></tr></thead>
    <tbody>{lift_rows}</tbody></table></div>

  <h3>7.2 Per-week behaviour</h3>
  <div class="card">
    <div id="fig-week"></div>
    <p class="cap">Every model collapses across the shutdown. What the
    tracking head recovers is the decision on individual post-shutdown weeks,
    not the ranking.</p>
  </div>

  <h3>7.3 Calibration</h3>
  <div class="card">
    <div id="fig-cal"></div>
    <p class="cap">Prior shift moves the decision boundary; it does not by
    itself miscalibrate the score. Separating the two is what rules out
    temperature scaling as a fix.</p>
  </div>

  <h3>7.4 Decision economics</h3>
  <p>F1 weights a missed illicit transaction and a frozen legitimate customer
  equally, which no compliance desk does. Cost-minimising threshold for the
  stack-plus-tracking system on the post-shutdown window.</p>
  <div class="tablewrap"><table>
    <thead><tr><th class="n">FN : FP cost</th><th class="n">Best threshold</th>
    <th class="n">Cost per node at best</th><th class="n">Cost at 0.5</th></tr></thead>
    <tbody>{cost_rows}</tbody></table></div>

  <h3>7.5 Temporal window</h3>
  <div class="card">
    <div id="fig-win"></div>
    <p class="cap">Zero Elliptic transaction nodes appear in more than one
    timestep, so there is no per-node history to sweep. The original's window
    prepends zero vectors; the corrected one prepends the preceding weeks'
    population-mean embeddings, so W carries actual context. Both are
    near-flat — but only the second is measuring temporal context at all.</p>
  </div>

  <h3>7.6 Paired across-seed tests</h3>
  <div class="tablewrap"><table>
    <thead><tr><th>Comparison</th><th class="n">Median paired difference</th>
    <th class="n">Wins</th><th class="n">Sign-test p</th></tr></thead>
    <tbody>{paired_rows}</tbody></table></div>
</section>

<section id="recon">
  <h2>8 · Reconciliation</h2>
  <div class="tiles">
    <div class="tile"><div class="lab">Consistent</div>
      <div class="val" style="color:var(--good)">{n_consistent}</div>
      <div class="ci">paper inside our 95% interval</div></div>
    <div class="tile"><div class="lab">Close</div>
      <div class="val" style="color:var(--good)">{n_close}</div>
      <div class="ci">within 5%</div></div>
    <div class="tile"><div class="lab">Shifted</div>
      <div class="val" style="color:var(--warning)">{n_shifted}</div>
      <div class="ci">5–20% away</div></div>
    <div class="tile"><div class="lab">Diverges</div>
      <div class="val" style="color:var(--critical)">{n_diverges}</div>
      <div class="ci">more than 20% away</div></div>
  </div>
  <p><strong>{ok} of {n_paper}</strong> published quantities reproduce or sit
  inside this run's interval, after six methodology corrections. Where a row
  moves, the audit below names which correction moved it.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Quantity</th><th class="n">Paper</th>
    <th class="n">Original run</th><th class="n">Corrected</th>
    <th class="n">95% CI</th><th>vs paper</th><th>vs original</th></tr></thead>
    <tbody>{recon_rows}</tbody></table></div>
</section>

<section id="audit">
  <h2>9 · Audit of the original notebook</h2>
  {audit}
</section>

<footer>
  Real Elliptic Bitcoin dataset (Weber et al., 2019). Oracle-threshold figures
  use the evaluated window's own labels and are upper bounds; deployable and
  prior-matched figures use no test labels. Produced by
  <code>scripts/build_enrichment_cache.py</code> →
  <code>scripts/train_gnn_bases.py</code> →
  <code>scripts/run_ensemble_study.py</code> →
  <code>scripts/aggregate_ensemble_study.py</code> →
  <code>scripts/build_ensemble_report.py</code>.
</footer>
</div>
<script>
const FIGS = {figs_json};
const BASE = {base_layout};
function merge(a,b){{const o=Object.assign({{}},a);
  for(const k of Object.keys(b||{{}})){{o[k]=(b[k]&&typeof b[k]==='object'&&!Array.isArray(b[k]))?merge(a[k]||{{}},b[k]):b[k];}}
  return o;}}
for(const [id,fig] of Object.entries(FIGS)){{
  const el=document.getElementById(id); if(!el) continue;
  Plotly.newPlot(el, fig.data, merge(BASE, fig.layout),
                 {{displayModeBar:false, responsive:true}});
}}
</script>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="out/aggregate.json")
    ap.add_argument("--audit", default="AUDIT_ENSEMBLE.md")
    ap.add_argument("--out", default="out/ensemble_report.html")
    ap.add_argument("--generated", default="")
    args = ap.parse_args()
    A = json.load(open(args.study))
    audit = (Path(args.audit).read_text() if Path(args.audit).exists()
             else "")
    html = build(A, audit, args.generated)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html)
    print(f"wrote {args.out} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
