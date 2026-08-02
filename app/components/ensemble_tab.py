"""Ensemble tab: the corrected stacking study, its deeper analysis, and the
three-way reconciliation against the paper and the original notebook run.

Reads ``artefacts/ensemble_study.json`` (written by
``scripts/aggregate_ensemble_study.py``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.utils.theme import (
    CHART,
    PLOTLY_CONFIG,
    apply_plotly_layout,
    section_close,
    section_open,
)

BASE_LABEL = {"RF": "Random Forest", "XGB": "XGBoost", "LGBM": "LightGBM",
              "LR": "Logistic Regression", "GNN": "GCN-GRU"}
META_LABEL = {"DT": "Decision tree", "LR": "Logistic regression",
              "RF": "Random Forest", "XGBs": "XGBoost"}
SYSTEMS = [("track.stack_RF", "Stack (RF meta) + tracking", CHART["violet"]),
           ("track.stack_XGBs", "Stack (XGB meta) + tracking", CHART["sky"]),
           ("track.rf_alone", "Random Forest + tracking", CHART["amber"]),
           ("track.gnn_alone", "GCN-GRU + tracking", CHART["emerald"])]
CONVENTIONS = [("f1_post_wholewin", "whole-window oracle", CHART["amber"]),
               ("f1_post", "post-subset oracle", CHART["violet"]),
               ("f1_post_deployable", "validation-fitted", CHART["sky"]),
               ("f1_post_prior_matched", "prior-matched", CHART["emerald"])]


def study_path(artefact_paths: dict) -> Path:
    override = os.getenv("STGNN_ENSEMBLE_PATH")
    if override:
        return Path(override)
    return Path(artefact_paths["metrics"]).parent / "ensemble_study.json"


def _load(artefact_paths: dict) -> dict | None:
    p = study_path(artefact_paths)
    if not p.exists():
        return None
    try:
        with open(p) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def load_study(artefact_paths: dict) -> dict | None:
    """Public accessor for the aggregated ensemble study (or None)."""
    return _load(artefact_paths)


def _f(v, d=3) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if x != x else f"{x:.{d}f}"


# --------------------------------------------------------------------------
def _headline_chart(A: dict) -> go.Figure:
    S, raw = A["summary"], A.get("raw", {})
    rows = [(k, lbl, c) for k, lbl, c in SYSTEMS if f"{k}.f1_post" in S]
    fig = go.Figure()
    fig.add_bar(
        x=[lbl for _, lbl, _ in rows],
        y=[S[f"{k}.f1_post"]["median"] for k, _, _ in rows],
        marker=dict(color=[c for _, _, c in rows], cornerradius=4),
        width=0.5, name="median", showlegend=False,
        text=[_f(S[f"{k}.f1_post"]["median"]) for k, _, _ in rows],
        textposition="outside",
        error_y=dict(
            type="data", symmetric=False,
            array=[S[f"{k}.f1_post"]["hi"] - S[f"{k}.f1_post"]["median"]
                   for k, _, _ in rows],
            arrayminus=[S[f"{k}.f1_post"]["median"] - S[f"{k}.f1_post"]["lo"]
                        for k, _, _ in rows],
            color=CHART["ink"], thickness=1.4, width=7),
        hovertemplate="%{x}<br>median %{y:.3f}<extra></extra>")
    for i, (k, lbl, _) in enumerate(rows):
        vals = [v for v in raw.get(f"{k}.f1_post", []) if v == v]
        fig.add_scatter(x=[lbl] * len(vals), y=vals, mode="markers",
                        name="individual seed", showlegend=i == 0,
                        marker=dict(size=7, color="rgba(11,15,26,0.32)",
                                    line=dict(width=1.5,
                                              color=CHART["surface"])),
                        hovertemplate="seed %{y:.3f}<extra></extra>")
    apply_plotly_layout(fig, height=400,
                        title="Post-shutdown illicit F1 — post-subset "
                              "threshold on every bar")
    fig.update_yaxes(title="F1 (illicit)")
    fig.update_layout(hovermode="closest")
    return fig


def _convention_chart(A: dict) -> go.Figure:
    S = A["summary"]
    sys_rows = [(k, lbl) for k, lbl, _ in SYSTEMS[:1] + SYSTEMS[2:3]]
    fig = go.Figure()
    for ck, clbl, colour in CONVENTIONS:
        fig.add_bar(x=[lbl for _, lbl in sys_rows],
                    y=[S.get(f"{k}.{ck}", {}).get("median") for k, _ in sys_rows],
                    name=clbl, marker=dict(color=colour, cornerradius=4),
                    text=[_f(S.get(f"{k}.{ck}", {}).get("median"))
                          for k, _ in sys_rows],
                    textposition="outside",
                    hovertemplate=clbl + ": %{y:.3f}<extra></extra>")
    apply_plotly_layout(fig, height=380,
                        title="The same systems under four thresholding "
                              "conventions")
    fig.update_layout(barmode="group", bargap=0.42)
    fig.update_yaxes(title="post-shutdown illicit F1")
    return fig


def _meta_chart(A: dict) -> go.Figure:
    S = A["summary"]
    paper = {r["key"]: r["paper"] for r in A["reconciliation"]}
    orig = {r["key"]: r["original"] for r in A["reconciliation"]}
    keys = [("meta.DT.pr_auc", "Decision tree"),
            ("meta.LR.pr_auc", "Logistic regression"),
            ("meta.RF.pr_auc", "Random Forest"),
            ("meta.XGBs.pr_auc", "XGBoost"),
            ("la.lgbm_pr_auc", "Best single base")]
    fig = go.Figure()
    fig.add_bar(x=[lbl for _, lbl in keys],
                y=[paper.get(k) for k, _ in keys], name="paper",
                marker=dict(color=CHART["slate"], cornerradius=4), width=0.26)
    fig.add_bar(x=[lbl for _, lbl in keys],
                y=[orig.get(k) for k, _ in keys], name="original run",
                marker=dict(color=CHART["amber"], cornerradius=4), width=0.26)
    fig.add_bar(x=[lbl for _, lbl in keys],
                y=[S.get(k, {}).get("median") for k, _ in keys],
                name="corrected run",
                marker=dict(color=CHART["sky"], cornerradius=4), width=0.26)
    apply_plotly_layout(fig, height=380,
                        title="Meta-learner ablation — PR-AUC on the "
                              "leak-aware test window")
    fig.update_layout(barmode="group", bargap=0.3)
    fig.update_yaxes(title="PR-AUC")
    return fig


def _corr_chart(A: dict) -> go.Figure:
    ec = A["error_correlation"]
    labels = [BASE_LABEL.get(n, n) for n in ec["names"]]
    fig = go.Figure(go.Heatmap(
        z=ec["matrix"], x=labels, y=labels, zmin=0, zmax=1, xgap=2, ygap=2,
        colorscale=[[0.0, "#FFFFFF"], [0.5, "#9ec5f4"], [1.0, "#0d366b"]],
        text=[[f"{v:.2f}" for v in row] for row in ec["matrix"]],
        texttemplate="%{text}",
        hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
        colorbar=dict(title="corr", thickness=11)))
    apply_plotly_layout(fig, height=430,
                        title="Base-learner error correlation "
                              "(median over seeds)")
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(autorange="reversed")
    return fig


def _traj_chart(A: dict) -> go.Figure:
    traj = A.get("trajectories", {})
    true_rate = A.get("meta", {}).get("true_rate", {})
    fig = go.Figure()
    ts = sorted(int(t) for t in true_rate if int(t) >= 42)
    if ts:
        fig.add_scatter(x=ts, y=[true_rate[str(t)] for t in ts],
                        name="true illicit rate", mode="lines+markers",
                        line=dict(color=CHART["ink"], width=2.4),
                        marker=dict(size=8))
    for key, lbl, colour in (("stack_RF", "tracked — stack (RF meta)",
                              CHART["violet"]),
                             ("rf_alone", "tracked — Random Forest",
                              CHART["amber"]),
                             ("gnn_alone", "tracked — GCN-GRU",
                              CHART["emerald"])):
        srs = traj.get(key) or {}
        xs = sorted(int(t) for t in srs)
        if xs:
            fig.add_scatter(x=xs, y=[srs[str(t)] for t in xs], name=lbl,
                            mode="lines+markers", marker=dict(size=8),
                            line=dict(color=colour, width=2, dash="dash"))
    fig.add_vline(x=42.5, line_width=1, line_dash="dot",
                  line_color=CHART["crimson"])
    apply_plotly_layout(fig, height=380,
                        title="Tracked versus true illicit rate")
    fig.update_yaxes(title="illicit rate")
    fig.update_xaxes(title="timestep (week)")
    return fig


def _week_chart(A: dict) -> go.Figure:
    rows = A.get("per_week_median", [])
    fig = go.Figure()
    for key, lbl, colour in (("rf", "Random Forest", CHART["amber"]),
                             ("gnn", "GCN-GRU", CHART["emerald"]),
                             ("stack", "stack (RF meta)", CHART["sky"]),
                             ("stack_tracking", "stack + tracking",
                              CHART["violet"])):
        xs = [r["t"] for r in rows if r.get(key) is not None]
        if xs:
            fig.add_scatter(x=xs, y=[r[key] for r in rows
                                     if r.get(key) is not None],
                            name=lbl, mode="lines+markers",
                            marker=dict(size=8),
                            line=dict(color=colour, width=2))
    fig.add_vline(x=42.5, line_width=1, line_dash="dot",
                  line_color=CHART["crimson"])
    apply_plotly_layout(fig, height=380, title="Per-week illicit F1")
    fig.update_yaxes(title="F1 (illicit)")
    fig.update_xaxes(title="timestep (week)")
    return fig


def _cal_chart(A: dict) -> go.Figure:
    cal = A.get("calibration", {})
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], name="perfect", mode="lines",
                    line=dict(color=CHART["line"], width=1.5, dash="dot"),
                    hoverinfo="skip")
    for key, lbl, colour in (("rf", "Random Forest", CHART["amber"]),
                             ("gnn", "GCN-GRU", CHART["emerald"]),
                             ("stack_tracking", "stack + tracking",
                              CHART["violet"])):
        c = cal.get(key)
        if not c:
            continue
        fig.add_scatter(x=c.get("bin_conf", []), y=c.get("bin_acc", []),
                        name=f"{lbl} · ECE {_f(c.get('ece'))}",
                        mode="lines+markers", marker=dict(size=8),
                        line=dict(color=colour, width=2))
    apply_plotly_layout(fig, height=390,
                        title="Reliability — post-shutdown window")
    fig.update_xaxes(title="predicted illicit probability")
    fig.update_yaxes(title="observed illicit fraction")
    fig.update_layout(hovermode="closest")
    return fig


# --------------------------------------------------------------------------
def render_ensemble_summary(artefact_paths: dict, *,
                            key_prefix: str = "results") -> None:
    """Compact ensemble comparison, embedded in the Results tab.

    Silently does nothing when artefacts/ensemble_study.json is absent.
    """
    A = _load(artefact_paths)
    if A is None:
        return
    S = A.get("summary", {})
    rows = [(k, lbl) for k, lbl, _ in SYSTEMS if f"{k}.f1_post" in S]
    if not rows:
        return

    st.markdown(section_open(
        "Ensemble comparison",
        f"{A.get('n_seeds', '?')} seeds · median over seeds · "
        f"full study in the Ensemble tab",
        "model"), unsafe_allow_html=True)

    table = []
    for k, lbl in rows:
        post = S.get(f"{k}.f1_post", {})
        dep = S.get(f"{k}.f1_post_deployable", {})
        rho = S.get(f"{k}.rho", {})
        table.append({
            "System": lbl,
            "post-43 F1": post.get("median"),
            "95% CI": f"[{_f(post.get('lo'))}, {_f(post.get('hi'))}]",
            "post-43 F1 (deployable)": dep.get("median"),
            "tracking rho": rho.get("median"),
        })
    st.dataframe(pd.DataFrame(table).round(4), use_container_width=True,
                 hide_index=True)
    st.plotly_chart(_headline_chart(A), use_container_width=True,
                    config=PLOTLY_CONFIG,
                    key=f"{key_prefix}_ensemble_headline")
    st.caption(
        "Post-subset oracle threshold on every bar. The deployable column "
        "uses no test labels. The single-run numbers elsewhere on this page "
        "are one seed; these are medians across seeds."
    )
    st.markdown(section_close(), unsafe_allow_html=True)


# --------------------------------------------------------------------------
def render_ensemble_tab(artefact_paths: dict) -> None:
    A = _load(artefact_paths)
    if A is None:
        st.info(
            "No ensemble study found. Build it with\n\n"
            "```bash\n"
            "python -m scripts.build_enrichment_cache\n"
            "python -m scripts.train_gnn_bases\n"
            "python -m scripts.run_ensemble_study\n"
            "python -m scripts.aggregate_ensemble_study "
            "--out artefacts/ensemble_study.json\n"
            "```")
        return

    S = A["summary"]
    meta = A.get("meta", {})
    paired = A.get("paired_seed_tests", {})
    n = A["n_seeds"]

    st.markdown(section_open(
        "The claim",
        f"{n} seeds · leak-aware split (train 1-34, validate 35-41, "
        f"test 42-49) · {meta.get('n_features_total', '?')} features · "
        f"causal wavelet"), unsafe_allow_html=True)

    a = S.get("track.stack_RF.f1_post", {})
    b = S.get("track.rf_alone.f1_post", {})
    h2h = paired.get("stackRF_vs_rfalone__post_f1", {})
    rho_h = paired.get("stackRF_vs_rfalone__rho", {})
    mae_h = paired.get("stackRF_vs_rfalone__rate_mae", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stack (RF meta) + tracking", _f(a.get("median")),
              f"post-43 F1 · [{_f(a.get('lo'))}, {_f(a.get('hi'))}]")
    c2.metric("Random Forest + tracking", _f(b.get("median")),
              f"post-43 F1 · [{_f(b.get('lo'))}, {_f(b.get('hi'))}]")
    c3.metric("Median paired gain", _f(h2h.get("median_diff"), 4),
              f"wins {h2h.get('wins', 0)}/{h2h.get('n', 0)} · "
              f"sign p={_f(h2h.get('sign_p'), 4)}")
    c4.metric("Paired change in ρ", _f(rho_h.get("median_diff"), 4),
              f"wins {rho_h.get('wins', 0)}/{rho_h.get('n', 0)}")

    overlap = a.get("lo", 0) <= b.get("hi", 0)
    st.markdown(
        f"**Verdict.** The intervals "
        f"{'overlap' if overlap else 'do not overlap'}, so the post-shutdown "
        f"F1 gain from feeding the tracking head a decorrelated score is "
        f"{'not resolvable' if overlap else 'resolvable'} at this seed "
        f"count. Tracking fidelity is a separate quantity: paired change in "
        f"rate MAE {_f(mae_h.get('median_diff'), 4)} "
        f"({mae_h.get('wins', 0)}/{mae_h.get('n', 0)} seeds).")
    st.plotly_chart(_headline_chart(A), use_container_width=True,
                    config=PLOTLY_CONFIG)
    st.markdown(section_close(), unsafe_allow_html=True)

    # ---------------- thresholding conventions ----------------
    st.markdown(section_open("One convention at a time",
                             "the same systems, four thresholds"),
                unsafe_allow_html=True)
    st.plotly_chart(_convention_chart(A), use_container_width=True,
                    config=PLOTLY_CONFIG)
    rows = []
    for k, lbl, _ in SYSTEMS:
        rows.append({"System": lbl, **{
            cl: S.get(f"{k}.{ck}", {}).get("median")
            for ck, cl, _ in CONVENTIONS}})
    st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True,
                 hide_index=True)
    st.caption("Only the last two columns use no test labels. Compare down a "
               "column, never across a row — the original notebook printed "
               "the first two side by side, which reads as a tracking gain "
               "that is partly a change of convention.")
    st.markdown(section_close(), unsafe_allow_html=True)

    # ---------------- meta ablation ----------------
    st.markdown(section_open("Meta-learner ablation",
                             "which combiner earns its place"),
                unsafe_allow_html=True)
    st.plotly_chart(_meta_chart(A), use_container_width=True,
                    config=PLOTLY_CONFIG)
    st.markdown(section_close(), unsafe_allow_html=True)

    # ---------------- decorrelation ----------------
    st.markdown(section_open("Where the decorrelation comes from",
                             "GCN-GRU retrained per seed"),
                unsafe_allow_html=True)
    ec = A.get("error_correlation", {})
    st.plotly_chart(_corr_chart(A), use_container_width=True,
                    config=PLOTLY_CONFIG)
    st.caption(
        f"Disagreement fraction {_f(ec.get('disagreement_fraction'))} — the "
        f"share of test nodes where at least one, but not every, base learner "
        f"is wrong. The original reused one frozen GCN-GRU artefact across "
        f"all seeds, so its error pattern came from a single draw.")
    st.markdown(section_close(), unsafe_allow_html=True)

    # ---------------- tracker sensitivity ----------------
    sens = A.get("tracker_sensitivity", {})
    if sens:
        st.markdown(section_open("What the tracking head is sensitive to",
                                 "regulariser and EM population"),
                    unsafe_allow_html=True)
        rows = []
        for k, v in sens.items():
            if v.get("f1_post") is None:
                continue
            sysname, cfg, pop = k.split("|")
            rows.append({"System": sysname.replace("_", " "),
                         "Regulariser": cfg.replace("_", " "),
                         "EM population": pop.replace("_", " "),
                         "post-43 F1": v["f1_post"],
                         "tracking ρ": v["rho"], "rate MAE": v["rate_mae"]})
        st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True,
                     hide_index=True)
        st.caption(
            "The paper states Beta(5, 10); the original code passed no α/β "
            "and ran at the Beta(0.2, 1.8) defaults. ρ has n = 8 weekly "
            "points here — the exact two-sided permutation p for |ρ| ≥ 0.333 "
            "at n = 8 is 0.428, so rank correlation cannot separate these "
            "systems. Rate MAE uses the magnitudes and does not degrade.")
        st.plotly_chart(_traj_chart(A), use_container_width=True,
                        config=PLOTLY_CONFIG)
        st.markdown(section_close(), unsafe_allow_html=True)

    # ---------------- diagnostics ----------------
    st.markdown(section_open("Diagnostics", "per-week and calibration"),
                unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        st.plotly_chart(_week_chart(A), use_container_width=True,
                        config=PLOTLY_CONFIG)
    with d2:
        st.plotly_chart(_cal_chart(A), use_container_width=True,
                        config=PLOTLY_CONFIG)
    lift = [{"Learner": BASE_LABEL.get(n_, n_),
             "raw 166 F1": S.get(f"base.raw166.{n_}.f1", {}).get("median"),
             "enriched F1": S.get(f"base.enriched.{n_}.f1", {}).get("median"),
             "lift": S.get(f"lift.{n_}.f1", {}).get("median")}
            for n_ in ("RF", "XGB", "LGBM", "LR", "GNN")]
    st.dataframe(pd.DataFrame(lift).round(4), use_container_width=True,
                 hide_index=True)
    st.caption("Feature enrichment against the raw 166 columns, with the "
               "wavelet block computed causally.")
    st.markdown(section_close(), unsafe_allow_html=True)

    # ---------------- reconciliation ----------------
    recon = A.get("reconciliation", [])
    if recon:
        st.markdown(section_open("Reconciliation",
                                 "paper · original run · corrected run"),
                    unsafe_allow_html=True)
        df = pd.DataFrame([{
            "Quantity": r["label"], "Paper": r["paper"],
            "Original run": r["original"], "Corrected": r["median"],
            "95% CI": f"[{r['lo']:.3f}, {r['hi']:.3f}]",
            "vs paper": r["vs_paper"], "vs original": r["vs_original"],
        } for r in recon])
        counts = df["vs paper"].value_counts().to_dict()
        cols = st.columns(4)
        for col, name in zip(cols, ["consistent", "close", "shifted",
                                    "diverges"]):
            col.metric(name.title(), counts.get(name, 0))
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown(section_close(), unsafe_allow_html=True)

    if paired:
        st.markdown(section_open("Paired across-seed tests", ""),
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            {"Comparison": k.replace("__", " · ").replace("_", " "),
             "Median paired difference": v.get("median_diff"),
             "Wins": f"{v.get('wins', 0)}/{v.get('n', 0)}",
             "Sign-test p": v.get("sign_p")}
            for k, v in paired.items()]).round(4),
            use_container_width=True, hide_index=True)
        st.markdown(section_close(), unsafe_allow_html=True)
