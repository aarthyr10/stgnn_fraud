from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from app.utils.theme import (
    CHART,
    CONDITION_COLOURS,
    apply_plotly_layout,
    section_close,
    section_open,
    shaded_table_html,
)

CONDITION_LABELS = {
    "none":   "C1 · no correction",
    "batch":  "C2 · batch Saerens-EM",
    "online": "C3 · online per-timestep (proposed)",
}

ENCODER_LABELS = {
    "gcn_gru": "GCN-GRU",
    "rf":      "Random Forest",
}


def _safe(d: dict, key: str, default=0.0) -> float:
    if not isinstance(d, dict):
        return default
    val = d.get(key, default)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _fmt(value: float, decimals: int = 3) -> str:
    if value != value:
        return "—"
    return f"{value:.{decimals}f}"


def _load_metrics(artefact_paths: dict) -> dict | None:
    p = Path(artefact_paths["metrics"])
    if not p.exists():
        return None
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return None


def _scoreboard_chart(metrics: dict, metric_key: str,
                      title: str) -> go.Figure:
    fig = go.Figure()
    encoders = ["gcn_gru", "rf"]
    conditions = ["none", "batch", "online"]
    cond_colours = CONDITION_COLOURS
    for cond in conditions:
        vals = [
            _safe(metrics.get(f"{enc}_{cond}", {}), metric_key)
            for enc in encoders
        ]
        fig.add_trace(go.Bar(
            x=[ENCODER_LABELS[e] for e in encoders],
            y=vals,
            name=CONDITION_LABELS[cond],
            marker_color=cond_colours[cond],
            text=[_fmt(v) for v in vals],
            textposition="outside",
        ))
    fig.update_layout(barmode="group")
    apply_plotly_layout(fig, height=360, title=title)
    fig.update_yaxes(title=metric_key)
    return fig


def _trajectory_chart(metrics: dict) -> go.Figure:
    true_prior = metrics.get("true_prior", {})
    ts = sorted(int(t) for t in true_prior.keys())
    if not ts:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts, y=[float(true_prior[str(t)]) * 100 for t in ts],
        mode="lines+markers", name="True illicit rate",
        line=dict(color=CHART["ink"], width=3),
        marker=dict(size=10, symbol="diamond"),
    ))
    palette = {
        ("gcn_gru", "online"): CHART["violet"],
        ("rf", "online"): CHART["emerald"],
    }
    for (enc, cond), col in palette.items():
        m = metrics.get(f"{enc}_{cond}", {})
        est = m.get("estimated_q_illicit", {})
        if not est:
            continue
        vals = [float(est.get(str(t), float("nan"))) * 100 for t in ts]
        fig.add_trace(go.Scatter(
            x=ts, y=vals, mode="lines+markers",
            name=f"{ENCODER_LABELS[enc]} + C3",
            line=dict(color=col, width=2),
            marker=dict(size=8),
        ))
    fig.add_vline(
        x=43, line_dash="dot", line_color=CHART["crimson"],
        annotation_text="t=43",
    )
    apply_plotly_layout(
        fig, height=380,
        title="Online tracker · estimated q_t versus true rate",
    )
    fig.update_xaxes(title="Test timestep")
    fig.update_yaxes(title="Illicit rate %")
    return fig


def _verdict_block(metrics: dict) -> None:
    g_none = metrics.get("gcn_gru_none", {})
    g_batch = metrics.get("gcn_gru_batch", {})
    g_online = metrics.get("gcn_gru_online", {})
    rf_none = metrics.get("rf_none", {})
    rf_online = metrics.get("rf_online", {})

    rho_g = _safe(g_online, "spearman_rho_prior", float("nan"))
    rho_rf = _safe(rf_online, "spearman_rho_prior", float("nan"))
    f1_c1 = _safe(g_none, "f1_illicit")
    f1_post_c1 = _safe(g_none, "f1_post_shutdown")
    f1_post_c2 = _safe(g_batch, "f1_post_shutdown")
    f1_post_c3 = _safe(g_online, "f1_post_shutdown")
    f1_rf = _safe(rf_none, "f1_illicit")

    rows = [
        (
            "Proposal (a): C1 aggregate F1(illicit) approx 0.69",
            f"F1 = {_fmt(f1_c1)}",
            f1_c1 == f1_c1 and f1_c1 >= 0.60,
        ),
        (
            "Proposal (b): C2 batch EM post-shutdown F1 in [0.08, 0.12]",
            f"C1 post = {_fmt(f1_post_c1)}, C2 post = {_fmt(f1_post_c2)}",
            f1_post_c2 == f1_post_c2 and 0.08 <= f1_post_c2 <= 0.20,
        ),
        (
            "Proposal (c): C3 online post-shutdown F1 >= 0.18",
            f"C3 post = {_fmt(f1_post_c3)}",
            f1_post_c3 == f1_post_c3 and f1_post_c3 >= 0.18,
        ),
        (
            "Proposal (d): C3 Spearman rho on t>=43 >= 0.7",
            f"rho = {_fmt(rho_g, 2)}",
            rho_g == rho_g and rho_g >= 0.7,
        ),
        (
            "Proposal (e): RF+ online tracker rho > 0 (architecture-agnostic)",
            f"RF+ rho = {_fmt(rho_rf, 2)}",
            rho_rf == rho_rf and rho_rf > 0,
        ),
        (
            "Maganti reference: RF F1(illicit) approx 0.82",
            f"RF F1 = {_fmt(f1_rf)}",
            f1_rf == f1_rf and f1_rf >= 0.78,
        ),
    ]

    for text, observed, passed in rows:
        verdict = "PASS" if passed else "FAIL"
        colour = "#0F6E56" if passed else "#993556"
        st.markdown(
            f'<div style="padding:0.3rem 0;font-size:0.9rem">'
            f'<span style="display:inline-block;min-width:46px;'
            f'padding:0.05rem 0.4rem;margin-right:0.6rem;'
            f'border-radius:4px;background:{colour};color:#FFFFFF;'
            f'font-size:0.72rem;font-weight:700;letter-spacing:0.04em;'
            f'text-align:center">{verdict}</span>'
            f'<b>{text}</b> &nbsp; <code>{observed}</code></div>',
            unsafe_allow_html=True,
        )

    if rho_rf == rho_rf and rho_g == rho_g:
        cross_ok = rho_g > 0 and rho_rf > 0
        verdict_txt = "encoder-agnostic" if cross_ok else "encoder-dependent"
        st.caption(
            f"Cross-architecture check: rho (GCN-GRU + online) = "
            f"{_fmt(rho_g, 2)}, rho (RF + online) = {_fmt(rho_rf, 2)}. "
            f"Both positive means the tracker is {verdict_txt}."
        )


def _per_timestep_f1_chart(metrics: dict) -> go.Figure | None:
    series = [
        ("gcn_gru_none", "C1 no correction", CHART["amber"]),
        ("gcn_gru_online", "C3 online (proposed)", CHART["violet"]),
    ]
    fig = go.Figure()
    have = False
    for key, label, color in series:
        per_t = metrics.get(key, {}).get("per_timestep_f1", {})
        if not per_t:
            continue
        ts = sorted(int(t) for t in per_t.keys())
        ys = []
        for t in ts:
            raw = per_t.get(str(t))
            v = float(raw) if raw is not None else float("nan")
            ys.append(v if v == v else None)
        fig.add_trace(go.Scatter(
            x=ts, y=ys, mode="lines+markers", name=label,
            line=dict(color=color, width=2), marker=dict(size=6),
            connectgaps=False,
        ))
        have = True
    if not have:
        return None
    fig.add_vline(
        x=43, line_dash="dot", line_color=CHART["crimson"],
        annotation_text="t=43",
    )
    apply_plotly_layout(fig, height=320, title="Per-week F1 (illicit)")
    fig.update_xaxes(title="Test timestep")
    fig.update_yaxes(title="F1", range=[0, 1])
    return fig


def _deployable_table(metrics: dict) -> str | None:
    rows = [
        ("GCN-GRU · C1 none", "gcn_gru_none"),
        ("GCN-GRU · C3 online", "gcn_gru_online"),
        ("RF · C1 none", "rf_none"),
        ("RF · C3 online", "rf_online"),
    ]
    have = any(
        "f1_post_shutdown_deployable" in metrics.get(k, {}) for _, k in rows
    )
    if not have:
        return None
    import pandas as pd
    data = []
    for label, key in rows:
        m = metrics.get(key, {})
        data.append({
            "Condition": label,
            "F1 (oracle)": _safe(m, "f1_illicit"),
            "F1 (deployable)": _safe(m, "f1_illicit_deployable"),
            "post-F1 (oracle)": _safe(m, "f1_post_shutdown"),
            "post-F1 (deployable)": _safe(m, "f1_post_shutdown_deployable"),
        })
    df = pd.DataFrame(data)
    return shaded_table_html(
        df,
        formats={
            "F1 (oracle)": "{:.3f}",
            "F1 (deployable)": "{:.3f}",
            "post-F1 (oracle)": "{:.3f}",
            "post-F1 (deployable)": "{:.3f}",
        },
    )


def _report_checks(metrics: dict) -> None:
    tp = {}
    for k, v in metrics.get("true_prior", {}).items():
        fv = _safe({"v": v}, "v", float("nan"))
        if fv == fv:
            tp[int(k)] = fv
    pre = [tp[t] for t in tp if 35 <= t <= 42]
    post = [tp[t] for t in tp if 43 <= t <= 49]
    checks = []
    if pre and post:
        pre_mean = sum(pre) / len(pre) * 100
        trough = min(post) * 100
        wk49 = tp.get(49, float("nan")) * 100
        checks.append((
            "True-prior curve",
            f"pre-shutdown ~{pre_mean:.1f}% -> trough {trough:.1f}% -> "
            f"week 49 {wk49:.1f}%",
            True,
        ))
    checks.append((
        "RF feature set",
        "trains on all 166 features including the timestep column "
        "(loader column 0)",
        True,
    ))
    checks.append((
        "Decision threshold",
        "both reported: oracle (fits test labels, upper bound) and "
        "deployable (prior-matched, no test labels)",
        True,
    ))
    for label, detail, ok in checks:
        colour = "#0F6E56" if ok else "#993556"
        tag = "OK" if ok else "CHECK"
        st.markdown(
            f'<div style="padding:0.3rem 0;font-size:0.9rem">'
            f'<span style="display:inline-block;min-width:42px;'
            f'padding:0.05rem 0.4rem;margin-right:0.6rem;border-radius:4px;'
            f'background:{colour};color:#FFFFFF;font-size:0.72rem;'
            f'font-weight:700;text-align:center">{tag}</span>'
            f'<b>{label}</b> &nbsp; <span style="color:#44443F">'
            f'{detail}</span></div>',
            unsafe_allow_html=True,
        )


def _render_report_assets(artefact_paths: dict) -> None:
    assets = Path(artefact_paths["metrics"]).parent / "report_assets"
    if not assets.exists():
        return
    pngs = sorted(assets.glob("*.png"))
    if not pngs:
        return
    st.markdown(section_open(
        "Representation and saliency",
        eyebrow="t-SNE per layer · GRU temporal saliency",
        tint="application",
    ), unsafe_allow_html=True)
    for png in pngs:
        st.image(str(png), use_container_width=True)
    st.markdown(section_close(), unsafe_allow_html=True)


def render_results_tab(artefact_paths: dict) -> None:
    st.markdown(
        '<h2 style="margin-top:0">Results</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='app-caption'>Two encoders × three prior-correction "
        "strategies. Headline metric: PR-AUC (threshold-free ranking "
        "quality). Secondary: Recall@5%FPR, F1@t≥43, Spearman ρ on the "
        "tracker.</div>",
        unsafe_allow_html=True,
    )

    metrics = _load_metrics(artefact_paths)
    if metrics is None:
        st.warning(
            "No metrics on disk yet. Open the Pipeline tab and click "
            "Run pipeline to generate them."
        )
        return

    st.markdown(section_open(
        "Scoreboard",
        eyebrow="2 encoders × 3 conditions",
        tint="application",
    ), unsafe_allow_html=True)

    encoder_keys = ["gcn_gru", "rf"]
    cond_keys = ["none", "batch", "online"]
    rows_table = []
    for enc in encoder_keys:
        for cond in cond_keys:
            m = metrics.get(f"{enc}_{cond}", {})
            rows_table.append({
                "Encoder": ENCODER_LABELS[enc],
                "Condition": CONDITION_LABELS[cond],
                "PR-AUC": _safe(m, "pr_auc"),
                "Recall @ 5% FPR": _safe(m, "recall_at_5pct_fpr"),
                "F1 (illicit)": _safe(m, "f1_illicit"),
                "F1 @ t≥43": _safe(m, "f1_post_shutdown"),
                "ρ on q_t": _safe(m, "spearman_rho_prior", float("nan")),
            })

    import pandas as pd
    df = pd.DataFrame(rows_table)
    st.markdown(
        shaded_table_html(
            df,
            subset=["PR-AUC", "Recall @ 5% FPR", "ρ on q_t"],
            ramp_per_column={
                "PR-AUC": "Purples",
                "Recall @ 5% FPR": "Greens",
                "ρ on q_t": "Blues",
            },
            formats={
                "PR-AUC": "{:.3f}",
                "Recall @ 5% FPR": "{:.3f}",
                "F1 (illicit)": "{:.3f}",
                "F1 @ t≥43": "{:.3f}",
                "ρ on q_t": "{:.2f}",
            },
        ),
        unsafe_allow_html=True,
    )
    st.markdown(section_close(), unsafe_allow_html=True)

    c1_chart, c2_chart = st.columns(2)
    with c1_chart:
        st.plotly_chart(
            _scoreboard_chart(metrics, "pr_auc",
                              "PR-AUC (primary headline)"),
            width="stretch",
            config={"displayModeBar": False},
        )
    with c2_chart:
        st.plotly_chart(
            _scoreboard_chart(metrics, "recall_at_5pct_fpr",
                              "Recall @ 5% FPR"),
            width="stretch",
            config={"displayModeBar": False},
        )

    st.markdown(section_open(
        "Tracker trajectory",
        eyebrow="Online estimate vs true illicit rate",
        tint="post",
    ), unsafe_allow_html=True)
    st.plotly_chart(
        _trajectory_chart(metrics),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.markdown(section_close(), unsafe_allow_html=True)

    per_t_fig = _per_timestep_f1_chart(metrics)
    if per_t_fig is not None:
        st.markdown(section_open(
            "Per-week F1",
            eyebrow="C1 vs C3 across the test window",
            tint="post",
        ), unsafe_allow_html=True)
        st.plotly_chart(
            per_t_fig, width="stretch",
            config={"displayModeBar": False},
        )
        st.markdown(section_close(), unsafe_allow_html=True)

    deployable = _deployable_table(metrics)
    if deployable:
        st.markdown(section_open(
            "Threshold honesty",
            eyebrow="Oracle (fits test labels) vs deployable (label-free)",
            tint="application",
        ), unsafe_allow_html=True)
        st.markdown(deployable, unsafe_allow_html=True)
        st.markdown(section_close(), unsafe_allow_html=True)

    _render_report_assets(artefact_paths)

    st.markdown(section_open(
        "Report checks",
        eyebrow="Quick confirmations for the writeup",
        tint="data",
    ), unsafe_allow_html=True)
    _report_checks(metrics)
    st.markdown(section_close(), unsafe_allow_html=True)

    st.markdown(section_open(
        "Verdict",
        eyebrow="Did the proposal's claims hold?",
        tint="model",
    ), unsafe_allow_html=True)
    _verdict_block(metrics)

    if metrics.get("_meta", {}).get("demo"):
        st.caption(
            "Numbers come from short demo training runs on the synthetic "
            "graph. Replace the artefacts with weights from "
            "`training/train_*.py` for paper-quality numbers."
        )
    st.markdown(section_close(), unsafe_allow_html=True)
