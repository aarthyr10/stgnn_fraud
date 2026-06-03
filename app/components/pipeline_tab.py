from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F

from app.data.loader import LABEL_ILLICIT
from app.data.snapshots import time_split
from app.services.cache import (
    get_data_summary,
    get_node_ids,
    get_snapshots,
)
from app.services.demo_artefacts import (
    _effective_p_train_gru,
    _effective_p_train_rf,
    _maybe_precompute_embeddings,
    _maybe_train_gcn,
    _maybe_train_gru,
    _maybe_train_rf,
    _score_condition,
)
from app.services.history import append_run
from app.services.prior_tracker import (
    compute_true_prior_per_timestep,
    online_per_timestep_tracker,
    saerens_em_batch,
)
from app.services.rf_baseline import predict_rf_per_timestep
from app.utils.theme import (
    CHART,
    CONDITION_COLOURS,
    apply_plotly_layout,
    encoder_banner,
    section_close,
    section_open,
    step_card_body,
    step_row,
)


def _safe_get(d: dict, key: str, default=0.0):
    val = d.get(key, default) if isinstance(d, dict) else default
    if val is None:
        return default
    return val


def _format_metric(value: float, decimals: int = 3) -> str:
    if value != value:
        return "—"
    return f"{value:.{decimals}f}"


def _trajectory_chart(
    timesteps: list[int],
    true_rate: list[float],
    estimated: dict[str, list[float]],
    rho_by_method: dict[str, float],
    title: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timesteps, y=[r * 100 for r in true_rate],
        mode="lines+markers", name="True illicit rate",
        line=dict(color=CHART["ink"], width=3),
        marker=dict(size=10, symbol="diamond", color=CHART["ink"]),
    ))
    for label, vals in estimated.items():
        rho = rho_by_method.get(label, float("nan"))
        rho_text = f" (ρ={rho:.2f})" if rho == rho else ""
        fig.add_trace(go.Scatter(
            x=timesteps, y=[v * 100 for v in vals], mode="lines+markers",
            name=f"{label}{rho_text}",
            line=dict(color=CONDITION_COLOURS.get(label, CHART["violet"]),
                      width=2),
            marker=dict(size=7),
        ))
    fig.add_vline(
        x=43, line_dash="dot", line_color=CHART["crimson"],
        annotation_text="t=43",
    )
    apply_plotly_layout(fig, height=320, title=title)
    fig.update_xaxes(title="Test timestep")
    fig.update_yaxes(title="Estimated q_t(illicit) %")
    return fig


def _per_timestep_score_chart(
    metrics_by_label: dict[str, dict],
    title: str,
) -> go.Figure:
    fig = go.Figure()
    for label, m in metrics_by_label.items():
        per_t = m.get("per_timestep_f1", {})
        if not per_t:
            continue
        ts = sorted(int(t) for t in per_t.keys())
        vals = [per_t[str(t)] for t in ts]
        fig.add_trace(go.Scatter(
            x=ts, y=vals, mode="lines+markers",
            name=label,
            line=dict(color=CONDITION_COLOURS.get(label, CHART["violet"]),
                      width=2),
            marker=dict(size=7),
        ))
    fig.add_vline(
        x=43, line_dash="dot", line_color=CHART["crimson"],
        annotation_text="t=43",
    )
    apply_plotly_layout(fig, height=300, title=title)
    fig.update_xaxes(title="Timestep")
    fig.update_yaxes(title="F1 (illicit)", range=[0, 1])
    return fig


def _gcn_per_timestep_posteriors(snaps, gcn, timesteps):
    p_per_t: dict[int, np.ndarray] = {}
    y_per_t: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for t_eval in timesteps:
            snap = snaps[t_eval - 1]
            if snap.x.size(0) == 0:
                p_per_t[t_eval] = np.zeros((0, 2))
                y_per_t[t_eval] = np.zeros(0, dtype=np.int64)
                continue
            logits = gcn(snap.x, snap.edge_index)
            p = F.softmax(logits, dim=-1).cpu().numpy()
            p_per_t[t_eval] = p
            y_per_t[t_eval] = snap.y.cpu().numpy().astype(np.int64)
    return p_per_t, y_per_t


def _gru_per_timestep_posteriors(
    snaps, node_ids, embeds_df, gru, timesteps,
):
    embed_cols = [c for c in embeds_df.columns if c.startswith("e")]
    embed_by_id = {
        nid: g.sort_values("t") for nid, g in embeds_df.groupby("node_id")
    }
    p_per_t: dict[int, list] = {t: [] for t in timesteps}
    y_per_t: dict[int, list] = {t: [] for t in timesteps}
    n_per_t: dict[int, list] = {t: [] for t in timesteps}
    with torch.no_grad():
        for t_eval in timesteps:
            snap = snaps[t_eval - 1]
            if snap.x.size(0) == 0:
                continue
            ids = node_ids[snap.global_idx].cpu().numpy()
            ys = snap.y.cpu().numpy()
            for nid, y in zip(ids, ys):
                g = embed_by_id.get(int(nid))
                if g is None or g.empty:
                    continue
                seq = g[g["t"] <= t_eval]
                if seq.empty:
                    continue
                tensor = torch.tensor(
                    seq[embed_cols].values, dtype=torch.float32,
                ).unsqueeze(0)
                logits = gru(tensor)
                p = F.softmax(logits, dim=-1).cpu().numpy()[0]
                p_per_t[t_eval].append(p)
                y_per_t[t_eval].append(int(y))
                n_per_t[t_eval].append(int(nid))
    return (
        {t: (np.asarray(p_per_t[t]) if p_per_t[t]
             else np.zeros((0, 2))) for t in timesteps},
        {t: (np.asarray(y_per_t[t], dtype=np.int64) if y_per_t[t]
             else np.zeros(0, dtype=np.int64)) for t in timesteps},
        {t: (np.asarray(n_per_t[t], dtype=np.int64) if n_per_t[t]
             else np.zeros(0, dtype=np.int64)) for t in timesteps},
    )


def _evaluate_encoder(
    p_per_t,
    y_per_t,
    p_train_eff,
    true_prior,
    test_ts,
    alpha: float,
    beta: float,
    em_max_iter: int,
    init_mode: str = "blend",
    blend: float = 0.5,
    floor: float = 0.005,
):
    chunks = [p_per_t[t] for t in test_ts if p_per_t[t].size]
    none_metrics = _score_condition(
        "none", p_per_t, y_per_t, true_prior,
        estimated_q={t: float(p_train_eff[LABEL_ILLICIT]) for t in test_ts},
    )

    if chunks:
        batch_in = np.concatenate(chunks, axis=0)

        batch = saerens_em_batch(batch_in, p_train_eff,
                                 alpha=alpha, beta=beta, max_iter=em_max_iter)
        q_batch = float(batch.q[LABEL_ILLICIT])
        p_batch = {}
        ratio = batch.q / np.clip(p_train_eff, 1e-8, None)
        for t in test_ts:
            p = p_per_t[t]
            if p.size == 0:
                p_batch[t] = p
                continue
            weighted = p * ratio[None, :]
            p_batch[t] = weighted / np.clip(
                weighted.sum(axis=1, keepdims=True), 1e-12, None,
            )

        batch_metrics = _score_condition(
            "batch", p_batch, y_per_t, true_prior,
            estimated_q={t: q_batch for t in test_ts},
        )
    else:
        q_batch = float(p_train_eff[LABEL_ILLICIT])
        batch_metrics = none_metrics

    online = online_per_timestep_tracker(
        p_per_t, p_train_eff,
        alpha=alpha, beta=beta, max_iter=em_max_iter,
        timesteps=test_ts,
        init_mode=init_mode, blend=blend, floor=floor,
    )
    p_online = {
        t: (online.per_step[t].corrected
            if t in online.per_step else p_per_t[t])
        for t in test_ts
    }
    online_q = {
        t: (float(online.per_step[t].q[LABEL_ILLICIT])
            if t in online.per_step
            else float(p_train_eff[LABEL_ILLICIT]))
        for t in test_ts
    }
    online_metrics = _score_condition(
        "online", p_online, y_per_t, true_prior,
        estimated_q=online_q,
    )
    return {
        "none": none_metrics,
        "batch": batch_metrics,
        "online": online_metrics,
        "q_batch": q_batch,
        "q_online": online_q,
        "p_train_eff": p_train_eff.tolist(),
    }


def _render_step(
    container,
    number: int,
    state: str,
    status_eyebrow: str,
    status_title: str,
    status_detail: str,
    data_eyebrow: str,
    data_title: str,
    data_detail: str = "",
    data_stats=None,
    result_eyebrow: str = "",
    result_title: str = "",
    result_detail: str = "",
    result_stats=None,
):
    html = step_row(
        number, state,
        step_card_body(status_eyebrow, status_title, status_detail),
        step_card_body(data_eyebrow, data_title, data_detail, data_stats),
        step_card_body(result_eyebrow, result_title, result_detail,
                       result_stats),
    )
    container.markdown(html, unsafe_allow_html=True)


def _render_condition_card(
    name: str,
    label: str,
    metrics: dict,
    is_proposed: bool,
) -> str:
    pr_auc = _safe_get(metrics, "pr_auc")
    recall = _safe_get(metrics, "recall_at_5pct_fpr")
    rho = _safe_get(metrics, "spearman_rho_prior", float("nan"))
    f1_post = _safe_get(metrics, "f1_post_shutdown")
    rho_txt = _format_metric(rho, 2)
    highlight = " highlight" if is_proposed else ""
    return (
        f'<div class="result-cell{highlight}">'
        f'<div class="name">{name}</div>'
        f'<div class="big">{_format_metric(pr_auc)}</div>'
        f'<div class="sub">PR-AUC · {label}</div>'
        f'<div class="stat-grid" style="margin-top:0.4rem">'
        f'<div class="stat"><div class="lbl">Recall@5%FPR</div>'
        f'<div class="val">{_format_metric(recall)}</div></div>'
        f'<div class="stat"><div class="lbl">F1 @ t≥43</div>'
        f'<div class="val">{_format_metric(f1_post)}</div></div>'
        f'<div class="stat"><div class="lbl">ρ on q_t</div>'
        f'<div class="val">{rho_txt}</div></div>'
        f'</div>'
        f'</div>'
    )


def _render_run(
    artefact_paths: dict, data_dir: str, graph_cache: str,
    *, force_retrain: bool,
    alpha_override: float | None = None,
    beta_override: float | None = None,
    em_iter_override: int | None = None,
    seed_override: int | None = None,
    init_mode_override: str | None = None,
    blend_override: float | None = None,
    floor_override: float | None = None,
    note: str = "",
) -> dict:
    alpha = (
        float(alpha_override) if alpha_override is not None
        else float(st.session_state.get("beta_alpha", 0.2))
    )
    beta = (
        float(beta_override) if beta_override is not None
        else float(st.session_state.get("beta_beta", 1.8))
    )
    em_iter = (
        int(em_iter_override) if em_iter_override is not None
        else int(st.session_state.get("em_max_iter", 12))
    )
    seed = (
        int(seed_override) if seed_override is not None
        else int(st.session_state.get("seed", 42))
    )
    init_mode = (
        str(init_mode_override) if init_mode_override is not None
        else str(st.session_state.get("tracker_init_mode", "blend"))
    )
    blend = (
        float(blend_override) if blend_override is not None
        else float(st.session_state.get("tracker_blend", 0.5))
    )
    floor = (
        float(floor_override) if floor_override is not None
        else float(st.session_state.get("tracker_floor", 0.005))
    )

    if force_retrain:
        for k in ("gcn", "hybrid_head", "rf", "embeddings", "metrics"):
            p = Path(artefact_paths[k])
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    sequence_box = st.container()
    sequence_box.markdown(section_open(
        "Pipeline run",
        eyebrow="Both encoders, three prior-correction conditions each",
        tint="application",
    ), unsafe_allow_html=True)
    progress = sequence_box.progress(0.0)

    snaps, _ = get_snapshots(data_dir, graph_cache)
    node_ids = get_node_ids(data_dir, graph_cache)
    summary = get_data_summary(data_dir, graph_cache)
    _, _, test_range = time_split()
    test_ts = list(test_range)
    true_prior = compute_true_prior_per_timestep(
        {t: snaps[t - 1].y.cpu().numpy() for t in test_ts}
    )

    step_slots = []
    for i in range(8):
        step_slots.append(sequence_box.empty())

    _render_step(
        step_slots[0], 1, "running",
        "Step 1", "Load and slice dataset",
        f"{summary['n_nodes']:,} nodes across 49 weekly snapshots; "
        f"strict-inductive split.",
        "Data observed",
        "Snapshots ready",
        data_stats=[
            ("Train t", "1..34"),
            ("Eval t", "35..49"),
            ("Post-shutdown", "43..49"),
            ("Edges", f"{summary['n_edges']:,}"),
        ],
        result_eyebrow="Status",
        result_title="OK",
        result_detail="Dataset cached for both encoders.",
    )
    progress.progress(0.1)
    _render_step(
        step_slots[0], 1, "done",
        "Step 1", "Load and slice dataset",
        f"{summary['n_nodes']:,} nodes across 49 weekly snapshots; "
        f"strict-inductive split.",
        "Data observed",
        "Snapshots ready",
        data_stats=[
            ("Train t", "1..34"),
            ("Eval t", "35..49"),
            ("Post-shutdown", "43..49"),
            ("Edges", f"{summary['n_edges']:,}"),
        ],
        result_eyebrow="Status",
        result_title="OK",
        result_detail="Dataset cached for both encoders.",
    )

    sequence_box.markdown(
        encoder_banner(
            "GCN-GRU (Pareja 2020 backbone)",
            "Encoder 1 of 2", "Frozen subnet → GRU head",
        ),
        unsafe_allow_html=True,
    )

    gcn_path = Path(artefact_paths["gcn"])
    embed_path = Path(artefact_paths["embeddings"])
    gru_path = Path(artefact_paths["hybrid_head"])

    _render_step(
        step_slots[1], 2, "running",
        "Step 2", "Train spatial GCN subnet",
        "Two-layer GraphConv 166→128→64, ReLU, dropout 0.30.",
        "Training data",
        "Snapshots 1..34",
        data_stats=[("Epochs", "85"), ("Loss", "CE")],
        result_eyebrow="Awaiting", result_title="Running…",
    )
    progress.progress(0.2)
    t0 = time.time()
    gcn = _maybe_train_gcn(snaps, gcn_path, seed=seed)
    elapsed = time.time() - t0
    _render_step(
        step_slots[1], 2, "done",
        "Step 2", "Train spatial GCN subnet",
        "Two-layer GraphConv 166→128→64, ReLU, dropout 0.30.",
        "Training data",
        "Snapshots 1..34",
        data_stats=[("Epochs", "85"), ("Loss", "CE")],
        result_eyebrow="Artefact",
        result_title="gcn_subnet.pt",
        result_detail=f"Wrote weights in {elapsed:.1f}s.",
    )

    _render_step(
        step_slots[2], 3, "running",
        "Step 3", "Precompute frozen embeddings",
        "One 64-d embedding per (node, timestep).",
        "Coverage",
        "All snapshots",
        result_eyebrow="Awaiting", result_title="Running…",
    )
    progress.progress(0.3)
    t0 = time.time()
    embeds_df = _maybe_precompute_embeddings(
        snaps, node_ids, gcn, embed_path,
    )
    elapsed = time.time() - t0
    _render_step(
        step_slots[2], 3, "done",
        "Step 3", "Precompute frozen embeddings",
        "One 64-d embedding per (node, timestep).",
        "Coverage",
        "All snapshots",
        data_stats=[("Rows", f"{len(embeds_df):,}"), ("Dim", "64")],
        result_eyebrow="Artefact",
        result_title="embeddings.parquet",
        result_detail=f"Cached in {elapsed:.1f}s.",
    )

    _render_step(
        step_slots[3], 4, "running",
        "Step 4", "Train GRU head",
        "Two-layer GRU (hidden 64), linear classifier head.",
        "Sequences", "Per-node embedding history",
        data_stats=[("Epochs", "55"), ("Class weight", "inverse-freq")],
        result_eyebrow="Awaiting", result_title="Running…",
    )
    progress.progress(0.45)
    t0 = time.time()
    gru = _maybe_train_gru(
        snaps, node_ids, embeds_df, gcn_path, gru_path, seed=seed,
    )
    elapsed = time.time() - t0
    _render_step(
        step_slots[3], 4, "done",
        "Step 4", "Train GRU head",
        "Two-layer GRU (hidden 64), linear classifier head.",
        "Sequences", "Per-node embedding history",
        data_stats=[("Epochs", "55"), ("Class weight", "inverse-freq")],
        result_eyebrow="Artefact",
        result_title="gru_head.pt",
        result_detail=f"Trained in {elapsed:.1f}s.",
    )

    _render_step(
        step_slots[4], 5, "running",
        "Step 5", "Apply prior-correction heads · GCN-GRU",
        "Score the test window, then run C1, C2 and C3 on the same softmax.",
        "Test horizon", "t=35..49",
        result_eyebrow="Awaiting", result_title="Computing C1/C2/C3…",
    )
    progress.progress(0.6)
    p_gcn_only, y_gcn_only = _gcn_per_timestep_posteriors(
        snaps, gcn, test_ts,
    )
    gcn_only_metrics = _score_condition(
        "gcn_only", p_gcn_only, y_gcn_only, true_prior,
        estimated_q={t: 0.0 for t in test_ts},
    )
    p_gru, y_gru, _ = _gru_per_timestep_posteriors(
        snaps, node_ids, embeds_df, gru, test_ts,
    )
    p_train_gru = _effective_p_train_gru(snaps, node_ids, embeds_df, gru)
    gru_eval = _evaluate_encoder(
        p_gru, y_gru, p_train_gru, true_prior, test_ts,
        alpha=alpha, beta=beta, em_max_iter=em_iter,
        init_mode=init_mode, blend=blend, floor=floor,
    )
    _render_step(
        step_slots[4], 5, "done",
        "Step 5", "Apply prior-correction heads · GCN-GRU",
        "Score the test window, then run C1, C2 and C3 on the same softmax.",
        "Effective p_train",
        f"{p_train_gru[LABEL_ILLICIT] * 100:.1f}% illicit",
        data_stats=[
            ("q batch", f"{gru_eval['q_batch'] * 100:.2f}%"),
            ("EM iters", str(em_iter)),
            ("Beta α", f"{alpha:.2f}"),
            ("Beta β", f"{beta:.2f}"),
        ],
        result_eyebrow="Verdict",
        result_title="See condition cards below",
        result_detail="C3 should beat C1 / C2 on PR-AUC and ρ.",
    )

    gru_grid = (
        '<div class="result-grid">'
        + _render_condition_card(
            "C1", "no correction", gru_eval["none"], False)
        + _render_condition_card(
            "C2", "batch Saerens-EM", gru_eval["batch"], False)
        + _render_condition_card(
            "C3", "online per-t (proposed)", gru_eval["online"], True)
        + "</div>"
    )
    sequence_box.markdown(gru_grid, unsafe_allow_html=True)
    sequence_box.plotly_chart(
        _trajectory_chart(
            test_ts,
            [true_prior.get(t, float("nan")) for t in test_ts],
            {
                "none": [
                    float(p_train_gru[LABEL_ILLICIT]) for _ in test_ts
                ],
                "batch": [gru_eval["q_batch"] for _ in test_ts],
                "online": [gru_eval["q_online"][t] for t in test_ts],
            },
            {
                "none": float("nan"),
                "batch": float("nan"),
                "online": _safe_get(gru_eval["online"], "spearman_rho_prior",
                                    float("nan")),
            },
            "Estimated q_t · GCN-GRU (true rate in black)",
        ),
        width="stretch",
        config={"displayModeBar": False},
    )

    sequence_box.markdown(
        encoder_banner(
            "Random Forest (Maganti 2026 reference)",
            "Encoder 2 of 2", "Trees on raw 166-d features",
        ),
        unsafe_allow_html=True,
    )

    rf_path = Path(artefact_paths["rf"])
    _render_step(
        step_slots[5], 6, "running",
        "Step 6", "Train Random Forest",
        "500 trees, depth unconstrained — Maganti reference scale.",
        "Training data", "Snapshots 1..34",
        result_eyebrow="Awaiting", result_title="Running…",
    )
    progress.progress(0.75)
    t0 = time.time()
    rf = _maybe_train_rf(snaps, rf_path, seed=seed)
    elapsed = time.time() - t0
    _render_step(
        step_slots[5], 6, "done",
        "Step 6", "Train Random Forest",
        "500 trees, depth unconstrained — Maganti reference scale.",
        "Training data", "Snapshots 1..34",
        data_stats=[("Trees", "500"), ("Depth", "unbounded")],
        result_eyebrow="Artefact",
        result_title="rf_baseline.pkl",
        result_detail=f"Fitted in {elapsed:.1f}s.",
    )

    _render_step(
        step_slots[6], 7, "running",
        "Step 7", "Apply prior-correction heads · Random Forest",
        "Score test window, then run C1, C2 and C3 on RF posteriors.",
        "Test horizon", "t=35..49",
        result_eyebrow="Awaiting", result_title="Computing…",
    )
    progress.progress(0.9)
    rf_per_t = predict_rf_per_timestep(rf, snaps, test_ts)
    p_rf = {t: rf_per_t[t]["p"] for t in test_ts}
    y_rf = {t: rf_per_t[t]["y"] for t in test_ts}
    p_train_rf_eff = _effective_p_train_rf(rf, snaps)
    rf_eval = _evaluate_encoder(
        p_rf, y_rf, p_train_rf_eff, true_prior, test_ts,
        alpha=alpha, beta=beta, em_max_iter=em_iter,
        init_mode=init_mode, blend=blend, floor=floor,
    )
    _render_step(
        step_slots[6], 7, "done",
        "Step 7", "Apply prior-correction heads · Random Forest",
        "Score test window, then run C1, C2 and C3 on RF posteriors.",
        "Effective p_train",
        f"{p_train_rf_eff[LABEL_ILLICIT] * 100:.1f}% illicit",
        data_stats=[
            ("q batch", f"{rf_eval['q_batch'] * 100:.2f}%"),
            ("EM iters", str(em_iter)),
            ("Beta α", f"{alpha:.2f}"),
            ("Beta β", f"{beta:.2f}"),
        ],
        result_eyebrow="Verdict",
        result_title="See condition cards below",
        result_detail="Same tracker, different backbone — works or it does not.",
    )

    rf_grid = (
        '<div class="result-grid">'
        + _render_condition_card(
            "C1", "no correction", rf_eval["none"], False)
        + _render_condition_card(
            "C2", "batch Saerens-EM", rf_eval["batch"], False)
        + _render_condition_card(
            "C3", "online per-t (proposed)", rf_eval["online"], True)
        + "</div>"
    )
    sequence_box.markdown(rf_grid, unsafe_allow_html=True)
    sequence_box.plotly_chart(
        _trajectory_chart(
            test_ts,
            [true_prior.get(t, float("nan")) for t in test_ts],
            {
                "none": [
                    float(p_train_rf_eff[LABEL_ILLICIT]) for _ in test_ts
                ],
                "batch": [rf_eval["q_batch"] for _ in test_ts],
                "online": [rf_eval["q_online"][t] for t in test_ts],
            },
            {
                "none": float("nan"),
                "batch": float("nan"),
                "online": _safe_get(rf_eval["online"], "spearman_rho_prior",
                                    float("nan")),
            },
            "Estimated q_t · Random Forest",
        ),
        width="stretch",
        config={"displayModeBar": False},
    )

    _render_step(
        step_slots[7], 8, "running",
        "Step 8", "Persist metrics",
        "Aggregate the six-cell scoreboard into metrics.json.",
        "Output", "artefacts/metrics.json",
        result_eyebrow="Status", result_title="Writing…",
    )
    progress.progress(0.95)
    metrics = {
        "gcn_only":       gcn_only_metrics,
        "gcn_gru_none":   gru_eval["none"],
        "gcn_gru_batch":  gru_eval["batch"],
        "gcn_gru_online": gru_eval["online"],
        "rf_none":        rf_eval["none"],
        "rf_batch":       rf_eval["batch"],
        "rf_online":      rf_eval["online"],
        "true_prior": {
            str(t): float(true_prior.get(t, float("nan"))) for t in test_ts
        },
        "p_train_gru_effective": gru_eval["p_train_eff"],
        "p_train_rf_effective": rf_eval["p_train_eff"],
        "gru_lift_over_gcn": {
            "gcn_only_f1": float(gcn_only_metrics.get("f1_illicit", 0.0)),
            "gcn_gru_f1": float(gru_eval["none"].get("f1_illicit", 0.0)),
            "gcn_only_pr_auc": float(gcn_only_metrics.get("pr_auc", 0.0)),
            "gcn_gru_pr_auc": float(gru_eval["none"].get("pr_auc", 0.0)),
        },
        "_meta": {
            "generated_at": int(time.time()),
            "alpha": alpha, "beta": beta, "em_max_iter": em_iter,
            "seed": seed,
            "tracker_init_mode": init_mode,
            "tracker_blend": blend,
            "tracker_floor": floor,
            "demo": summary["is_demo"],
        },
    }
    metrics_path = Path(artefact_paths["metrics"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    progress.progress(1.0)
    _render_step(
        step_slots[7], 8, "done",
        "Step 8", "Persist metrics",
        "Aggregate the six-cell scoreboard into metrics.json.",
        "Output", "artefacts/metrics.json",
        result_eyebrow="Status", result_title="Saved",
        result_detail="Open the Results tab for the side-by-side scoreboard.",
    )
    sequence_box.markdown(section_close(), unsafe_allow_html=True)

    st.session_state.pipeline_metrics = metrics

    try:
        append_run(
            artefact_paths.get("history",
                               str(Path(metrics_path).parent / "run_history.jsonl")),
            params={
                "alpha": alpha, "beta": beta, "em_iter": em_iter,
                "seed": seed,
            },
            metrics=metrics,
            note=note,
        )
    except Exception:
        pass
    return metrics


def render_pipeline_tab(
    artefact_paths: dict, data_dir: str, graph_cache: str,
) -> None:
    st.markdown('<h2 style="margin-top:0">Pipeline</h2>',
                unsafe_allow_html=True)

    with st.expander("Tracker settings", expanded=False):
        cols = st.columns(4)
        with cols[0]:
            st.slider(
                "Beta alpha (illicit pseudo-count)",
                min_value=0.0, max_value=10.0, value=0.2, step=0.1,
                key="beta_alpha",
                help="Pseudo-count on the illicit class in the Beta "
                     "regulariser. Lower values let EM chase a deeper trough.",
            )
        with cols[1]:
            st.slider(
                "Beta beta (licit pseudo-count)",
                min_value=0.0, max_value=50.0, value=1.8, step=0.2,
                key="beta_beta",
                help="Beta-prior mode is alpha / (alpha + beta). "
                     "Defaults (0.2, 1.8) give a 10% mode with only 2 "
                     "pseudo-counts.",
            )
        with cols[2]:
            st.slider(
                "EM max iterations",
                min_value=1, max_value=50, value=12, step=1,
                key="em_max_iter",
                help="Per-timestep EM iteration cap.",
            )
        with cols[3]:
            st.number_input(
                "Random seed",
                min_value=0, max_value=10000, value=42, step=1,
                key="seed",
            )
        cols2 = st.columns(3)
        with cols2[0]:
            st.selectbox(
                "Tracker init mode",
                options=["blend", "prior", "prev"],
                index=0,
                key="tracker_init_mode",
                help="How each timestep's EM starts. 'prev' = warm-start "
                     "from q_{t-1} (traps EM in the post-shutdown trough). "
                     "'prior' = init from p_train every step. "
                     "'blend' = mix of both (recommended).",
            )
        with cols2[1]:
            st.slider(
                "Blend weight on q_{t-1}",
                min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                key="tracker_blend",
                help="Only used when init mode is 'blend'. 1.0 = pure "
                     "warm-start, 0.0 = pure p_train init.",
            )
        with cols2[2]:
            st.slider(
                "q_init floor",
                min_value=0.0, max_value=0.05, value=0.005, step=0.001,
                key="tracker_floor",
                format="%.3f",
                help="Lower-bound on q(illicit) used to seed EM. Prevents "
                     "the trough from pinning subsequent steps at zero.",
            )

    artefact_states = {
        k: Path(artefact_paths[k]).exists()
        for k in ("gcn", "hybrid_head", "rf", "embeddings", "metrics")
    }
    ready = all(artefact_states.values())

    if not ready:
        missing = [k for k, v in artefact_states.items() if not v]
        st.warning(f"Missing artefacts: {', '.join(missing)}")

    b1, b2, b3 = st.columns([2, 2, 2])
    with b1:
        run_clicked = st.button(
            "Run pipeline" if not ready else "Re-run evaluation",
            type="primary", width="stretch",
        )
    with b2:
        fresh_clicked = st.button(
            "Run fresh (wipe and retrain)",
            width="stretch",
        )
    with b3:
        seed_sweep_clicked = st.button(
            "Run 10-seed protocol",
            width="stretch",
            help="Train the full GCN+GRU+RF stack ten times with seeds "
                 "[42, 7, 1337, 2024, 11, 100, 200, 300, 400, 500] using "
                 "the current Tracker settings. Slow: every encoder is "
                 "retrained from scratch ten times. This is the "
                 "proposal's median-over-seeds bootstrap protocol.",
        )

    if not (run_clicked or fresh_clicked or seed_sweep_clicked):
        return

    try:
        if seed_sweep_clicked:
            seed_list = [42, 7, 1337, 2024, 11, 100, 200, 300, 400, 500]
            for i, s in enumerate(seed_list, 1):
                st.markdown(
                    f"<div style='margin-top:0.6rem;font-weight:700;"
                    f"color:#185FA5'>Seed protocol "
                    f"{i}/{len(seed_list)}: seed={s} (full retrain)</div>",
                    unsafe_allow_html=True,
                )
                _render_run(
                    artefact_paths, data_dir, graph_cache,
                    force_retrain=True,
                    seed_override=s,
                    note=f"seed:{s}",
                )
        else:
            _render_run(
                artefact_paths, data_dir, graph_cache,
                force_retrain=fresh_clicked,
                note="fresh" if fresh_clicked else "single",
            )
    except Exception as exc:
        st.error(
            f"Pipeline failed: {type(exc).__name__}: {exc}",
        )
