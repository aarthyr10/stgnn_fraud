from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.data.demo import ELLIPTIC_PRIOR_TRAJECTORY
from app.data.loader import LABEL_ILLICIT, LABEL_LICIT, LABEL_UNKNOWN
from app.services.cache import get_data_summary, get_graph, get_true_prior
from app.utils.theme import (
    CHART,
    PLOTLY_COLOURS,
    apply_plotly_layout,
    section_close,
    section_open,
)


def _prior_trajectory_chart(true_prior: dict[int, float]) -> go.Figure:
    ts = sorted(true_prior.keys())
    vals = [float(true_prior[t]) * 100 for t in ts]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts, y=vals, mode="lines+markers",
        line=dict(color=CHART["violet"], width=3),
        marker=dict(size=8, color=CHART["violet"]),
        name="Realised illicit rate",
    ))
    target_ts = sorted(ELLIPTIC_PRIOR_TRAJECTORY.keys())
    target_vals = [ELLIPTIC_PRIOR_TRAJECTORY[t] * 100 for t in target_ts]
    fig.add_trace(go.Scatter(
        x=target_ts, y=target_vals, mode="lines",
        line=dict(color=CHART["crimson"], width=1.5, dash="dot"),
        name="Generator target",
    ))
    fig.add_vrect(
        x0=34.5, x1=40.5, fillcolor="#EDEBFD", opacity=0.35,
        layer="below", line_width=0, annotation_text="val",
    )
    fig.add_vrect(
        x0=40.5, x1=49.5, fillcolor="#FBE7EE", opacity=0.35,
        layer="below", line_width=0, annotation_text="test",
    )
    fig.add_vline(
        x=43, line_dash="dot", line_color=CHART["crimson"],
        annotation_text="t=43 shutdown",
    )
    apply_plotly_layout(
        fig, height=360,
        title="Per-timestep illicit rate (labelled nodes only)",
    )
    fig.update_xaxes(title="Timestep (week)")
    fig.update_yaxes(title="Illicit rate %")
    return fig


def _class_balance_chart(rows: list[dict]) -> go.Figure:
    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["t"], y=df["illicit"], name="Illicit",
        marker_color=PLOTLY_COLOURS["illicit"],
    ))
    fig.add_trace(go.Bar(
        x=df["t"], y=df["licit"], name="Licit",
        marker_color=PLOTLY_COLOURS["licit"],
    ))
    fig.add_trace(go.Bar(
        x=df["t"], y=df["unknown"], name="Unlabelled",
        marker_color=PLOTLY_COLOURS["unknown"],
    ))
    fig.update_layout(barmode="stack")
    apply_plotly_layout(fig, height=320, title="Label composition per timestep")
    fig.update_xaxes(title="Timestep")
    fig.update_yaxes(title="Nodes")
    fig.add_vline(x=43, line_dash="dash", line_color=CHART["crimson"])
    return fig


def render_dataset_tab(
    artefact_paths: dict, data_dir: str, graph_cache: str,
) -> None:
    st.markdown(
        '<h2 style="margin-top:0">Dataset</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='app-caption'>Elliptic Bitcoin transaction graph, "
        "real or synthetic. Each row is a transaction in one of 49 weekly "
        "snapshots. The illicit rate among labelled nodes is the "
        "non-stationary deployment prior the tracker has to follow."
        "</div>",
        unsafe_allow_html=True,
    )

    try:
        data = get_graph(data_dir, graph_cache)
        summary = get_data_summary(data_dir, graph_cache)
        true_prior = get_true_prior(data_dir, graph_cache)
    except Exception as exc:
        st.error(f"Could not load the dataset: {type(exc).__name__}: {exc}")
        return

    is_demo = summary["is_demo"]

    cols = st.columns(4)
    cols[0].metric(
        "Source",
        "Synthetic demo" if is_demo else "Real Elliptic CSVs",
    )
    cols[1].metric("Nodes", f"{summary['n_nodes']:,}")
    cols[2].metric("Edges", f"{summary['n_edges']:,}")
    n_il = int((data.y == LABEL_ILLICIT).sum())
    n_li = int((data.y == LABEL_LICIT).sum())
    cols[3].metric("Illicit / Licit", f"{n_il:,} / {n_li:,}")

    st.markdown(section_open(
        "Prior trajectory",
        eyebrow="The non-stationary deployment prior",
        tint="post",
    ), unsafe_allow_html=True)

    pre_t = [t for t in true_prior if t < 43 and true_prior[t] == true_prior[t]]
    trough = min(
        ((true_prior[t], t) for t in true_prior if 43 <= t <= 47),
        default=(0.0, 0),
    )
    mid_cols = st.columns(3)
    if pre_t:
        pre_rate = sum(true_prior[t] for t in pre_t) / max(len(pre_t), 1)
        mid_cols[0].metric("Pre-shutdown rate", f"{pre_rate * 100:.1f}%")
    else:
        mid_cols[0].metric("Pre-shutdown rate", "—")
    mid_cols[1].metric(
        f"Trough at t={trough[1]}", f"{trough[0] * 100:.2f}%",
    )
    last = true_prior.get(49, 0.0)
    mid_cols[2].metric(
        "Recovery at t=49",
        f"{last * 100:.1f}%" if last == last else "—",
    )

    st.plotly_chart(
        _prior_trajectory_chart(true_prior),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.markdown(section_close(), unsafe_allow_html=True)

    st.markdown(section_open(
        "Label composition over time",
        eyebrow="Counts per timestep",
        tint="data",
    ), unsafe_allow_html=True)
    st.plotly_chart(
        _class_balance_chart(summary["per_timestep"]),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.markdown(section_close(), unsafe_allow_html=True)

    n_un = int((data.y == LABEL_UNKNOWN).sum())
    st.caption(
        f"Splits: train t=1..34, eval t=35..49 (post-shutdown t>=43). "
        f"Unlabelled rows in scope: {n_un:,}."
    )
