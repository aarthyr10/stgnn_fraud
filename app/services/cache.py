from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
import pandas as pd
import torch

from app.data.demo import generate_demo_graph
from app.data.loader import (
    LABEL_ILLICIT,
    LABEL_LICIT,
    LABEL_UNKNOWN,
    load_elliptic,
)
from app.data.preprocess import apply_scaler, fit_scaler
from app.data.snapshots import build_snapshots, time_split
from app.models.gcn import StaticGCN
from app.models.gcn_gru import GcnGruHybrid
from app.models.tgat import TGAT
from app.services.rf_baseline import RFBundle, load_rf

log = logging.getLogger(__name__)

T = TypeVar("T")


def _cache_or_passthrough(streamlit_decorator_name: str):
    try:
        import streamlit as st
        return getattr(st, streamlit_decorator_name)
    except ImportError:
        def passthrough(*args, **kwargs):
            def wrap(fn: Callable[..., T]) -> Callable[..., T]:
                return fn
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return wrap
        return passthrough


cache_resource = _cache_or_passthrough("cache_resource")
cache_data = _cache_or_passthrough("cache_data")


def _real_dataset_available(data_dir: str) -> bool:
    d = Path(data_dir)
    return (
        d.exists()
        and (d / "elliptic_txs_features.csv").exists()
        and (d / "elliptic_txs_edgelist.csv").exists()
        and (d / "elliptic_txs_classes.csv").exists()
    )


@cache_resource(show_spinner="Loading graph")
def get_graph(data_dir: str, cache_path: str):
    if _real_dataset_available(data_dir):
        log.info("Real Elliptic CSVs found at %s", data_dir)
        return load_elliptic(data_dir, cache_path=cache_path)
    log.warning(
        "Real Elliptic CSVs not found at %s — using synthetic demo graph.",
        data_dir,
    )
    return generate_demo_graph()


@cache_resource(show_spinner="Building 49 weekly snapshots")
def get_snapshots(data_dir: str, cache_path: str):
    data = get_graph(data_dir, cache_path)
    snaps = build_snapshots(data)
    train_range, _, _ = time_split()
    scaler = fit_scaler(snaps, train_range)
    return apply_scaler(snaps, scaler), scaler


@cache_resource(show_spinner="Loading node-id registry")
def get_node_ids(data_dir: str, cache_path: str) -> torch.Tensor:
    data = get_graph(data_dir, cache_path)
    return data.node_id.clone()


@cache_resource(show_spinner="Computing true per-timestep prior")
def get_true_prior(data_dir: str, cache_path: str) -> dict[int, float]:
    data = get_graph(data_dir, cache_path)
    out: dict[int, float] = {}
    for t in range(1, 50):
        mask_t = (data.t == t)
        if not mask_t.any():
            out[t] = float("nan")
            continue
        y_t = data.y[mask_t]
        labelled = (y_t == LABEL_ILLICIT) | (y_t == LABEL_LICIT)
        if not labelled.any():
            out[t] = float("nan")
        else:
            out[t] = float((y_t[labelled] == LABEL_ILLICIT).float().mean())
    return out


@cache_resource(show_spinner="Snapshot summary")
def get_data_summary(data_dir: str, cache_path: str) -> dict:
    data = get_graph(data_dir, cache_path)
    snaps = build_snapshots(data)
    rows = []
    for s in snaps:
        deg = torch.bincount(
            s.edge_index[0], minlength=s.x.size(0)
        ) if s.edge_index.numel() else torch.zeros(s.x.size(0),
                                                    dtype=torch.long)
        n_il = int((s.y == LABEL_ILLICIT).sum())
        n_li = int((s.y == LABEL_LICIT).sum())
        n_un = int((s.y == LABEL_UNKNOWN).sum())
        n_lab = n_il + n_li
        rows.append({
            "t": s.t,
            "nodes": int(s.x.size(0)),
            "edges": int(s.edge_index.size(1)),
            "illicit": n_il,
            "licit": n_li,
            "unknown": n_un,
            "illicit_rate_labelled": (n_il / n_lab) if n_lab else 0.0,
            "mean_degree": float(deg.float().mean()) if deg.numel() else 0.0,
            "max_degree": int(deg.max()) if deg.numel() else 0,
        })
    return {
        "is_demo": getattr(data, "is_demo", False),
        "n_nodes": int(data.num_nodes),
        "n_edges": int(data.edge_index.size(1)),
        "per_timestep": rows,
    }


def _full_artefact_set_present(artefact_paths: dict) -> bool:
    keys = ("gcn", "hybrid_head", "embeddings", "metrics", "rf")
    return all(Path(artefact_paths[k]).exists() for k in keys)


@cache_resource(show_spinner="Bootstrapping demo artefacts")
def ensure_artefacts(data_dir: str, cache_path: str,
                     artefact_paths_tuple: tuple) -> dict:
    from app.services.demo_artefacts import bootstrap_demo_artefacts
    artefact_paths = dict(artefact_paths_tuple)
    if _full_artefact_set_present(artefact_paths):
        log.info("All artefacts present at %s", artefact_paths)
        return {"used_demo": False}

    snaps, _ = get_snapshots(data_dir, cache_path)
    node_ids = get_node_ids(data_dir, cache_path)
    bootstrap_demo_artefacts(snaps, node_ids, artefact_paths)
    return {"used_demo": True}


@cache_resource(show_spinner="Loading GCN subnet")
def get_gcn(weights_path: str) -> StaticGCN:
    model = StaticGCN()
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


@cache_resource(show_spinner="Loading hybrid model")
def get_hybrid(gcn_path: str, head_path: str) -> GcnGruHybrid:
    model = GcnGruHybrid(gcn_weights_path=gcn_path)
    head_state = torch.load(head_path, map_location="cpu")
    if isinstance(head_state, dict) and "model" in head_state:
        head_state = head_state["model"]
    model.load_state_dict(head_state, strict=False)
    model.eval()
    return model


@cache_resource(show_spinner="Loading TGAT")
def get_tgat(weights_path: str) -> TGAT:
    model = TGAT()
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


@cache_resource(show_spinner="Loading Random Forest baseline")
def get_rf(weights_path: str) -> RFBundle:
    return load_rf(weights_path)


@cache_resource(show_spinner="Loading embedding table")
def get_embeddings(parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


@cache_resource(show_spinner="Loading metrics")
def get_metrics(json_path: str) -> dict:
    import json
    with open(json_path) as fh:
        return json.load(fh)


@cache_data(show_spinner="Computing 2D projection")
def cached_tsne(embeddings_bytes: bytes, perplexity: int = 30) -> "pd.DataFrame":
    import io

    from sklearn.manifold import TSNE
    arr = np.load(io.BytesIO(embeddings_bytes))
    if arr.shape[0] < 5:
        return pd.DataFrame(
            arr[:, :2] if arr.shape[1] >= 2 else np.zeros((arr.shape[0], 2)),
            columns=["x", "y"],
        )
    perp = min(perplexity, max(5, arr.shape[0] // 4))
    proj = TSNE(n_components=2, perplexity=perp, init="pca",
                random_state=0).fit_transform(arr)
    return pd.DataFrame(proj, columns=["x", "y"])
