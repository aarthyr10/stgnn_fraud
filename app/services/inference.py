from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F

from app.data.loader import LABEL_ILLICIT
from app.data.snapshots import Snapshot, time_split
from app.services.cache import (
    get_embeddings,
    get_hybrid,
    get_node_ids,
    get_rf,
    get_snapshots,
)
from app.services.prior_tracker import (
    correct_single_posterior,
    online_per_timestep_tracker,
    saerens_em_batch,
)
from app.services.rf_baseline import predict_rf_per_timestep

log = logging.getLogger(__name__)

ModelName = Literal["gcn_gru", "rf"]
PriorMode = Literal["none", "batch", "online"]


@dataclass
class Prediction:
    node_id: int
    timestep: int
    illicit_prob_raw: float
    illicit_prob: float
    label: int
    model: ModelName
    prior_mode: PriorMode
    q_train_illicit: float
    q_t_illicit: float
    embedding: Optional[np.ndarray] = None
    history: List[float] = field(default_factory=list)


def _global_node_idx(snapshots: List[Snapshot], node_id: int,
                     t: int) -> Optional[int]:
    snap = snapshots[t - 1]
    matches = (snap.global_idx == node_id).nonzero(as_tuple=False)
    if matches.numel() == 0:
        return None
    return int(matches[0])


def _gru_posteriors_per_test_t(
    snapshots: List[Snapshot], node_ids: torch.Tensor,
    artefact_paths: dict, timesteps: List[int],
) -> Dict[int, Dict[str, np.ndarray]]:
    embeds_df = get_embeddings(artefact_paths["embeddings"])
    embed_cols = [c for c in embeds_df.columns if c.startswith("e")]
    embed_by_id = {nid: g.sort_values("t")
                   for nid, g in embeds_df.groupby("node_id")}
    gru = get_hybrid(artefact_paths["gcn"], artefact_paths["hybrid_head"])

    out: Dict[int, Dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for t in timesteps:
            snap = snapshots[t - 1]
            if snap.x.size(0) == 0:
                out[t] = {"p": np.zeros((0, 2)),
                          "node_id": np.zeros(0, dtype=np.int64),
                          "y": np.zeros(0, dtype=np.int64)}
                continue
            ps, ys, nids = [], [], []
            ids = node_ids[snap.global_idx].cpu().numpy()
            yloc = snap.y.cpu().numpy()
            for nid, y in zip(ids, yloc):
                g = embed_by_id.get(int(nid))
                if g is None or g.empty:
                    continue
                seq = g[g["t"] <= t]
                if seq.empty:
                    continue
                tensor = torch.tensor(seq[embed_cols].values,
                                      dtype=torch.float32).unsqueeze(0)
                logits = gru(tensor)
                p = F.softmax(logits, dim=-1).cpu().numpy()[0]
                ps.append(p)
                ys.append(int(y))
                nids.append(int(nid))
            out[t] = {
                "p": np.asarray(ps) if ps else np.zeros((0, 2)),
                "y": np.asarray(ys, dtype=np.int64) if ys else
                     np.zeros(0, dtype=np.int64),
                "node_id": np.asarray(nids, dtype=np.int64) if nids else
                     np.zeros(0, dtype=np.int64),
            }
    return out


_GRU_POSTERIOR_CACHE: dict[str, dict] = {}


def gru_posteriors_cached(
    data_dir: str, graph_cache: str, artefact_paths: dict,
) -> Dict[int, Dict[str, np.ndarray]]:
    key = artefact_paths["hybrid_head"]
    if key in _GRU_POSTERIOR_CACHE:
        return _GRU_POSTERIOR_CACHE[key]
    snaps, _ = get_snapshots(data_dir, graph_cache)
    node_ids = get_node_ids(data_dir, graph_cache)
    _, _, test_range = time_split()
    _GRU_POSTERIOR_CACHE[key] = _gru_posteriors_per_test_t(
        snaps, node_ids, artefact_paths, list(test_range),
    )
    return _GRU_POSTERIOR_CACHE[key]


def _rf_posteriors_per_test_t(
    snapshots: List[Snapshot], node_ids: torch.Tensor,
    artefact_paths: dict, timesteps: List[int],
) -> Dict[int, Dict[str, np.ndarray]]:
    rf = get_rf(artefact_paths["rf"])
    raw = predict_rf_per_timestep(rf, snapshots, timesteps)
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for t in timesteps:
        snap = snapshots[t - 1]
        ids = node_ids[snap.global_idx].cpu().numpy() if snap.x.size(0)\
            else np.zeros(0, dtype=np.int64)
        out[t] = {
            "p": raw[t]["p"], "y": raw[t]["y"], "node_id": ids,
        }
    return out


_RF_POSTERIOR_CACHE: dict[str, dict] = {}


def rf_posteriors_cached(
    data_dir: str, graph_cache: str, artefact_paths: dict,
) -> Dict[int, Dict[str, np.ndarray]]:
    key = artefact_paths["rf"]
    if key in _RF_POSTERIOR_CACHE:
        return _RF_POSTERIOR_CACHE[key]
    snaps, _ = get_snapshots(data_dir, graph_cache)
    node_ids = get_node_ids(data_dir, graph_cache)
    _, _, test_range = time_split()
    _RF_POSTERIOR_CACHE[key] = _rf_posteriors_per_test_t(
        snaps, node_ids, artefact_paths, list(test_range),
    )
    return _RF_POSTERIOR_CACHE[key]


def compute_correction_for_timestep(
    p_per_t: Dict[int, Dict[str, np.ndarray]],
    p_train: np.ndarray,
    prior_mode: PriorMode,
    target_t: int,
    *,
    alpha: float = 0.2, beta: float = 1.8, em_max_iter: int = 12,
) -> tuple[np.ndarray, list[float]]:
    if prior_mode == "none":
        return p_train.copy(), [float(p_train[LABEL_ILLICIT])]
    if prior_mode == "batch":
        chunks = [p_per_t[t]["p"] for t in sorted(p_per_t.keys())
                  if p_per_t[t]["p"].size]
        if not chunks:
            return p_train.copy(), [float(p_train[LABEL_ILLICIT])]
        p_all = np.concatenate(chunks, axis=0)
        result = saerens_em_batch(p_all, p_train)
        return result.q, result.history

    p_dict = {t: p_per_t[t]["p"] for t in sorted(p_per_t.keys())}
    tracker = online_per_timestep_tracker(
        p_dict, p_train,
        alpha=alpha, beta=beta, max_iter=em_max_iter,
        timesteps=[t for t in sorted(p_dict.keys()) if t <= target_t],
    )
    if target_t in tracker.per_step:
        step = tracker.per_step[target_t]
        return step.q, step.history
    return p_train.copy(), [float(p_train[LABEL_ILLICIT])]


def predict(
    model_name: ModelName,
    node_id: int,
    timestep: int,
    artefact_paths: dict,
    data_dir: str,
    graph_cache_path: str,
    *,
    prior_mode: PriorMode = "none",
    alpha: float = 0.2, beta: float = 1.8, em_max_iter: int = 12,
) -> Prediction:
    snapshots, _ = get_snapshots(data_dir, graph_cache_path)
    snap = snapshots[timestep - 1]
    local_idx = _global_node_idx(snapshots, node_id, timestep)
    if local_idx is None:
        raise ValueError(
            f"Node {node_id} not present in timestep {timestep}. "
            "Pick a timestep in which the node appears."
        )
    label = int(snap.y[local_idx])

    if model_name == "gcn_gru":
        rf = get_rf(artefact_paths["rf"])
        p_train = rf.p_train
        per_t = gru_posteriors_cached(data_dir, graph_cache_path,
                                      artefact_paths)
        row_mask = per_t[timestep]["node_id"] == node_id
        if not row_mask.any():
            raise ValueError(f"No cached posterior for node {node_id} at "
                             f"t={timestep}")
        p_raw = per_t[timestep]["p"][row_mask][0]

        embeds_df = get_embeddings(artefact_paths["embeddings"])
        seq_rows = embeds_df[(embeds_df.node_id == node_id)
                             & (embeds_df.t <= timestep)].sort_values("t")
        embed_cols = [c for c in embeds_df.columns if c.startswith("e")]
        embedding = (seq_rows[embed_cols].values[-1]
                     if not seq_rows.empty else None)
    elif model_name == "rf":
        rf = get_rf(artefact_paths["rf"])
        p_train = rf.p_train
        per_t = rf_posteriors_cached(data_dir, graph_cache_path,
                                     artefact_paths)
        row_mask = per_t[timestep]["node_id"] == node_id
        if not row_mask.any():
            raise ValueError(f"No RF prediction for node {node_id} at "
                             f"t={timestep}")
        p_raw = per_t[timestep]["p"][row_mask][0]
        embedding = None
    else:
        raise ValueError(f"Unknown model: {model_name}")

    q_y, history = compute_correction_for_timestep(
        per_t, p_train, prior_mode, timestep,
        alpha=alpha, beta=beta, em_max_iter=em_max_iter,
    )
    p_corrected = correct_single_posterior(p_raw, q_y, p_train)

    return Prediction(
        node_id=node_id, timestep=timestep,
        illicit_prob_raw=float(p_raw[LABEL_ILLICIT]),
        illicit_prob=float(p_corrected[LABEL_ILLICIT]),
        label=label, model=model_name, prior_mode=prior_mode,
        q_train_illicit=float(p_train[LABEL_ILLICIT]),
        q_t_illicit=float(q_y[LABEL_ILLICIT]),
        embedding=embedding,
        history=history,
    )


def compute_lead_time(*args, **kwargs) -> Optional[int]:
    return None
