from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph

from app.data.loader import LABEL_ILLICIT
from app.data.snapshots import Snapshot
from app.models.gcn import StaticGCN
from app.models.gcn_gru import GcnGruHybrid


@dataclass
class GruSaliency:
    timesteps: List[int]
    saliency: np.ndarray


def gru_saliency(
    model: GcnGruHybrid,
    sequence: torch.Tensor,
    target_class: int = LABEL_ILLICIT,
) -> GruSaliency:
    seq = sequence.clone().detach().requires_grad_(True)
    model.eval()
    logits = model(seq)
    target = logits[0, target_class]
    target.backward()
    grads = seq.grad.detach().abs().squeeze(0)
    per_t = grads.sum(dim=-1).cpu().numpy()
    return GruSaliency(
        timesteps=list(range(1, per_t.shape[0] + 1)),
        saliency=per_t,
    )


def gcn_layer_embeddings(
    model: StaticGCN,
    snap: Snapshot,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        h1 = F.relu(model.conv1(snap.x, snap.edge_index))
        h2 = model.conv2(h1, snap.edge_index)
    return h1, h2


def neighbour_saliency(
    model: StaticGCN,
    snap: Snapshot,
    local_idx: int,
    num_hops: int = 2,
    target_class: int = LABEL_ILLICIT,
) -> dict:
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx=local_idx,
        num_hops=num_hops,
        edge_index=snap.edge_index,
        relabel_nodes=True,
        num_nodes=snap.x.size(0),
    )

    sub_x = snap.x[subset].clone().detach().requires_grad_(True)
    model.eval()
    logits = model(sub_x, sub_edge_index)
    target = logits[mapping.item(), target_class]
    target.backward()
    importance = sub_x.grad.detach().abs().sum(dim=-1).cpu().numpy()

    rng = importance.max() - importance.min()
    if rng > 0:
        importance = (importance - importance.min()) / rng

    return {
        "local_nodes": subset.cpu().numpy(),
        "edge_index": sub_edge_index.cpu().numpy(),
        "importance": importance,
        "target_local_idx": int(mapping.item()),
    }
