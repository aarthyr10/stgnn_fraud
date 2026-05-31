from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

import torch
from torch_geometric.data import Data

from app.data.loader import NUM_TIMESTEPS


@dataclass(frozen=True)
class Snapshot:
    t: int
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    global_idx: torch.Tensor


def build_snapshots(data: Data) -> List[Snapshot]:
    snapshots: List[Snapshot] = []

    masks: Dict[int, torch.Tensor] = {}
    global_to_local: Dict[int, torch.Tensor] = {}
    for t in range(1, NUM_TIMESTEPS + 1):
        mask = (data.t == t)
        masks[t] = mask

        lookup = torch.full((data.num_nodes,), -1, dtype=torch.long)
        lookup[mask] = torch.arange(int(mask.sum()))
        global_to_local[t] = lookup

    src, dst = data.edge_index[0], data.edge_index[1]
    edge_t = data.t[src]

    for t in range(1, NUM_TIMESTEPS + 1):
        mask = masks[t]
        lookup = global_to_local[t]
        global_idx = mask.nonzero(as_tuple=False).squeeze(1)

        edge_mask = (edge_t == t)
        local_src = lookup[src[edge_mask]]
        local_dst = lookup[dst[edge_mask]]
        local_edge_index = torch.stack([local_src, local_dst], dim=0)

        snapshots.append(Snapshot(
            t=t, x=data.x[mask], edge_index=local_edge_index,
            y=data.y[mask], global_idx=global_idx,
        ))

    return snapshots


@lru_cache(maxsize=1)
def time_split(train_end: int = 34, val_end: int = 34) -> tuple[range, range, range]:
    return (
        range(1, train_end + 1),
        range(train_end + 1, val_end + 1),
        range(val_end + 1, NUM_TIMESTEPS + 1),
    )
