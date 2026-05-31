from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from app.data.snapshots import Snapshot


@dataclass
class FeatureScaler:
    mean: torch.Tensor
    std: torch.Tensor

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std.clamp(min=1e-6)


def fit_scaler(snapshots: List[Snapshot], train_timesteps: range) -> FeatureScaler:
    train_x = torch.cat([s.x for s in snapshots if s.t in train_timesteps], dim=0)
    return FeatureScaler(mean=train_x.mean(0), std=train_x.std(0))


def apply_scaler(snapshots: List[Snapshot], scaler: FeatureScaler) -> List[Snapshot]:
    return [
        Snapshot(t=s.t, x=scaler.transform(s.x), edge_index=s.edge_index,
                 y=s.y, global_idx=s.global_idx)
        for s in snapshots
    ]


def add_feature_noise(
    x: torch.Tensor,
    sigma: float = 0.05,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if sigma <= 0:
        return x
    noise = torch.randn(x.shape, generator=generator) * sigma
    return x + noise


def drop_edges(
    edge_index: torch.Tensor,
    drop_rate: float = 0.1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if drop_rate <= 0:
        return edge_index
    e = edge_index.size(1)
    keep = torch.rand(e, generator=generator) > drop_rate
    return edge_index[:, keep]
