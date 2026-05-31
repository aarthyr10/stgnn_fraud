from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from app.data.preprocess import (
    FeatureScaler,
    add_feature_noise,
    drop_edges,
)
from app.data.snapshots import build_snapshots


@pytest.fixture
def toy_data():
    x = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5],
        [1, 0, 3, 2, 5, 4],
    ], dtype=torch.long)
    y = torch.tensor([1, 0, 1, -1, 0, 1])
    t = torch.tensor([1, 1, 2, 2, 3, 3])
    node_id = torch.tensor([100, 101, 102, 103, 104, 105])
    return Data(x=x, edge_index=edge_index, y=y, t=t,
                node_id=node_id, num_nodes=6)


def test_snapshot_count(toy_data):
    snaps = build_snapshots(toy_data)

    nonempty = [s for s in snaps if s.x.size(0) > 0]
    assert len(nonempty) == 3


def test_snapshot_local_indices(toy_data):
    snaps = build_snapshots(toy_data)
    s1 = snaps[0]
    assert s1.x.size(0) == 2

    assert s1.edge_index.max() < 2


def test_feature_scaler():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = FeatureScaler(mean=x.mean(0), std=x.std(0))
    z = scaler.transform(x)
    assert torch.allclose(z.mean(0), torch.zeros(2), atol=1e-6)


def test_noise_zero_sigma_identity():
    x = torch.randn(10, 3)
    out = add_feature_noise(x, sigma=0.0)
    assert torch.equal(x, out)


def test_edge_dropout_zero_identity():
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]])
    out = drop_edges(ei, drop_rate=0.0)
    assert torch.equal(ei, out)


def test_edge_dropout_roughly_correct_fraction():
    ei = torch.arange(2000).reshape(2, 1000)
    g = torch.Generator().manual_seed(0)
    out = drop_edges(ei, drop_rate=0.3, generator=g)

    kept_frac = out.size(1) / 1000
    assert 0.65 < kept_frac < 0.75
