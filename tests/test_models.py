from __future__ import annotations

import torch

from app.models.gcn import StaticGCN
from app.models.gcn_gru import GcnGruHybrid
from app.models.tgat import TGAT


def _toy_graph(n=20, in_dim=166):
    x = torch.randn(n, in_dim)

    src = torch.arange(n)
    dst = (torch.arange(n) + 1) % n
    edge_index = torch.stack([src, dst], dim=0)
    return x, edge_index


def test_gcn_forward_shape():
    x, ei = _toy_graph()
    model = StaticGCN()
    out = model(x, ei)
    assert out.shape == (20, 2)


def test_gcn_encode_shape():
    x, ei = _toy_graph()
    model = StaticGCN(embed_dim=32)
    h = model.encode(x, ei)
    assert h.shape == (20, 32)


def test_tgat_forward_shape():
    x, ei = _toy_graph()
    t = torch.full((20,), 5, dtype=torch.long)
    model = TGAT()
    out = model(x, ei, t)
    assert out.shape == (20, 2)


def test_hybrid_freeze_blocks_gradients():
    model = GcnGruHybrid(freeze_gcn=True)

    seq = torch.randn(1, 5, 64, requires_grad=True)
    out = model(seq).sum()
    out.backward()
    for name, p in model.gcn.named_parameters():
        assert p.grad is None, f"GCN param {name} got a gradient"


def test_hybrid_forward_shape():
    model = GcnGruHybrid()
    seq = torch.randn(4, 10, 64)
    out = model(seq)
    assert out.shape == (4, 2)
