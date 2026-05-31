from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from app.data.loader import NUM_FEATURES


class StaticGCN(nn.Module):
    def __init__(
        self,
        in_dim: int = NUM_FEATURES,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embed_dim)
        self.dropout = dropout
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.embed_dim = embed_dim

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encode(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)
