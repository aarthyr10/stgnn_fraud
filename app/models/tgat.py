from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class TimeEncoder(nn.Module):

    def __init__(self, dim: int = 16):
        super().__init__()
        freqs = 1.0 / (10000.0 ** (torch.arange(0, dim).float() / dim))
        self.freqs = nn.Parameter(freqs, requires_grad=True)
        self.phase = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().unsqueeze(-1)
        return torch.cos(t * self.freqs + self.phase)


class TGAT(nn.Module):

    def __init__(
        self,
        in_dim: int = 166,
        time_dim: int = 16,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.time_enc = TimeEncoder(time_dim)
        feat_dim = in_dim + time_dim
        self.attn1 = GATConv(feat_dim, hidden_dim, heads=num_heads,
                             dropout=dropout)
        self.attn2 = GATConv(hidden_dim * num_heads, out_dim, heads=1,
                             concat=False, dropout=dropout)
        self.head = nn.Linear(out_dim, num_classes)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        timesteps: torch.Tensor,
        return_attention: bool = False,
    ):
        t_enc = self.time_enc(timesteps)
        h = torch.cat([x, t_enc], dim=-1)

        if return_attention:
            h, (e1, a1) = self.attn1(h, edge_index,
                                     return_attention_weights=True)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h, (e2, a2) = self.attn2(h, edge_index,
                                     return_attention_weights=True)
            logits = self.head(h)
            return logits, {"layer1": (e1, a1), "layer2": (e2, a2)}

        h = F.elu(self.attn1(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.attn2(h, edge_index)
        return self.head(h)
