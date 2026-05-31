from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from app.models.gcn import StaticGCN


class GcnGruHybrid(nn.Module):

    def __init__(
        self,
        gcn_weights_path: Optional[str | Path] = None,
        embed_dim: int = 64,
        gru_hidden: int = 64,
        gru_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2,
        freeze_gcn: bool = True,
    ):
        super().__init__()

        self.gcn = StaticGCN(embed_dim=embed_dim)
        if gcn_weights_path is not None:
            state = torch.load(gcn_weights_path, map_location="cpu")

            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.gcn.load_state_dict(state)

        if freeze_gcn:
            for p in self.gcn.parameters():
                p.requires_grad_(False)

            self.gcn.eval()

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            dropout=dropout if gru_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(gru_hidden, num_classes)
        self.embed_dim = embed_dim
        self.gru_hidden = gru_hidden

    @torch.no_grad()
    def embed_snapshot(
        self, x: torch.Tensor, edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.gcn.encode(x, edge_index)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(sequence)
        return self.head(h_n[-1])

    def predict_from_snapshots(
        self, node_sequences: List[torch.Tensor],
    ) -> torch.Tensor:
        batch = torch.stack(node_sequences, dim=0)
        return self.forward(batch)
