"""Baseline TransformerConv stack for OPF regression (edge-aware)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv


class TransformerBaseline(nn.Module):
    """3-layer TransformerConv stack with PReLU activations."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        out_dim: int = 4,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.0,
        edge_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()

        for i in range(num_layers):
            in_ch = dims[i]
            out_ch = hidden_dim if i < num_layers - 1 else out_dim
            concat = i < num_layers - 1
            conv = TransformerConv(
                in_ch,
                out_ch // heads if concat else out_ch,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim,
                concat=concat,
            )
            self.convs.append(conv)
            self.acts.append(nn.PReLU())

        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:  # pragma: no cover - simple wrapper
        x, edge_index, edge_attr = data.x, data.edge_index, getattr(data, "edge_attr", None)
        for conv, act in zip(self.convs[:-1], self.acts[:-1]):
            x = conv(x, edge_index, edge_attr)
            x = act(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index, edge_attr)
        return x
