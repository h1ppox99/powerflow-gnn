"""Physics-informed GraphSAGE baseline aligned with the PowerGrid dataset contract."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class GraphSAGE_PI(nn.Module):
    """Original GraphSAGE architecture (Hamilton et al., 2017) for node-level OPF regression."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        aggr: Literal["mean", "max", "add"] = "mean",
        use_layer_norm: bool = True,
    ) -> None:
        if num_layers < 2:
            raise ValueError("GraphSAGE_PI expects at least two layers.")
        super().__init__()

        dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        self.convs = nn.ModuleList(
            SAGEConv(dims[i], dims[i + 1], aggr=aggr) for i in range(len(dims) - 1)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(dims[i + 1]) if use_layer_norm else nn.Identity()
            for i in range(len(dims) - 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.activation_fn = torch.relu
        self.head = nn.Linear(dims[-1], out_dim)

        # Previously the model exposed a buffer for optional angle scaling.
        # Keep a commented note here for now; uncomment if a loss or other
        # component requires it in the future.
        # self.register_buffer("angle_scale", torch.tensor(1.0))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, data: Data) -> torch.Tensor:
        """Return normalized node-level predictions matching `data.y` shape."""

        x, edge_index = data.x, data.edge_index
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = self.activation_fn(x)
            x = self.dropout(x)
        return self.head(x)
