"""PNA-based physics-informed model compatible with the PowerGrid dataset."""

from __future__ import annotations

from typing import Iterable, Literal, Sequence

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import PNAConv
from torch_geometric.utils import degree

_Activation = Literal["relu", "elu"]


def compute_degree_histogram(dataset: Sequence[Data]) -> torch.Tensor:
    """Return degree histogram required by PNAConv (counts per degree)."""

    hist = torch.zeros(1, dtype=torch.long)
    for sample in dataset:
        edge_index = sample.edge_index.to("cpu")
        deg = degree(edge_index[1], num_nodes=sample.num_nodes, dtype=torch.long)
        max_degree = int(deg.max().item())
        if max_degree + 1 > hist.numel():
            hist = torch.nn.functional.pad(hist, (0, max_degree + 1 - hist.numel()))
        hist += torch.bincount(deg, minlength=hist.numel())
    return hist if hist.sum() > 0 else torch.ones(1, dtype=torch.long)


class PNA_PI(nn.Module):
    """Principal Neighbourhood Aggregation model for node-level OPF regression."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        aggrs: Iterable[str] = ("mean", "max", "min", "std"),
        scalers: Iterable[str] = ("identity", "amplification", "attenuation"),
        deg_histogram: torch.Tensor | None = None,
        edge_dim: int | None = None,
        towers: int = 1,
        pre_layers: int = 1,
        post_layers: int = 1,
        activation: _Activation = "relu",
        use_layer_norm: bool = True,
    ) -> None:
        if num_layers < 2:
            raise ValueError("PNA_PI expects at least two layers.")
        if deg_histogram is None:
            raise ValueError("PNA_PI requires a precomputed degree histogram (`deg_histogram`).")
        if edge_dim is None:
            raise ValueError("PNA_PI requires `edge_dim` matching `data.edge_attr.size(-1)`.")
        super().__init__()

        deg_hist = deg_histogram.to(torch.float)
        dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        self.convs = nn.ModuleList(
            PNAConv(
                dims[i],
                dims[i + 1],
                aggregators=tuple(aggrs),
                scalers=tuple(scalers),
                deg=deg_hist,
                edge_dim=edge_dim,
                towers=towers,
                pre_layers=pre_layers,
                post_layers=post_layers,
            )
            for i in range(len(dims) - 1)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(dims[i + 1]) if use_layer_norm else nn.Identity()
            for i in range(len(dims) - 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.head = nn.Linear(dims[-1], out_dim)

        self.register_buffer("angle_scale", torch.tensor(1.0))
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
        """Return normalized node-level predictions adhering to `data.y` layout."""

        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is None:
            raise ValueError("PNA_PI expects `edge_attr` in each Data sample.")

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = self._activate(x)
            x = self.dropout(x)
        return self.head(x)

    def _activate(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return torch.relu(tensor)
        if self.activation == "elu":
            return torch.elu(tensor)
        raise ValueError(f"Unsupported activation {self.activation!r}.")
