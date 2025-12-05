"""TransformerConv stack augmented with a global virtual node for OPF regression."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool


class TransformerConvVN(nn.Module):
    """TransformerConv with a virtual node that pools/broadcasts per graph each layer."""

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

        self.hidden_dim = hidden_dim
        dims = [hidden_dim] * num_layers
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.vn_pre = nn.ModuleList()
        self.vn_post = nn.ModuleList()

        for i in range(num_layers):
            in_ch = dims[i]
            out_ch = hidden_dim if i < num_layers - 1 else out_dim
            concat = i < num_layers - 1
            self.convs.append(
                TransformerConv(
                    in_ch,
                    out_ch // heads if concat else out_ch,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    concat=concat,
                )
            )
            self.acts.append(nn.PReLU())
            self.vn_pre.append(nn.Linear(in_ch, hidden_dim))
            self.vn_post.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, in_ch)))

        self.vn_emb = nn.Parameter(torch.zeros(hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:  # pragma: no cover - simple wrapper
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Project raw features to the hidden dimension so VN + residuals stay aligned.
        x = self.input_proj(x)

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        vn_state = self.vn_emb.unsqueeze(0).expand(num_graphs, -1)  # [G, hidden_dim]

        for idx, (conv, act, pre, post) in enumerate(zip(self.convs, self.acts, self.vn_pre, self.vn_post)):
            # Read: pool node features per graph.
            pooled = global_mean_pool(x, batch)  # [G, in_ch]

            # Update global: combine pooled features with prior VN state through an MLP.
            vn_hidden = pre(pooled) + vn_state  # project pooled to hidden_dim and add previous VN
            vn_state = torch.relu(vn_hidden)

            # Write: broadcast the updated virtual node to every node in its graph.
            vn_broadcast = post(vn_state)[batch]  # map VN back to node feature dimension
            x_enriched = x + vn_broadcast

            # Local + global update via TransformerConv, with residual when dimensions match.
            out = conv(x_enriched, edge_index, edge_attr)
            if idx < len(self.convs) - 1:
                out = act(out)
                out = self.dropout(out)
                x = x_enriched + out
            else:
                x = out
        return x


__all__ = ["TransformerConvVN"]
