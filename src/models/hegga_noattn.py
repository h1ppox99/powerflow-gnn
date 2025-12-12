"""HeGGA variant without any global attention (pure heterogeneous MPNN backbone)."""

from __future__ import annotations

import hashlib
import torch
import torch.nn as nn
from torch_geometric.data import Data

from .hh_mpnn import HHMPNNConfig, _infer_node_types, _make_mlp


class HeGGANoAttn(nn.Module):
    """Typed encoders/decoders + edge-aware MLP updates, but no global attention."""

    def __init__(self, cfg: HHMPNNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim

        self.node_encoders = nn.ModuleList(
            [_make_mlp(cfg.in_dim + cfg.pe_dims, h, h) for _ in range(3)]
        )

        if cfg.edge_dim is not None:
            self.edge_encoder = _make_mlp(cfg.edge_dim, h, h)
        else:
            self.edge_encoder = None
            self.edge_bias = nn.Parameter(torch.zeros(h))

        self.edge_updates = nn.ModuleList()
        self.node_updates = nn.ModuleList()
        self.fusions = nn.ModuleList()
        for _ in range(cfg.num_layers):
            self.edge_updates.append(_make_mlp(3 * h, h, h))
            self.node_updates.append(_make_mlp(2 * h, h, h))
            self.fusions.append(_make_mlp(h, h, h))

        self.dropout = nn.Dropout(cfg.dropout)
        self.decoders = nn.ModuleList([_make_mlp(h, h, cfg.out_dim) for _ in range(3)])
        self._pe_cache: dict[str, torch.Tensor] = {}

    def _compute_lap_pe(self, edge_index: torch.Tensor, node_mask: torch.Tensor, k: int) -> torch.Tensor:
        device = edge_index.device
        dtype = torch.float
        idx = torch.nonzero(node_mask, as_tuple=False).flatten()
        n = idx.numel()
        if n == 0:
            return torch.zeros(0, k, device=device, dtype=dtype)

        node_id_map = -torch.ones(int(node_mask.numel()), device=device, dtype=torch.long)
        node_id_map[idx] = torch.arange(n, device=device)
        src, dst = edge_index
        edge_mask = node_mask[src] & node_mask[dst]
        src_local = node_id_map[src[edge_mask]]
        dst_local = node_id_map[dst[edge_mask]]
        if src_local.numel() == 0:
            return torch.zeros(n, k, device=device, dtype=dtype)

        A = torch.zeros((n, n), device=device, dtype=dtype)
        A[src_local, dst_local] = 1.0
        A = torch.maximum(A, A.t())
        deg = A.sum(dim=1)
        deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
        D_inv = torch.diag(deg_inv_sqrt)
        L = torch.eye(n, device=device, dtype=dtype) - D_inv @ A @ D_inv
        _, evecs = torch.linalg.eigh(L)
        evecs = evecs[:, 1 : min(k + 1, n)]
        if evecs.size(1) < k:
            pad = torch.zeros(n, k - evecs.size(1), device=device, dtype=dtype)
            evecs = torch.cat([evecs, pad], dim=1)
        return evecs[:, :k]

    def _get_pe(self, edge_index: torch.Tensor, batch: torch.Tensor, num_nodes: int, k: int) -> torch.Tensor:
        pe = torch.zeros(num_nodes, k, device=edge_index.device, dtype=torch.float)
        unique_graphs = batch.unique(sorted=True)
        for g in unique_graphs:
            mask = batch == g
            edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
            key_tensor = edge_index[:, edge_mask]
            key = hashlib.sha1(key_tensor.cpu().numpy().tobytes()).hexdigest()
            if key not in self._pe_cache:
                self._pe_cache[key] = self._compute_lap_pe(edge_index, mask, k=k)
            pe[mask] = self._pe_cache[key]
        return pe

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        node_types = _infer_node_types(x[:, 2])

        pe = getattr(data, "pe", None)
        if pe is None:
            pe = self._get_pe(edge_index, batch, num_nodes=x.size(0), k=self.cfg.pe_dims).to(x.dtype)

        h_nodes = x.new_zeros(x.size(0), self.node_encoders[0][0].out_features)
        for t in range(len(self.node_encoders)):
            mask = node_types == t
            if mask.any():
                h_nodes[mask] = self.node_encoders[t](torch.cat([x[mask], pe[mask]], dim=-1))

        if self.edge_encoder is not None and edge_attr is not None:
            h_edges = self.edge_encoder(edge_attr)
        else:
            h_edges = self.edge_bias.unsqueeze(0).expand(edge_index.size(1), -1)

        for edge_upd, node_upd, fusion in zip(self.edge_updates, self.node_updates, self.fusions):
            src, dst = edge_index
            m_edge = edge_upd(torch.cat([h_nodes[src], h_nodes[dst], h_edges], dim=-1))
            h_edges = h_edges + m_edge

            m_node = torch.zeros_like(h_nodes)
            m_node.index_add_(0, dst, h_edges)
            h_node_upd = node_upd(torch.cat([h_nodes, m_node], dim=-1))
            local = h_nodes + h_node_upd

            h_nodes = fusion(local)
            h_nodes = self.dropout(h_nodes)

        out = h_nodes.new_zeros(h_nodes.size(0), self.decoders[0][-1].out_features)
        for t in range(len(self.decoders)):
            mask = node_types == t
            if mask.any():
                out[mask] = self.decoders[t](h_nodes[mask])
        return out


def build_model(cfg: dict, dataset) -> HeGGANoAttn:
    """Factory compatible with load_model; expects cfg['model'] entries."""
    data = dataset[0]
    model_cfg = HHMPNNConfig(
        in_dim=data.x.size(-1),
        out_dim=data.y.size(-1),
        hidden_dim=cfg["model"].get("hidden", 256),
        num_layers=cfg["model"].get("num_layers", 5),
        heads=cfg["model"].get("heads", 4),
        dropout=cfg["model"].get("dropout", 0.0),
        edge_dim=data.edge_attr.size(-1) if getattr(data, "edge_attr", None) is not None else None,
        attention=cfg["model"].get("attention", "mha"),
        pe_dims=cfg["model"].get("pe_dims", 5),
    )
    return HeGGANoAttn(model_cfg)
