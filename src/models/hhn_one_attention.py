"""HeGGA variant with a single global attention applied after message passing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

try:
    from performer_pytorch import SelfAttention
except ImportError:  # pragma: no cover - optional
    SelfAttention = None


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def _infer_node_types(node_type_feat: torch.Tensor) -> torch.Tensor:
    scaled = torch.round(node_type_feat * 3.0).clamp(min=1, max=3)
    return (scaled - 1).long()


@dataclass
class HHNOneAttentionConfig:
    in_dim: int
    out_dim: int
    hidden_dim: int = 256
    num_layers: int = 5
    heads: int = 4
    dropout: float = 0.0
    edge_dim: Optional[int] = None
    attention: str = "mha"  # "mha" or "performer"
    pe_dims: int = 5


class HHNOneAttention(nn.Module):
    """Message passing stack, then a single global attention and fusion."""

    def __init__(self, cfg: HHNOneAttentionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim
        attn_type = cfg.attention.lower()
        self.use_performer = attn_type == "performer" and SelfAttention is not None
        self._pe_cache: dict[str, torch.Tensor] = {}

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
        for _ in range(cfg.num_layers):
            self.edge_updates.append(_make_mlp(3 * h, h, h))
            self.node_updates.append(_make_mlp(2 * h, h, h))

        if self.use_performer:
            self.attn = SelfAttention(
                dim=h,
                heads=cfg.heads,
                dim_head=max(1, h // cfg.heads),
                nb_features=64,
                dropout=cfg.dropout,
            )
        else:
            self.attn = nn.MultiheadAttention(h, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.fusion = _make_mlp(h, h, h)
        self.dropout = nn.Dropout(cfg.dropout)

        self.decoders = nn.ModuleList([_make_mlp(h, h, cfg.out_dim) for _ in range(3)])

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

    def _global_attention(self, local: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x_padded, mask = to_dense_batch(local, batch)
        if self.use_performer:
            attn_out = self.attn(x_padded, mask=mask)
            global_out = x_padded + attn_out
        else:
            attn_out, _ = self.attn(x_padded, x_padded, x_padded, key_padding_mask=~mask)
            global_out = x_padded + attn_out
        return global_out[mask]

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

        for edge_upd, node_upd in zip(self.edge_updates, self.node_updates):
            src, dst = edge_index
            m_edge = edge_upd(torch.cat([h_nodes[src], h_nodes[dst], h_edges], dim=-1))
            h_edges = h_edges + m_edge

            m_node = torch.zeros_like(h_nodes)
            m_node.index_add_(0, dst, h_edges)
            h_node_upd = node_upd(torch.cat([h_nodes, m_node], dim=-1))
            h_nodes = h_nodes + h_node_upd

        global_flat = self._global_attention(h_nodes, batch)
        h_nodes = self.fusion(h_nodes + global_flat)
        h_nodes = self.dropout(h_nodes)

        out = h_nodes.new_zeros(h_nodes.size(0), self.decoders[0][-1].out_features)
        for t in range(len(self.decoders)):
            mask = node_types == t
            if mask.any():
                out[mask] = self.decoders[t](h_nodes[mask])
        return out


def build_model(cfg: dict, dataset) -> HHNOneAttention:
    data = dataset[0]
    model_cfg = HHNOneAttentionConfig(
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
    return HHNOneAttention(model_cfg)
