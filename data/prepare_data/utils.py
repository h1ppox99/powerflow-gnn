"""Utility helpers reused by the PowerGraph dataset."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.utils import to_scipy_sparse_matrix


def padded_datalist(data_list, adj_list, max_num_nodes):
    """Pad a list of ``Data`` objects to share the same node count."""
    for i, data in enumerate(data_list):
        data.adj_padded = padding_graphs(adj_list[i], max_num_nodes)
        data.x_padded = padding_features(data.x, max_num_nodes)
    return data_list


def padding_graphs(adj, max_num_nodes):
    num_nodes = adj.shape[0]
    adj_padded = np.eye(max_num_nodes)
    adj_padded[:num_nodes, :num_nodes] = adj.cpu()
    return torch.tensor(adj_padded, dtype=torch.long)


def padding_features(features, max_num_nodes):
    feat_dim = features.shape[1]
    num_nodes = features.shape[0]
    features_padded = np.zeros((max_num_nodes, feat_dim))
    features_padded[:num_nodes] = features.cpu()
    return torch.tensor(features_padded, dtype=torch.float)


def from_edge_index_to_adj(edge_index, edge_weight, max_n):
    """Convert an edge index representation to a dense adjacency matrix."""
    adj = to_scipy_sparse_matrix(edge_index, edge_attr=edge_weight).toarray()
    if len(adj) < max_n:
        adj = np.pad(adj, (0, max_n - len(adj)), mode="constant")
    return torch.FloatTensor(adj)
