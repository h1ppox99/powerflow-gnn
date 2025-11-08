"""Unit tests for models under src/models."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from src.models.graphsage_pi import GraphSAGE_PI
from src.models.pna_pi import PNA_PI, compute_degree_histogram


def test_graphsage_pi_forward_shapes() -> None:
    """GraphSAGE_PI should return per-node predictions with the expected shape."""

    num_nodes, in_dim, out_dim = 5, 3, 4
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    data = Data(x=torch.randn(num_nodes, in_dim), edge_index=edge_index)

    model = GraphSAGE_PI(in_dim=in_dim, hidden_dim=16, out_dim=out_dim, num_layers=3, dropout=0.0)
    output = model(data)

    assert output.shape == (num_nodes, out_dim)
    output.sum().backward()


def test_pna_pi_forward_shapes() -> None:
    """PNA_PI should respect node-level output shape and backpropagate."""

    num_nodes, in_dim, out_dim, edge_dim = 6, 3, 4, 2
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]],
        dtype=torch.long,
    )
    edge_attr = torch.randn(edge_index.size(1), edge_dim)
    sample = Data(x=torch.randn(num_nodes, in_dim), edge_index=edge_index, edge_attr=edge_attr)

    deg_hist = compute_degree_histogram([sample])
    model = PNA_PI(
        in_dim=in_dim,
        hidden_dim=16,
        out_dim=out_dim,
        num_layers=3,
        dropout=0.0,
        deg_histogram=deg_hist,
        edge_dim=edge_dim,
    )

    output = model(sample)
    assert output.shape == (num_nodes, out_dim)
    output.sum().backward()
