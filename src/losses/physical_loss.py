"""Physics-informed penalties enforcing Kirchhoff's laws."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor


def incidence_matrix(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Return the (sparse) node-edge incidence matrix for a directed edge list."""

    device = edge_index.device
    num_edges = edge_index.size(1)
    row = torch.cat([edge_index[0], edge_index[1]], dim=0)
    col = torch.cat([torch.arange(num_edges, device=device)] * 2, dim=0)
    values = torch.cat(
        [torch.ones(num_edges, device=device), -torch.ones(num_edges, device=device)],
        dim=0,
    )
    indices = torch.stack([row, col])
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_edges), device=device)


def nodal_imbalance(
    net_injection: Tensor,
    edge_index: Tensor,
    edge_flows: Tensor,
) -> Tensor:
    """Compute ΔP and ΔQ imbalances: B @ flow - net_injection."""

    if edge_flows.dim() == 1:
        edge_flows = edge_flows.unsqueeze(-1)
    incidence = incidence_matrix(edge_index, net_injection.size(0))
    aggregated = torch.sparse.mm(incidence, edge_flows)
    return aggregated - net_injection


def kcl_quadratic_penalty(
    net_injection: Tensor,
    edge_index: Tensor,
    edge_flows: Tensor,
    *,
    mask: Optional[Tensor] = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> Tensor:
    """λ * Σ_i (ΔPi² + ΔQi²) style penalty from the project proposal."""

    imbalance = nodal_imbalance(net_injection, edge_index, edge_flows)
    if mask is not None:
        imbalance = imbalance[mask]
    penalties = (imbalance**2).sum(dim=-1)
    if reduction == "sum":
        return penalties.sum()
    return penalties.mean()


@torch.no_grad()
def check_kcl_residuals(
    edge_index: Tensor,
    edge_flows: Tensor,
    net_injection: Tensor,
    *,
    tol: float = 1e-3,
    verbose: bool = True,
) -> Tensor:
    """Return node-wise KCL residuals and emit a simple pass/fail summary."""

    residual = nodal_imbalance(net_injection, edge_index, edge_flows)
    max_violation = residual.abs().max().item()
    if verbose:
        print(f"[KCL] max violation = {max_violation:.3e}")
        if max_violation > tol:
            print("[KCL] WARN: Kirchhoff's current law violated at some nodes.")
        else:
            print("[KCL] PASS within tolerance.")
    return residual
