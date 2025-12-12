"""Physics-informed penalties enforcing Kirchhoff's laws."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor
from torch_scatter import scatter_add


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



def physics_loss(pred, edge_index, edge_attr, maxs_y):
    """
    Calculates the power mismatch based on Kirchhoff's laws (AC Power Flow).
    
    Args:
        pred: Model predictions [N, 4] -> [Pg, Qg, V, Theta] (Normalized)
        edge_index: Graph topology [2, E]
        edge_attr: Edge attributes [E, 2] -> [Conductance G, Susceptance B] (Unnormalized!)
        maxs_y: Normalization factors for Y [1, 4]
        
    Returns:
        Scalar Tensor representing the average physical error (MSE of the mismatch).
    """
    # 1. Denormalization
    P_pred = pred[:, 0] * maxs_y[0]
    Q_pred = pred[:, 1] * maxs_y[1]
    V_pred = pred[:, 2] * maxs_y[2]
    theta_pred = pred[:, 3] * maxs_y[3]

    src, dst = edge_index
    G = edge_attr[:, 0]
    B = edge_attr[:, 1]

    # --- CALCULATION OF BRANCH FLOWS ---
    
    # Pre-calculations
    delta_theta = theta_pred[src] - theta_pred[dst]
    cos_t = torch.cos(delta_theta)
    sin_t = torch.sin(delta_theta)
    
    # Quadratic term (V_i^2) for the source node
    # Represents the power related to the voltage at the node itself
    vv_self = V_pred[src] ** 2 
    
    # Cross term (V_i * V_j)
    vv_cross = V_pred[src] * V_pred[dst]

    # Full Branch Flow Equations
    # P_ij = G * V_i^2 - V_i*V_j * (G*cos + B*sin)
    p_flow = (G * vv_self) - vv_cross * (G * cos_t + B * sin_t)
    
    # Q_ij = -B * V_i^2 - V_i*V_j * (G*sin - B*cos)
    # (Note: The sign of B depends on the dataset convention; B is often negative for inductive lines)
    q_flow = (-B * vv_self) - vv_cross * (G * sin_t - B * cos_t)

    # ------------------------------

    # 5. Aggregation (Sum of outgoing flows)
    # Aggregates flows from edges back to source nodes
    P_out_lines = scatter_add(p_flow, src, dim=0, dim_size=pred.size(0))
    Q_out_lines = scatter_add(q_flow, src, dim=0, dim_size=pred.size(0))

    # 6. Mismatch: Injection - Outflow = 0
    # P_pred is the NET injection (Generation - Load). 
    # Therefore, P_pred must equal P_out_lines (power flowing into the grid).
    diff_P = P_pred - P_out_lines
    diff_Q = Q_pred - Q_out_lines

    # Calculate Mean Squared Error of the physical violation
    loss_phy = torch.mean(diff_P**2 + diff_Q**2)
    return loss_phy

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
