"""Data-driven regression losses used by the PowerGraph models."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def _apply_mask(pred: Tensor, target: Tensor, mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
    if mask is None:
        return pred, target
    return pred[mask], target[mask]


def rmse(pred: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Root mean squared error supporting optional masks."""

    pred_sel, target_sel = _apply_mask(pred, target, mask)
    return torch.sqrt(F.mse_loss(pred_sel, target_sel))


def circular_rmse(
    theta_pred: Tensor,
    theta_target: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """RMSE on phase angles expressed in radians, invariant to 2π wrapping."""

    pred_sel, target_sel = _apply_mask(theta_pred, theta_target, mask)
    diff = torch.atan2(torch.sin(pred_sel - target_sel), torch.cos(pred_sel - target_sel))
    return torch.sqrt(torch.mean(diff**2))


def opf_regression_components(
    pred: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    *,
    angle_index: int = 3,
) -> Dict[str, Tensor]:
    """Return per-variable losses matching the PowerGrid node-level target layout.

    The ``data.prepare_data.PowerGrid`` dataset stores node-level targets in
    ``data.y`` with columns ``[P_G, Q_G, |V|, theta]``. Each ``idx`` in the loop
    below therefore corresponds to:

    - ``idx=0`` → ``"p_active"`` (active power injections P_G)
    - ``idx=1`` → ``"q_reactive"`` (reactive power injections Q_G)
    - ``idx=2`` → ``"voltage"`` (voltage magnitude |V|)
    - ``idx=3`` → ``"angle"`` (voltage phase θ, treated via ``circular_rmse``)

    The same ordering is consumed by the GNN models in ``src/models`` because they
    output tensors shaped exactly like ``data.y``. Masks—if provided—must share
    the ``(num_nodes, 4)`` layout emitted by the dataset, ensuring the loss only
    considers valid nodes/targets.
    """

    components = {}
    for idx, key in enumerate(("p_active", "q_reactive", "voltage", "angle")):
        col_mask = mask[..., idx] if mask is not None else None
        if idx == angle_index:
            components[key] = circular_rmse(pred[..., idx], target[..., idx], col_mask)
        else:
            components[key] = rmse(pred[..., idx], target[..., idx], col_mask)
    return components


def opf_regression_loss(
    pred: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    *,
    angle_index: int = 3,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Sum of RMSE terms for [P_G, Q_G, |V|, θ] along with individual metrics."""

    components = opf_regression_components(pred, target, mask, angle_index=angle_index)
    total = sum(components.values())
    components["total"] = total
    return total, components
