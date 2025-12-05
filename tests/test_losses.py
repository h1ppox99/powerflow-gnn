"""Unit tests for regression and physics losses."""

from __future__ import annotations

import torch

from src.losses import (
    circular_rmse,
    kcl_quadratic_penalty,
    opf_regression_loss,
    rmse,
)
from src.training.train import _compute_loss


def test_rmse_with_mask() -> None:
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.zeros_like(pred)
    mask = torch.tensor([[True, False], [False, True]])
    loss = rmse(pred, target, mask)
    expected = torch.sqrt(torch.mean(torch.tensor([1.0**2, 4.0**2])))
    torch.testing.assert_close(loss, expected)


def test_circular_rmse_wrap_invariance() -> None:
    theta_true = torch.tensor([[0.0], [torch.pi]])
    theta_pred = torch.tensor([[2 * torch.pi], [-torch.pi]])
    loss = circular_rmse(theta_pred, theta_true)
    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_opf_regression_loss_components() -> None:
    pred = torch.tensor(
        [
            [1.0, 0.5, 1.02, 0.1],
            [0.9, 0.4, 0.98, -0.2],
        ]
    )
    target = torch.ones_like(pred)
    total, components = opf_regression_loss(pred, target)
    assert set(components) == {"p_active", "q_reactive", "voltage", "angle", "total"}
    torch.testing.assert_close(total, components["total"])


def test_kcl_quadratic_penalty_zero_for_balanced_flow() -> None:
    # Single directed edge 0 -> 1 with matching injections
    edge_index = torch.tensor([[0], [1]])
    edge_flow = torch.tensor([[2.0, 1.5]])
    net_injection = torch.tensor([[2.0, 1.5], [-2.0, -1.5]])
    penalty = kcl_quadratic_penalty(net_injection, edge_index, edge_flow)
    torch.testing.assert_close(penalty, torch.tensor(0.0))


def test_kcl_quadratic_penalty_positive_when_unbalanced() -> None:
    edge_index = torch.tensor([[0], [1]])
    edge_flow = torch.tensor([[1.0, 0.5]])
    net_injection = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    penalty = kcl_quadratic_penalty(net_injection, edge_index, edge_flow, reduction="sum")
    torch.testing.assert_close(penalty, torch.tensor(1.0**2 + 0.5**2 + 1.0**2 + 0.5**2))


def test_physics_enhanced_loss_zero_when_balanced() -> None:
    # Edge 0 -> 1 carries the exact predicted injections, so KCL penalty should vanish.
    edge_index = torch.tensor([[0], [1]])
    edge_attr = torch.tensor([[1.0, 1.0]])
    pred = torch.tensor(
        [
            [1.0, 0.5, 1.5, 1.0],   # node 0
            [-1.0, -0.5, 1.0, 0.0], # node 1
        ]
    )
    target = pred.clone()
    mask = torch.ones((2, 4), dtype=torch.bool)

    class DummyBatch:
        def __init__(self) -> None:
            self.edge_index = edge_index
            self.edge_attr = edge_attr
            self.physics_weight = 1.0

    loss = _compute_loss(pred, target, mask, "physics_enhanced", batch=DummyBatch())
    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_physics_enhanced_loss_penalizes_imbalance() -> None:
    # No angle/voltage error, but net injections are not balanced by flows.
    edge_index = torch.tensor([[0], [1]])
    pred = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],  # node 0 injects active power
            [-1.0, 0.0, 1.0, 0.0], # node 1 draws active power
        ]
    )
    target = pred.clone()
    mask = torch.ones((2, 4), dtype=torch.bool)

    class DummyBatch:
        def __init__(self) -> None:
            self.edge_index = edge_index
            self.edge_attr = None
            self.physics_weight = 1.0

    loss = _compute_loss(pred, target, mask, "physics_enhanced", batch=DummyBatch())
    # KCL term: imbalance per node = +/-1, squared and averaged => mean 1.0.
    torch.testing.assert_close(loss, torch.tensor(1.0))
