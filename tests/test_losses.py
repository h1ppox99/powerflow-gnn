"""Unit tests for regression and physics losses."""

from __future__ import annotations

import torch

from src.losses import (
    circular_rmse,
    kcl_quadratic_penalty,
    opf_regression_loss,
    rmse,
)


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
