# tests/test_training.py
import math
import types
import sys
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# ---------------------------------------------------------------------
# Ensure the loss imports used by the training script exist.
# If your project already provides these, this block is harmless.
# ---------------------------------------------------------------------
if "src.losses.regression_loss" not in sys.modules:
    m = types.ModuleType("src.losses.regression_loss")
    def rmse(pred, target):
        return torch.sqrt(torch.mean((pred - target) ** 2) + 1e-12)
    def circular_rmse(pred_theta, true_theta):
        # angles in radians, wrap into [-pi, pi]
        diff = (pred_theta - true_theta + math.pi) % (2 * math.pi) - math.pi
        return torch.sqrt(torch.mean(diff ** 2) + 1e-12)
    m.rmse = rmse
    m.circular_rmse = circular_rmse
    sys.modules["src"] = types.ModuleType("src")
    sys.modules["src.losses"] = types.ModuleType("src.losses")
    sys.modules["src.losses.regression_loss"] = m

# Import the functions under test from your training script.
# Adjust the import path to match your repo layout.
from src.training.train import train_one_epoch, evaluate, fit


# -----------------------------
# Helpers: tiny synthetic setup
# -----------------------------
class TinyNodeModel(nn.Module):
    """A minimal node-level model with forward(Data)->(N,4)."""
    def __init__(self, in_dim=5, hidden=16, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        return self.net(data.x)


def make_synthetic_dataset(
    n_graphs=24, min_nodes=6, max_nodes=10, in_dim=5, seed=7
):
    """
    Create a list of Data objects with:
      y[:, :3] = W * x + b (noisy)
      y[:, 3]  = angle in radians (wrapped)
    """
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(in_dim, 3, generator=g) * 0.5
    b = torch.randn(3, generator=g) * 0.1

    dataset = []
    for _ in range(n_graphs):
        n = int(torch.randint(min_nodes, max_nodes + 1, (1,), generator=g))
        x = torch.randn(n, in_dim, generator=g)

        y_num = x @ W + b  # (n, 3)
        # Build a smooth angle from a linear projection, then wrap to [-pi, pi]
        theta_lin = (x @ torch.randn(in_dim, 1, generator=g)).squeeze(-1) * 0.1
        theta = (theta_lin + math.pi) % (2 * math.pi) - math.pi  # in radians

        y = torch.cat([y_num, theta.unsqueeze(-1)], dim=1)  # (n, 4)

        # A simple chain graph (edge_index) to keep PyG happy
        row = torch.arange(0, n - 1, dtype=torch.long)
        col = torch.arange(1, n, dtype=torch.long)
        edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)

        data = Data(x=x, y=y, edge_index=edge_index)
        dataset.append(data)

    return dataset


# -----------------------------
# Tests
# -----------------------------
def test_train_one_epoch_updates_params():
    torch.manual_seed(0)
    dataset = make_synthetic_dataset(n_graphs=10)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = TinyNodeModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Snapshot a parameter tensor to verify it changes
    with torch.no_grad():
        before = next(model.parameters()).clone().cpu()

    loss_avg = train_one_epoch(model, loader, opt, device)

    assert isinstance(loss_avg, float)
    assert math.isfinite(loss_avg)

    with torch.no_grad():
        after = next(model.parameters()).clone().cpu()

    # Ensure at least one parameter changed
    assert not torch.allclose(before, after), "Parameters did not update during training"


def test_evaluate_returns_expected_keys_and_finite():
    torch.manual_seed(0)
    dataset = make_synthetic_dataset(n_graphs=8)
    loader = DataLoader(dataset, batch_size=4)

    model = TinyNodeModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    metrics = evaluate(model, loader, device)
    assert set(metrics.keys()) == {"rmse_num", "rmse_theta"}
    assert math.isfinite(metrics["rmse_num"])
    assert math.isfinite(metrics["rmse_theta"])


def test_fit_end_to_end_returns_test_metrics():
    torch.manual_seed(123)
    # Ensure we have non-empty train/val/test splits (80/10/10)
    dataset = make_synthetic_dataset(n_graphs=30)
    model = TinyNodeModel()

    cfg = {
        "train": {
            "batch_size": 4,
            "lr": 1e-2,            # float, not string
            "weight_decay": 0.0,
            "epochs": 3,           # keep it quick
        }
    }

    test_metrics = fit(model, dataset, cfg)

    # Sanity checks on return
    assert isinstance(test_metrics, dict)
    assert set(test_metrics.keys()) == {"rmse_num", "rmse_theta"}
    assert math.isfinite(test_metrics["rmse_num"])
    assert math.isfinite(test_metrics["rmse_theta"])

    # Reasonable bounds for this tiny synthetic problem
    assert 0.0 <= test_metrics["rmse_num"] < 5.0
    assert 0.0 <= test_metrics["rmse_theta"] < math.pi
