"""Standard supervised training loop with scientific logging and metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.losses.physical_loss import kcl_quadratic_penalty, nodal_imbalance, physics_loss
from src.visualization.visualize_losses import plot_training_curves


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    if pred.numel() == 0:
        return torch.tensor(0.0, device=target.device)
    return F.mse_loss(pred, target)


def _masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor], beta: float) -> torch.Tensor:
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    if pred.numel() == 0:
        return torch.tensor(0.0, device=target.device)
    return F.smooth_l1_loss(pred, target, beta=beta)


def _compute_loss(pred, target, mask, loss_type: str, huber_beta: float = 1.0, batch=None, *, physics_weight: float = 1.0) -> torch.Tensor:
    """
    Loss over normalized outputs [P, Q, V, theta] with optional node mask.
    Supports: mse (default) or huber.
    """
    if loss_type == "huber":
        return _masked_huber(pred, target, mask, beta=huber_beta)
    if loss_type == "mse":
        return _masked_mse(pred, target, mask)
    elif loss_type == "physics_mse" :
        if batch is None:
            raise ValueError("batch must be provided for physics_mse loss computation.")
        maxs = batch.maxs # Nécessaire pour dénormaliser si stocké dans batch
        
        phy_term = physics_loss(
            pred=pred,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr_real, # ATTENTION: Doit être les vrais G/B
            maxs_y=maxs[0] if maxs.dim() > 1 else maxs # Gestion du batching de maxs
        )
        return _masked_mse(pred, target, mask) + physics_weight * phy_term
    elif loss_type == "physics_huber":
        if batch is None:
            raise ValueError("batch must be provided for physics_huber loss")
        maxs = batch.maxs # Nécessaire pour dénormaliser si stocké dans batch
        
        phy_term = physics_loss(
            pred=pred,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr_real, # ATTENTION: Doit être les vrais G/B
            maxs_y=maxs[0] if maxs.dim() > 1 else maxs # Gestion du batching de maxs
        )
        return _masked_huber(pred, target, mask, beta=huber_beta) + physics_weight * phy_term

    raise ValueError(f"Unsupported loss_type {loss_type!r}; use 'mse' or 'huber' or 'physics_mse' or 'physics_huber'.")


class MetricTracker:
    """Accumulate per-component MSE/RMSE and KCL residual."""

    def __init__(self, loss_type: str = "mse", huber_beta: float = 1.0, physics_weight: float = 1.0) -> None:
        self.loss_type = loss_type
        self.huber_beta = huber_beta
        self.physics_weight = physics_weight
        self.reset()

    def reset(self) -> None:
        self.sse = {k: 0.0 for k in ("p", "q", "v", "theta")}
        self.count = {k: 0.0 for k in ("p", "q", "v", "theta")}
        self.loss_sum = 0.0
        self.nodes = 0
        self.kcl_sum = 0.0
        self.kcl_batches = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor], batch, *, physics_weight: Optional[float] = None) -> None:
        for (k, idx) in (("p", 0), ("q", 1), ("v", 2)):
            diff = pred[:, idx] - target[:, idx]
            if mask is not None:
                diff = diff[mask[:, idx]]
            if diff.numel() == 0:
                continue
            self.sse[k] += torch.sum(diff**2).item()
            self.count[k] += diff.numel()

        diff_theta = torch.atan2(
            torch.sin(pred[:, 3] - target[:, 3]),
            torch.cos(pred[:, 3] - target[:, 3]),
        )
        if mask is not None:
            diff_theta = diff_theta[mask[:, 3]]
        if diff_theta.numel() > 0:
            self.sse["theta"] += torch.sum(diff_theta**2).item()
            self.count["theta"] += diff_theta.numel()

        weight = self.physics_weight if physics_weight is None else physics_weight
        loss = _compute_loss(pred, target, mask, self.loss_type, batch=batch, physics_weight=weight, huber_beta=self.huber_beta)
        self.loss_sum += loss.item() * pred.size(0)
        self.nodes += pred.size(0)

        with torch.no_grad():
            edge_flows = _estimate_edge_flows(pred, batch)
            net_injection = torch.stack([pred[:, 0], pred[:, 1]], dim=-1)
            residual = nodal_imbalance(net_injection, batch.edge_index, edge_flows)
            self.kcl_sum += residual.abs().mean().item()
            self.kcl_batches += 1

    def compute(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key in ("p", "q", "v", "theta"):
            if self.count[key] > 0:
                mse = self.sse[key] / self.count[key]
                out[f"mse_{key}"] = mse
                out[f"rmse_{key}"] = mse**0.5
            else:
                out[f"mse_{key}"] = float("nan")
                out[f"rmse_{key}"] = float("nan")
        # Backward-compatibility keys
        out["rmse_p_active"] = out["rmse_p"]
        out["rmse_q_reactive"] = out["rmse_q"]
        out["rmse_voltage"] = out["rmse_v"]

        num_count = self.count["p"] + self.count["q"] + self.count["v"]
        num_sse = self.sse["p"] + self.sse["q"] + self.sse["v"]
        out["rmse_num"] = (num_sse / num_count) ** 0.5 if num_count > 0 else float("nan")
        out["rmse_theta"] = (self.sse["theta"] / self.count["theta"]) ** 0.5 if self.count["theta"] > 0 else float("nan")

        total_nodes = max(self.nodes, 1)
        out["loss"] = self.loss_sum / total_nodes
        out["mean_kcl_residual"] = self.kcl_sum / max(self.kcl_batches, 1)
        return out


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    *,
    debug: bool = False,
    debug_batches: int = 1,
    loss_type: str = "mse",
    huber_beta: float = 1.0,
    physics_weight: float = 1.0
):
    model.train()
    total = 0.0
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        batch = batch.to(device)
        mask = getattr(batch, "mask", None)
        y_hat = model(batch)
        if debug and step < debug_batches:
            _debug_report("train", y_hat, batch.y, mask)
        loss = _compute_loss(y_hat, batch.y, mask, loss_type, huber_beta=huber_beta, batch=batch, physics_weight=physics_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * batch.num_nodes
    return total / sum(b.num_nodes for b in loader.dataset)

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    *,
    debug: bool = False,
    debug_batches: int = 1,
    kcl_tolerance: float = 1e-2,
    loss_type: str = "mse",
    huber_beta: float = 1.0,
    return_loss: bool = False,
    physics_weight: float = 1.0,
):
    model.eval()
    tracker = MetricTracker(loss_type=loss_type, huber_beta=huber_beta, physics_weight=physics_weight)
    for step, batch in enumerate(tqdm(loader, desc="eval", leave=False)):
        batch = batch.to(device)
        y_hat = model(batch)
        mask = getattr(batch, "mask", None)
        if debug and step < debug_batches:
            _debug_report("eval", y_hat, batch.y, mask)
        tracker.update(y_hat, batch.y, mask, batch, physics_weight=physics_weight)
    return tracker.compute()

def fit(model, dataset, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logging_cfg = cfg.get("logging", {})
    debug_cfg = cfg.get("debug", {})
    debug_enable = debug_cfg.get("enable", True)
    debug_batches = int(debug_cfg.get("batches", 1))
    kcl_tol = float(debug_cfg.get("kcl_tolerance", 1e-2))
    history: List[dict[str, float]] = []
    loss_type = cfg.get("train", {}).get("loss", "mse").lower()
    huber_beta = float(cfg.get("train", {}).get("huber_beta", 1.0))

    n = len(dataset)
    split_cfg = cfg["train"].get("split", {"train": 0.8, "val": 0.1, "test": 0.1})
    train_frac = float(split_cfg.get("train", 0.8))
    val_frac = float(split_cfg.get("val", 0.1))
    test_frac = float(split_cfg.get("test", 0.1))
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        train_frac /= total
        val_frac /= total
        test_frac /= total
    n_train = max(1, int(train_frac * n))
    n_val = max(1, int(val_frac * n))
    n_test = max(1, n - n_train - n_val)
    seed = int(cfg["train"].get("seed", 7))
    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=cfg["train"]["batch_size"])
    test_loader  = DataLoader(test_set,  batch_size=cfg["train"]["batch_size"])
    physics_weight = float(cfg["train"].get("physics_weight", 1.0))

    opt = Adam(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    sched_cfg = cfg["train"].get("scheduler")
    scheduler = None
    if sched_cfg and sched_cfg.get("type", "").lower() == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(
            opt,
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 5)),
            verbose=False,
        )

    best_val, best_state = float("inf"), None
    header_printed = False
    colw = 12
    fmt = (
        f"{{:^6}} | "  # epoch
        f"{{:^{colw}}} | "  # train_loss
        f"{{:^{colw}}} | "  # val_loss
        f"{{:^{colw}}} | "  # mse_P
        f"{{:^{colw}}} | "  # mse_Q
        f"{{:^{colw}}} | "  # mse_V
        f"{{:^{colw}}} | "  # mse_theta
        f"{{:^{colw}}}"     # KCL_mean
    )
    for epoch in range(1, cfg['train']['epochs'] + 1):
        tr = train_one_epoch(
            model,
            train_loader,
            opt,
            device,
            debug=debug_enable and epoch == 1,
            debug_batches=debug_batches,
            loss_type=loss_type,
            physics_weight=physics_weight,
            huber_beta=huber_beta,
        )
        val = evaluate(
            model,
            val_loader,
            device,
            debug=debug_enable and epoch == 1,
            debug_batches=debug_batches,
            kcl_tolerance=kcl_tol,
            loss_type=loss_type,
            huber_beta=huber_beta,
            physics_weight=physics_weight,
        )
        if scheduler is not None:
            scheduler.step(val["loss"] if isinstance(scheduler, ReduceLROnPlateau) else None)

        row = {
            "epoch": float(epoch),
            "train_loss": float(tr),
            "val_loss": float(val["loss"]),
            "val_mse_p": float(val["mse_p"]),
            "val_mse_q": float(val["mse_q"]),
            "val_mse_v": float(val["mse_v"]),
            "val_mse_theta": float(val["mse_theta"]),
            "val_rmse_num": float(val["rmse_num"]),
            "val_rmse_theta": float(val["rmse_theta"]),
            "val_mean_kcl": float(val["mean_kcl_residual"]),
        }
        history.append(row)

        if not header_printed:
            header_printed = True
            tqdm.write(
                fmt.format(
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "mse_P",
                    "mse_Q",
                    "mse_V",
                    "mse_theta",
                    "KCL_mean",
                )
            )
        tqdm.write(
            fmt.format(
                f"{epoch:03d}",
                f"{tr:.4e}",
                f"{val['loss']:.4e}",
                f"{val['mse_p']:.4e}",
                f"{val['mse_q']:.4e}",
                f"{val['mse_v']:.4e}",
                f"{val['mse_theta']:.4e}",
                f"{val['mean_kcl_residual']:.4e}",
            )
        )

        val_loss = val.get("loss", val["rmse_num"] + val["rmse_theta"])
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test = evaluate(
        model,
        test_loader,
        device,
        debug=debug_enable,
        debug_batches=debug_batches,
        kcl_tolerance=kcl_tol,
        loss_type=loss_type,
        huber_beta=huber_beta,
        return_loss=True,
        physics_weight=physics_weight,
    )
    tqdm.write(
        "[TEST] loss {loss:.4e} | rmse_num {num:.4e} | rmse_theta {theta:.4e} | "
        "mse_p {p:.4e} | mse_q {q:.4e} | mse_v {v:.4e} | mse_theta {t:.4e} | mean_kcl {kcl:.4e}".format(
            loss=test["loss"],
            num=test["rmse_num"],
            theta=test["rmse_theta"],
            p=test["mse_p"],
            q=test["mse_q"],
            v=test["mse_v"],
            t=test["mse_theta"],
            kcl=test["mean_kcl_residual"],
        )
    )
    _log_history(history, logging_cfg)
    _baseline_comparison(test)
    _plot_results(history, logging_cfg)
    return test


def _log_history(history: List[Mapping[str, float]], logging_cfg: Mapping | None) -> None:
    if not history:
        return
    out_dir = Path(logging_cfg.get("output_dir", "output")) if logging_cfg else Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "metrics.jsonl"
    csv_path = out_dir / "metrics.csv"

    with jsonl_path.open("w") as handle:
        for row in history:
            handle.write(json.dumps(row))
            handle.write("\n")

    # CSV
    keys = sorted(history[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def _plot_results(history: List[Mapping[str, float]], logging_cfg: Mapping | None) -> None:
    if not history:
        return
    out_dir = Path(logging_cfg.get("output_dir", "output")) if logging_cfg else Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(epochs, train_loss, label="train_loss")
    axs[0].plot(epochs, val_loss, label="val_loss")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss (log scale)")
    axs[0].legend()
    axs[0].grid(True, ls="--", alpha=0.5)

    for key, label in (
        ("val_mse_p", "MSE_P"),
        ("val_mse_q", "MSE_Q"),
        ("val_mse_v", "MSE_V"),
        ("val_mse_theta", "MSE_theta"),
    ):
        axs[1].plot(epochs, [h[key] for h in history], label=label)
    axs[1].set_yscale("log")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Validation MSE (log scale)")
    axs[1].legend()
    axs[1].grid(True, ls="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=200)
    plt.close(fig)


def _baseline_comparison(test_metrics: Mapping[str, float]) -> None:
    target_rows = [
        ("P/Q MSE (target ~1e-4)", test_metrics["mse_p"], test_metrics["mse_q"], "≈1e-4"),
        ("V/θ MSE (target ~1e-5)", test_metrics["mse_v"], test_metrics["mse_theta"], "≈1e-5"),
        ("Total target", test_metrics["loss"], None, "≈3.75e-5"),
    ]
    tqdm.write("\nBaseline comparison (PowerGraph IEEE118 targets):")
    for name, v1, v2, ref in target_rows:
        if v2 is None:
            tqdm.write(f"{name:30s}: {v1:.4e} (ref {ref})")
        else:
            tqdm.write(f"{name:30s}: {v1:.4e} / {v2:.4e} (ref {ref})")


@torch.no_grad()
def _debug_report(prefix: str, preds: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> None:
    preds = preds.detach().cpu()
    target = target.detach().cpu()
    zero_frac = float((preds.abs() < 1e-5).float().mean().item())
    diff = (preds - target).abs()
    msg = (
        f"[DEBUG:{prefix}] preds shape {tuple(preds.shape)} | "
        f"mean {preds.mean():.4f} | std {preds.std():.4f} | zero_frac {zero_frac:.3f} | "
        f"mean_abs_err {diff.mean():.4f}"
    )
    if mask is not None:
        valid = mask.detach().cpu().float().mean().item()
        msg += f" | mask_active {valid:.3f}"
    tqdm.write(msg)


def _estimate_edge_flows(preds: torch.Tensor, batch) -> torch.Tensor:
    theta = preds[:, 3]
    voltage = preds[:, 2]
    src, dst = batch.edge_index

    edge_attr = getattr(batch, "edge_attr", None)
    if edge_attr is not None and edge_attr.size(-1) >= 2:
        weight_p = edge_attr[:, 0]
        weight_q = edge_attr[:, 1]
    else:
        weight_p = torch.ones_like(src, dtype=theta.dtype, device=theta.device)
        weight_q = weight_p

    flow_p = weight_p * (theta[src] - theta[dst])
    flow_q = weight_q * (voltage[src] - voltage[dst])
    return torch.stack([flow_p, flow_q], dim=-1)
