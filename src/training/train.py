"""Standard supervised training loop with lightweight logging helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import torch
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.losses.physical_loss import check_kcl_residuals
from src.losses.regression_loss import rmse, circular_rmse
from src.visualization.visualize_losses import plot_training_curves

def train_one_epoch(model, loader, optimizer, device, *, debug: bool = False, debug_batches: int = 1):
    model.train()
    total = 0.0
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        batch = batch.to(device)
        mask = getattr(batch, "mask", None)
        y_hat = model(batch)
        if debug and step < debug_batches:
            _debug_report("train", y_hat, batch.y, mask)
        # assume y = [P_G, Q_G, |V|, theta] in columns 0..3 (adjust later)
        loss_num = rmse(
            y_hat[:, :3],
            batch.y[:, :3],
            mask[:, :3] if mask is not None else None,
        )
        loss_ang = circular_rmse(
            y_hat[:, 3],
            batch.y[:, 3],
            mask[:, 3] if mask is not None else None,
        )
        loss = loss_num + loss_ang

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
):
    model.eval()
    rmse_totals = _init_rmse_accumulator()
    for step, batch in enumerate(tqdm(loader, desc="eval", leave=False)):
        batch = batch.to(device)
        y_hat = model(batch)
        mask = getattr(batch, "mask", None)
        if debug and step < debug_batches:
            _debug_report("eval", y_hat, batch.y, mask)
            _kcl_debug(batch, y_hat, tol=kcl_tolerance)
        _accumulate_rmse(rmse_totals, y_hat, batch.y, mask)
    return _finalize_rmse(rmse_totals)

def fit(model, dataset, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logging_cfg = cfg.get("logging", {})
    debug_cfg = cfg.get("debug", {})
    debug_enable = debug_cfg.get("enable", True)
    debug_batches = int(debug_cfg.get("batches", 1))
    kcl_tol = float(debug_cfg.get("kcl_tolerance", 1e-2))
    history: List[dict[str, float]] = []

    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(7))

    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=cfg["train"]["batch_size"])
    test_loader  = DataLoader(test_set,  batch_size=cfg["train"]["batch_size"])

    opt = Adam(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])

    best_val, best_state = float("inf"), None
    for epoch in range(1, cfg['train']['epochs'] + 1):
        tr = train_one_epoch(
            model,
            train_loader,
            opt,
            device,
            debug=debug_enable and epoch == 1,
            debug_batches=debug_batches,
        )
        val = evaluate(
            model,
            val_loader,
            device,
            debug=debug_enable and epoch == 1,
            debug_batches=debug_batches,
            kcl_tolerance=kcl_tol,
        )
        print(f"epoch {epoch:03d} | train_loss ~ {tr:.4f} | val_num {val['rmse_num']:.4f} | val_theta {val['rmse_theta']:.4f}")

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(tr),
                "val_rmse_num": float(val["rmse_num"]),
                "val_rmse_theta": float(val["rmse_theta"]),
            }
        )
        if val["rmse_num"] + val["rmse_theta"] < best_val:
            best_val = val["rmse_num"] + val["rmse_theta"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test = evaluate(
        model,
        test_loader,
        device,
        debug=debug_enable,
        debug_batches=debug_batches,
        kcl_tolerance=kcl_tol,
    )
    print(f"[TEST] num {test['rmse_num']:.4f} | theta {test['rmse_theta']:.4f}")
    print(
        "[TEST details] P {:.4f} | Q {:.4f} | |V| {:.4f}".format(
            test["rmse_p_active"],
            test["rmse_q_reactive"],
            test["rmse_voltage"],
        )
    )
    _log_history(history, logging_cfg)
    return test


def _log_history(history: List[Mapping[str, float]], logging_cfg: Mapping | None) -> None:
    if not history or not logging_cfg:
        return

    history_path = logging_cfg.get("history_path")
    if history_path:
        _write_history(history, Path(history_path))

    if logging_cfg.get("plot_losses"):
        plot_path = logging_cfg.get("loss_plot_path")
        if plot_path is None and history_path is not None:
            plot_path = str(Path(history_path).with_suffix(".png"))
        if plot_path is None:
            plot_path = "output/loss_curves.png"
        plot_training_curves(
            history,
            metrics=logging_cfg.get("plot_metrics"),
            smoothing=logging_cfg.get("plot_smoothing", 1),
            title=logging_cfg.get("plot_title"),
            save_path=plot_path,
            show=logging_cfg.get("show_plots", False),
        )


def _write_history(history: Iterable[Mapping[str, float]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in history:
            handle.write(json.dumps(row))
            handle.write("\n")


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
    print(msg)


def _init_rmse_accumulator() -> Dict[str, Dict[str, float]]:
    return {
        "sse": {key: 0.0 for key in ("p_active", "q_reactive", "voltage", "angle")},
        "count": {key: 0.0 for key in ("p_active", "q_reactive", "voltage", "angle")},
    }


def _accumulate_rmse(
    totals: Dict[str, Dict[str, float]],
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> None:
    for key, idx in (("p_active", 0), ("q_reactive", 1), ("voltage", 2)):
        diff = pred[:, idx] - target[:, idx]
        if mask is not None:
            diff = diff[mask[:, idx]]
        if diff.numel() == 0:
            continue
        totals["sse"][key] += torch.sum(diff**2).item()
        totals["count"][key] += diff.numel()

    diff_theta = torch.atan2(
        torch.sin(pred[:, 3] - target[:, 3]),
        torch.cos(pred[:, 3] - target[:, 3]),
    )
    if mask is not None:
        diff_theta = diff_theta[mask[:, 3]]
    if diff_theta.numel() > 0:
        totals["sse"]["angle"] += torch.sum(diff_theta**2).item()
        totals["count"]["angle"] += diff_theta.numel()


def _finalize_rmse(totals: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    results: Dict[str, float] = {}
    num_sse = 0.0
    num_count = 0.0
    for key in ("p_active", "q_reactive", "voltage"):
        count = totals["count"][key]
        if count > 0:
            rmse_val = (totals["sse"][key] / count) ** 0.5
            results[f"rmse_{key}"] = rmse_val
            num_sse += totals["sse"][key]
            num_count += count
        else:
            results[f"rmse_{key}"] = float("nan")

    if num_count > 0:
        results["rmse_num"] = (num_sse / num_count) ** 0.5
    else:
        results["rmse_num"] = float("nan")

    angle_count = totals["count"]["angle"]
    if angle_count > 0:
        results["rmse_theta"] = (totals["sse"]["angle"] / angle_count) ** 0.5
    else:
        results["rmse_theta"] = float("nan")
    results["rmse_angle"] = results["rmse_theta"]
    return results


@torch.no_grad()
def _kcl_debug(batch, preds, tol: float) -> None:
    edge_flows = _estimate_edge_flows(preds, batch)
    net_injection = torch.stack([preds[:, 0], preds[:, 1]], dim=-1)
    residual = check_kcl_residuals(
        batch.edge_index,
        edge_flows,
        net_injection,
        tol=tol,
        verbose=True,
    )
    print(
        "[DEBUG:KCL] mean |residual| {:.3e} | std {:.3e}".format(
            residual.abs().mean().item(),
            residual.std().item(),
        )
    )


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
