# Standard supervised training loop

from __future__ import annotations
import torch
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.losses.regression_loss import rmse, circular_rmse

def train_one_epoch(model, loader, optimizer, device, angle_col=None):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch.to(device)
        y_hat = model(batch)
        # assume y = [P_G, Q_G, |V|, theta] in columns 0..3 (adjust later)
        loss_num = rmse(y_hat[:, :3], batch.y[:, :3])
        loss_ang = circular_rmse(y_hat[:, 3], batch.y[:, 3])
        loss = loss_num + loss_ang

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * batch.num_nodes
    return total / sum(b.num_nodes for b in loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_num, total_ang, count = 0.0, 0.0, 0
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = batch.to(device)
        y_hat = model(batch)
        total_num += rmse(y_hat[:, :3], batch.y[:, :3]).item() * batch.num_nodes
        total_ang += circular_rmse(y_hat[:, 3], batch.y[:, 3]).item() * batch.num_nodes
        count += batch.num_nodes
    return dict(rmse_num=total_num / count, rmse_theta=total_ang / count)

def fit(model, dataset, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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
        tr = train_one_epoch(model, train_loader, opt, device)
        val = evaluate(model, val_loader, device)
        print(f"epoch {epoch:03d} | train_loss ~ {tr:.4f} | val_num {val['rmse_num']:.4f} | val_theta {val['rmse_theta']:.4f}")
        if val["rmse_num"] + val["rmse_theta"] < best_val:
            best_val = val["rmse_num"] + val["rmse_theta"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device)
    print(f"[TEST] num {test['rmse_num']:.4f} | theta {test['rmse_theta']:.4f}")
    return test