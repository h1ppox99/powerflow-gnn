# RMSE, circular phase angle loss

import torch
import torch.nn.functional as F

def rmse(pred, target):
    return torch.sqrt(F.mse_loss(pred, target))

def circular_rmse(theta_pred, theta_true):
    """
    RMSE on angles in radians, invariant to 2π wrap.
    """
    diff = torch.atan2(torch.sin(theta_pred - theta_true), torch.cos(theta_pred - theta_true))
    return torch.sqrt(torch.mean(diff**2))