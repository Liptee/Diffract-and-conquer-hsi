from __future__ import annotations

import torch


def spectral_angle_mapper(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean SAM in degrees; pred/target [B, C, H, W]."""
    p = pred / (pred.norm(dim=1, keepdim=True).clamp_min(eps))
    t = target / (target.norm(dim=1, keepdim=True).clamp_min(eps))
    cos = (p * t).sum(dim=1).clamp(-1.0, 1.0)
    return torch.acos(cos).mean() * (180.0 / torch.pi)


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float("inf"), device=pred.device)
    return 10.0 * torch.log10((max_val**2) / mse)
