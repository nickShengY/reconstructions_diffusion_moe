from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim


def _to_01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x * 0.5 + 0.5, 0.0, 1.0)


def batch_mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, y, reduction="none").mean(dim=(1, 2, 3))


def batch_psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = batch_mse(x, y).clamp_min(eps)
    return 10.0 * torch.log10(1.0 / mse)


def batch_ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x01 = _to_01(x).detach().cpu().numpy().transpose(0, 2, 3, 1)
    y01 = _to_01(y).detach().cpu().numpy().transpose(0, 2, 3, 1)
    vals: List[float] = []
    for i in range(x01.shape[0]):
        vals.append(float(ssim(x01[i], y01[i], channel_axis=-1, data_range=1.0)))
    return torch.tensor(vals, dtype=torch.float32)


def latent_cosine(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_flat = x.reshape(x.shape[0], -1)
    y_flat = y.reshape(y.shape[0], -1)
    return F.cosine_similarity(x_flat, y_flat, dim=1)


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_flat = x.reshape(x.shape[0], -1)
    y_flat = y.reshape(y.shape[0], -1)
    x_center = x_flat - x_flat.mean(dim=0, keepdim=True)
    y_center = y_flat - y_flat.mean(dim=0, keepdim=True)
    k = x_center @ x_center.t()
    l = y_center @ y_center.t()
    hsic = (k * l).sum()
    norm = torch.sqrt((k * k).sum().clamp_min(eps) * (l * l).sum().clamp_min(eps))
    return hsic / norm


def summarize(metric_map: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in metric_map.items():
        if value.ndim == 0:
            out[key] = float(value.item())
            continue
        out[f"{key}_mean"] = float(value.mean().item())
        out[f"{key}_std"] = float(value.std(unbiased=False).item())
        out[f"{key}_p10"] = float(torch.quantile(value, 0.10).item())
        out[f"{key}_p05"] = float(torch.quantile(value, 0.05).item())
    return out
