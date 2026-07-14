"""Classical pixel-space image comparison: MSE and SSIM (Wang et al. 2004)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _load_image_tensor(path: Path | str, resize: tuple[int, int] | None = None) -> torch.Tensor:
    im = Image.open(path).convert("RGB")
    if resize is not None:
        im = im.resize(resize, Image.BICUBIC)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def pixel_mse(path_a: Path | str, path_b: Path | str) -> float:
    """Mean squared error between two RGB images in ``[0, 1]`` pixel space."""
    a = _load_image_tensor(path_a)
    b = _load_image_tensor(path_b)
    if a.shape != b.shape:
        b = _load_image_tensor(path_b, resize=(a.shape[-1], a.shape[-2]))
    return F.mse_loss(a, b).item()


def _gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(
    path_a: Path | str,
    path_b: Path | str,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Single-scale SSIM (Wang et al. 2004), averaged over RGB channels."""
    a = _load_image_tensor(path_a)
    b = _load_image_tensor(path_b)
    if a.shape != b.shape:
        b = _load_image_tensor(path_b, resize=(a.shape[-1], a.shape[-2]))

    channels = a.shape[1]
    window = _gaussian_window(window_size, sigma, channels)
    pad = window_size // 2

    mu_a = F.conv2d(a, window, padding=pad, groups=channels)
    mu_b = F.conv2d(b, window, padding=pad, groups=channels)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    sigma_a2 = F.conv2d(a * a, window, padding=pad, groups=channels) - mu_a2
    sigma_b2 = F.conv2d(b * b, window, padding=pad, groups=channels) - mu_b2
    sigma_ab = F.conv2d(a * b, window, padding=pad, groups=channels) - mu_ab

    data_range = 1.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / (
        (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    )
    return ssim_map.mean().item()
