from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gaussian_kernel_1d(sigma: float, *, radius: int | None = None) -> torch.Tensor:
    if radius is None:
        radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(x ** 2) / (2.0 * sigma * sigma))
    k = k / k.sum()
    return k


def smooth_targets_time(targets: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return targets

    B, T, C = targets.shape
    kernel = gaussian_kernel_1d(sigma).to(targets.device, dtype=targets.dtype)
    radius = (kernel.numel() - 1) // 2

    x = targets.permute(0, 2, 1).reshape(B * C, 1, T)
    k = kernel.view(1, 1, -1)
    smoothed = F.conv1d(x, k, padding=radius)
    smoothed = smoothed.view(B, C, T).permute(0, 2, 1).contiguous()

    peak = kernel.max().clamp(min=1e-8)
    smoothed = smoothed / peak
    return smoothed.clamp(0.0, 1.0)


def rhythm_game_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: torch.Tensor,
    timing_sigma: float = 1.0,
) -> torch.Tensor:
    soft = smooth_targets_time(targets, timing_sigma)
    return F.binary_cross_entropy_with_logits(
        logits, soft, pos_weight=pos_weight, reduction="mean",
    )


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.exp(-bce)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    loss = alpha_t * focal_weight * bce
    return loss.mean()


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    smooth: float = 1.0,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(0, 1))
    union = probs.sum(dim=(0, 1)) + targets.sum(dim=(0, 1))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def rhythm_game_focal(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: torch.Tensor,
    gamma: float = 2.0,
    dice_weight: float = 0.3,
    timing_sigma: float = 0.5,
) -> torch.Tensor:
    if timing_sigma > 0:
        targets = smooth_targets_time(targets, timing_sigma)

    focal = focal_bce_with_logits(logits, targets, alpha=alpha, gamma=gamma)
    d_loss = dice_loss(logits, targets)

    return (1.0 - dice_weight) * focal + dice_weight * d_loss
