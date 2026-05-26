"""Loss functions for drum transcription model training.

Provides timing-aware BCE, focal loss, dice loss, and a combined
rhythm-game focal-dice loss with optional Gaussian target smoothing.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gaussian_kernel_1d(sigma: float, *, radius: int | None = None) -> torch.Tensor:
    """Create a 1-D Gaussian kernel for temporal smoothing.

    Args:
        sigma: Standard deviation of the Gaussian.
        radius: Half-width of the kernel (inferred as 3*sigma if None).

    Returns:
        Normalised 1-D float32 tensor of shape (2*radius+1,).
    """
    if radius is None:
        radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(x ** 2) / (2.0 * sigma * sigma))
    k = k / k.sum()
    return k


def smooth_targets_time(targets: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply 1-D Gaussian smoothing along the time axis of target labels.

    Args:
        targets: Binary label tensor of shape (B, T, C).
        sigma: Standard deviation for the Gaussian kernel. If <= 0,
               returns targets unchanged.

    Returns:
        Smoothed float tensor in [0, 1] of the same shape as targets.
    """
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
    """Binary cross-entropy with timing-smoothed targets and class weighting.

    Args:
        logits: Raw model logits of shape (B, T, C).
        targets: Binary targets of shape (B, T, C).
        pos_weight: Per-class positive-weight tensor.
        timing_sigma: Sigma for Gaussian temporal smoothing of targets.

    Returns:
        Scalar BCE loss.
    """
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
    """Focal binary cross-entropy loss with per-class alpha weighting.

    Args:
        logits: Raw model logits of shape (B, T, C).
        targets: Binary targets of shape (B, T, C).
        alpha: Per-class balancing weight tensor.
        gamma: Focusing parameter (default 2.0).

    Returns:
        Scalar focal loss.
    """
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
    """Dice (F1) loss for binary segmentation.

    Args:
        logits: Raw model logits of shape (B, T, C).
        targets: Binary targets of shape (B, T, C).
        smooth: Smoothing constant to avoid division by zero.

    Returns:
        Scalar dice loss.
    """
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
    """Combined focal + dice loss for rhythm-game transcription.

    Optionally applies temporal smoothing to targets before computing
    the weighted combination.

    Args:
        logits: Raw model logits of shape (B, T, C).
        targets: Binary targets of shape (B, T, C).
        alpha: Per-class focal alpha weights.
        gamma: Focal loss gamma parameter.
        dice_weight: Weight of dice loss in the combination.
        timing_sigma: Sigma for temporal smoothing (0 to disable).

    Returns:
        Scalar combined loss.
    """
    if timing_sigma > 0:
        targets = smooth_targets_time(targets, timing_sigma)

    focal = focal_bce_with_logits(logits, targets, alpha=alpha, gamma=gamma)
    d_loss = dice_loss(logits, targets)

    return (1.0 - dice_weight) * focal + dice_weight * d_loss