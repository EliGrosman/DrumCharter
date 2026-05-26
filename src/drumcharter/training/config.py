"""Training configuration dataclass.

Centralises all hyperparameters for the two-phase training pipeline
(warmup followed by fine-tuning) including loss function selection,
learning rates, data dimensions, and early-stopping criteria.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_CLASS_WEIGHTS_8: tuple[float, ...] = (
    1.2,  # 0 kick
    1.2,  # 1 snare
    1.0,  # 2 hi-hat
    2.0,  # 3 yellow tom
    1.5,  # 4 ride
    2.0,  # 5 blue tom
    1.8,  # 6 crash
    2.5,  # 7 floor tom
)


@dataclass
class TrainConfig:
    """Hyperparameter configuration for drum transcription model training.

    Attributes:
        cache_dir: Directory containing precomputed spectrograms and labels.
        output_dir: Directory for checkpoints, logs, and evaluation results.
        variant: Drumkit variant string (default "pro8").
        window_frames: Number of frames per training window.
        stride_frames: Stride between consecutive windows.
        batch_size: Number of windows per training batch.
        num_workers: Dataloader worker processes.
        seed: Random seed for reproducibility.
        warmup_epochs: Number of frozen-CNN warmup epochs.
        warmup_lr_gru: Learning rate for GRU layers during warmup.
        warmup_lr_head: Learning rate for output head during warmup.
        finetune_epochs: Number of full fine-tuning epochs.
        finetune_lr: Learning rate for GRU / context layers during fine-tune.
        onecycle_pct_start: OneCycleLR percentage of cycle spent ramping up.
        onecycle_div_factor: Initial learning rate divisor for OneCycleLR.
        onecycle_final_div: Final learning rate divisor for OneCycleLR.
        loss_fn: Loss function name ("bce" or "focal+dice").
        focal_gamma: Focal loss focusing parameter.
        dice_weight: Weight of dice loss in the combined loss.
        finetune_lr_cnn_early: Learning rate for early CNN blocks.
        finetune_lr_cnn_late: Learning rate for late CNN blocks.
        finetune_lr_head: Learning rate for output head during fine-tune.
        weight_decay: AdamW weight decay.
        onset_weight: Multiplier for onset-class positive weights.
        timing_sigma: Gaussian smoothing sigma for timing targets.
        class_weights: Per-class loss weights (overrides default if set).
        patience: Early-stopping patience in epochs.
        tolerance_ms_primary: Primary evaluation tolerance in milliseconds.
        tolerance_ms_secondary: Secondary evaluation tolerance in milliseconds.
        amp: Whether to use automatic mixed precision.
        device: Training device string ("cuda" or "cpu").
        harmonix_only: If True, exclude RBN community-charted songs.
    """
    """Hyperparameter configuration for drum transcription model training.

    Attributes:
        cache_dir: Directory containing precomputed spectrograms and labels.
        output_dir: Directory for checkpoints, logs, and evaluation results.
        variant: Drumkit variant string (default "pro8").
        window_frames: Number of frames per training window.
        stride_frames: Stride between consecutive windows.
        batch_size: Number of windows per training batch.
        num_workers: Dataloader worker processes.
        seed: Random seed for reproducibility.
        warmup_epochs: Number of frozen-CNN warmup epochs.
        warmup_lr_gru: Learning rate for GRU layers during warmup.
        warmup_lr_head: Learning rate for output head during warmup.
        finetune_epochs: Number of full fine-tuning epochs.
        finetune_lr: Learning rate for GRU / context layers during fine-tune.
        onecycle_pct_start: OneCycleLR percentage of cycle spent ramping up.
        onecycle_div_factor: Initial learning rate divisor for OneCycleLR.
        onecycle_final_div: Final learning rate divisor for OneCycleLR.
        loss_fn: Loss function name ("bce" or "focal+dice").
        focal_gamma: Focal loss focusing parameter.
        dice_weight: Weight of dice loss in the combined loss.
        finetune_lr_cnn_early: Learning rate for early CNN blocks.
        finetune_lr_cnn_late: Learning rate for late CNN blocks.
        finetune_lr_head: Learning rate for output head during fine-tune.
        weight_decay: AdamW weight decay.
        onset_weight: Multiplier for onset-class positive weights.
        timing_sigma: Gaussian smoothing sigma for timing targets.
        class_weights: Per-class loss weights (overrides default if set).
        patience: Early-stopping patience in epochs.
        tolerance_ms_primary: Primary evaluation tolerance in milliseconds.
        tolerance_ms_secondary: Secondary evaluation tolerance in milliseconds.
        amp: Whether to use automatic mixed precision.
        device: Training device string ("cuda" or "cpu").
        harmonix_only: If True, exclude RBN community-charted songs.
    """
    cache_dir: Path
    output_dir: Path

    variant: str = "pro8"
    window_frames: int = 100
    stride_frames: int = 50
    batch_size: int = 128
    num_workers: int = 4
    seed: int = 42

    warmup_epochs: int = 5
    warmup_lr_gru: float = 5e-4
    warmup_lr_head: float = 3e-3

    finetune_epochs: int = 30
    finetune_lr: float = 3e-4

    onecycle_pct_start: float = 0.1
    onecycle_div_factor: float = 10.0
    onecycle_final_div: float = 100.0

    loss_fn: str = "focal+dice"
    focal_gamma: float = 2.0
    dice_weight: float = 0.3

    finetune_lr_cnn_early: float = 3e-5
    finetune_lr_cnn_late: float = 1e-4
    finetune_lr_head: float = 1e-3

    weight_decay: float = 1e-4

    onset_weight: float = 1.0
    timing_sigma: float = 0.5
    class_weights: tuple[float, ...] | None = None

    patience: int = 8

    tolerance_ms_primary: int = 20
    tolerance_ms_secondary: int = 30

    amp: bool = True
    device: str = "cuda"

    harmonix_only: bool = False

    def num_classes(self) -> int:
        """Return the number of drum classes (hardcoded to 8 for pro8)."""

    def resolved_class_weights(self) -> tuple[float, ...]:
        """Return the effective per-class loss weights.

        Uses the configured weights if set, otherwise falls back to the
        DEFAULT_CLASS_WEIGHTS_8 default.
        """
