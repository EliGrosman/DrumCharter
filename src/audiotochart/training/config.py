from __future__ import annotations

from dataclasses import dataclass, field
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
        return 8

    def resolved_class_weights(self) -> tuple[float, ...]:
        if self.class_weights is not None:
            return self.class_weights
        return DEFAULT_CLASS_WEIGHTS_8
