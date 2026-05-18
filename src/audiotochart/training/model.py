from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from audiotochart.adtof_model import (
    build_adtof_frame_rnn,
    forward_adtof_features,
    forward_adtof_logits,
    freeze_adtof_cnn,
    load_adtof_pretrained_weights,
)

log = logging.getLogger(__name__)


def build_finetune_model(
    weights_path: Path | None = None,
    *,
    num_classes: int = 8,
    freeze_cnn: bool = True,
) -> nn.Module:
    model = build_adtof_frame_rnn(num_classes=num_classes)

    resolved = load_adtof_pretrained_weights(model, weights_path)
    if resolved is None:
        log.warning("No pretrained weights found at %s; training from scratch", weights_path)
    if freeze_cnn:
        freeze_adtof_cnn(model)

    return model


def forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return forward_adtof_logits(model, x)


def forward_encoder_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return forward_adtof_features(model, x)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
