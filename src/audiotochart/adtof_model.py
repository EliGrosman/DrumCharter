from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def build_adtof_frame_rnn(*, num_classes: int = 8) -> nn.Module:
    from adtof_pytorch import calculate_n_bins, create_frame_rnn_model

    model = create_frame_rnn_model(calculate_n_bins())
    in_features = model.output_layer.in_features
    head = nn.Linear(in_features, num_classes)
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)
    model.output_layer = head
    return model


def load_adtof_pretrained_weights(
    model: nn.Module,
    weights_path: Path | None = None,
) -> Path | None:
    from adtof_pytorch import get_default_weights_path
    from adtof_pytorch.model import load_pytorch_weights

    resolved = weights_path
    if resolved is None:
        default = get_default_weights_path()
        if default is not None:
            resolved = Path(default)

    if resolved is None or not Path(resolved).exists():
        return None

    load_pytorch_weights(model, str(resolved), strict=False)
    return Path(resolved)


def freeze_adtof_cnn(model: nn.Module) -> None:
    for param in model.cnn_blocks.parameters():
        param.requires_grad = False


def forward_adtof_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    batch_size, time_steps, _freq_bins, _channels = x.shape
    x = x.permute(0, 3, 1, 2)
    for cnn_block in model.cnn_blocks:
        x = cnn_block(x)
    x = x.permute(0, 2, 3, 1)
    features = x.shape[2] * x.shape[3]
    x = x.reshape(batch_size, time_steps, features)
    if getattr(model, "context_layer", None) is not None:
        x = model.context_layer(x)
    for gru in model.gru_layers:
        x, _ = gru(x)
    return x


def forward_adtof_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model.output_layer(forward_adtof_features(model, x))
