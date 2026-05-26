"""ADTOF model builder and forward helpers.

Wraps the ``adtof_pytorch`` library to build an ADTOF Frame RNN model,
load pretrained weights, freeze the CNN backbone, and extract features
or logits for inference.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def build_adtof_frame_rnn(*, num_classes: int = 8) -> nn.Module:
    """Build an ADTOF Frame RNN model with a custom output head.

    Creates the default Frame RNN from ``adtof_pytorch``, replaces the
    output layer with a linear head for *num_classes* drum classes,
    and initializes the new weights.

    Args:
        num_classes: Number of output drum classes. Defaults to 8.

    Returns:
        A PyTorch ``nn.Module`` with the new output head.
    """
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
    """Load pretrained ADTOF weights into a model.

    If *weights_path* is None, uses the default ADTOF weights location.
    Does not raise on failure — returns None if no weights are found.

    Args:
        model: The model to load weights into.
        weights_path: Optional path to a weights file.

    Returns:
        The resolved path to the weights file, or None if no weights were loaded.
    """
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
    """Freeze all parameters in the ADTOF CNN backbone.

    Sets ``requires_grad=False`` on every parameter in ``model.cnn_blocks``.

    Args:
        model: An ADTOF Frame RNN model with a ``cnn_blocks`` attribute.
    """
    for param in model.cnn_blocks.parameters():
        param.requires_grad = False


def forward_adtof_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Extract features from an ADTOF model without the output layer.

    Passes the input through the CNN blocks, context layer (if present),
    and GRU layers, returning the final hidden states.

    Args:
        model: An ADTOF Frame RNN model.
        x: Input tensor of shape ``(B, T, F, C)``.

    Returns:
        Feature tensor of shape ``(B, T, features)``.
    """
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
    """Compute output logits from an ADTOF model.

    Shorthand for ``model.output_layer(forward_adtof_features(model, x))``.

    Args:
        model: An ADTOF Frame RNN model.
        x: Input tensor of shape ``(B, T, F, C)``.

    Returns:
        Logits tensor of shape ``(B, T, num_classes)``.
    """
    return model.output_layer(forward_adtof_features(model, x))
