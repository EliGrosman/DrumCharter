"""Model bundle loading and checkpoint handling.

Supports loading model bundles (``config.json``, ``weights.pt``/``best.pt``,
``labels.json``, ``thresholds.json``) for both the ``simple_cnn`` test
architecture and the ``adtof_frame_rnn`` pro8 architecture.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from drumcharter.chart.drum_vocab import PRO8_LABELS

log = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when a model bundle cannot be loaded."""


@dataclass
class ModelBundle:
    """Container for a loaded model, its labels, config, and device.

    Attributes:
        model: The PyTorch model instance.
        labels: List of instrument label strings.
        config: Model configuration dict from ``config.json``.
        device: The torch device the model is on.
    """

    model: object
    labels: list[str]
    config: dict = field(default_factory=dict)
    device: str = "cpu"


PRO8_VARIANT = "pro8"
"""Variant string for the built-in pro8 model."""

PRO8_ARCHITECTURE = "adtof_frame_rnn"
"""Architecture name for the built-in pro8 model (ADTOF Frame RNN)."""


def _build_simple_cnn(config: dict) -> object:
    """Build a simple 2D CNN for test models.

    A minimal architecture with two Conv2d layers followed by a Linear head.
    Used exclusively in tests and for simple architectures.

    Args:
        config: Config dict with ``num_classes`` and ``hidden_dim`` keys.

    Returns:
        A ``_SimpleCNN`` module instance.
    """
    import torch.nn as nn

    num_classes: int = config.get("num_classes", 8)
    hidden: int = config.get("hidden_dim", 8)

    class _SimpleCNN(nn.Module):
        """Two-layer 2D CNN for simple test architectures."""

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, hidden, kernel_size=(3, 3), padding=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(hidden, hidden, kernel_size=(3, 3), padding=(1, 1)),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((None, 1)),
            )
            self.fc = nn.Linear(hidden, num_classes)

        def forward(self, x):
            """Forward pass through the CNN.

            Args:
                x: Input tensor of shape ``(B, T, F, C)``.

            Returns:
                Logits of shape ``(B, T, num_classes)``.
            """
            B, T, F, C = x.shape
            x = x.permute(0, 3, 1, 2)
            x = self.conv(x)
            x = x.squeeze(-1).permute(0, 2, 1)
            return self.fc(x)

    return _SimpleCNN()


def _build_adtof_frame_rnn(config: dict) -> object:
    """Build an ADTOF Frame_RNN with an 8-class output head.

    Requires the ``ai`` extra (``adtof_pytorch`` and ``torch``).
    The returned model outputs **logits** (pre-sigmoid).
    """
    from drumcharter.adtof_model import build_adtof_frame_rnn, forward_adtof_logits

    num_classes: int = config.get("num_classes", 8)
    model = build_adtof_frame_rnn(num_classes=num_classes)

    # Wrap forward to return logits (ADTOF's built-in forward applies sigmoid).
    def _logit_forward(self, x):
        return forward_adtof_logits(self, x)

    model.forward = _logit_forward.__get__(model, type(model))

    return model


def _is_known_architecture(architecture: str) -> bool:
    return architecture == "simple_cnn" or architecture == PRO8_ARCHITECTURE


def _known_architectures_text() -> str:
    return f"simple_cnn, {PRO8_ARCHITECTURE}"


def _build_model_for_architecture(architecture: str, config: dict) -> object:
    """Build a model for a known architecture name."""
    if architecture == "simple_cnn":
        return _build_simple_cnn(config)
    if architecture == PRO8_ARCHITECTURE:
        return _build_adtof_frame_rnn(config)
    raise ModelLoadError(
        f"Unknown architecture {architecture!r}. Supported: {_known_architectures_text()}"
    )


def _validate_metadata_list_length(
    key: str,
    value: object,
    expected_len: int,
    source: Path,
) -> None:
    """Validate that a metadata value is a list of a specific length.

    Args:
        key: The field name (for error messages).
        value: The value to validate.
        expected_len: The expected list length.
        source: The source file path (for error messages).

    Raises:
        ModelLoadError: If *value* is not a list or has the wrong length.
    """
    if not isinstance(value, list):
        raise ModelLoadError(f"{source.name} field {key!r} must be a list")
    if len(value) != expected_len:
        raise ModelLoadError(
            f"{source.name} field {key!r} has {len(value)} entries, "
            f"expected {expected_len}"
        )


def load_model_bundle(model_dir: Path, *, device: str = "cpu") -> ModelBundle:
    """Load a model bundle from *model_dir*.

    The directory must contain:

    * ``config.json`` — model metadata including an ``"architecture"`` key
      (or ``"variant": "pro8"`` for the built-in CloneHero-ChartGen model).
    * ``weights.pt`` or ``best.pt`` — ``state_dict`` for the model.
    * ``labels.json`` (optional) — list of instrument label strings.
      If missing, labels are derived from ``variant`` (``"pro8"``).

    Supported architecture names are ``"simple_cnn"`` for tests and
    ``"adtof_frame_rnn"`` for the built-in pro8 model.
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise ModelLoadError(f"Model directory not found: {model_dir}")

    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        raise ModelLoadError(f"Missing config.json in {model_dir}")
    config = json.loads(cfg_path.read_text(encoding="utf-8"))

    arch_name = config.get("architecture")
    variant = config.get("variant")

    if variant is not None and variant != PRO8_VARIANT:
        raise ModelLoadError(
            f"Unsupported variant {variant!r}. Known variants: {PRO8_VARIANT}"
        )

    if arch_name is None and variant == PRO8_VARIANT:
        arch_name = PRO8_ARCHITECTURE
    if not arch_name:
        raise ModelLoadError(
            f"config.json must contain an 'architecture' key. "
            f"Found keys: {list(config.keys())}"
        )

    if not _is_known_architecture(arch_name):
        raise ModelLoadError(
            f"Unknown architecture {arch_name!r}. Supported: {_known_architectures_text()}"
        )

    weights_candidates = [
        model_dir / "weights.pt",
        model_dir / "best.pt",
    ]
    weights_path: Path | None = None
    for w in weights_candidates:
        if w.is_file():
            weights_path = w
            break
    if weights_path is None:
        raise ModelLoadError(
            f"No weights file found in {model_dir}. "
            f"Looked for: {[p.name for p in weights_candidates]}"
        )

    labels_path = model_dir / "labels.json"
    if labels_path.is_file():
        labels: list[str] = json.loads(labels_path.read_text(encoding="utf-8"))
        if not isinstance(labels, list) or not all(isinstance(s, str) for s in labels):
            raise ModelLoadError("labels.json must contain a list of strings")
    elif variant == PRO8_VARIANT:
        labels = list(PRO8_LABELS)
        log.info("labels.json not found; using variant %r labels: %s", variant, labels)
    else:
        raise ModelLoadError(
            f"Missing labels.json in {model_dir} and variant {variant!r} "
            f"has no default labels"
        )

    num_classes = config.get("num_classes", len(labels))

    # Merge thresholds.json into config so peak-picking can find them.
    thr_path = model_dir / "thresholds.json"
    if thr_path.is_file():
        thr_data = json.loads(thr_path.read_text(encoding="utf-8"))
        if "thresholds" in thr_data:
            _validate_metadata_list_length(
                "thresholds",
                thr_data["thresholds"],
                num_classes,
                thr_path,
            )
            config.setdefault("thresholds", thr_data["thresholds"])
            log.info("Loaded %d thresholds from %s", len(thr_data["thresholds"]), thr_path)
        if "confidence_gates" in thr_data:
            _validate_metadata_list_length(
                "confidence_gates",
                thr_data["confidence_gates"],
                num_classes,
                thr_path,
            )
            config.setdefault("confidence_gates", thr_data["confidence_gates"])
            log.info("Loaded %d confidence_gates from %s", len(thr_data["confidence_gates"]), thr_path)

    if "thresholds" in config:
        _validate_metadata_list_length(
            "thresholds",
            config["thresholds"],
            num_classes,
            cfg_path,
        )
    if "confidence_gates" in config:
        _validate_metadata_list_length(
            "confidence_gates",
            config["confidence_gates"],
            num_classes,
            cfg_path,
        )

    log.info(
        "Building model architecture %r (num_classes=%d, n_mels=%d)",
        arch_name,
        num_classes,
        config.get("n_mels", 84),
    )
    model = _build_model_for_architecture(arch_name, config)

    import torch

    state = torch.load(str(weights_path), map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    log.info(
        "Loaded model bundle from %s (architecture=%r, %d labels, device=%s)",
        model_dir,
        arch_name,
        len(labels),
        device,
    )
    return ModelBundle(model=model, labels=labels, config=config, device=device)
