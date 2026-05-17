from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from audiotochart.chart.drum_vocab import PRO8_LABELS

log = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when a model bundle cannot be loaded."""


@dataclass
class ModelBundle:
    model: object
    labels: list[str]
    config: dict = field(default_factory=dict)
    device: str = "cpu"


PRO8_VARIANT = "pro8"
PRO8_ARCHITECTURE = "adtof_frame_rnn"


def _build_simple_cnn(config: dict) -> object:
    import torch.nn as nn

    n_mels: int = config.get("n_mels", 84)
    num_classes: int = config.get("num_classes", 8)
    hidden: int = config.get("hidden_dim", 8)

    class _SimpleCNN(nn.Module):
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
    import torch.nn as nn
    from adtof_pytorch import create_frame_rnn_model, calculate_n_bins

    num_classes: int = config.get("num_classes", 8)
    n_bins = calculate_n_bins()
    model = create_frame_rnn_model(n_bins)

    in_features = model.output_layer.in_features
    new_head = nn.Linear(in_features, num_classes)
    nn.init.xavier_uniform_(new_head.weight)
    nn.init.zeros_(new_head.bias)
    model.output_layer = new_head

    # Wrap forward to return logits (ADTOF's built-in forward applies sigmoid).
    import torch

    def _logit_forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, _freq_bins, _channels = x.shape
        h = x.permute(0, 3, 1, 2)
        for block in self.cnn_blocks:
            h = block(h)
        h = h.permute(0, 2, 3, 1)
        features = h.shape[2] * h.shape[3]
        h = h.reshape(batch_size, time_steps, features)
        if getattr(self, "context_layer", None) is not None:
            h = self.context_layer(h)
        for gru in self.gru_layers:
            h, _ = gru(h)
        return self.output_layer(h)

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
    import torch

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

    # Merge thresholds.json into config so peak-picking can find them.
    thr_path = model_dir / "thresholds.json"
    if thr_path.is_file():
        thr_data = json.loads(thr_path.read_text(encoding="utf-8"))
        if "thresholds" in thr_data:
            config.setdefault("thresholds", thr_data["thresholds"])
            log.info("Loaded %d thresholds from %s", len(thr_data["thresholds"]), thr_path)

    num_classes = config.get("num_classes", len(labels))
    log.info(
        "Building model architecture %r (num_classes=%d, n_mels=%d)",
        arch_name,
        num_classes,
        config.get("n_mels", 84),
    )
    model = _build_model_for_architecture(arch_name, config)

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
