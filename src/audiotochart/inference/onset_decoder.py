from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from audiotochart.inference.checkpoint import (
    ModelBundle,
    PRO8_ARCHITECTURE,
    _build_model_for_architecture,
)
from audiotochart.onset_decoder_common import (
    CHORD_BOS,
    CHORD_NULL,
    NUM_CHORD_CLASSES,
    ChordVocabulary,
    aggregate_chord_features,
    build_chord_vocabulary,
    build_onset_conditioned_model,
    build_onset_feature_rows,
    classes_to_mask as classes_to_mask,
    mask_to_classes,
    mask_to_name as mask_to_name,
)

log = logging.getLogger(__name__)


class OnsetDecoderError(RuntimeError):
    """Raised when an onset decoder bundle cannot be used."""


@dataclass(slots=True)
class ChordDecoderBundle:
    model: object
    config: dict
    vocab: ChordVocabulary
    device: str
    source_dir: Path


def load_chord_decoder_bundle(
    decoder_dir: Path,
    *,
    base_bundle: ModelBundle,
    device: str,
) -> ChordDecoderBundle:
    """Load a standalone chord decoder bundle.

    Supports checkpoints with a full ``model_state`` first, then falls back to
    ``decoder_state`` using the already-loaded base encoder.
    """

    decoder_dir = Path(decoder_dir)
    if not decoder_dir.is_dir():
        raise OnsetDecoderError(f"Onset decoder directory not found: {decoder_dir}")

    cfg_path = decoder_dir / "config.json"
    if not cfg_path.is_file():
        raise OnsetDecoderError(f"Missing config.json in onset decoder directory: {decoder_dir}")
    config = json.loads(cfg_path.read_text(encoding="utf-8"))

    if config.get("use_structure"):
        raise OnsetDecoderError(
            "Onset decoder config has use_structure=true, which is not supported "
            "by AudioToChart v1 live transcription"
        )

    if "chord_masks" not in config:
        raise OnsetDecoderError(
            "Unsupported onset decoder bundle: only chord decoder configs with "
            "'chord_masks' are supported"
        )

    weights_path = decoder_dir / "best.pt"
    if not weights_path.is_file():
        raise OnsetDecoderError(f"Missing best.pt in onset decoder directory: {decoder_dir}")

    raw_masks = config["chord_masks"]
    if not isinstance(raw_masks, list) or not all(isinstance(m, int) for m in raw_masks):
        raise OnsetDecoderError("onset decoder config field 'chord_masks' must be a list of integers")
    vocab = ChordVocabulary(
        masks=tuple(int(mask) for mask in raw_masks),
        blocklist_policy=config.get("blocklist_policy", "none"),
    )
    configured_vocab_size = config.get("vocab_size")
    if configured_vocab_size is not None and int(configured_vocab_size) != vocab.vocab_size:
        raise OnsetDecoderError(
            "onset decoder config vocab_size "
            f"({configured_vocab_size}) does not match chord_masks ({vocab.vocab_size})"
        )

    model = build_onset_conditioned_model(
        _build_decoder_encoder(),
        config=config,
        vocab_size=vocab.vocab_size,
    )

    import torch

    ckpt = torch.load(str(weights_path), map_location=device, weights_only=True)
    load_error: Exception | None = None
    full_state = None
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        full_state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and _looks_like_full_model_state(ckpt):
        full_state = ckpt

    if full_state is not None:
        try:
            model.load_state_dict(full_state, strict=True)
        except Exception as exc:
            load_error = exc
        else:
            model.to(device)
            model.eval()
            log.info("Loaded chord onset decoder full model_state from %s", weights_path)
            return ChordDecoderBundle(
                model=model,
                config=config,
                vocab=vocab,
                device=device,
                source_dir=decoder_dir,
            )

    if isinstance(ckpt, dict) and "decoder_state" in ckpt:
        fallback_model = build_onset_conditioned_model(
            base_bundle.model,
            config=config,
            vocab_size=vocab.vocab_size,
        )
        try:
            fallback_model.decoder.load_state_dict(ckpt["decoder_state"], strict=True)
        except Exception as exc:
            raise OnsetDecoderError(
                f"Failed to load onset decoder checkpoint {weights_path}: {exc}"
            ) from exc
        fallback_model.to(device)
        fallback_model.eval()
        log.info("Loaded chord onset decoder decoder_state from %s", weights_path)
        return ChordDecoderBundle(
            model=fallback_model,
            config=config,
            vocab=vocab,
            device=device,
            source_dir=decoder_dir,
        )

    if load_error is not None:
        raise OnsetDecoderError(
            f"Failed to load onset decoder full model_state from {weights_path}: {load_error}"
        ) from load_error
    raise OnsetDecoderError(
        f"Onset decoder checkpoint {weights_path} must contain model_state or decoder_state"
    )


def refine_chord_onsets(
    decoder_bundle: ChordDecoderBundle,
    onsets: list[tuple[float, int, float]],
    acts: np.ndarray,
    spec: np.ndarray,
    *,
    fps: float,
    thresholds: Sequence[float],
) -> list[tuple[float, int, float]]:
    """Apply the chord decoder and return refined ``(time, class, confidence)`` onsets."""

    if not onsets:
        return []

    T = spec.shape[0]
    if T == 0:
        return onsets

    onset_frames = [min(int(round(t * fps)), T - 1) for t, _c, _conf in onsets]
    onset_classes = [int(c) for _t, c, _conf in onsets]
    onset_features = build_onset_feature_rows(
        acts,
        onset_frames,
        onset_classes,
        thresholds=thresholds,
    )
    baseline_onsets = list(zip(onset_frames, onset_classes))

    refined = decode_chord_hybrid_onsets(
        decoder_bundle.model,
        baseline_onsets=baseline_onsets,
        onset_features=onset_features,
        spec=spec,
        device=decoder_bundle.device,
        window_frames=decoder_bundle.config.get("window_frames", 1000),
        stride_frames=decoder_bundle.config.get("stride_frames", 500),
        max_onsets=decoder_bundle.config.get("max_onsets", 256),
        vocab=decoder_bundle.vocab,
    )

    confidence_map = {
        (frame, cls): conf
        for frame, cls, (_time, _class_idx, conf) in zip(
            onset_frames,
            onset_classes,
            onsets,
        )
    }
    out: list[tuple[float, int, float]] = []
    for frame, cls in refined:
        confidence = confidence_map.get((frame, cls))
        if confidence is None:
            safe_frame = min(max(int(frame), 0), acts.shape[0] - 1)
            confidence = (
                float(acts[safe_frame, cls])
                if acts.shape[0] > 0 and 0 <= cls < acts.shape[1]
                else 0.5
            )
        out.append((float(frame) / fps, int(cls), float(confidence)))
    out.sort(key=lambda item: (item[0], item[1]))
    return out


def decode_chord_hybrid_onsets(
    model: object,
    baseline_onsets: list[tuple[int, int]],
    onset_features: np.ndarray,
    spec: np.ndarray,
    *,
    device: str,
    window_frames: int = 1000,
    stride_frames: int = 500,
    max_onsets: int = 256,
    vocab: ChordVocabulary | None = None,
) -> list[tuple[int, int]]:
    """Run a chord decoder over baseline onset timings and expand tokens to events."""

    if not baseline_onsets:
        return []

    import torch

    vocab = vocab or build_chord_vocabulary()
    torch_device = torch.device(device)
    t_frames = spec.shape[0]
    decoder_predictions: dict[int, tuple[int, int]] = {}

    for win_start in range(0, t_frames, stride_frames):
        win_end = min(win_start + window_frames, t_frames)
        if win_end - win_start < 100:
            break

        win_indices = [
            idx
            for idx, (frame, _class_idx) in enumerate(baseline_onsets)
            if win_start <= frame < win_end
        ]
        if not win_indices:
            continue

        group_frames, group_features = _group_window_onsets(
            baseline_onsets,
            onset_features,
            win_indices,
            win_start=win_start,
        )
        if not group_frames:
            continue

        chunk = spec[win_start:win_end]
        if chunk.shape[0] < window_frames:
            chunk = np.pad(
                chunk,
                ((0, window_frames - chunk.shape[0]), (0, 0), (0, 0)),
            )

        x = torch.from_numpy(chunk).float().unsqueeze(0).to(torch_device)
        enc = model.encode(x)

        pred_tokens = _greedy_decode_chords(
            model.decoder,
            enc,
            group_frames,
            device=torch_device,
            max_onsets=max_onsets,
            onset_features=torch.from_numpy(group_features.astype(np.float32))
            .unsqueeze(0)
            .to(torch_device),
        )

        win_center = win_start + window_frames // 2
        for local_idx, local_frame in enumerate(group_frames):
            if local_idx >= len(pred_tokens):
                break
            global_frame = win_start + int(local_frame)
            dist = abs(global_frame - win_center)
            prev = decoder_predictions.get(global_frame)
            if prev is None or dist < prev[1]:
                decoder_predictions[global_frame] = (int(pred_tokens[local_idx]), dist)

    hybrid_onsets: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for frame in sorted(decoder_predictions):
        token = decoder_predictions[frame][0]
        if token == CHORD_NULL:
            continue
        mask = vocab.mask_for_token(token)
        if mask is None:
            continue
        for class_idx in mask_to_classes(mask):
            key = (int(frame), int(class_idx))
            if key in seen:
                continue
            seen.add(key)
            hybrid_onsets.append(key)

    hybrid_onsets.sort()
    return hybrid_onsets


def _group_window_onsets(
    baseline_onsets: list[tuple[int, int]],
    onset_features: np.ndarray,
    win_indices: list[int],
    *,
    win_start: int,
) -> tuple[list[int], np.ndarray]:
    grouped: dict[int, dict[str, list]] = {}
    for global_idx in win_indices:
        frame, class_idx = baseline_onsets[global_idx]
        local_frame = int(frame) - win_start
        bucket = grouped.setdefault(local_frame, {"classes": [], "features": []})
        bucket["classes"].append(int(class_idx))
        if global_idx < len(onset_features):
            bucket["features"].append(onset_features[global_idx])

    frames: list[int] = []
    features: list[np.ndarray] = []
    for local_frame in sorted(grouped):
        classes = grouped[local_frame]["classes"]
        feature_rows = np.asarray(grouped[local_frame]["features"], dtype=np.float32)
        frames.append(local_frame)
        features.append(aggregate_chord_features(feature_rows, classes))

    return frames, np.asarray(features, dtype=np.float32)


def _greedy_decode_chords(
    decoder: object,
    encoder_features: object,
    onset_frames: list[int],
    *,
    device: object,
    max_onsets: int,
    onset_features: object | None = None,
) -> list[int]:
    import torch

    n_onsets = min(len(onset_frames), max_onsets)
    if n_onsets == 0:
        return []

    frames_t = torch.tensor([onset_frames[:n_onsets]], dtype=torch.long, device=device)
    tokens = [CHORD_BOS]
    for _step in range(n_onsets):
        tgt = torch.tensor([tokens], dtype=torch.long, device=device)
        cur_frames = frames_t[:, : len(tokens)]
        if cur_frames.shape[1] < len(tokens):
            cur_frames = torch.cat(
                [
                    cur_frames[:, :1].expand(-1, len(tokens) - cur_frames.shape[1]),
                    cur_frames,
                ],
                dim=1,
            )
        logits = decoder(
            encoder_features,
            cur_frames,
            tgt,
            onset_features=onset_features[:, : len(tokens)]
            if onset_features is not None
            else None,
        )
        tokens.append(int(logits[0, -1].argmax().item()))
    return tokens[1:]


def _looks_like_full_model_state(state: dict) -> bool:
    return any(isinstance(key, str) and key.startswith("encoder.") for key in state) and any(
        isinstance(key, str) and key.startswith("decoder.") for key in state
    )


def _build_decoder_encoder() -> object:
    return _build_model_for_architecture(PRO8_ARCHITECTURE, {"num_classes": NUM_CHORD_CLASSES})
