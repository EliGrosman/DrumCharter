"""Precomputation of frame-model candidate onsets for decoder training.

Runs a trained frame-level model over training audio to extract candidate
onset positions and their feature representations, which are saved as
.npz files for chord-decoder training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drumcharter.onset_decoder_common import build_onset_feature_rows
from drumcharter.training.dataset import _load_entries
from drumcharter.training.thresholds import pick_peaks

log = logging.getLogger(__name__)


@dataclass
class OnsetPrecomputeConfig:
    cache_dir: Path
    frame_model_dir: Path
    output_dir: Path
    device: str = "cuda"
    harmonix_only: bool = False
    force: bool = False
    max_chunk_frames: int = 2000


@dataclass(frozen=True, slots=True)
class OnsetPrecomputeResult:
    written: int = 0
    skipped: int = 0
    failed: int = 0


def run_precompute_onsets(cfg: OnsetPrecomputeConfig) -> OnsetPrecomputeResult:
    import torch

    from drumcharter.inference.checkpoint import load_model_bundle

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    entries = _load_entries(cfg.cache_dir, harmonix_only=cfg.harmonix_only)
    if not entries:
        raise ValueError(f"No entries found in cache: {cfg.cache_dir}")

    if cfg.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available; falling back to CPU")
        cfg.device = "cpu"

    bundle = load_model_bundle(cfg.frame_model_dir, device=cfg.device)
    model = bundle.model
    model.eval()
    thresholds = list(bundle.config.get("thresholds", [0.5] * len(bundle.labels)))
    confidence_gates = list(bundle.config.get("confidence_gates", [None] * len(bundle.labels)))

    written = 0
    skipped = 0
    failed = 0
    for entry in entries:
        out_path = cfg.output_dir / f"{entry.song_hash}.npz"
        if out_path.exists() and not cfg.force:
            skipped += 1
            continue

        try:
            spec = np.load(str(entry.spec_path))
            labels = np.load(str(entry.label_path), mmap_mode="r")
            t_frames = min(spec.shape[0], labels.shape[0])
            spec = spec[:t_frames]
            if spec.ndim == 2:
                spec = spec[:, :, np.newaxis]

            acts_chunks: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, t_frames, cfg.max_chunk_frames):
                    chunk = spec[start : start + cfg.max_chunk_frames]
                    x = torch.from_numpy(chunk).float().unsqueeze(0).to(cfg.device)
                    logits = model(x)
                    probs = torch.sigmoid(logits)[0].cpu().numpy()
                    acts_chunks.append(probs)
            acts = (
                np.concatenate(acts_chunks, axis=0).astype(np.float32)
                if acts_chunks
                else np.zeros((0, len(bundle.labels)), dtype=np.float32)
            )

            onset_events: list[tuple[int, int]] = []
            for class_idx in range(min(acts.shape[1], len(bundle.labels))):
                gate = confidence_gates[class_idx] if class_idx < len(confidence_gates) else None
                if gate is not None and acts.shape[0] > 0:
                    if float(acts[:, class_idx].max()) < float(gate):
                        continue
                threshold = thresholds[class_idx] if class_idx < len(thresholds) else 0.5
                for frame in pick_peaks(acts[:, class_idx], float(threshold)):
                    onset_events.append((int(frame), int(class_idx)))

            onset_events.sort()
            onset_frames = np.asarray([frame for frame, _class_idx in onset_events], dtype=np.int32)
            onset_classes = np.asarray([class_idx for _frame, class_idx in onset_events], dtype=np.int32)
            onset_features = build_onset_feature_rows(
                acts,
                onset_frames,
                onset_classes,
                thresholds=thresholds,
            )
            np.savez_compressed(
                out_path,
                onset_frames=onset_frames,
                onset_classes=onset_classes,
                onset_features=onset_features,
                num_frames=np.asarray(t_frames, dtype=np.int32),
            )
            written += 1
            log.info("Wrote %s (%d candidate onsets)", out_path, len(onset_frames))
        except Exception:
            failed += 1
            log.exception("Failed to precompute onsets for %s", entry.song_name)

    return OnsetPrecomputeResult(written=written, skipped=skipped, failed=failed)
