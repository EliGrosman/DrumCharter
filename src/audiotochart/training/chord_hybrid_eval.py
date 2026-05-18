"""Hybrid evaluation of chord-conditioned onset decoder models.

Compares baseline (frame-level peak-picking) transcription against
the chord-hybrid decoder output using F-measure, CQS, and tom
consistency metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from audiotochart.inference.onset_decoder import decode_chord_hybrid_onsets
from audiotochart.onset_decoder_common import (
    ChordVocabulary,
    build_chord_vocabulary,
    build_onset_feature_rows,
)
from audiotochart.training.cqs import aggregate_cqs, compute_cqs
from audiotochart.training.dataset import SongEntry
from audiotochart.training.evaluate import CLASS_NAMES_8
from audiotochart.training.thresholds import (
    fmeasure_with_tolerance,
    labels_to_frame_list,
    pick_peaks,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedChordHybridSong:
    """Precomputed per-song data used for hybrid onset decoder evaluation.

    Attributes:
        entry: The song entry with spec and label paths.
        baseline_onsets: Frame-level peak-picked onsets (frame, class).
        onset_features: Feature vectors for each baseline onset.
    """
    entry: SongEntry
    baseline_onsets: tuple[tuple[int, int], ...]
    onset_features: np.ndarray


@dataclass(frozen=True)
class ChordHybridEvalReport:
    """Evaluation report comparing baseline vs hybrid transcription.

    Attributes:
        n_songs: Number of songs evaluated.
        baseline_macro_f: Macro-averaged F-measure for baseline.
        hybrid_macro_f: Macro-averaged F-measure for hybrid decoder.
        baseline_per_class: Per-class F-measures for baseline.
        hybrid_per_class: Per-class F-measures for hybrid decoder.
        baseline_tom_consistency: Tom consistency ratio for baseline.
        hybrid_tom_consistency: Tom consistency ratio for hybrid decoder.
        baseline_cqs: Mean Chart Quality Score for baseline.
        hybrid_cqs: Mean Chart Quality Score for hybrid decoder.
        baseline_cqs_components: Full CQS breakdown for baseline.
        hybrid_cqs_components: Full CQS breakdown for hybrid decoder.
    """
    """Evaluation report comparing baseline vs hybrid transcription.

    Attributes:
        n_songs: Number of songs evaluated.
        baseline_macro_f: Macro-averaged F-measure for baseline.
        hybrid_macro_f: Macro-averaged F-measure for hybrid decoder.
        baseline_per_class: Per-class F-measures for baseline.
        hybrid_per_class: Per-class F-measures for hybrid decoder.
        baseline_tom_consistency: Tom consistency ratio for baseline.
        hybrid_tom_consistency: Tom consistency ratio for hybrid decoder.
        baseline_cqs: Mean Chart Quality Score for baseline.
        hybrid_cqs: Mean Chart Quality Score for hybrid decoder.
        baseline_cqs_components: Full CQS breakdown for baseline.
        hybrid_cqs_components: Full CQS breakdown for hybrid decoder.
    """
    n_songs: int
    baseline_macro_f: float
    hybrid_macro_f: float
    baseline_per_class: dict[str, float]
    hybrid_per_class: dict[str, float]
    baseline_tom_consistency: float
    hybrid_tom_consistency: float
    baseline_cqs: float
    hybrid_cqs: float
    baseline_cqs_components: dict[str, float | int | None]
    hybrid_cqs_components: dict[str, float | int | None]

def as_dict(self) -> dict[str, object]:
        """Serialize the report to a flat dictionary."""
        return {
            "n_songs": self.n_songs,
            "baseline_macro_f": self.baseline_macro_f,
            "hybrid_macro_f": self.hybrid_macro_f,
            "baseline_per_class": self.baseline_per_class,
            "hybrid_per_class": self.hybrid_per_class,
            "baseline_tom_consistency": self.baseline_tom_consistency,
            "hybrid_tom_consistency": self.hybrid_tom_consistency,
            "baseline_cqs": self.baseline_cqs,
            "hybrid_cqs": self.hybrid_cqs,
            "baseline_cqs_components": self.baseline_cqs_components,
            "hybrid_cqs_components": self.hybrid_cqs_components,
        }


@torch.no_grad()
def prepare_chord_hybrid_eval_songs(
    frame_model: torch.nn.Module,
    entries: list[SongEntry],
    *,
    thresholds: list[float],
    confidence_gates: list[float | None] | None = None,
    device: str = "cuda",
    max_songs: int = 0,
    max_chunk_frames: int = 2000,
) -> list[PreparedChordHybridSong]:
    """Run frame-level peak-picking to prepare baseline onsets for hybrid eval.

    Args:
        frame_model: Pretrained frame-level transcription model.
        entries: Song entries to evaluate.
        thresholds: Per-class peak-picking thresholds.
        confidence_gates: Per-class confidence gates (None disables gating).
        device: Device string for inference.
        max_songs: Maximum songs to process (0 for all).
        max_chunk_frames: Chunk size for long spectrograms.

    Returns:
List of PreparedChordHybridSong with baseline onsets and features.
    """
    selected_entries = list(entries[:max_songs]) if max_songs > 0 else list(entries)
    if not selected_entries:
        return []

    frame_model.eval()
    gates = confidence_gates or [None] * len(thresholds)
    prepared: list[PreparedChordHybridSong] = []
    for entry in selected_entries:
        acts = _collect_activations(
            frame_model,
            entry.spec_path,
            device=device,
            max_chunk_frames=max_chunk_frames,
        )
        baseline_onsets: list[tuple[int, int]] = []
        for class_idx, threshold in enumerate(thresholds):
            if class_idx >= acts.shape[1]:
                break
            gate = gates[class_idx] if class_idx < len(gates) else None
            class_acts = acts[:, class_idx]
            if gate is not None and (class_acts.size == 0 or float(np.max(class_acts)) < float(gate)):
                continue
            peaks = pick_peaks(class_acts, float(threshold))
            baseline_onsets.extend((int(frame), class_idx) for frame in peaks)
        baseline_onsets.sort()
        prepared.append(
            PreparedChordHybridSong(
                entry=entry,
                baseline_onsets=tuple(baseline_onsets),
                onset_features=build_onset_feature_rows(
                    acts,
                    onset_frames=[frame for frame, _class_idx in baseline_onsets],
                    onset_classes=[class_idx for _frame, class_idx in baseline_onsets],
                    thresholds=thresholds,
                ),
            )
        )
    return prepared


@torch.no_grad()
def _collect_activations(
    model: torch.nn.Module,
    spec_path: Path,
    *,
    device: str,
    max_chunk_frames: int,
) -> np.ndarray:
    """Run a frame model over a spectrogram in chunks and return sigmoid activations.

    Args:
        model: Frame-level transcription model.
        spec_path: Path to the .npy spectrogram file.
        device: Torch device for inference.
        max_chunk_frames: Maximum frames per chunk.

    Returns:
        Float32 array of shape (T, 8) with sigmoid activations.
    """
    spec = np.load(str(spec_path))
    if spec.ndim == 2:
        spec = spec[:, :, np.newaxis]

    chunks: list[np.ndarray] = []
    for start in range(0, spec.shape[0], max_chunk_frames):
        chunk = spec[start : start + max_chunk_frames]
        x = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
        logits = model(x)
        chunks.append(torch.sigmoid(logits)[0].cpu().numpy())
    if not chunks:
        return np.zeros((0, 8), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32)


@torch.no_grad()
def evaluate_prepared_chord_hybrid(
    model: torch.nn.Module,
    songs: list[PreparedChordHybridSong],
    *,
    device: torch.device,
    window_frames: int = 1000,
    stride_frames: int = 500,
    max_onsets: int = 256,
    vocab: ChordVocabulary | None = None,
) -> ChordHybridEvalReport:
    """Evaluate baseline vs hybrid chord-conditioned decoder performance.

    Computes per-class F-measure, CQS, and tom consistency for both
    baseline and hybrid models.

    Args:
        model: Chord-conditioned onset decoder model.
        songs: Pre-prepared hybrid evaluation songs.
        device: Torch device for inference.
        window_frames: Window size for the decoder.
        stride_frames: Stride between decoder windows.
        max_onsets: Maximum number of onsets per window.
        vocab: Chord vocabulary (built from default if None).

    Returns:
        A ChordHybridEvalReport comparing baseline and hybrid metrics.
    """
    model.eval()
    vocab = vocab or build_chord_vocabulary()

    baseline_f: list[list[float]] = [[] for _ in range(8)]
    hybrid_f: list[list[float]] = [[] for _ in range(8)]
    baseline_cqs_rows = []
    hybrid_cqs_rows = []
    baseline_tom_pairs = {"same": 0, "total": 0}
    hybrid_tom_pairs = {"same": 0, "total": 0}

    for song in songs:
        labels = np.load(str(song.entry.label_path))
        spec = np.load(str(song.entry.spec_path))
        if spec.ndim == 2:
            spec = spec[:, :, np.newaxis]
        t_frames = min(labels.shape[0], spec.shape[0])
        labels = labels[:t_frames]
        spec = spec[:t_frames]

        baseline_onsets = [
            (frame, class_idx)
            for frame, class_idx in song.baseline_onsets
            if 0 <= frame < t_frames
        ]
        baseline_features = np.asarray(
            [
                song.onset_features[idx]
                for idx, (frame, _class_idx) in enumerate(song.baseline_onsets)
                if 0 <= frame < t_frames
            ],
            dtype=np.float32,
        )
        hybrid_onsets = decode_chord_hybrid_onsets(
            model,
            baseline_onsets=baseline_onsets,
            onset_features=baseline_features,
            spec=spec,
            device=str(device),
            window_frames=window_frames,
            stride_frames=stride_frames,
            max_onsets=max_onsets,
            vocab=vocab,
        )

        baseline_picks = _onsets_to_picks(baseline_onsets)
        hybrid_picks = _onsets_to_picks(hybrid_onsets)
        baseline_cqs_rows.append(compute_cqs(baseline_picks, labels))
        hybrid_cqs_rows.append(compute_cqs(hybrid_picks, labels))

        for class_idx in range(8):
            gt = labels_to_frame_list(labels[:, class_idx])
            _bp, _br, baseline_score = fmeasure_with_tolerance(
                baseline_picks[class_idx],
                gt,
                tolerance_frames=2,
            )
            _hp, _hr, hybrid_score = fmeasure_with_tolerance(
                hybrid_picks[class_idx],
                gt,
                tolerance_frames=2,
            )
            baseline_f[class_idx].append(baseline_score)
            hybrid_f[class_idx].append(hybrid_score)

        _update_tom_consistency(baseline_onsets, baseline_tom_pairs)
        _update_tom_consistency(hybrid_onsets, hybrid_tom_pairs)

    baseline_per_class = {
        CLASS_NAMES_8[class_idx]: float(np.mean(scores)) if scores else 0.0
        for class_idx, scores in enumerate(baseline_f)
    }
    hybrid_per_class = {
        CLASS_NAMES_8[class_idx]: float(np.mean(scores)) if scores else 0.0
        for class_idx, scores in enumerate(hybrid_f)
    }
    baseline_cqs = aggregate_cqs(baseline_cqs_rows)
    hybrid_cqs = aggregate_cqs(hybrid_cqs_rows)
    return ChordHybridEvalReport(
        n_songs=len(songs),
        baseline_macro_f=(
            float(np.mean(list(baseline_per_class.values())))
            if baseline_per_class
            else 0.0
        ),
        hybrid_macro_f=(
            float(np.mean(list(hybrid_per_class.values())))
            if hybrid_per_class
            else 0.0
        ),
        baseline_per_class=baseline_per_class,
        hybrid_per_class=hybrid_per_class,
        baseline_tom_consistency=_tom_ratio(baseline_tom_pairs),
        hybrid_tom_consistency=_tom_ratio(hybrid_tom_pairs),
        baseline_cqs=float(baseline_cqs["cqs"]),
        hybrid_cqs=float(hybrid_cqs["cqs"]),
        baseline_cqs_components=baseline_cqs,
        hybrid_cqs_components=hybrid_cqs,
    )


def hybrid_selection_value(report: ChordHybridEvalReport, metric: str) -> float:
    """Extract a scalar selection value from an eval report.

    Args:
        report: Chord-hybrid evaluation report.
        metric: Name of the metric ("hybrid_macro_f" or "hybrid_cqs").

    Returns:
        The requested metric value.

    Raises:
        ValueError: If the metric is not supported.
    """
    if metric == "hybrid_macro_f":
        return report.hybrid_macro_f
    if metric == "hybrid_cqs":
        return report.hybrid_cqs
    raise ValueError(f"Unsupported hybrid selection metric: {metric}")


def _onsets_to_picks(onsets: list[tuple[int, int]]) -> dict[int, np.ndarray]:
    """Convert a flat onset list to per-class NumPy arrays of frame indices.

    Args:
        onsets: List of (frame, class_idx) onset events.

    Returns:
        Mapping from class index (0-7) to sorted int64 frame arrays.
    """
    per_class: dict[int, list[int]] = {class_idx: [] for class_idx in range(8)}
    for frame, class_idx in onsets:
        if 0 <= class_idx < 8:
            per_class[class_idx].append(int(frame))
    return {
        class_idx: np.asarray(sorted(frames), dtype=np.int64)
        for class_idx, frames in per_class.items()
    }


def _update_tom_consistency(onsets: list[tuple[int, int]], tracker: dict[str, int]) -> None:
    """Update a tom-consistency tracker by examining close-together tom hits.

    Counts consecutive tom events (classes 3, 5, 7) within 20 frames;
    when they share the same class it counts as a "same" pair.

    Args:
        onsets: Full list of (frame, class_idx) events.
        tracker: Dict with "same" and "total" keys to update in place.
    """
    tom_events = sorted(
        (frame, class_idx)
        for frame, class_idx in onsets
        if class_idx in {3, 5, 7}
    )
    for idx in range(len(tom_events) - 1):
        frame_0, class_0 = tom_events[idx]
        frame_1, class_1 = tom_events[idx + 1]
        if frame_1 - frame_0 <= 20:
            tracker["total"] += 1
            if class_0 == class_1:
                tracker["same"] += 1


def _tom_ratio(tracker: dict[str, int]) -> float:
    """Compute the tom-consistency ratio from a tracker dict.

    Args:
        tracker: Dict with "same" and "total" integer counts.

    Returns:
        Ratio of same-class tom pairs to total tom pairs.
    """
    return tracker["same"] / max(1, tracker["total"])
