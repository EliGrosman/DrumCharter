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
    entry: SongEntry
    baseline_onsets: tuple[tuple[int, int], ...]
    onset_features: np.ndarray


@dataclass(frozen=True)
class ChordHybridEvalReport:
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
    if metric == "hybrid_macro_f":
        return report.hybrid_macro_f
    if metric == "hybrid_cqs":
        return report.hybrid_cqs
    raise ValueError(f"Unsupported hybrid selection metric: {metric}")


def _onsets_to_picks(onsets: list[tuple[int, int]]) -> dict[int, np.ndarray]:
    per_class: dict[int, list[int]] = {class_idx: [] for class_idx in range(8)}
    for frame, class_idx in onsets:
        if 0 <= class_idx < 8:
            per_class[class_idx].append(int(frame))
    return {
        class_idx: np.asarray(sorted(frames), dtype=np.int64)
        for class_idx, frames in per_class.items()
    }


def _update_tom_consistency(onsets: list[tuple[int, int]], tracker: dict[str, int]) -> None:
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
    return tracker["same"] / max(1, tracker["total"])
