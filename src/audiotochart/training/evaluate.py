"""Evaluation of frame-level drum transcription models.

Collects per-song activations, computes F-measure at multiple tolerances,
and formats human-readable reports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from audiotochart.training.dataset import SongEntry
from audiotochart.training.model import forward_logits
from audiotochart.training.thresholds import (
    fmeasure_with_tolerance,
    labels_to_frame_list,
    pick_peaks,
)

log = logging.getLogger(__name__)

CLASS_NAMES_8 = ["kick", "snare", "hihat", "y_tom", "ride", "b_tom", "crash", "f_tom"]


@dataclass
class EvalReport:
    """Evaluation report containing per-class and macro F-measure scores.

    Attributes:
        num_classes: Number of drum classes evaluated.
        tolerance_ms_primary: Primary tolerance in milliseconds.
        tolerance_ms_secondary: Secondary tolerance in milliseconds.
        per_class_primary: (precision, recall, f) per class at primary tolerance.
        per_class_secondary: (precision, recall, f) per class at secondary tolerance.
        macro_primary: Macro-averaged (precision, recall, f) at primary tolerance.
        macro_secondary: Macro-averaged (precision, recall, f) at secondary tolerance.
        num_songs: Number of songs evaluated.
        thresholds: Per-class peak-picking thresholds used.
    """

    num_classes: int
    tolerance_ms_primary: int
    tolerance_ms_secondary: int
    per_class_primary: list[tuple[float, float, float]] = field(default_factory=list)
    per_class_secondary: list[tuple[float, float, float]] = field(default_factory=list)
    macro_primary: tuple[float, float, float] = (0.0, 0.0, 0.0)
    macro_secondary: tuple[float, float, float] = (0.0, 0.0, 0.0)
    num_songs: int = 0
    thresholds: list[float] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize the report to a JSON string."""
        return json.dumps(asdict(self), indent=2)

    def save(self, path: Path) -> None:
        """Write the JSON report to a file, creating parent directories if needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())


def _macro(triples: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Compute macro-averaged precision, recall, and F-measure over triples.

    Args:
        triples: List of (precision, recall, f) tuples.

    Returns:
        Mean (precision, recall, f) across all provided triples.
    """
    if not triples:
        return 0.0, 0.0, 0.0
    p = sum(t[0] for t in triples) / len(triples)
    r = sum(t[1] for t in triples) / len(triples)
    f = sum(t[2] for t in triples) / len(triples)
    return p, r, f


@torch.no_grad()
def collect_song_activations(
    model: torch.nn.Module,
    entries: list[SongEntry],
    *,
    device: str = "cuda",
    max_chunk_frames: int = 2000,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run a frame model over all song entries and collect sigmoid activations.

    Processes each spectrogram in chunks to avoid OOM on long songs.

    Args:
        model: Frame-level transcription model.
        entries: Song entries to process.
        device: Device string for inference.
        max_chunk_frames: Maximum frames per chunk.

    Returns:
        List of (activations, labels) tuples with truncated to matching lengths.
    """
    model.eval()
    out: list[tuple[np.ndarray, np.ndarray]] = []

    for entry in entries:
        spec = np.load(str(entry.spec_path))
        labels = np.load(str(entry.label_path))

        T = min(spec.shape[0], labels.shape[0])
        spec = spec[:T]
        labels = labels[:T]

        if spec.ndim == 2:
            spec = spec[:, :, np.newaxis]

        acts_chunks: list[np.ndarray] = []
        for start in range(0, T, max_chunk_frames):
            chunk = spec[start : start + max_chunk_frames]
            x = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
            logits = forward_logits(model, x)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            acts_chunks.append(probs)

        acts = (
            np.concatenate(acts_chunks, axis=0)
            if acts_chunks
            else np.zeros((0, labels.shape[1]), dtype=np.float32)
        )
        out.append((acts.astype(np.float32), labels.astype(np.float32)))

    return out


def evaluate(
    model: torch.nn.Module,
    entries: list[SongEntry],
    thresholds: list[float],
    *,
    fps: int = 100,
    tolerance_ms_primary: int = 20,
    tolerance_ms_secondary: int = 30,
    device: str = "cuda",
) -> EvalReport:
    """Run a full evaluation of a frame model against song entries.

    Peak-picks at the given thresholds then computes F-measure at
    both primary and secondary tolerances for every class.

    Args:
        model: Frame-level transcription model.
        entries: Song entries to evaluate.
        thresholds: Per-class peak-picking thresholds.
        fps: Frames per second for tolerance conversion.
        tolerance_ms_primary: Primary tolerance in milliseconds.
        tolerance_ms_secondary: Secondary tolerance in milliseconds.
        device: Device string for inference.

    Returns:
        An EvalReport with per-class and macro metrics.
    """
    num_classes = len(thresholds)
    tol_primary = max(1, round(tolerance_ms_primary * fps / 1000))
    tol_secondary = max(1, round(tolerance_ms_secondary * fps / 1000))

    pairs = collect_song_activations(model, entries, device=device)

    primary_triples: list[list[tuple[float, float, float]]] = [[] for _ in range(num_classes)]
    secondary_triples: list[list[tuple[float, float, float]]] = [[] for _ in range(num_classes)]

    for acts, labels in pairs:
        for c in range(num_classes):
            picks = pick_peaks(acts[:, c], thresholds[c])
            gt = labels_to_frame_list(labels[:, c])
            primary_triples[c].append(fmeasure_with_tolerance(picks, gt, tolerance_frames=tol_primary))
            secondary_triples[c].append(fmeasure_with_tolerance(picks, gt, tolerance_frames=tol_secondary))

    per_class_primary = [_macro(triples) for triples in primary_triples]
    per_class_secondary = [_macro(triples) for triples in secondary_triples]

    return EvalReport(
        num_classes=num_classes,
        tolerance_ms_primary=tolerance_ms_primary,
        tolerance_ms_secondary=tolerance_ms_secondary,
        per_class_primary=per_class_primary,
        per_class_secondary=per_class_secondary,
        macro_primary=_macro(per_class_primary),
        macro_secondary=_macro(per_class_secondary),
        num_songs=len(entries),
        thresholds=list(thresholds),
    )


def format_report(report: EvalReport, class_names: list[str] | None = None) -> str:
    """Format an EvalReport into a human-readable multi-line string.

    Args:
        report: The evaluation report to format.
        class_names: Optional list of class display names.

    Returns:
        A formatted multi-line string with macro and per-class metrics.
    """
    lines = []
    lines.append(
        f"Eval over {report.num_songs} songs, "
        f"primary tolerance ±{report.tolerance_ms_primary}ms, "
        f"secondary ±{report.tolerance_ms_secondary}ms"
    )
    lines.append(
        f"  Macro F  primary={report.macro_primary[2]:.4f}  "
        f"secondary={report.macro_secondary[2]:.4f}"
    )
    lines.append("  Per-class (P / R / F at primary tolerance):")
    for c, (p, r, f) in enumerate(report.per_class_primary):
        name = class_names[c] if class_names and c < len(class_names) else f"class_{c}"
        thr = report.thresholds[c] if c < len(report.thresholds) else float("nan")
        lines.append(f"    {c} {name:<14} thr={thr:.2f}  P={p:.3f}  R={r:.3f}  F={f:.3f}")
    return "\n".join(lines)