"""Peak-picking, F-measure evaluation, and threshold optimisation for drum transcription.

Provides utilities for converting continuous activations into discrete
onset predictions and for grid-searching optimal per-class thresholds.
"""

from __future__ import annotations

import numpy as np


def pick_peaks(activation: np.ndarray, threshold: float) -> np.ndarray:
    """Find local-maximum peak frames above a threshold in a 1-D activation array.

    Args:
        activation: 1-D array of frame-level activations.
        threshold: Minimum activation value for a peak.

    Returns:
        Int64 array of frame indices where peaks occur.
    """
    if activation.size == 0:
        return np.empty(0, dtype=np.int64)

    left = np.concatenate(([-np.inf], activation[:-1]))
    right = np.concatenate((activation[1:], [-np.inf]))
    is_peak = (activation > left) & (activation >= right) & (activation >= threshold)
    return np.flatnonzero(is_peak).astype(np.int64)


def fmeasure_with_tolerance(
    pred_frames: np.ndarray,
    gt_frames: np.ndarray,
    *,
    tolerance_frames: int,
) -> tuple[float, float, float]:
    """Compute precision, recall, and F-measure with frame-level tolerance.

    Each predicted frame is matched to at most one ground-truth frame
    within the given tolerance using a greedy nearest-neighbour strategy.

    Args:
        pred_frames: 1-D array of predicted frame indices.
        gt_frames: 1-D array of ground-truth frame indices.
        tolerance_frames: Maximum frame difference for a valid match.

    Returns:
        Tuple of (precision, recall, f_measure).
    """
    pred_frames = np.asarray(pred_frames, dtype=np.int64)
    gt_frames = np.asarray(gt_frames, dtype=np.int64)
    if pred_frames.ndim == 0:
        pred_frames = pred_frames.reshape(1)
    if gt_frames.ndim == 0:
        gt_frames = gt_frames.reshape(1)

    if pred_frames.size == 0 and gt_frames.size == 0:
        return 1.0, 1.0, 1.0
    if pred_frames.size == 0:
        return 0.0, 0.0, 0.0
    if gt_frames.size == 0:
        return 0.0, 0.0, 0.0

    gt_sorted = np.sort(gt_frames)
    used = np.zeros(gt_sorted.size, dtype=bool)
    tp = 0

    for p in np.sort(pred_frames):
        idx = np.searchsorted(gt_sorted, p)
        candidates = []
        if idx < gt_sorted.size:
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        best = -1
        best_dist = tolerance_frames + 1
        for c in candidates:
            if used[c]:
                continue
            d = abs(int(gt_sorted[c]) - int(p))
            if d <= tolerance_frames and d < best_dist:
                best = c
                best_dist = d
        if best >= 0:
            used[best] = True
            tp += 1

    fp = pred_frames.size - tp
    fn = gt_sorted.size - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f


def labels_to_frame_list(label_track: np.ndarray) -> np.ndarray:
    """Convert a binary label track to an array of frame indices.

    Args:
        label_track: 1-D array with 1.0 at onset frames.

    Returns:
        Int64 array of frame indices where the track exceeds 0.5.
    """
    return np.flatnonzero(label_track > 0.5).astype(np.int64)


def optimize_thresholds(
    activations_per_class: list[np.ndarray],
    gt_per_class: list[np.ndarray],
    *,
    tolerance_frames: int = 2,
    grid: np.ndarray | None = None,
) -> tuple[list[float], list[float]]:
    """Grid-search per-class thresholds to maximise F-measure.

    Args:
        activations_per_class: Per-class concatenated activation arrays.
        gt_per_class: Per-class ground-truth frame indices (offset-adjusted).
        tolerance_frames: Frame tolerance for F-measure computation.
        grid: 1-D array of candidate thresholds. Defaults to arange(0.10, 0.61, 0.02).

    Returns:
        Tuple of (best_thresholds, best_f_scores) for each class.
    """
    if grid is None:
        grid = np.arange(0.10, 0.61, 0.02)

    num_classes = len(activations_per_class)
    best_thresholds: list[float] = []
    best_scores: list[float] = []

    for c in range(num_classes):
        act = activations_per_class[c]
        gt = gt_per_class[c]

        f_curve = np.empty(len(grid), dtype=np.float64)
        for i, t in enumerate(grid):
            picks = pick_peaks(act, float(t))
            _, _, f = fmeasure_with_tolerance(picks, gt, tolerance_frames=tolerance_frames)
            f_curve[i] = f

        best_idx = int(np.argmax(f_curve))
        best_thresholds.append(float(grid[best_idx]))
        best_scores.append(float(f_curve[best_idx]))

    return best_thresholds, best_scores