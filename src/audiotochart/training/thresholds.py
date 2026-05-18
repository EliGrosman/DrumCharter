from __future__ import annotations

import numpy as np


def pick_peaks(activation: np.ndarray, threshold: float) -> np.ndarray:
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
    return np.flatnonzero(label_track > 0.5).astype(np.int64)


def optimize_thresholds(
    activations_per_class: list[np.ndarray],
    gt_per_class: list[np.ndarray],
    *,
    tolerance_frames: int = 2,
    grid: np.ndarray | None = None,
) -> tuple[list[float], list[float]]:
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
