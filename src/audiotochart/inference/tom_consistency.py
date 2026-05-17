from __future__ import annotations

from collections import Counter

import numpy as np

TOM_CLASSES = frozenset({3, 5, 7})
FPS = 100

ANCHOR_THRESHOLD = 0.7
MIN_ANCHORS = 5
MIN_ANCHOR_FRACTION = 0.15
REASSIGN_THRESHOLD = 0.5
FILL_IOI_MAX = 0.18
FILL_MIN_HITS = 4


def apply_tom_consistency(
    onsets: list[tuple[float, int]],
    acts: np.ndarray,
    *,
    fps: float = FPS,
    anchor_threshold: float = ANCHOR_THRESHOLD,
    min_anchors: int = MIN_ANCHORS,
    min_anchor_fraction: float = MIN_ANCHOR_FRACTION,
    reassign_threshold: float = REASSIGN_THRESHOLD,
    fill_ioi_max: float = FILL_IOI_MAX,
    fill_min_hits: int = FILL_MIN_HITS,
) -> tuple[list[tuple[float, int]], dict[str, int]]:
    """Song-level tom consistency: reassign low-confidence tom sub-classes
    to match the song's dominant tom convention.

    Returns the (possibly modified) onset list and a stats dict for logging.
    """
    stats: dict[str, int | list[int]] = {
        "n_tom_hits": 0,
        "n_anchors": 0,
        "n_reassigned": 0,
        "convention": [],
    }

    tom_data: list[tuple[int, float, int]] = []
    for i, (t, c) in enumerate(onsets):
        if c in TOM_CLASSES:
            tom_data.append((i, t, c))
    stats["n_tom_hits"] = len(tom_data)
    if len(tom_data) < 3 or acts.shape[0] == 0:
        return onsets, stats  # type: ignore[return-value]

    anchors: list[tuple[int, float, int]] = []
    for idx, t, c in tom_data:
        frame = min(int(round(t * fps)), acts.shape[0] - 1)
        if float(acts[frame, c]) >= anchor_threshold:
            anchors.append((idx, t, c))
    stats["n_anchors"] = len(anchors)
    if len(anchors) < min_anchors:
        return onsets, stats  # type: ignore[return-value]

    anchor_counts = Counter(c for _, _, c in anchors)
    total_anchors = sum(anchor_counts.values())
    convention = {c for c, n in anchor_counts.items() if n / total_anchors >= min_anchor_fraction}
    stats["convention"] = sorted(convention)
    if not convention or convention == TOM_CLASSES:
        return onsets, stats  # type: ignore[return-value]

    fill_regions: list[tuple[float, float]] = []
    i = 0
    while i < len(tom_data):
        j = i + 1
        while j < len(tom_data) and tom_data[j][1] - tom_data[j - 1][1] < fill_ioi_max:
            j += 1
        if j - i >= fill_min_hits:
            fill_regions.append((tom_data[i][1], tom_data[j - 1][1]))
        i = max(j, i + 1)

    def _in_fill(t: float) -> bool:
        return any(start <= t <= end for start, end in fill_regions)

    new_onsets = list(onsets)
    for idx, t, c in tom_data:
        if c in convention:
            continue
        frame = min(int(round(t * fps)), acts.shape[0] - 1)
        if float(acts[frame, c]) >= reassign_threshold:
            continue
        if _in_fill(t):
            continue
        best_c = max(convention, key=lambda cc: float(acts[frame, cc]))
        new_onsets[idx] = (t, best_c)
        stats["n_reassigned"] += 1

    return new_onsets, stats  # type: ignore[return-value]
