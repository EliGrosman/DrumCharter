"""Song-level tom drum consistency post-processing.

Ensures that low-confidence tom sub-classes (high/mid/low toms) are
reassigned to match the dominant tom convention detected across the
entire song. This prevents the model from flipping tom assignments
mid-song.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

TOM_CLASSES = frozenset({3, 5, 7})
"""Indexes of tom drum classes (high=3, mid=5, low=7) in the 8-class output."""

FPS = 100
"""Feature frames per second used for frame-to-time conversion."""

ANCHOR_THRESHOLD = 0.7
"""Activation threshold for a tom hit to be considered an anchor."""

MIN_ANCHORS = 5
"""Minimum number of anchor hits required to establish a convention."""

MIN_ANCHOR_FRACTION = 0.15
"""Minimum fraction of anchors a class must represent to be in the convention."""

REASSIGN_THRESHOLD = 0.5
"""Activation threshold below which a non-convention tom hit is reassigned."""

FILL_IOI_MAX = 0.18
"""Maximum inter-onset-interval (seconds) for fill detection."""

FILL_MIN_HITS = 4
"""Minimum tom hits in a row for a sequence to be a fill (skipped by reassign)."""


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
    """Song-level tom consistency post-processing.

    Identifies the dominant tom convention from high-confidence ``anchor``
    hits, then reassigns low-confidence tom hits outside of fills to match
    that convention.

    Args:
        onsets: List of ``(time_sec, class_index)`` pairs.
        acts: Frame-level activations array of shape ``(frames, 8)``.
        fps: Feature frames per second. Defaults to 100.
        anchor_threshold: Minimum activation for a tom hit to be an anchor.
        min_anchors: Minimum anchor count to establish a convention.
        min_anchor_fraction: Min fraction of anchors a class needs for convention.
        reassign_threshold: Activation threshold for skipping reassignment.
        fill_ioi_max: Max seconds between hits for fill detection.
        fill_min_hits: Min consecutive hits for a fill region.

    Returns:
        A tuple of ``(modified_onsets, stats_dict)``.
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
