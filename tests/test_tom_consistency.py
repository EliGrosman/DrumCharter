from __future__ import annotations

import numpy as np

from drumcharter.inference.tom_consistency import apply_tom_consistency


def test_tom_consistency_empty_input() -> None:
    acts = np.zeros((100, 8), dtype=np.float32)
    onsets, stats = apply_tom_consistency([], acts)
    assert onsets == []
    assert stats["n_tom_hits"] == 0


def test_tom_consistency_no_tom_hits() -> None:
    """Hits without any tom classes should pass through unchanged."""
    acts = np.zeros((100, 8), dtype=np.float32)
    onsets = [(0.1, 0), (0.2, 1), (0.3, 2)]  # kick, snare, hihat
    result, stats = apply_tom_consistency(onsets, acts)
    assert result == onsets
    assert stats["n_tom_hits"] == 0


def test_tom_consistency_few_tom_hits_bails_out() -> None:
    """Fewer than 3 tom hits should not trigger consistency logic."""
    acts = np.zeros((100, 8), dtype=np.float32)
    onsets = [(0.1, 3), (0.2, 5)]  # only 2 tom hits
    result, stats = apply_tom_consistency(onsets, acts)
    assert result == onsets
    assert stats["n_tom_hits"] == 2


def test_tom_consistency_few_anchors_bails_out() -> None:
    """Fewer than min_anchors (5) high-confidence anchors: no change."""
    acts = np.zeros((100, 8), dtype=np.float32)
    # 6 tom hits, all low confidence (< 0.7)
    acts[10, 3] = 0.1
    acts[20, 5] = 0.2
    acts[30, 7] = 0.3
    acts[40, 3] = 0.15
    acts[50, 5] = 0.25
    acts[60, 7] = 0.35
    onsets = [(0.1, 3), (0.2, 5), (0.3, 7), (0.4, 3), (0.5, 5), (0.6, 7)]

    result, stats = apply_tom_consistency(onsets, acts)
    assert result == onsets
    assert stats["n_tom_hits"] == 6
    assert stats["n_anchors"] == 0


def test_tom_consistency_reassigns_low_confidence_tom() -> None:
    """Low-confidence tom hits outside the convention should be reassigned
    to the best-matching convention class."""
    T = 200
    acts = np.zeros((T, 8), dtype=np.float32)

    # Establish y_tom (class 3) as convention: 10 high-confidence y_tom anchors
    for i in range(10):
        frame = i * 10 + 5
        acts[frame, 3] = 0.85  # y_tom anchor
        acts[frame, 5] = 0.1
        acts[frame, 7] = 0.1

    # One low-confidence b_tom (class 5) — should be reassigned, and when there
    # is no strong activation in the convention classes at this frame (acts[195, 3] = 0),
    # the pick among convention classes picks the one with highest activation.
    # Since all are 0, max returns the first in iteration order, which depends on
    # Python's dict ordering but is guaranteed to be deterministic.
    acts[195, 5] = 0.01

    onsets = [
        (0.05, 3), (0.15, 3), (0.25, 3), (0.35, 3), (0.45, 3),
        (0.55, 3), (0.65, 3), (0.75, 3), (0.85, 3), (0.95, 3),
        (1.95, 5),  # low-confidence b_tom — should be reassigned
    ]

    result, stats = apply_tom_consistency(onsets, acts)

    assert stats["n_reassigned"] == 1
    assert stats["n_tom_hits"] == 11
    assert stats.get("convention") == [3]
    # The b_tom should be reassigned to the convention class
    assert result[-1][1] == 3


def test_tom_consistency_uses_supplied_fps() -> None:
    fps = 20
    acts = np.zeros((50, 8), dtype=np.float32)

    anchor_onsets = []
    for i in range(10):
        t = 0.05 + i * 0.1
        frame = int(round(t * fps))
        acts[frame, 3] = 0.85
        anchor_onsets.append((t, 3))

    acts[39, 5] = 0.01
    onsets = anchor_onsets + [(1.95, 5)]

    result, stats = apply_tom_consistency(onsets, acts, fps=fps)

    assert stats["n_reassigned"] == 1
    assert result[-1][1] == 3


def test_tom_consistency_preserves_fills() -> None:
    """Rapid tom clusters (drum fills) should not be reassigned."""
    fps = 100
    T = 200
    acts = np.zeros((T, 8), dtype=np.float32)

    # Convention: y_tom (class 3)
    for i in range(10):
        frame = i * 10 + 5
        acts[frame, 3] = 0.85

    # A rapid fill: 5 tom hits in quick succession (0.12s apart)
    fill_start = 1.5
    fill_onsets = []
    for i in range(5):
        t = fill_start + i * 0.12
        frame = int(round(t * fps))
        acts[frame, 5] = 0.01  # b_tom, low confidence
        fill_onsets.append((t, 5))

    onsets = [
        (0.05, 3), (0.15, 3), (0.25, 3), (0.35, 3), (0.45, 3),
        (0.55, 3), (0.65, 3), (0.75, 3), (0.85, 3), (0.95, 3),
    ] + fill_onsets

    result, stats = apply_tom_consistency(onsets, acts)

    # Fill hits should NOT be reassigned
    for t, c in fill_onsets:
        assert (t, c) in result, f"Fill hit ({t}, {c}) was reassigned"


def test_tom_consistency_keeps_high_confidence_outliers() -> None:
    """High-confidence non-convention tom hits should be preserved."""
    T = 200
    acts = np.zeros((T, 8), dtype=np.float32)

    # Convention: y_tom (class 3)
    for i in range(10):
        frame = i * 10 + 5
        acts[frame, 3] = 0.85

    # A high-confidence b_tom — should be kept even though not in convention
    acts[150, 5] = 0.9

    onsets = [
        (0.05, 3), (0.15, 3), (0.25, 3), (0.35, 3), (0.45, 3),
        (0.55, 3), (0.65, 3), (0.75, 3), (0.85, 3), (0.95, 3),
        (1.50, 5),
    ]

    result, stats = apply_tom_consistency(onsets, acts)

    assert stats["n_reassigned"] == 0
    assert (1.50, 5) in result


def test_tom_consistency_all_three_classes_anchored_bails() -> None:
    """If anchors span all three tom classes, no convention can be
    established — bail out."""
    T = 200
    acts = np.zeros((T, 8), dtype=np.float32)

    for i in range(6):
        acts[i * 10 + 5, 3] = 0.85  # y_tom
        acts[i * 10 + 5, 5] = 0.85  # b_tom
        acts[i * 10 + 5, 7] = 0.85  # f_tom

    onsets = [
        (0.05, 3), (0.05, 5), (0.05, 7),
        (0.15, 3), (0.15, 5), (0.15, 7),
        (0.25, 3), (0.25, 5), (0.25, 7),
        (0.35, 3), (0.35, 5), (0.35, 7),
        (0.45, 3), (0.45, 5), (0.45, 7),
        (0.55, 3), (0.55, 5), (0.55, 7),
    ]

    result, stats = apply_tom_consistency(onsets, acts)
    assert result == onsets
    assert stats["convention"] == [3, 5, 7]
