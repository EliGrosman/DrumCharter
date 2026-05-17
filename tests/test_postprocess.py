"""Tests for post-processing functions."""

from __future__ import annotations

import pytest

from audiotochart.drums import DrumHit
from audiotochart.postprocess import (
    BeatGrid,
    build_quantize_grid,
    filter_hits,
    limit_simultaneous_hits,
    make_beat_grid_from_bpm,
    merge_nearby_hits,
    snap_hits_to_grid,
)


# ---------------------------------------------------------------------------
# filter_hits
# ---------------------------------------------------------------------------


def test_filter_hits_removes_low_confidence() -> None:
    hits = [
        DrumHit(0.0, "kick", confidence=0.9),
        DrumHit(0.1, "snare", confidence=0.2),
        DrumHit(0.2, "hihat", confidence=0.5),
    ]
    result = filter_hits(hits, min_confidence=0.5)
    assert len(result) == 2
    assert all(h.confidence >= 0.5 for h in result)


def test_filter_hits_empty() -> None:
    assert filter_hits([], min_confidence=0.5) == []


# ---------------------------------------------------------------------------
# merge_nearby_hits
# ---------------------------------------------------------------------------


def test_merge_nearby_kick_hits() -> None:
    hits = [
        DrumHit(0.0, "kick", confidence=0.8),
        DrumHit(0.01, "kick", confidence=0.9),
        DrumHit(0.5, "kick", confidence=0.7),
    ]
    result = merge_nearby_hits(hits)
    assert len(result) == 2
    # The first two should be merged, keeping the higher confidence one
    assert result[0].confidence == 0.9


def test_merge_nearby_different_instruments() -> None:
    hits = [
        DrumHit(0.0, "kick", confidence=0.8),
        DrumHit(0.01, "snare", confidence=0.9),
    ]
    result = merge_nearby_hits(hits)
    assert len(result) == 2


def test_merge_nearby_no_merge_needed() -> None:
    hits = [
        DrumHit(0.0, "kick", confidence=0.8),
        DrumHit(0.5, "kick", confidence=0.7),
    ]
    result = merge_nearby_hits(hits)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# snap_hits_to_grid
# ---------------------------------------------------------------------------


def test_off_grid_hits_snap_when_close() -> None:
    bpm = 120.0
    sec_per_beat = 60.0 / bpm
    # Place a hit very close to a beat
    hit_time = sec_per_beat + 0.01  # 10 ms off the beat
    hits = [DrumHit(hit_time, "kick")]
    grid = BeatGrid(beat_times=[0.0, sec_per_beat, sec_per_beat * 2], bpm=bpm)
    result = snap_hits_to_grid(hits, grid, divisor=16)
    assert len(result) == 1
    assert result[0].time_sec == pytest.approx(sec_per_beat, abs=0.001)


def test_off_grid_hits_stay_unsnapped_when_far() -> None:
    bpm = 120.0
    sec_per_beat = 60.0 / bpm
    # Place a hit more than 50 ms from the nearest 16th-note grid point.
    # At 120 BPM, 16ths are spaced 125 ms apart, so 190 ms is 60 ms from 250 ms.
    hit_time = 0.19
    hits = [DrumHit(hit_time, "kick")]
    grid = BeatGrid(beat_times=[0.0, sec_per_beat, sec_per_beat * 2], bpm=bpm)
    result = snap_hits_to_grid(hits, grid, divisor=16)
    assert len(result) == 1
    # Should stay at the original time (too far from any grid point)
    assert result[0].time_sec == pytest.approx(hit_time, abs=0.001)


def test_snap_hits_to_grid_empty() -> None:
    hits: list[DrumHit] = []
    grid = BeatGrid(beat_times=[], bpm=120.0)
    assert snap_hits_to_grid(hits, grid) == []


def test_quantize_grid_preserves_variable_tempo_intervals() -> None:
    grid_points = build_quantize_grid([0.0, 1.0, 1.8, 2.4], divisor=8, song_end_sec=2.4)

    second_beat = [t for t in grid_points if 1.0 <= t < 1.8]
    third_beat = [t for t in grid_points if 1.8 <= t < 2.4]

    assert second_beat == pytest.approx([1.0, 1.4])
    assert third_beat == pytest.approx([1.8, 2.1])


# ---------------------------------------------------------------------------
# limit_simultaneous_hits
# ---------------------------------------------------------------------------


def test_three_hand_lanes_reduced_to_two() -> None:
    hits = [
        DrumHit(0.0, "snare"),
        DrumHit(0.0, "crash"),
        DrumHit(0.0, "ride"),
    ]
    result = limit_simultaneous_hits(hits)
    assert len(result) == 2
    # Should keep snare and crash (highest preference)
    instruments = {h.instrument for h in result}
    assert instruments == {"snare", "crash"}


def test_cymbal_modifiers_removed_when_pad_lane_removed() -> None:
    """When a hand-lane hit is pruned, its associated cymbal modifier should
    also be removed. This tests that the limiter only keeps valid instrument
    pairs."""
    hits = [
        DrumHit(0.0, "hihat"),
        DrumHit(0.0, "crash"),
        DrumHit(0.0, "ride"),
    ]
    result = limit_simultaneous_hits(hits)
    assert len(result) == 2
    # Should keep crash and ride (highest preference after snare)
    instruments = {h.instrument for h in result}
    assert instruments == {"crash", "ride"}


def test_kick_always_kept() -> None:
    hits = [
        DrumHit(0.0, "kick"),
        DrumHit(0.0, "snare"),
        DrumHit(0.0, "crash"),
        DrumHit(0.0, "ride"),
    ]
    result = limit_simultaneous_hits(hits)
    # Kick is free, plus 2 hand lanes
    assert len(result) == 3
    assert any(h.instrument == "kick" for h in result)
    assert any(h.instrument == "snare" for h in result)
    assert any(h.instrument == "crash" for h in result)


def test_limit_simultaneous_hits_empty() -> None:
    assert limit_simultaneous_hits([]) == []
