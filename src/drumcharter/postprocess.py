"""Post-processing for raw drum predictions before chart conversion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from drumcharter.chart.drum_vocab import HAND_INSTRUMENT_PRIORITY, KICK_LABEL
from drumcharter.drums import DrumHit

QUANTIZE_DIVISORS = (4, 8, 16, 32)
QUANTIZE_CHOICES = {
    "none": None,
    **{f"1/{divisor}": divisor for divisor in QUANTIZE_DIVISORS},
}
DEFAULT_SNAP_DISTANCE_SEC = 0.05


@dataclass(frozen=True)
class BeatGrid:
    """A simple beat grid: list of beat times in seconds."""

    beat_times: Sequence[float]
    bpm: float


def make_beat_grid_from_bpm(bpm: float, start_time: float = 0.0, duration_sec: float = 0.0) -> BeatGrid:
    """Create a regular beat grid from a BPM value."""
    if duration_sec <= start_time:
        return BeatGrid(beat_times=[], bpm=bpm)
    sec_per_beat = 60.0 / bpm
    num_beats = int((duration_sec - start_time) / sec_per_beat) + 2
    beats = [start_time + i * sec_per_beat for i in range(num_beats)]
    return BeatGrid(beat_times=beats, bpm=bpm)


def normalize_beat_times(beat_times: Sequence[float] | None) -> list[float]:
    """Return sorted unique beat times as plain floats."""
    if beat_times is None:
        return []

    beats = sorted(float(t) for t in beat_times)
    out: list[float] = []
    for beat in beats:
        if not out or abs(beat - out[-1]) > 1e-9:
            out.append(beat)
    return out


def build_quantize_grid(
    beat_times: Sequence[float],
    *,
    divisor: int,
    song_end_sec: float | None = None,
) -> list[float]:
    """Build musical grid points from actual beat positions.

    ``divisor=16`` means sixteenth notes: four subdivisions inside each
    quarter-note beat interval. The grid uses the detected inter-beat intervals
    directly, so tempo changes are preserved instead of collapsed into average
    BPM spacing.
    """
    if divisor not in QUANTIZE_DIVISORS:
        raise ValueError(f"divisor must be one of {QUANTIZE_DIVISORS}, got {divisor}")

    beats = normalize_beat_times(beat_times)
    if len(beats) < 2:
        return []

    sub_per_beat = divisor // 4
    intervals = [beats[i + 1] - beats[i] for i in range(len(beats) - 1)]
    positive = [interval for interval in intervals if interval > 1e-9]
    if not positive:
        return beats

    first_ibi = intervals[0] if intervals[0] > 1e-9 else positive[0]
    last_ibi = intervals[-1] if intervals[-1] > 1e-9 else positive[-1]

    grid: list[float] = []

    pre_beat = beats[0] - first_ibi
    while pre_beat >= -first_ibi:
        for index in range(sub_per_beat):
            time_sec = pre_beat + index * (first_ibi / sub_per_beat)
            if time_sec >= 0:
                grid.append(time_sec)
        pre_beat -= first_ibi

    for index, beat_start in enumerate(beats[:-1]):
        beat_end = beats[index + 1]
        step = (beat_end - beat_start) / sub_per_beat
        for subdivision in range(sub_per_beat):
            grid.append(beat_start + subdivision * step)
    grid.append(beats[-1])

    target_end = song_end_sec if song_end_sec is not None else beats[-1] + last_ibi
    post_beat = beats[-1] + last_ibi
    while post_beat <= target_end + last_ibi:
        for index in range(sub_per_beat):
            grid.append(post_beat + index * (last_ibi / sub_per_beat))
        post_beat += last_ibi

    ordered = sorted(grid)
    unique: list[float] = []
    for time_sec in ordered:
        if not unique or abs(time_sec - unique[-1]) > 1e-9:
            unique.append(time_sec)
    return unique


def snap_hits_to_grid(
    hits: list[DrumHit],
    beat_grid: BeatGrid,
    divisor: int = 16,
    max_distance_sec: float = DEFAULT_SNAP_DISTANCE_SEC,
) -> list[DrumHit]:
    """Snap hit times to the nearest grid subdivision.

    Only snaps if the hit is within ``max_distance_sec`` of a grid point.
    Hits too far from any grid point are left unsnapped.
    """
    if not hits:
        return hits

    song_end_sec = max(h.time_sec for h in hits)
    grid_points = build_quantize_grid(
        beat_grid.beat_times,
        divisor=divisor,
        song_end_sec=song_end_sec,
    )
    if not grid_points:
        return hits

    def _snap_time(time_sec: float) -> float:
        best_diff = float("inf")
        best_grid_point = time_sec
        for grid_point in grid_points:
            diff = abs(time_sec - grid_point)
            if diff < best_diff:
                best_diff = diff
                best_grid_point = grid_point
        if best_diff <= max_distance_sec:
            return best_grid_point
        return time_sec

    return [
        DrumHit(
            time_sec=_snap_time(h.time_sec),
            instrument=h.instrument,
            confidence=h.confidence,
        )
        for h in hits
    ]


# Preference order for resolving conflicts (highest preference first)
_HAND_PREF = list(HAND_INSTRUMENT_PRIORITY)


def limit_simultaneous_hits(hits: list[DrumHit]) -> list[DrumHit]:
    """Limit simultaneous notes to playable combinations.

    Rules:
    - Kick is always free (no hand lane limit)
    - Max two hand-lane instruments per tick (30 ms window)
    - When over limit, prefer snare > crash > ride > hihat > toms
    """
    if not hits:
        return []

    # Group hits by tick (30 ms window)
    sorted_hits = sorted(hits, key=lambda h: h.time_sec)

    # Group into time windows
    windows: list[list[DrumHit]] = []
    for hit in sorted_hits:
        placed = False
        for window in windows:
            # Check if this hit falls within the window
            if any(abs(hit.time_sec - w.time_sec) < 0.03 for w in window):
                window.append(hit)
                placed = True
                break
        if not placed:
            windows.append([hit])

    result: list[DrumHit] = []
    for window in windows:
        # Separate kick from hand-lane hits
        kicks = [h for h in window if h.instrument == KICK_LABEL]
        hands = [h for h in window if h.instrument != KICK_LABEL]

        # Sort hand hits by preference (highest first)
        hands.sort(key=lambda h: _HAND_PREF.index(h.instrument) if h.instrument in _HAND_PREF else 999)

        # Keep at most 2 hand-lane hits
        kept_hands = hands[:2]

        result.extend(kicks)
        result.extend(kept_hands)

    return result
