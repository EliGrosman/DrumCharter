"""Convert drum hits from various sources into Clone Hero chart documents."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby

from drumcharter.chart.format import (
    ChartDocument,
    DrumDifficulty,
    DrumNote,
    SyncTrackEvent,
    bpm_to_chart_integer,
)
from drumcharter.chart.drum_vocab import (
    CYMBAL_BY_PAD,
    HAND_LANE_PRIORITY,
    HAND_PAD_NOTES,
    INSTRUMENT_TO_CHART_NOTES,
    KICK,
    MAX_HAND_LANES,
)
from drumcharter.drums import DrumHit
from drumcharter.postprocess import (
    BeatGrid,
    make_beat_grid_from_bpm,
    normalize_beat_times,
    limit_simultaneous_hits,
    snap_hits_to_grid,
)


def seconds_to_tick(
    time_sec: float,
    bpm: float,
    resolution: int,
    snap_ticks: int = 1,
) -> int:
    """Convert a time in seconds to a chart tick at the given BPM and resolution.

    Args:
        time_sec: Time in seconds to convert.
        bpm: Tempo in beats per minute.
        resolution: Ticks per beat (e.g. 192).
        snap_ticks: Snap to the nearest multiple of this value.

    Returns:
        The chart tick corresponding to *time_sec*.
    """
    sec_per_beat = 60.0 / bpm
    sec_per_tick = sec_per_beat / float(resolution)
    if snap_ticks < 1:
        snap_ticks = 1
    tick = round(time_sec / sec_per_tick / snap_ticks)
    return int(max(0, tick * snap_ticks))


@dataclass(frozen=True)
class BeatTempoMap:
    """Beat-aligned tempo map used for sync-track emission and tick conversion.

    Attributes:
        beat_times: Detected beat positions in seconds.
        beat_ticks: Corresponding tick positions for each beat.
        first_interval_sec: Duration of the first beat interval in seconds.
        last_interval_sec: Duration of the last beat interval in seconds.
        resolution: Ticks per beat.
    """

    beat_times: list[float]
    beat_ticks: list[int]
    first_interval_sec: float
    last_interval_sec: float
    resolution: int


def build_beat_tempo_map(
    beat_times: Sequence[float] | None,
    *,
    resolution: int,
) -> BeatTempoMap | None:
    """Build a variable-tempo map from detected beat positions.

    Computes beat intervals from consecutive detected beat times and maps
    them to chart-relative ticks at the given resolution.

    Args:
        beat_times: Detected beat positions in seconds, or None.
        resolution: Ticks per beat.

    Returns:
        A :class:`BeatTempoMap` if at least two valid beats exist, or None.
    """
    beats = normalize_beat_times(beat_times)
    if len(beats) < 2:
        return None

    intervals = [beats[i + 1] - beats[i] for i in range(len(beats) - 1)]
    positive = [interval for interval in intervals if interval > 1e-9]
    if not positive:
        return None

    first_interval = intervals[0] if intervals[0] > 1e-9 else positive[0]
    last_interval = intervals[-1] if intervals[-1] > 1e-9 else positive[-1]
    first_tick = max(0, int(round(beats[0] / first_interval * resolution)))
    beat_ticks = [first_tick + i * resolution for i in range(len(beats))]

    return BeatTempoMap(
        beat_times=beats,
        beat_ticks=beat_ticks,
        first_interval_sec=first_interval,
        last_interval_sec=last_interval,
        resolution=resolution,
    )


def seconds_to_tick_tempo_map(time_sec: float, tempo_map: BeatTempoMap) -> int:
    """Convert seconds to ticks using a beat-derived variable-tempo map.

    Uses linear interpolation between beat boundaries when the tempo
    varies across beats.

    Args:
        time_sec: Time in seconds to convert.
        tempo_map: A :class:`BeatTempoMap` built from detected beats.

    Returns:
        The chart tick corresponding to *time_sec*.
    """
    time_sec = float(time_sec)
    if time_sec <= 0:
        return 0

    beats = tempo_map.beat_times
    ticks = tempo_map.beat_ticks
    resolution = tempo_map.resolution

    if time_sec < beats[0]:
        tick = ticks[0] + ((time_sec - beats[0]) / tempo_map.first_interval_sec) * resolution
        return int(max(0, round(tick)))

    index = 0
    while index + 1 < len(beats) and beats[index + 1] <= time_sec:
        index += 1

    if index >= len(beats) - 1:
        tick = ticks[-1] + ((time_sec - beats[-1]) / tempo_map.last_interval_sec) * resolution
        return int(max(0, round(tick)))

    interval = beats[index + 1] - beats[index]
    if interval <= 1e-9:
        return int(max(0, ticks[index]))
    tick = ticks[index] + ((time_sec - beats[index]) / interval) * resolution
    return int(max(0, round(tick)))


def tick_to_seconds_tempo_map(tick: int, tempo_map: BeatTempoMap) -> float:
    """Convert a chart tick back to seconds using a beat-derived tempo map.

    Args:
        tick: Chart tick to convert.
        tempo_map: A :class:`BeatTempoMap` built from detected beats.

    Returns:
        Time in seconds corresponding to *tick*.
    """
    tick = max(0, int(tick))

    beats = tempo_map.beat_times
    ticks = tempo_map.beat_ticks
    resolution = tempo_map.resolution

    if tick <= ticks[0]:
        return beats[0] + ((tick - ticks[0]) / resolution) * tempo_map.first_interval_sec

    index = 0
    while index + 1 < len(ticks) and ticks[index + 1] <= tick:
        index += 1

    if index >= len(ticks) - 1:
        return beats[-1] + ((tick - ticks[-1]) / resolution) * tempo_map.last_interval_sec

    interval = beats[index + 1] - beats[index]
    if interval <= 1e-9:
        return beats[index]
    return beats[index] + ((tick - ticks[index]) / resolution) * interval


def build_sync_track_from_beats(
    beat_times: Sequence[float] | None,
    *,
    resolution: int,
) -> list[SyncTrackEvent]:
    """Emit a variable-tempo SyncTrack from detected beat times.

    Produces a time-signature event and one or more tempo events (``B <bpm>``)
    for each detected beat where the tempo changes.

    Args:
        beat_times: Detected beat positions in seconds, or ``None`` for a
            constant-tempo fallback.
        resolution: Ticks per beat (e.g. 192).

    Returns:
        A list of :class:`SyncTrackEvent` objects for the chart's ``[SyncTrack]``
        section.
    """
    tempo_map = build_beat_tempo_map(beat_times, resolution=resolution)
    if tempo_map is None:
        return [SyncTrackEvent(0, "TS 4")]

    events = [SyncTrackEvent(0, "TS 4")]
    active_bpm = 60.0 / tempo_map.first_interval_sec
    events.append(SyncTrackEvent(0, f"B {bpm_to_chart_integer(active_bpm)}"))

    for index, interval in enumerate(
        tempo_map.beat_times[i + 1] - tempo_map.beat_times[i]
        for i in range(len(tempo_map.beat_times) - 1)
    ):
        if interval <= 1e-9:
            continue
        next_bpm = 60.0 / interval
        if bpm_to_chart_integer(next_bpm) == bpm_to_chart_integer(active_bpm):
            continue
        events.append(
            SyncTrackEvent(
                int(tempo_map.beat_ticks[index]),
                f"B {bpm_to_chart_integer(next_bpm)}",
            )
        )
        active_bpm = next_bpm

    return events


INSTRUMENT_MAP = INSTRUMENT_TO_CHART_NOTES


def _cap_simultaneous_notes(notes: list[DrumNote]) -> list[DrumNote]:
    """Limit chart notes to kick plus at most two hand lanes per tick.

    When more than ``MAX_HAND_LANES`` hand-lane pads occur at the same
    tick, lower-priority pads are dropped (cymbal modifiers are preserved
    for kept lanes).

    Args:
        notes: Drum notes to filter.

    Returns:
        Filtered notes with simultaneous hits capped.
    """
    out: list[DrumNote] = []
    sorted_notes = sorted(notes, key=lambda note: (note.tick, note.note, note.length))

    for tick, group in groupby(sorted_notes, key=lambda note: note.tick):
        notes_at_tick = [note.note for note in group]
        has_kick = KICK in notes_at_tick
        hand_lanes = sorted({note for note in notes_at_tick if note in HAND_PAD_NOTES})

        if len(hand_lanes) <= MAX_HAND_LANES:
            out.extend(DrumNote(tick, note) for note in notes_at_tick)
            continue

        keep_lanes = set(
            sorted(hand_lanes, key=lambda lane: HAND_LANE_PRIORITY.get(lane, lane))[
                :MAX_HAND_LANES
            ]
        )

        if has_kick:
            out.append(DrumNote(tick, KICK))
        for lane in sorted(keep_lanes):
            out.append(DrumNote(tick, lane))
            cymbal = CYMBAL_BY_PAD.get(lane)
            if cymbal is not None and cymbal in notes_at_tick:
                out.append(DrumNote(tick, cymbal))

    return out


def _dedupe_notes(notes: list[DrumNote]) -> list[DrumNote]:
    """Remove duplicate DrumNote entries (same tick, note, and length).

    Args:
        notes: Drum notes to deduplicate.

    Returns:
        A new list with duplicates removed, sorted by (tick, note, length).
    """
    seen: set[tuple[int, int, int]] = set()
    deduped: list[DrumNote] = []
    for note in sorted(notes, key=lambda n: (n.tick, n.note, n.length)):
        key = (note.tick, note.note, note.length)
        if key not in seen:
            seen.add(key)
            deduped.append(note)
    return deduped


def hits_to_chart_document(
    hits: list[DrumHit],
    *,
    song,
    bpm: float,
    resolution: int = 192,
    beat_times: Sequence[float] | None = None,
    quantize_divisor: int | None = None,
) -> ChartDocument:
    """Convert drum hits into a full Clone Hero chart document.

    Applies quantization, simultaneous-hit limiting, instrument-to-note
    mapping, variable-tempo sync track generation, and builds the final
    :class:`ChartDocument` with Expert drums populated.

    Args:
        hits: List of drum hits to convert.
        song: :class:`SongMetadata` for the chart.
        bpm: Beats per minute (used as fallback if no beat_times).
        resolution: Ticks per beat. Defaults to 192.
        beat_times: Detected beat positions for variable-tempo sync.
        quantize_divisor: If set, snap hits to a grid (e.g. 16 for 1/16 notes).

    Returns:
        A :class:`ChartDocument` with the generated chart data.
    """
    if quantize_divisor is not None and hits:
        quantize_beats = normalize_beat_times(beat_times)
        if len(quantize_beats) < 2:
            duration_sec = max(hit.time_sec for hit in hits) + (60.0 / bpm)
            quantize_beats = list(
                make_beat_grid_from_bpm(bpm, duration_sec=duration_sec).beat_times
            )
        hits = snap_hits_to_grid(
            hits,
            BeatGrid(beat_times=quantize_beats, bpm=bpm),
            divisor=quantize_divisor,
        )

    hits = limit_simultaneous_hits(hits)

    tempo_map = build_beat_tempo_map(beat_times, resolution=resolution)

    notes: list[DrumNote] = []
    for hit in hits:
        if hit.instrument not in INSTRUMENT_MAP:
            raise ValueError(f"Unknown instrument: {hit.instrument!r}")
        note_num, cymbal = INSTRUMENT_MAP[hit.instrument]
        if tempo_map is None:
            tick = seconds_to_tick(hit.time_sec, bpm, resolution)
        else:
            tick = seconds_to_tick_tempo_map(hit.time_sec, tempo_map)
        notes.append(DrumNote(tick, note_num))
        if cymbal is not None:
            notes.append(DrumNote(tick, cymbal))

    deduped = _dedupe_notes(_cap_simultaneous_notes(_dedupe_notes(notes)))
    if tempo_map is None:
        sync = [
            SyncTrackEvent(0, "TS 4"),
            SyncTrackEvent(0, f"B {bpm_to_chart_integer(bpm)}"),
        ]
    else:
        sync = build_sync_track_from_beats(beat_times, resolution=resolution)
    events: list = []

    return ChartDocument(
        song=song,
        sync=sync,
        events=events,
        drums={DrumDifficulty.EXPERT: deduped},
    )
