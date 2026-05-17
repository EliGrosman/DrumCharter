from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Iterable

from audiotochart.chart.convert import hits_to_chart_document
from audiotochart.chart.drum_vocab import (
    CRASH_LABEL,
    HIHAT_LABEL,
    KICK_LABEL,
    RIDE_LABEL,
    SNARE_LABEL,
    TOM_BLUE_LABEL,
    TOM_GREEN_LABEL,
    TOM_YELLOW_LABEL,
)
from audiotochart.chart.format import ChartDocument, SongMetadata
from audiotochart.drums import DrumHit


MIDI_DRUM_MAP: dict[int, str] = {
    35: KICK_LABEL,
    36: KICK_LABEL,
    38: SNARE_LABEL,
    40: SNARE_LABEL,
    42: HIHAT_LABEL,
    44: HIHAT_LABEL,
    46: HIHAT_LABEL,
    48: TOM_YELLOW_LABEL,
    50: TOM_YELLOW_LABEL,
    45: TOM_BLUE_LABEL,
    47: TOM_BLUE_LABEL,
    51: RIDE_LABEL,
    59: RIDE_LABEL,
    49: CRASH_LABEL,
    57: CRASH_LABEL,
    41: TOM_GREEN_LABEL,
    43: TOM_GREEN_LABEL,
}

CHART_MIDI_DRUM_TRACK_NAMES = {
    "PART DRUMS",
    "PART REAL_DRUMS",
}

CHART_MIDI_EXPERT_DRUM_MAP: dict[int, str] = {
    96: KICK_LABEL,
    97: SNARE_LABEL,
    98: HIHAT_LABEL,
    99: RIDE_LABEL,
    100: CRASH_LABEL,
}

CHART_MIDI_TOM_MARKERS: dict[int, tuple[int, str]] = {
    110: (98, TOM_YELLOW_LABEL),
    111: (99, TOM_BLUE_LABEL),
    112: (100, TOM_GREEN_LABEL),
}


class MidiError(RuntimeError):
    """Raised when MIDI input cannot be read or the optional dependency is missing."""


def _load_pretty_midi():
    try:
        import pretty_midi
    except ImportError as exc:
        raise MidiError(
            "MIDI conversion requires pretty_midi. Install it with the 'midi' extra."
        ) from exc
    return pretty_midi


def _is_chart_midi_drum_track(name: str) -> bool:
    return name.strip().upper() in CHART_MIDI_DRUM_TRACK_NAMES


def midi_pitch_to_instrument(pitch: int) -> str | None:
    """Return the project drum instrument for a General MIDI drum pitch."""
    return MIDI_DRUM_MAP.get(pitch)


def _iter_general_midi_hits(notes: Iterable[object]) -> list[DrumHit]:
    hits: list[DrumHit] = []
    for note in notes:
        drum_instrument = midi_pitch_to_instrument(note.pitch)
        if drum_instrument is None:
            continue
        hits.append(
            DrumHit(
                time_sec=float(note.start),
                instrument=drum_instrument,
                confidence=float(note.velocity) / 127.0,
            )
        )
    return hits


def _iter_chart_midi_hits(notes: Iterable[object]) -> list[DrumHit]:
    notes_by_start: dict[float, dict[int, object]] = {}
    for note in notes:
        start = round(float(note.start), 6)
        by_pitch = notes_by_start.setdefault(start, {})
        existing = by_pitch.get(note.pitch)
        if existing is None or note.velocity > existing.velocity:
            by_pitch[note.pitch] = note

    hits: list[DrumHit] = []
    for start, notes_by_pitch in sorted(notes_by_start.items()):
        tom_marker_lanes = {
            lane: instrument
            for marker, (lane, instrument) in CHART_MIDI_TOM_MARKERS.items()
            if marker in notes_by_pitch
        }

        for pitch, default_instrument in CHART_MIDI_EXPERT_DRUM_MAP.items():
            note = notes_by_pitch.get(pitch)
            if note is None:
                continue
            hits.append(
                DrumHit(
                    time_sec=start,
                    instrument=tom_marker_lanes.get(pitch, default_instrument),
                    confidence=float(note.velocity) / 127.0,
                )
            )

    return hits


def iter_drum_midi_hits(path: Path) -> list[DrumHit]:
    """Read General MIDI drum notes from *path* as neutral ``DrumHit`` objects."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {path}")

    pretty_midi = _load_pretty_midi()
    try:
        midi = pretty_midi.PrettyMIDI(str(path))
    except Exception as exc:
        raise MidiError(f"Failed to read MIDI {path}: {exc}") from exc

    hits: list[DrumHit] = []
    for instrument in midi.instruments:
        if _is_chart_midi_drum_track(instrument.name):
            hits.extend(_iter_chart_midi_hits(instrument.notes))
        elif instrument.is_drum:
            hits.extend(_iter_general_midi_hits(instrument.notes))

    return sorted(hits, key=lambda hit: (hit.time_sec, hit.instrument))


def midi_to_chart_document(
    path: Path,
    *,
    song: SongMetadata,
    bpm: float,
    resolution: int = 192,
    beat_times: Sequence[float] | None = None,
    quantize_divisor: int | None = None,
) -> ChartDocument:
    """Convert a General MIDI drum file into a Clone Hero chart document."""
    return hits_to_chart_document(
        iter_drum_midi_hits(path),
        song=song,
        bpm=bpm,
        resolution=resolution,
        beat_times=beat_times,
        quantize_divisor=quantize_divisor,
    )
