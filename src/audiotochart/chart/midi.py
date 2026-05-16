from __future__ import annotations

from pathlib import Path
from typing import Iterable

from audiotochart.chart.convert import hits_to_chart_document
from audiotochart.chart.format import ChartDocument, SongMetadata
from audiotochart.drums import DrumHit


MIDI_DRUM_MAP: dict[int, str] = {
    35: "kick",
    36: "kick",
    38: "snare",
    40: "snare",
    42: "hihat",
    44: "hihat",
    46: "hihat",
    48: "tom_yellow",
    50: "tom_yellow",
    45: "tom_blue",
    47: "tom_blue",
    51: "ride",
    59: "ride",
    49: "crash",
    57: "crash",
    41: "tom_green",
    43: "tom_green",
}

CHART_MIDI_DRUM_TRACK_NAMES = {
    "PART DRUMS",
    "PART REAL_DRUMS",
}

CHART_MIDI_EXPERT_DRUM_MAP: dict[int, str] = {
    96: "kick",
    97: "snare",
    98: "hihat",
    99: "ride",
    100: "crash",
}

CHART_MIDI_TOM_MARKERS: dict[int, tuple[int, str]] = {
    110: (98, "tom_yellow"),
    111: (99, "tom_blue"),
    112: (100, "tom_green"),
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
) -> ChartDocument:
    """Convert a General MIDI drum file into a Clone Hero chart document."""
    return hits_to_chart_document(
        iter_drum_midi_hits(path),
        song=song,
        bpm=bpm,
        resolution=resolution,
    )
