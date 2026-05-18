from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

NUM_CLASSES = 8
CLASS_NAMES = [
    "Kick",
    "Snare",
    "Hi-Hat",
    "Yellow Tom",
    "Ride",
    "Blue Tom",
    "Crash",
    "Floor Tom",
]

_EXPERT_PITCHES = frozenset(range(95, 101))
_MARKER_TO_PAD: dict[int, int] = {110: 98, 111: 99, 112: 100}
_MARKER_PITCHES = frozenset(_MARKER_TO_PAD)

_PAD_CLASS_MAP: dict[int, tuple[int, int]] = {
    98: (2, 3),
    99: (4, 5),
    100: (6, 7),
}

_DRUM_TRACK_NAMES = {"PART DRUMS", "PART DRUMS_2X"}


@dataclass(frozen=True, slots=True)
class DrumOnset:
    time: float
    class_id: int


def _find_drum_track(pm: object) -> object | None:
    for inst in pm.instruments:
        if inst.name.strip().upper() in _DRUM_TRACK_NAMES:
            return inst
    return None


def _resolve_class(pitch: int, has_marker: bool) -> int | None:
    if pitch == 96:
        return 0
    if pitch == 97:
        return 1
    if pitch == 95:
        return 0
    pair = _PAD_CLASS_MAP.get(pitch)
    if pair is not None:
        return pair[1] if has_marker else pair[0]
    return None


def parse_rb_drum_onsets(midi_path: Path) -> list[DrumOnset]:
    import mido
    import pretty_midi

    midi_path = Path(midi_path)
    if not midi_path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except (OSError, ValueError) as exc:
        if "data byte must be in range" in str(exc):
            log.debug("Retrying %s with mido clip=True", midi_path.name)
            mido_obj = mido.MidiFile(str(midi_path), clip=True)
            pm = pretty_midi.PrettyMIDI(mido_object=mido_obj)
        else:
            raise

    track = _find_drum_track(pm)
    if track is None:
        raise ValueError(f"No PART DRUMS track found in {midi_path}")

    by_tick: dict[int, set[int]] = defaultdict(set)
    marker_intervals: dict[int, list[tuple[float, float]]] = {
        m: [] for m in _MARKER_PITCHES
    }
    _MARKER_END_EPS = 1e-3

    for note in track.notes:
        pitch = note.pitch
        if pitch in _MARKER_PITCHES:
            marker_intervals[pitch].append((note.start, note.end))
            continue
        if pitch not in _EXPERT_PITCHES:
            continue
        tick = int(pm.time_to_tick(note.start))
        by_tick[tick].add(pitch)

    for m in marker_intervals:
        marker_intervals[m].sort()

    _PAD_TO_MARKER = {p: m for m, p in _MARKER_TO_PAD.items()}

    def _pad_has_marker(pad_pitch: int, time_sec: float) -> bool:
        marker_pitch = _PAD_TO_MARKER.get(pad_pitch)
        if marker_pitch is None:
            return False
        for start, end in marker_intervals[marker_pitch]:
            if start - _MARKER_END_EPS <= time_sec < end + _MARKER_END_EPS:
                return True
            if start > time_sec + _MARKER_END_EPS:
                break
        return False

    onsets: list[DrumOnset] = []
    for tick in sorted(by_tick):
        pads = by_tick[tick]
        time_sec = pm.tick_to_time(tick)
        for pad in sorted(pads):
            has_marker = _pad_has_marker(pad, time_sec)
            cls = _resolve_class(pad, has_marker)
            if cls is not None:
                onsets.append(DrumOnset(time=time_sec, class_id=cls))

    onsets.sort(key=lambda o: (o.time, o.class_id))
    return onsets


def has_pro_markers(midi_path: Path) -> bool:
    import mido
    import pretty_midi

    midi_path = Path(midi_path)
    if not midi_path.is_file():
        return False

    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except (OSError, ValueError) as exc:
        if "data byte must be in range" in str(exc):
            mido_obj = mido.MidiFile(str(midi_path), clip=True)
            pm = pretty_midi.PrettyMIDI(mido_object=mido_obj)
        else:
            return False

    track = _find_drum_track(pm)
    if track is None:
        return False
    return any(note.pitch in _MARKER_PITCHES for note in track.notes)


def onset_stats(onsets: list[DrumOnset]) -> dict[int, int]:
    return dict(Counter(o.class_id for o in onsets))
