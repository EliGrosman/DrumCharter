"""Generate lower pro-drum difficulties from an Expert pro-drum chart."""

from __future__ import annotations

from audiotochart.chart.format import ChartDocument, DrumDifficulty, DrumNote

KICK = 0
SNARE = 1
YELLOW_PAD = 2
BLUE_PAD = 3
GREEN_PAD = 4
CYMBAL_YELLOW = 66
CYMBAL_BLUE = 67
CYMBAL_GREEN = 68

_PAD_NOTES = {KICK, SNARE, YELLOW_PAD, BLUE_PAD, GREEN_PAD}
_CYMBAL_MODIFIERS = {CYMBAL_YELLOW, CYMBAL_BLUE, CYMBAL_GREEN}
_PAD_FOR_CYMBAL = {
    CYMBAL_YELLOW: YELLOW_PAD,
    CYMBAL_BLUE: BLUE_PAD,
    CYMBAL_GREEN: GREEN_PAD,
}


def _group_by_tick(notes: list[DrumNote]) -> dict[int, list[DrumNote]]:
    groups: dict[int, list[DrumNote]] = {}
    for note in notes:
        groups.setdefault(note.tick, []).append(note)
    return groups


def _tick_pad_notes(tick_notes: list[DrumNote]) -> set[int]:
    return {note.note for note in tick_notes if note.note in _PAD_NOTES}


def _has_cymbal_at_tick(tick_notes: list[DrumNote], pad: int) -> bool:
    modifier = {
        YELLOW_PAD: CYMBAL_YELLOW,
        BLUE_PAD: CYMBAL_BLUE,
        GREEN_PAD: CYMBAL_GREEN,
    }.get(pad)
    if modifier is None:
        return False
    return any(note.note == modifier for note in tick_notes)


def _append_pad_with_cymbal(
    out: list[DrumNote],
    *,
    tick: int,
    pad: int,
    tick_notes: list[DrumNote],
) -> None:
    out.append(DrumNote(tick=tick, note=pad))
    modifier = {
        YELLOW_PAD: CYMBAL_YELLOW,
        BLUE_PAD: CYMBAL_BLUE,
        GREEN_PAD: CYMBAL_GREEN,
    }.get(pad)
    if modifier is not None and any(note.note == modifier for note in tick_notes):
        out.append(DrumNote(tick=tick, note=modifier))


def _thin_notes(
    notes: list[DrumNote],
    min_gap: int,
    priority: set[int] | None = None,
) -> list[DrumNote]:
    last_tick_by_pad: dict[int, int] = {}
    out: list[DrumNote] = []

    for note in sorted(notes, key=lambda n: (n.tick, n.note, n.length)):
        if note.note in _CYMBAL_MODIFIERS:
            continue
        if priority and note.note in priority:
            last_tick_by_pad[note.note] = note.tick
            out.append(note)
            continue
        previous_tick = last_tick_by_pad.get(note.note)
        if previous_tick is not None and note.tick - previous_tick < min_gap:
            continue
        last_tick_by_pad[note.note] = note.tick
        out.append(note)

    surviving_pads = {(note.tick, note.note) for note in out}
    for note in notes:
        pad = _PAD_FOR_CYMBAL.get(note.note)
        if pad is not None and (note.tick, pad) in surviving_pads:
            out.append(note)

    return sorted(out, key=lambda n: (n.tick, n.note, n.length))


def _generate_hard(expert: list[DrumNote], resolution: int) -> list[DrumNote]:
    thirty_second = resolution // 8
    return _thin_notes(expert, min_gap=thirty_second)


def _generate_medium(expert: list[DrumNote], resolution: int) -> list[DrumNote]:
    eighth = resolution // 2
    groups = _group_by_tick(expert)
    kept: list[DrumNote] = []

    for tick in sorted(groups):
        tick_notes = groups[tick]
        pads = _tick_pad_notes(tick_notes)
        if KICK in pads:
            kept.append(DrumNote(tick=tick, note=KICK))
        if SNARE in pads:
            kept.append(DrumNote(tick=tick, note=SNARE))
        if YELLOW_PAD in pads and _has_cymbal_at_tick(tick_notes, YELLOW_PAD):
            _append_pad_with_cymbal(
                kept,
                tick=tick,
                pad=YELLOW_PAD,
                tick_notes=tick_notes,
            )

    return _thin_notes(kept, min_gap=eighth, priority={KICK, SNARE})


def _generate_easy(expert: list[DrumNote], resolution: int) -> list[DrumNote]:
    quarter = resolution
    groups = _group_by_tick(expert)
    kept: list[DrumNote] = []
    last_kick = -quarter
    last_snare = -quarter
    last_hihat = -quarter

    for tick in sorted(groups):
        tick_notes = groups[tick]
        pads = _tick_pad_notes(tick_notes)
        beat_in_bar = round(tick / quarter) % 4

        if KICK in pads and beat_in_bar in (0, 2) and tick - last_kick >= quarter:
            kept.append(DrumNote(tick=tick, note=KICK))
            last_kick = tick
        elif SNARE in pads and beat_in_bar in (1, 3) and tick - last_snare >= quarter:
            kept.append(DrumNote(tick=tick, note=SNARE))
            last_snare = tick
        elif KICK in pads and tick - last_kick >= quarter:
            kept.append(DrumNote(tick=tick, note=KICK))
            last_kick = tick
        elif SNARE in pads and tick - last_snare >= quarter:
            kept.append(DrumNote(tick=tick, note=SNARE))
            last_snare = tick

        if (
            YELLOW_PAD in pads
            and _has_cymbal_at_tick(tick_notes, YELLOW_PAD)
            and tick - last_hihat >= quarter
        ):
            _append_pad_with_cymbal(
                kept,
                tick=tick,
                pad=YELLOW_PAD,
                tick_notes=tick_notes,
            )
            last_hihat = tick

    return kept


def generate_difficulties(doc: ChartDocument) -> None:
    """Populate Hard, Medium, and Easy pro drums from Expert in place."""
    expert = doc.drums.get(DrumDifficulty.EXPERT, [])
    if not expert:
        return

    resolution = doc.song.resolution
    doc.drums[DrumDifficulty.HARD] = _generate_hard(expert, resolution)
    doc.drums[DrumDifficulty.MEDIUM] = _generate_medium(expert, resolution)
    doc.drums[DrumDifficulty.EASY] = _generate_easy(expert, resolution)
