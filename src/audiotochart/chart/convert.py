from __future__ import annotations

from audiotochart.chart.format import (
    ChartDocument,
    DrumDifficulty,
    DrumNote,
    SectionEvent,
    SyncTrackEvent,
    bpm_to_chart_integer,
)
from audiotochart.drums import DrumHit


def seconds_to_tick(time_sec: float, bpm: float, resolution: int) -> int:
    sec_per_beat = 60.0 / bpm
    sec_per_tick = sec_per_beat / resolution
    return max(0, round(time_sec / sec_per_tick))


INSTRUMENT_MAP: dict[str, tuple[int, int | None]] = {
    "kick": (0, None),
    "snare": (1, None),
    "hihat": (2, 66),
    "tom_yellow": (2, None),
    "ride": (3, 67),
    "tom_blue": (3, None),
    "crash": (4, 68),
    "tom_green": (4, None),
}


def hits_to_chart_document(
    hits: list[DrumHit],
    *,
    song,
    bpm: float,
    resolution: int = 192,
) -> ChartDocument:
    notes: list[DrumNote] = []
    for hit in hits:
        if hit.instrument not in INSTRUMENT_MAP:
            raise ValueError(f"Unknown instrument: {hit.instrument!r}")
        note_num, cymbal = INSTRUMENT_MAP[hit.instrument]
        tick = seconds_to_tick(hit.time_sec, bpm, resolution)
        notes.append(DrumNote(tick, note_num))
        if cymbal is not None:
            notes.append(DrumNote(tick, cymbal))

    # Deduplicate notes
    seen: set[tuple[int, int, int]] = set()
    deduped: list[DrumNote] = []
    for n in sorted(notes, key=lambda n: (n.tick, n.note, n.length)):
        key = (n.tick, n.note, n.length)
        if key not in seen:
            seen.add(key)
            deduped.append(n)

    sync = [
        SyncTrackEvent(0, "TS 4"),
        SyncTrackEvent(0, f"B {bpm_to_chart_integer(bpm)}"),
    ]
    events = [SectionEvent(0, "section Intro")]

    return ChartDocument(
        song=song,
        sync=sync,
        events=events,
        drums={DrumDifficulty.EXPERT: deduped},
    )
