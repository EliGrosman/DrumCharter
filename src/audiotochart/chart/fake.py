from __future__ import annotations

from audiotochart.chart.format import (
    ChartDocument,
    DrumDifficulty,
    DrumNote,
    SectionEvent,
    SongMetadata,
    SyncTrackEvent,
    bpm_to_chart_integer,
)


def create_fake_drum_chart(
    *,
    song: SongMetadata,
    bpm: float = 120.0,
    measures: int = 16,
) -> ChartDocument:
    """Create a simple playable Expert drum chart for early CLI testing."""
    beat = song.resolution
    eighth = beat // 2
    bar = beat * 4

    notes: list[DrumNote] = []
    for measure in range(measures):
        start = measure * bar

        # Eighth-note hi-hat pulse.
        for step in range(8):
            tick = start + step * eighth
            notes.append(DrumNote(tick, 2))
            notes.append(DrumNote(tick, 66))

        # Basic rock backbeat: kick on 1 and 3, snare on 2 and 4.
        notes.append(DrumNote(start, 0))
        notes.append(DrumNote(start + 2 * beat, 0))
        notes.append(DrumNote(start + beat, 1))
        notes.append(DrumNote(start + 3 * beat, 1))

    return ChartDocument(
        song=song,
        sync=[
            SyncTrackEvent(0, "TS 4"),
            SyncTrackEvent(0, f"B {bpm_to_chart_integer(bpm)}"),
        ],
        events=[SectionEvent(0, "section Intro")],
        drums={DrumDifficulty.EXPERT: notes},
    )
