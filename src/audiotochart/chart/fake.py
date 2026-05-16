from audiotochart.chart.convert import hits_to_chart_document
from audiotochart.chart.format import ChartDocument, SongMetadata
from audiotochart.drums import DrumHit


def create_fake_drum_chart(
    *,
    song: SongMetadata,
    duration_sec: float,
    bpm: float = 120.0,
) -> ChartDocument:
    beats_per_measure = 4
    eighth = 0.5
    bar = beats_per_measure * 1.0

    hits: list[DrumHit] = []
    measure = 0
    while True:
        start = measure * bar
        # Stop if this measure starts after the duration
        if start >= duration_sec:
            break

        # Eighth-note hi-hat pulse
        for step in range(8):
            time = start + step * eighth
            if time >= duration_sec:
                break
            hits.append(DrumHit(time, "hihat"))

        # Kick on 1 and 3
        hits.append(DrumHit(start, "kick"))
        if start + 2.0 < duration_sec:
            hits.append(DrumHit(start + 2.0, "kick"))

        # Snare on 2 and 4
        hits.append(DrumHit(start + 1.0, "snare"))
        if start + 3.0 < duration_sec:
            hits.append(DrumHit(start + 3.0, "snare"))

        measure += 1

    return hits_to_chart_document(
        hits,
        song=song,
        bpm=bpm,
        resolution=song.resolution,
    )
