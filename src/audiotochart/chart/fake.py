from audiotochart.chart.convert import hits_to_chart_document
from audiotochart.chart.format import ChartDocument, SongMetadata
from audiotochart.drums import DrumHit


def create_fake_drum_chart(
    *,
    song: SongMetadata,
    bpm: float = 120.0,
    measures: int = 16,
) -> ChartDocument:
    beats_per_measure = 4
    eighth = 0.5
    bar = beats_per_measure * 1.0

    hits: list[DrumHit] = []
    for measure in range(measures):
        start = measure * bar

        # Eighth-note hi-hat pulse
        for step in range(8):
            time = start + step * eighth
            hits.append(DrumHit(time, "hihat"))

        # Kick on 1 and 3
        hits.append(DrumHit(start, "kick"))
        hits.append(DrumHit(start + 2.0, "kick"))

        # Snare on 2 and 4
        hits.append(DrumHit(start + 1.0, "snare"))
        hits.append(DrumHit(start + 3.0, "snare"))

    return hits_to_chart_document(
        hits,
        song=song,
        bpm=bpm,
        resolution=song.resolution,
    )
