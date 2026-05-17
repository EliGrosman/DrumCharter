from collections.abc import Sequence

from audiotochart.chart.convert import hits_to_chart_document
from audiotochart.chart.format import ChartDocument, SongMetadata
from audiotochart.chart.drum_vocab import HIHAT_LABEL, KICK_LABEL, SNARE_LABEL
from audiotochart.drums import DrumHit


def make_fake_drum_hits(duration_sec: float) -> list[DrumHit]:
    beats_per_measure = 4
    eighth = 0.5
    bar = beats_per_measure * 1.0

    hits: list[DrumHit] = []
    measure = 0
    while True:
        start = measure * bar
        if start >= duration_sec:
            break

        for step in range(8):
            time = start + step * eighth
            if time >= duration_sec:
                break
            hits.append(DrumHit(time, HIHAT_LABEL))

        hits.append(DrumHit(start, KICK_LABEL))
        if start + 2.0 < duration_sec:
            hits.append(DrumHit(start + 2.0, KICK_LABEL))

        hits.append(DrumHit(start + 1.0, SNARE_LABEL))
        if start + 3.0 < duration_sec:
            hits.append(DrumHit(start + 3.0, SNARE_LABEL))

        measure += 1

    return hits


def create_fake_drum_chart(
    *,
    song: SongMetadata,
    duration_sec: float,
    bpm: float = 120.0,
    beat_times: Sequence[float] | None = None,
    quantize_divisor: int | None = None,
) -> ChartDocument:
    return hits_to_chart_document(
        make_fake_drum_hits(duration_sec),
        song=song,
        bpm=bpm,
        resolution=song.resolution,
        beat_times=beat_times,
        quantize_divisor=quantize_divisor,
    )
