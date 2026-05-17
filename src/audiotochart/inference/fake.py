from __future__ import annotations

from pathlib import Path

from audiotochart.audio import get_audio_duration_sec
from audiotochart.drums import DrumHit
from audiotochart.tempo import TempoError, detect_beat_grid


class FakeTranscriber:
    def transcribe(self, audio_path: Path) -> list[DrumHit]:
        duration_sec = get_audio_duration_sec(audio_path)

        try:
            beat_grid = detect_beat_grid(audio_path)
            bpm = beat_grid.bpm
        except TempoError:
            bpm = 120.0

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
                hits.append(DrumHit(time, "hihat"))

            hits.append(DrumHit(start, "kick"))
            if start + 2.0 < duration_sec:
                hits.append(DrumHit(start + 2.0, "kick"))

            hits.append(DrumHit(start + 1.0, "snare"))
            if start + 3.0 < duration_sec:
                hits.append(DrumHit(start + 3.0, "snare"))

            measure += 1

        return hits
