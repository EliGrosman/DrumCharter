from __future__ import annotations

from pathlib import Path

from audiotochart.audio import get_audio_duration_sec
from audiotochart.chart.fake import make_fake_drum_hits
from audiotochart.drums import DrumHit


class FakeTranscriber:
    def transcribe(self, audio_path: Path) -> list[DrumHit]:
        duration_sec = get_audio_duration_sec(audio_path)
        return make_fake_drum_hits(duration_sec)
