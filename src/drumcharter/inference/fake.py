"""Fake transcriber backend for testing and format checks.

Generates a simple rock-beat drum pattern without requiring a model or
audio processing. Useful for integration tests and pipeline validation.
"""

from __future__ import annotations

from pathlib import Path

from drumcharter.audio import get_audio_duration_sec
from drumcharter.chart.fake import make_fake_drum_hits
from drumcharter.drums import DrumHit


class FakeTranscriber:
    """Transcriber that produces a deterministic fake drum pattern.

    Ignores the actual audio content and returns a pre-defined beat
    pattern based solely on the audio duration.
    """

    def transcribe(self, audio_path: Path) -> list[DrumHit]:
        """Generate a fake drum pattern for the given audio duration.

        Args:
            audio_path: Path to audio file (only duration is used).

        Returns:
            A list of :class:`DrumHit` objects forming a simple rock beat.
        """
        duration_sec = get_audio_duration_sec(audio_path)
        return make_fake_drum_hits(duration_sec)
