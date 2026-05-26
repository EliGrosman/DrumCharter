"""Abstract interface for drum transcribers.

Defines the :class:`DrumTranscriber` protocol that all backends must satisfy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from drumcharter.drums import DrumHit


class DrumTranscriber(Protocol):
    """Protocol for audio-to-drum-hit transcribers.

    All inference backends (fake, ADTOF, model) implement this interface.
    """

    def transcribe(self, audio_path: Path) -> list[DrumHit]:
        """Transcribe an audio file into a list of drum hits.

        Args:
            audio_path: Path to a WAV or other audio file.

        Returns:
            A list of :class:`DrumHit` objects.
        """
        ...
