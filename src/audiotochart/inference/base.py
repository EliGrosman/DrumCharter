from __future__ import annotations

from pathlib import Path
from typing import Protocol

from audiotochart.drums import DrumHit


class DrumTranscriber(Protocol):
    def transcribe(self, audio_path: Path) -> list[DrumHit]: ...
