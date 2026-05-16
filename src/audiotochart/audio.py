from __future__ import annotations

import logging
from pathlib import Path

import librosa

logger = logging.getLogger(__name__)


class AudioError(Exception):
    """Raised when audio cannot be read or has an unsupported format."""
    pass


def get_audio_duration_sec(path: Path) -> float:
    """Return the duration of *path* in seconds.

    Uses ``librosa.load`` which supports wav, mp3, ogg, flac and more.
    """
    path = Path(path)
    try:
        y, sr = librosa.load(path, sr=None)
        if sr is None or sr == 0:
            raise AudioError(f"Could not determine sample rate for {path}")
        return float(len(y)) / sr
    except FileNotFoundError:
        raise AudioError(f"Audio file not found: {path}")
    except Exception as exc:
        raise AudioError(f"Failed to read audio {path}: {exc}") from exc
