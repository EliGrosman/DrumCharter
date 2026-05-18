"""Audio I/O utilities.

Provides helpers for reading audio file duration via librosa with a
WAV-header fallback when librosa is unavailable.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


class AudioError(Exception):
    """Raised when audio cannot be read or has an unsupported format."""
    pass


def _fallback_duration_sec(path: Path) -> float | None:
    """Try to get duration from WAV header without librosa.

    Args:
        path: Path to a WAV file.

    Returns:
        Duration in seconds, or None on failure.
    """
    try:
        with wave.open(str(path), "r") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return frames / rate
    except Exception:
        pass
    return None


def get_audio_duration_sec(path: Path) -> float:
    """Return the duration of *path* in seconds.

    Uses ``librosa.load`` which supports wav, mp3, ogg, flac and more.
    Falls back to WAV header parsing when librosa is unavailable.
    """
    path = Path(path)
    if _HAS_LIBROSA:
        try:
            y, sr = librosa.load(path, sr=None)
            if sr is None or sr == 0:
                raise AudioError(f"Could not determine sample rate for {path}")
            return float(len(y)) / sr
        except FileNotFoundError:
            raise AudioError(f"Audio file not found: {path}")
        except Exception as exc:
            raise AudioError(f"Failed to read audio {path}: {exc}") from exc
    else:
        duration = _fallback_duration_sec(path)
        if duration is not None:
            return duration
        raise AudioError(
            f"Cannot determine duration of {path} without librosa. "
            "Install it with: pip install audiotochart[audio]"
        )
