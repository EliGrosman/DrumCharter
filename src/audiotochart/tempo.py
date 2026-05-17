from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


class TempoError(Exception):
    """Raised when tempo detection fails or is unavailable."""
    pass


@dataclass(frozen=True)
class BeatGrid:
    bpm: float
    beat_times: np.ndarray


def _detect_beat_grid_librosa(path: Path) -> BeatGrid:
    """Estimate tempo and beat positions using librosa."""
    y, sr = librosa.load(path, sr=None, mono=True)
    if len(y) == 0:
        raise TempoError(f"Audio file is empty: {path}")

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo_est, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    # beat_track returns a numpy array of tempo estimates
    if isinstance(tempo_est, np.ndarray):
        if len(tempo_est) == 0:
            raise TempoError(f"Could not estimate tempo for {path}")
        bpm = float(tempo_est[0])
    else:
        bpm = float(tempo_est)

    # Validate BPM is in a reasonable range
    if bpm <= 0 or bpm < 20 or bpm > 300:
        raise TempoError(f"Detected BPM {bpm} is out of range [20, 300] for {path}")

    # Get beat positions using the estimated tempo.
    _tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        start_bpm=bpm,
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beat_times) < 2:
        raise TempoError(f"Insufficient beats detected for {path}")

    return BeatGrid(bpm=bpm, beat_times=beat_times)


def detect_beat_grid(path: Path) -> BeatGrid:
    """Detect tempo and beat positions from an audio file.

    Uses librosa for tempo estimation and beat tracking.
    Raises TempoError if librosa is not installed or detection fails.
    """
    path = Path(path)
    if not path.is_file():
        raise TempoError(f"Audio file not found: {path}")

    if not _HAS_LIBROSA:
        raise TempoError(
            "librosa is not installed. Install it with: "
            "pip install audiotochart[audio] or use --bpm to specify manually"
        )

    try:
        return _detect_beat_grid_librosa(path)
    except TempoError:
        raise
    except Exception as exc:
        raise TempoError(f"Failed to detect tempo for {path}: {exc}") from exc
