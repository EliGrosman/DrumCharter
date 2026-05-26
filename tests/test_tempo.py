"""Tests for tempo detection."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from drumcharter.tempo import BeatGrid, TempoError, detect_beat_grid


def _make_wav(tmp_path: Path, name: str, duration_sec: float, sample_rate: int = 44100) -> Path:
    """Create a minimal silent WAV file and return its path."""
    path = tmp_path / name
    num_samples = int(duration_sec * sample_rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_samples)
    return path


def test_missing_file_raises_tempo_error(tmp_path: Path) -> None:
    with pytest.raises(TempoError, match="not found"):
        detect_beat_grid(tmp_path / "doesnotexist.wav")


def test_beatgrid_produces_valid_sync_track_event() -> None:
    """BeatGrid can be used to construct a valid SyncTrackEvent."""
    from drumcharter.chart.format import SyncTrackEvent, bpm_to_chart_integer

    grid = BeatGrid(bpm=128.0, beat_times=[])
    sync_event = SyncTrackEvent(0, f"B {bpm_to_chart_integer(grid.bpm)}")
    assert sync_event.tick == 0
    assert sync_event.payload == "B 128000"
    assert "B 128000" in sync_event.line()


def test_insufficient_beat_data_raises_tempo_error() -> None:
    """Silent audio may yield few/no beats; detect_beat_grid raises TempoError."""
    wav_path = _make_wav(Path("."), "_temp_silent.wav", duration_sec=0.1)
    try:
        with pytest.raises(TempoError):
            detect_beat_grid(wav_path)
    finally:
        wav_path.unlink(missing_ok=True)


def test_beatgrid_is_frozen_dataclass() -> None:
    """BeatGrid should be immutable."""
    import numpy as np

    grid = BeatGrid(bpm=120.0, beat_times=np.array([0.0, 0.5, 1.0]))
    with pytest.raises(Exception):
        grid.bpm = 130.0
    with pytest.raises(Exception):
        grid.beat_times = np.array([0.0])


def test_detect_beat_grid_real_audio(tmp_path: Path) -> None:
    """Detect tempo on a short WAV with a rhythmic noise burst signal."""
    import numpy as np

    sr = 44100
    bpm = 120.0
    beat_interval = 60.0 / bpm
    duration = 4.0

    # Create a signal with rhythmic onset bursts at 120 BPM
    audio_data = np.zeros(int(sr * duration), dtype=np.float32)
    num_beats = int(duration / beat_interval)
    for i in range(num_beats):
        beat_time = i * beat_interval
        beat_samples = int(beat_time * sr)
        # Short noise burst on each beat
        burst_len = min(int(0.02 * sr), len(audio_data) - beat_samples)
        if burst_len > 0:
            audio_data[beat_samples:beat_samples + burst_len] = np.random.randn(burst_len).astype(np.float32) * 0.5

    wav_path = tmp_path / "rhythmic.wav"
    import soundfile as sf
    sf.write(str(wav_path), audio_data, sr)

    grid = detect_beat_grid(wav_path)
    assert grid.bpm > 0
    assert grid.bpm > 20 and grid.bpm < 300
    assert isinstance(grid.beat_times, np.ndarray)
    assert len(grid.beat_times) > 0


def test_tempo_error_message_when_not_found(tmp_path: Path) -> None:
    """Missing file should produce a clear error message."""
    wav_path = tmp_path / "missing.wav"
    with pytest.raises(TempoError, match=str(wav_path)):
        detect_beat_grid(wav_path)
