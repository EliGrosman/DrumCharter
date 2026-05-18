"""Tests for audio duration extraction using librosa."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from audiotochart.audio import AudioError, get_audio_duration_sec


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


def test_get_audio_duration_sec_returns_correct_duration(tmp_path: Path) -> None:
    wav_path = _make_wav(tmp_path, "test.wav", duration_sec=3.0)
    duration = get_audio_duration_sec(wav_path)
    assert duration == pytest.approx(3.0, abs=0.01)


def test_get_audio_duration_sec_short_file(tmp_path: Path) -> None:
    wav_path = _make_wav(tmp_path, "short.wav", duration_sec=0.5)
    duration = get_audio_duration_sec(wav_path)
    assert duration == pytest.approx(0.5, abs=0.01)


def test_get_audio_duration_sec_unsupported_extension(tmp_path: Path) -> None:
    bad_file = tmp_path / "data.xyz"
    bad_file.write_bytes(b"not audio")
    with pytest.raises(AudioError, match="Failed to read audio"):
        get_audio_duration_sec(bad_file)


def test_get_audio_duration_sec_nonexistent(tmp_path: Path) -> None:
    with pytest.raises(AudioError, match="Audio file not found"):
        get_audio_duration_sec(tmp_path / "doesnotexist.wav")
