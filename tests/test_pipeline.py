"""Pipeline regression tests for the first self-contained generator."""

from __future__ import annotations

import wave
from pathlib import Path
from unittest.mock import patch

import drumcharter.pipeline as pipeline


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


def test_generate_drum_chart_folder_writes_clone_hero_song_folder(tmp_path: Path) -> None:
    source_audio = _make_wav(tmp_path, "song.wav", duration_sec=4.0)

    folder = pipeline.generate_drum_chart_folder(
        source_audio=source_audio,
        output_parent=tmp_path / "out",
        song_name="Song",
        artist_name="Artist",
        bpm=128.0,
    )

    assert folder == tmp_path / "out" / "Artist - Song"
    assert (folder / "song.wav").read_bytes() == source_audio.read_bytes()
    ini = (folder / "song.ini").read_text(encoding="utf-8")
    assert "name = Song" in ini
    assert "artist = Artist" in ini
    assert "charter = DrumCharter (AI)" in ini
    assert "diff_drums = 4" in ini
    assert "song_length = 4000" in ini

    chart = (folder / "notes.chart").read_text(encoding="utf-8")
    assert 'Name = "Song"' in chart
    assert 'Artist = "Artist"' in chart
    assert 'MusicStream = "song.wav"' in chart
    assert "0 = B 128000" in chart
    assert "[ExpertDrums]" in chart

    # Verify notes span close to the full 4-second duration
    # At 128 BPM, 1 beat = 60/128 = 0.46875s, 1 bar = 1.875s
    # 4 seconds = ~2.13 bars, so last eighth-note should be near tick ~4s worth
    expert_section = chart.split("[ExpertDrums]")[1].split("}")[0]
    note_lines = [line.strip() for line in expert_section.strip().splitlines() if "= N" in line]
    ticks = [int(line.split("=")[0].strip()) for line in note_lines]
    # Resolution=192, 128 BPM: 1 tick = 0.46875/192 = 0.00244s
    # 4 seconds ≈ tick 1638
    max_tick = max(ticks) if ticks else 0
    assert max_tick >= 1200  # at least ~2.9 seconds worth of ticks
    assert max_tick <= 2000  # not way past the duration


def test_manual_bpm_bypasses_tempo_detection(tmp_path: Path) -> None:
    """When bpm is provided, detect_beat_grid should not be called."""
    source_audio = _make_wav(tmp_path, "song.wav", duration_sec=2.0)

    with patch("drumcharter.pipeline.detect_beat_grid") as mock_detect:
        folder = pipeline.generate_drum_chart_folder(
            source_audio=source_audio,
            output_parent=tmp_path / "out",
            song_name="Test",
            artist_name="Test",
            bpm=140.0,
        )
        mock_detect.assert_not_called()

    chart = (folder / "notes.chart").read_text(encoding="utf-8")
    assert "0 = B 140000" in chart


def test_detected_beat_times_drive_variable_sync_track(tmp_path: Path) -> None:
    source_audio = _make_wav(tmp_path, "song.wav", duration_sec=2.0)

    class FakeBeatGrid:
        bpm = 120.0
        beat_times = [0.5, 1.0, 1.4]

    with patch("drumcharter.pipeline.detect_beat_grid", return_value=FakeBeatGrid()):
        folder = pipeline.generate_drum_chart_folder(
            source_audio=source_audio,
            output_parent=tmp_path / "out",
            song_name="Tempo",
            artist_name="Tests",
        )

    chart = (folder / "notes.chart").read_text(encoding="utf-8")
    assert "0 = B 120000" in chart
    assert "384 = B 150000" in chart
