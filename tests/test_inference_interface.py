from __future__ import annotations

import wave
from pathlib import Path
from unittest.mock import Mock

from click.testing import CliRunner

from audiotochart.cli import cli
from audiotochart.drums import DrumHit
from audiotochart.inference.base import DrumTranscriber
from audiotochart.inference.fake import FakeTranscriber


def _make_wav(tmp_path: Path, name: str, duration_sec: float, sample_rate: int = 44100) -> Path:
    path = tmp_path / name
    num_samples = int(duration_sec * sample_rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_samples)
    return path


def test_fake_transcriber_returns_hits(tmp_path: Path) -> None:
    audio = _make_wav(tmp_path, "song.wav", duration_sec=8.0)
    transcriber = FakeTranscriber()
    hits = transcriber.transcribe(audio)
    assert isinstance(hits, list)
    assert len(hits) > 0
    assert all(isinstance(h, DrumHit) for h in hits)


def test_pipeline_calls_transcriber_once(tmp_path: Path) -> None:
    import audiotochart.pipeline as pipeline
    audio = _make_wav(tmp_path, "song.wav", duration_sec=4.0)
    mock_transcriber = Mock(spec=DrumTranscriber)
    mock_transcriber.transcribe.return_value = [
        DrumHit(0.0, "kick"),
        DrumHit(0.5, "hihat"),
    ]
    pipeline.generate_drum_chart_folder(
        source_audio=audio,
        output_parent=tmp_path / "out",
        song_name="Song",
        artist_name="Artist",
        bpm=120.0,
        transcriber=mock_transcriber,
    )
    mock_transcriber.transcribe.assert_called_once_with(audio)


def test_backend_output_flows_to_chart(tmp_path: Path) -> None:
    import audiotochart.pipeline as pipeline
    audio = _make_wav(tmp_path, "song.wav", duration_sec=4.0)
    folder = pipeline.generate_drum_chart_folder(
        source_audio=audio,
        output_parent=tmp_path / "out",
        song_name="Song",
        artist_name="Artist",
        bpm=120.0,
    )
    chart = (folder / "notes.chart").read_text(encoding="utf-8")
    assert "[ExpertDrums]" in chart
    assert "= N" in chart


def test_unknown_backend_cli_error() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "--backend", "bogus"])
    assert result.exit_code != 0
    assert "bogus" in result.output.lower() or "invalid choice" in result.output.lower()


def test_fake_transcriber_produces_all_instruments(tmp_path: Path) -> None:
    audio = _make_wav(tmp_path, "song.wav", duration_sec=4.0)
    transcriber = FakeTranscriber()
    hits = transcriber.transcribe(audio)
    instruments = {h.instrument for h in hits}
    assert "kick" in instruments
    assert "snare" in instruments
    assert "hihat" in instruments
