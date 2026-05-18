"""Tests for the ADTOF inference backend."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest
from click.testing import CliRunner

from audiotochart.cli import BACKENDS, _resolve_backend, cli
from audiotochart.drums import DrumHit
from audiotochart.inference.adtof import TranscriptionError


def _make_wav(tmp_path: Path, name: str, duration_sec: float, sample_rate: int = 44100) -> Path:
    path = tmp_path / name
    num_samples = int(duration_sec * sample_rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_samples)
    return path


def _fake_transcribe(drums_wav: Path, midi_out: Path, **kwargs) -> Path:
    """Write a minimal valid MIDI with one kick note at 0.0s."""
    pretty_midi = pytest.importorskip("pretty_midi")
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    midi.instruments.append(inst)
    midi.write(str(midi_out))
    return Path(midi_out)


def _fake_transcribe_empty(drums_wav: Path, midi_out: Path, **kwargs) -> Path:
    raise TranscriptionError(
        "ADTOF produced an empty MIDI file — no drum hits detected"
    )


def test_transcriber_returns_hits(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "audiotochart.inference.adtof._transcribe_drums_to_midi",
        _fake_transcribe,
    )
    from audiotochart.inference.adtof import AdtofTranscriber

    audio = _make_wav(tmp_path, "song.wav", duration_sec=4.0)
    hits = AdtofTranscriber().transcribe(audio)

    assert isinstance(hits, list)
    assert len(hits) == 1
    assert hits[0] == DrumHit(time_sec=0.0, instrument="kick", confidence=100 / 127)


def test_empty_midi_raises_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "audiotochart.inference.adtof._transcribe_drums_to_midi",
        _fake_transcribe_empty,
    )
    from audiotochart.inference.adtof import AdtofTranscriber, TranscriptionError

    audio = _make_wav(tmp_path, "song.wav", duration_sec=4.0)
    with pytest.raises(TranscriptionError, match="empty MIDI"):
        AdtofTranscriber().transcribe(audio)


def test_missing_deps_clear_error(tmp_path: Path) -> None:
    """When ADTOF deps are absent, the user gets a clear install hint."""
    try:
        import torch  # noqa: F401
        import adtof_pytorch  # noqa: F401
        pytest.skip("ADTOF deps are installed — cannot test missing-deps path")
    except ImportError:
        pass

    from audiotochart.inference.adtof import AdtofTranscriber

    audio = _make_wav(tmp_path, "song.wav", duration_sec=4.0)
    with pytest.raises(ImportError, match="uv sync --extra ai"):
        AdtofTranscriber().transcribe(audio)


def test_resolve_backend_available() -> None:
    cls = BACKENDS.get("adtof")
    if cls is None:
        with pytest.raises(SystemExit):
            _resolve_backend("adtof")
    else:
        assert _resolve_backend("adtof") is cls


def test_cli_backend_adtof(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "audiotochart.inference.adtof._transcribe_drums_to_midi",
        _fake_transcribe,
    )

    audio = _make_wav(tmp_path, "song.wav", duration_sec=4.0)
    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(cli, [
        "generate", str(audio), "--backend", "adtof", "-o", str(out_dir),
        "--song", "Test", "--artist", "Tester", "--bpm", "120",
    ])

    if result.exit_code != 0:
        assert "uv sync --extra ai" in result.output
    else:
        chart = out_dir / "Tester - Test" / "notes.chart"
        assert chart.exists()
        assert "[ExpertDrums]" in chart.read_text(encoding="utf-8")


def test_cli_backend_fake_is_default(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "--help"])
    assert result.exit_code == 0
    assert "--backend" in result.output
    assert "fake" in result.output
    assert "adtof" in result.output
