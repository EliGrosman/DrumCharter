from __future__ import annotations

import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_wav(tmp_path: Path, name: str, duration_sec: float = 1.0, sample_rate: int = 44100) -> Path:
    path = tmp_path / name
    num_samples = int(duration_sec * sample_rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_samples)
    return path


def _mock_demucs_env():
    demucs = MagicMock()
    demucs.apply.apply_model = MagicMock()
    demucs.pretrained.get_model = MagicMock()
    demucs.separate.load_track = MagicMock()

    sources_mock = ["drums", "bass", "other", "vocals"]
    model_mock = MagicMock()
    model_mock.sources = sources_mock
    model_mock.audio_channels = 2
    model_mock.samplerate = 44100
    demucs.pretrained.get_model.return_value = model_mock

    torch = MagicMock()
    torch.cuda.is_available.return_value = False

    soundfile = MagicMock()

    modules = {
        "demucs": demucs,
        "demucs.apply": demucs.apply,
        "demucs.pretrained": demucs.pretrained,
        "demucs.separate": demucs.separate,
        "torch": torch,
        "soundfile": soundfile,
    }
    originals = {}
    for mod_name, mock in modules.items():
        originals[mod_name] = sys.modules.get(mod_name)
        sys.modules[mod_name] = mock
    return modules, originals


def _restore_modules(originals: dict[str, object]) -> None:
    for mod_name, original in originals.items():
        if original is None:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = original


class TestIsolateDrums:
    def test_writes_output_path(self, tmp_path: Path):
        modules, originals = _mock_demucs_env()
        try:
            from audiotochart.separation import isolate_drums

            audio = _make_wav(tmp_path, "input.wav")
            out = tmp_path / "drums.wav"

            tensor_mock = MagicMock()
            tensor_mock.cpu.return_value.clamp.return_value.numpy.return_value.T = [[0.0]]
            modules["demucs"].apply.apply_model.return_value = [tensor_mock]

            result = isolate_drums(audio, out, progress=False)
            assert result == out

            modules["soundfile"].write.assert_called_once()
        finally:
            _restore_modules(originals)

    def test_file_not_found(self, tmp_path: Path):
        from audiotochart.separation import isolate_drums

        with pytest.raises(FileNotFoundError):
            isolate_drums(tmp_path / "nonexistent.wav", tmp_path / "out.wav")


class TestPipelineIntegration:
    def test_skip_separation_passes_original_to_transcriber(self, tmp_path: Path):
        import audiotochart.pipeline as pipeline

        audio = _make_wav(tmp_path, "song.wav")
        transcriber = MagicMock()
        transcriber.transcribe.return_value = []

        pipeline.generate_drum_chart_folder(
            source_audio=audio,
            output_parent=tmp_path / "out",
            song_name="Test",
            artist_name="Test",
            bpm=120.0,
            transcriber=transcriber,
            separate_drums=False,
        )

        transcriber.transcribe.assert_called_once_with(audio)

    def test_separation_calls_isolate_drums(self, tmp_path: Path):
        import audiotochart.pipeline as pipeline

        audio = _make_wav(tmp_path, "song.wav")
        transcriber = MagicMock()
        transcriber.transcribe.return_value = []

        with patch("audiotochart.separation.isolate_drums") as mock_iso:
            mock_iso.return_value = tmp_path / "drums.wav"
            pipeline.generate_drum_chart_folder(
                source_audio=audio,
                output_parent=tmp_path / "out2",
                song_name="Test",
                artist_name="Test",
                bpm=120.0,
                transcriber=transcriber,
                separate_drums=True,
            )

        mock_iso.assert_called_once()
        transcriber.transcribe.assert_called_once()

    def test_temp_workdir_cleaned_by_default(self, tmp_path: Path):
        import audiotochart.pipeline as pipeline

        audio = _make_wav(tmp_path, "song.wav")
        transcriber = MagicMock()
        transcriber.transcribe.return_value = []

        with patch("audiotochart.separation.isolate_drums") as mock_iso:
            mock_iso.return_value = tmp_path / "drums.wav"
            with patch("audiotochart.pipeline.shutil.rmtree") as mock_rmtree:
                pipeline.generate_drum_chart_folder(
                    source_audio=audio,
                    output_parent=tmp_path / "out3",
                    song_name="Test",
                    artist_name="Test",
                    bpm=120.0,
                    transcriber=transcriber,
                    separate_drums=True,
                    keep_workdir=False,
                )
                mock_rmtree.assert_called_once()

    def test_keep_workdir_preserves_temp_dir(self, tmp_path: Path):
        import audiotochart.pipeline as pipeline

        audio = _make_wav(tmp_path, "song.wav")
        transcriber = MagicMock()
        transcriber.transcribe.return_value = []

        with patch("audiotochart.separation.isolate_drums") as mock_iso:
            mock_iso.return_value = tmp_path / "drums.wav"
            with patch("audiotochart.pipeline.shutil.rmtree") as mock_rmtree:
                pipeline.generate_drum_chart_folder(
                    source_audio=audio,
                    output_parent=tmp_path / "out4",
                    song_name="Test",
                    artist_name="Test",
                    bpm=120.0,
                    transcriber=transcriber,
                    separate_drums=True,
                    keep_workdir=True,
                )
                mock_rmtree.assert_not_called()
