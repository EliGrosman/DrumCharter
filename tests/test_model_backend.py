from __future__ import annotations

import json
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from audiotochart.drums import DrumHit
from audiotochart.inference.checkpoint import (
    ModelLoadError,
    PRO8_ARCHITECTURE,
    load_model_bundle,
    PRO8_LABELS,
    _build_model_for_architecture,
)
from audiotochart.inference.model import (
    ModelTranscriber,
    ModelTranscriberError,
    _compute_adtof_spectrogram,
    _pick_peaks_simple,
    _pick_peaks_original,
)


def _make_wav(tmp_path: Path, name: str, duration_sec: float, sample_rate: int = 44100) -> Path:
    path = tmp_path / name
    num_samples = int(duration_sec * sample_rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_samples)
    return path


def _make_model_dir(
    tmp_path: Path,
    *,
    architecture: str = "simple_cnn",
    num_classes: int = 3,
    labels: list[str] | None = None,
    n_mels: int = 10,
    sample_rate: int = 8000,
    hop_length: int = 160,
    thresholds: list[float] | None = None,
    extra_config: dict | None = None,
) -> Path:
    import torch

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    if labels is None:
        labels = ["kick", "snare", "hihat"]

    config = {
        "architecture": architecture,
        "num_classes": num_classes,
        "n_mels": n_mels,
        "sample_rate": sample_rate,
        "hop_length": hop_length,
        "n_fft": 400,
        "fmin": 20.0,
        "fmax": 4000.0,
        "thresholds": thresholds or [0.01] * num_classes,
        "min_peak_distance": 2,
    }
    if extra_config:
        config.update(extra_config)
    (model_dir / "config.json").write_text(json.dumps(config))

    model = _build_model_for_architecture(architecture, config)
    torch.save(model.state_dict(), model_dir / "weights.pt")

    (model_dir / "labels.json").write_text(json.dumps(labels))

    return model_dir


# ---------------------------------------------------------------------------
# checkpoint.py — loading errors
# ---------------------------------------------------------------------------


def test_missing_model_dir() -> None:
    with pytest.raises(ModelLoadError, match="not found"):
        load_model_bundle(Path("/nonexistent/path"))


def test_missing_config_json(tmp_path: Path) -> None:
    d = tmp_path / "model"
    d.mkdir()
    with pytest.raises(ModelLoadError, match="Missing config.json"):
        load_model_bundle(d)


def test_missing_weights(tmp_path: Path) -> None:
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"architecture": "simple_cnn"}))
    with pytest.raises(ModelLoadError, match="No weights file found"):
        load_model_bundle(d)


def test_missing_labels(tmp_path: Path) -> None:
    import torch
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"architecture": "simple_cnn", "num_classes": 3}))
    torch.save({"dummy": torch.zeros(1)}, d / "weights.pt")
    with pytest.raises(ModelLoadError, match="Missing labels.json"):
        load_model_bundle(d)


def test_unknown_architecture(tmp_path: Path) -> None:
    import torch
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"architecture": "bogus_arch"}))
    torch.save({"dummy": torch.zeros(1)}, d / "weights.pt")
    (d / "labels.json").write_text(json.dumps(["kick"]))
    with pytest.raises(ModelLoadError, match="Unknown architecture"):
        load_model_bundle(d)


def test_loads_from_best_pt(tmp_path: Path) -> None:
    """Should pick up best.pt when weights.pt is absent."""
    import torch
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "architecture": "simple_cnn", "num_classes": 3, "n_mels": 10,
    }))
    m = _build_model_for_architecture("simple_cnn", {"num_classes": 3, "n_mels": 10})
    torch.save({"model_state": m.state_dict()}, d / "best.pt")
    (d / "labels.json").write_text(json.dumps(["kick", "snare", "hihat"]))
    bundle = load_model_bundle(d)
    assert bundle.labels == ["kick", "snare", "hihat"]


def test_labels_from_variant(tmp_path: Path) -> None:
    """When labels.json is missing, derive from variant."""
    import torch
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "architecture": "simple_cnn", "variant": "pro8", "num_classes": 8, "n_mels": 84,
    }))
    m = _build_model_for_architecture("simple_cnn", {"num_classes": 8, "n_mels": 84})
    torch.save(m.state_dict(), d / "weights.pt")
    bundle = load_model_bundle(d)
    assert bundle.labels == PRO8_LABELS


def test_explicit_label_bundle_can_have_fewer_outputs(tmp_path: Path) -> None:
    """Custom adapters can provide any supported label subset via labels.json."""
    labels = ["kick", "snare", "hihat", "ride", "crash"]
    model_dir = _make_model_dir(
        tmp_path,
        num_classes=len(labels),
        labels=labels,
        n_mels=84,
    )

    bundle = load_model_bundle(model_dir)

    assert bundle.labels == labels


def test_unknown_variant_errors(tmp_path: Path) -> None:
    """An unrecognized variant should produce a clear error."""
    import torch
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "architecture": "simple_cnn", "variant": "bogus_variant", "num_classes": 3,
    }))
    torch.save({"dummy": torch.zeros(1)}, d / "weights.pt")
    (d / "labels.json").write_text(json.dumps(["kick", "snare", "hihat"]))
    with pytest.raises(ModelLoadError, match="Unsupported variant"):
        load_model_bundle(d)


def test_loads_checkpoint_with_original_metadata_pattern(tmp_path: Path) -> None:
    """Load a checkpoint using the same metadata pattern as phase3_harmonix_b:
    variant-only config (no explicit architecture), best.pt with model_state
    wrapper, thresholds.json, and no labels.json (variant-derived).
    """
    import torch
    num_classes = 8
    thresholds = [0.12, 0.28, 0.24, 0.16, 0.48, 0.22, 0.1, 0.1]

    d = tmp_path / "model"
    d.mkdir()

    (d / "config.json").write_text(json.dumps({
        "variant": "pro8",
        "num_classes": num_classes,
    }))

    (d / "thresholds.json").write_text(json.dumps({
        "thresholds": thresholds,
        "val_f_scores": [0.95] * num_classes,
        "tolerance_frames": 2,
    }))

    model = _build_model_for_architecture(PRO8_ARCHITECTURE, {"num_classes": num_classes})
    torch.save({"model_state": model.state_dict()}, d / "best.pt")

    bundle = load_model_bundle(d)

    assert bundle.labels == PRO8_LABELS
    assert bundle.config["thresholds"] == thresholds
    assert bundle.model is not None
    assert not bundle.model.training


def test_thresholds_json_length_mismatch_errors(tmp_path: Path) -> None:
    model_dir = _make_model_dir(
        tmp_path,
        num_classes=3,
        labels=["kick", "snare", "hihat"],
    )
    (model_dir / "thresholds.json").write_text(json.dumps({
        "thresholds": [0.1, 0.2],
    }))

    with pytest.raises(ModelLoadError, match="thresholds.*2 entries.*expected 3"):
        load_model_bundle(model_dir)


def test_confidence_gates_length_mismatch_errors(tmp_path: Path) -> None:
    model_dir = _make_model_dir(
        tmp_path,
        num_classes=3,
        labels=["kick", "snare", "hihat"],
    )
    (model_dir / "thresholds.json").write_text(json.dumps({
        "thresholds": [0.1, 0.2, 0.3],
        "confidence_gates": [None, 0.9],
    }))

    with pytest.raises(ModelLoadError, match="confidence_gates.*2 entries.*expected 3"):
        load_model_bundle(model_dir)


# ---------------------------------------------------------------------------
# ModelTranscriber — transcribe path
# ---------------------------------------------------------------------------


def test_transcriber_no_model_dir_errors() -> None:
    t = ModelTranscriber()
    with pytest.raises(ModelTranscriberError, match="no model_dir"):
        t._ensure_loaded()


def test_transcriber_missing_audio_file(tmp_path: Path) -> None:
    model_dir = _make_model_dir(tmp_path)
    t = ModelTranscriber(model_dir=model_dir)
    with pytest.raises(FileNotFoundError, match="not found"):
        t.transcribe(tmp_path / "nope.wav")


def test_transcriber_returns_drumhits(tmp_path: Path) -> None:
    """With a tiny simple_cnn model, verify the hit list structure."""
    import torch

    model_dir = _make_model_dir(
        tmp_path,
        labels=["kick", "snare", "hihat"],
        n_mels=4,
        sample_rate=4000,
        hop_length=200,
        thresholds=[0.01, 0.01, 0.01],
    )
    audio = _make_wav(tmp_path, "song.wav", duration_sec=0.2, sample_rate=4000)

    t = ModelTranscriber(model_dir=model_dir)
    hits = t.transcribe(audio)

    assert isinstance(hits, list)
    if hits:
        assert all(isinstance(h, DrumHit) for h in hits)
        instruments = {h.instrument for h in hits}
        assert instruments.issubset({"kick", "snare", "hihat"})


def test_transcriber_label_mapping(tmp_path: Path) -> None:
    """Verify that class 0 maps to the first label, class 1 to second, etc."""
    import torch

    custom_labels = ["crash", "ride"]
    model_dir = _make_model_dir(
        tmp_path,
        num_classes=2,
        labels=custom_labels,
        thresholds=[0.5, 0.5],
        n_mels=4,
        sample_rate=4000,
        hop_length=200,
    )
    audio = _make_wav(tmp_path, "song.wav", duration_sec=0.2, sample_rate=4000)

    bundle = load_model_bundle(model_dir)

    def _fake_forward(x):
        import torch as _torch
        B, T = x.shape[0], x.shape[1]
        logits = _torch.zeros(B, T, 2)
        logits[:, :, 0] = 10.0
        logits[:, :, 1] = -10.0
        return logits

    bundle.model.forward = _fake_forward

    with patch.object(ModelTranscriber, "_ensure_loaded", return_value=bundle):
        t = ModelTranscriber(model_dir=model_dir)
        hits = t.transcribe(audio)

    assert len(hits) > 0
    for h in hits:
        assert h.instrument == "crash"


def test_transcriber_empty_output_when_no_peaks(tmp_path: Path) -> None:
    """When thresholds are very high, no hits should be produced."""
    model_dir = _make_model_dir(
        tmp_path,
        num_classes=1,
        labels=["kick"],
        thresholds=[999.0],
        n_mels=4,
        sample_rate=4000,
        hop_length=200,
    )
    audio = _make_wav(tmp_path, "song.wav", duration_sec=0.2, sample_rate=4000)

    t = ModelTranscriber(model_dir=model_dir)
    hits = t.transcribe(audio)
    assert hits == []


def test_peak_picker_keeps_local_max_after_threshold_crossing() -> None:
    acts = np.array([[0.1], [0.6], [0.9], [0.2]], dtype=np.float32)

    hits = _pick_peaks_simple(
        acts,
        thresholds=[0.5],
        min_distance=2,
        fps=10.0,
        num_classes=1,
    )

    assert len(hits) == 1
    assert hits[0][0] == pytest.approx(0.2)
    assert hits[0][1] == 0
    assert hits[0][2] == pytest.approx(0.9)


def test_pick_peaks_original_matches_expected_peak_frames() -> None:
    acts = np.array(
        [
            [0.6, 0.1],
            [0.4, 0.7],
            [0.8, 0.3],
            [0.8, 0.6],
            [0.7, 0.2],
            [0.2, 0.6],
            [0.9, 0.6],
        ],
        dtype=np.float32,
    )

    result = _pick_peaks_original(acts, thresholds=[0.5, 0.6], fps=10.0)

    assert result == [
        (0.0, 0, pytest.approx(0.6)),
        (0.1, 1, pytest.approx(0.7)),
        (0.2, 0, pytest.approx(0.8)),
        (0.3, 1, pytest.approx(0.6)),
        (0.5, 1, pytest.approx(0.6)),
        (0.6, 0, pytest.approx(0.9)),
    ]


def test_adtof_spectrogram_uses_training_time_normalization(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("adtof_pytorch.audio")
    pytest.importorskip("librosa")

    captured_audio: list[np.ndarray] = []

    class _FakeAudioProcessor:
        fps = 100

        def compute_stft(self, audio: np.ndarray) -> np.ndarray:
            captured_audio.append(audio.copy())
            return np.ones((2, 3), dtype=np.float32)

        def apply_filterbank(self, stft: np.ndarray) -> np.ndarray:
            assert stft.shape == (2, 3)
            return np.ones((4, 3), dtype=np.float32)

    monkeypatch.setattr(
        "librosa.load",
        lambda *_args, **_kwargs: (np.array([0.5, -1.0], dtype=np.float32), 44100),
    )
    monkeypatch.setattr("adtof_pytorch.audio.AudioProcessor", _FakeAudioProcessor)

    audio = _make_wav(tmp_path, "song.wav", duration_sec=0.01)
    spec, fps = _compute_adtof_spectrogram(audio)

    assert fps == 100.0
    assert spec.shape == (3, 4, 1)
    assert np.max(np.abs(captured_audio[0])) == pytest.approx(0.95)


def test_confidence_gates_suppress_class_when_below_gate(tmp_path: Path) -> None:
    """A class whose max activation is below its confidence gate should yield zero hits."""
    import torch
    model_dir = _make_model_dir(
        tmp_path,
        labels=["kick", "snare", "hihat"],
        n_mels=4,
        sample_rate=4000,
        hop_length=200,
        thresholds=[0.01, 0.01, 0.01],
        extra_config={
            "confidence_gates": [None, 0.9, None],
        },
    )
    bundle = load_model_bundle(model_dir)

    def _fake_forward(x):
        import torch as _torch
        B, T = x.shape[0], x.shape[1]
        logits = _torch.zeros(B, T, 3)
        logits[:, :, 0] = 10.0
        logits[:, :, 1] = -10.0
        logits[:, :, 2] = 10.0
        return logits

    bundle.model.forward = _fake_forward

    from unittest.mock import patch
    with patch.object(ModelTranscriber, "_ensure_loaded", return_value=bundle):
        t = ModelTranscriber(model_dir=model_dir)
        audio = _make_wav(tmp_path, "song.wav", duration_sec=0.2, sample_rate=4000)
        hits = t.transcribe(audio)

    instruments = {h.instrument for h in hits}
    assert "snare" not in instruments
    assert "kick" in instruments
    assert "hihat" in instruments


def test_confidence_gates_absent_does_nothing(tmp_path: Path) -> None:
    """When no confidence_gates in config, all classes should produce hits as normal."""
    import torch
    model_dir = _make_model_dir(
        tmp_path,
        num_classes=2,
        labels=["kick", "snare"],
        n_mels=4,
        sample_rate=4000,
        hop_length=200,
        thresholds=[0.01, 0.01],
    )
    bundle = load_model_bundle(model_dir)

    def _fake_forward(x):
        import torch as _torch
        B, T = x.shape[0], x.shape[1]
        logits = _torch.ones(B, T, 2) * 10.0
        return logits

    bundle.model.forward = _fake_forward

    from unittest.mock import patch
    with patch.object(ModelTranscriber, "_ensure_loaded", return_value=bundle):
        t = ModelTranscriber(model_dir=model_dir)
        audio = _make_wav(tmp_path, "song.wav", duration_sec=0.2, sample_rate=4000)
        hits = t.transcribe(audio)

    assert len(hits) > 0
    instruments = {h.instrument for h in hits}
    assert "kick" in instruments
    assert "snare" in instruments


def test_confidence_gates_partial_gating(tmp_path: Path) -> None:
    """Only specified gates are applied; classes without gates pass through."""
    import torch
    model_dir = _make_model_dir(
        tmp_path,
        labels=["kick", "snare", "hihat"],
        n_mels=4,
        sample_rate=4000,
        hop_length=200,
        thresholds=[0.01, 0.01, 0.01],
        extra_config={
            "confidence_gates": [0.9, None, None],
        },
    )
    bundle = load_model_bundle(model_dir)

    def _fake_forward(x):
        import torch as _torch
        B, T = x.shape[0], x.shape[1]
        logits = _torch.zeros(B, T, 3)
        logits[:, :, 0] = -10.0
        logits[:, :, 1] = 10.0
        logits[:, :, 2] = 10.0
        return logits

    bundle.model.forward = _fake_forward

    from unittest.mock import patch
    with patch.object(ModelTranscriber, "_ensure_loaded", return_value=bundle):
        t = ModelTranscriber(model_dir=model_dir)
        audio = _make_wav(tmp_path, "song.wav", duration_sec=0.2, sample_rate=4000)
        hits = t.transcribe(audio)

    instruments = {h.instrument for h in hits}
    assert "kick" not in instruments
    assert "snare" in instruments
    assert "hihat" in instruments


def test_tom_consistency_is_off_by_default(tmp_path: Path) -> None:
    model_dir = _make_model_dir(
        tmp_path,
        num_classes=8,
        labels=PRO8_LABELS,
        n_mels=4,
        sample_rate=4000,
        hop_length=200,
        thresholds=[0.01] * 8,
    )
    bundle = load_model_bundle(model_dir)

    def _fake_forward(x):
        import torch as _torch
        B, T = x.shape[0], x.shape[1]
        return _torch.ones(B, T, 8) * 10.0

    bundle.model.forward = _fake_forward

    with (
        patch.object(ModelTranscriber, "_ensure_loaded", return_value=bundle),
        patch("audiotochart.inference.tom_consistency.apply_tom_consistency") as apply_tc,
    ):
        t = ModelTranscriber(model_dir=model_dir)
        audio = _make_wav(tmp_path, "song.wav", duration_sec=0.2, sample_rate=4000)
        hits = t.transcribe(audio)

    assert hits
    apply_tc.assert_not_called()


def test_tom_consistency_opt_in_runs_for_pro8_labels(tmp_path: Path) -> None:
    model_dir = _make_model_dir(
        tmp_path,
        num_classes=8,
        labels=PRO8_LABELS,
        n_mels=4,
        sample_rate=4000,
        hop_length=200,
        thresholds=[0.01] * 8,
    )
    bundle = load_model_bundle(model_dir)

    def _fake_forward(x):
        import torch as _torch
        B, T = x.shape[0], x.shape[1]
        return _torch.ones(B, T, 8) * 10.0

    bundle.model.forward = _fake_forward

    with (
        patch.object(ModelTranscriber, "_ensure_loaded", return_value=bundle),
        patch("audiotochart.inference.tom_consistency.apply_tom_consistency") as apply_tc,
    ):
        apply_tc.side_effect = lambda onsets, acts, **_: (
            onsets,
            {"n_reassigned": 0, "n_tom_hits": 0, "convention": []},
        )
        t = ModelTranscriber(model_dir=model_dir, tom_consistency=True)
        audio = _make_wav(tmp_path, "song.wav", duration_sec=0.2, sample_rate=4000)
        hits = t.transcribe(audio)

    assert hits
    apply_tc.assert_called_once()
    assert apply_tc.call_args.kwargs["fps"] == pytest.approx(20.0)


def test_tom_consistency_skips_custom_8_class_models(tmp_path: Path) -> None:
    labels = [f"class_{i}" for i in range(8)]
    model_dir = _make_model_dir(
        tmp_path,
        num_classes=8,
        labels=labels,
        n_mels=4,
        sample_rate=4000,
        hop_length=200,
        thresholds=[0.01] * 8,
    )
    bundle = load_model_bundle(model_dir)

    def _fake_forward(x):
        import torch as _torch
        B, T = x.shape[0], x.shape[1]
        return _torch.ones(B, T, 8) * 10.0

    bundle.model.forward = _fake_forward

    with (
        patch.object(ModelTranscriber, "_ensure_loaded", return_value=bundle),
        patch("audiotochart.inference.tom_consistency.apply_tom_consistency") as apply_tc,
    ):
        t = ModelTranscriber(model_dir=model_dir, tom_consistency=True)
        audio = _make_wav(tmp_path, "song.wav", duration_sec=0.2, sample_rate=4000)
        hits = t.transcribe(audio)

    assert hits
    apply_tc.assert_not_called()


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_backend_model_missing_model_dir(tmp_path: Path) -> None:
    from click.testing import CliRunner
    from audiotochart.cli import cli

    audio = _make_wav(tmp_path, "song.wav", duration_sec=2.0)
    runner = CliRunner()
    result = runner.invoke(cli, [
        "generate", str(audio), "--backend", "model", "-o", str(tmp_path / "out"),
        "--song", "Test", "--artist", "Tester", "--bpm", "120",
    ])
    assert result.exit_code != 0
    assert "--model-dir" in result.output


def test_cli_backend_model_load_error_is_clean(tmp_path: Path) -> None:
    from click.testing import CliRunner
    from audiotochart.cli import cli

    audio = _make_wav(tmp_path, "song.wav", duration_sec=2.0)
    model_dir = tmp_path / "broken-model"
    model_dir.mkdir()
    runner = CliRunner()

    result = runner.invoke(cli, [
        "generate", str(audio), "--backend", "model",
        "--model-dir", str(model_dir),
        "-o", str(tmp_path / "out"),
        "--song", "Test", "--artist", "Tester", "--bpm", "120",
    ])

    assert result.exit_code != 0
    assert "Missing config.json" in result.output
    assert "Traceback" not in result.output


def test_cli_backend_model_help_shows_option(tmp_path: Path) -> None:
    from click.testing import CliRunner
    from audiotochart.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "--help"])
    assert result.exit_code == 0
    assert "--model-dir" in result.output
    assert "model" in result.output
    assert "--tom-consistency" in result.output
    assert "--no-tom-consistency" in result.output
