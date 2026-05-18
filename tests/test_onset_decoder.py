from __future__ import annotations

import json
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from audiotochart.chart.drum_vocab import PRO8_LABELS
from audiotochart.inference.checkpoint import ModelBundle
from audiotochart.inference.model import ModelTranscriber
from audiotochart.inference.onset_decoder import (
    CHORD_NULL,
    aggregate_chord_features,
    build_chord_vocabulary,
    build_onset_feature_rows,
    classes_to_mask,
    load_chord_decoder_bundle,
    mask_to_classes,
    mask_to_name,
    OnsetDecoderError,
)


def _make_wav(tmp_path: Path, name: str = "song.wav") -> Path:
    path = tmp_path / name
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(4000)
        wf.writeframes(b"\x00\x00" * 800)
    return path


def test_onset_feature_rows_match_expected_layout() -> None:
    acts = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        ],
        dtype=np.float32,
    )

    rows = build_onset_feature_rows(
        acts,
        onset_frames=[1],
        onset_classes=[4],
        thresholds=[0.1] * 8,
    )

    assert rows.shape == (1, 18)
    np.testing.assert_allclose(rows[0, :8], acts[1], atol=1e-6)
    assert rows[0, 8 + 4] == 1.0
    assert rows[0, 16] == acts[1, 4]
    assert rows[0, 17] == pytest.approx(float(acts[1, 4] - 0.1))


def test_chord_vocabulary_maps_masks_and_classes() -> None:
    vocab = build_chord_vocabulary()

    assert vocab.vocab_size == 54
    mask = classes_to_mask([0, 1, 6])
    token = vocab.token_for_mask(mask)

    assert token is not None
    assert vocab.mask_for_token(token) == mask
    assert mask_to_classes(mask) == [0, 1, 6]
    assert mask_to_name(mask) == "KRGc"
    assert vocab.mask_for_token(CHORD_NULL) is None


def test_chord_vocabulary_blocklist_policy() -> None:
    vocab = build_chord_vocabulary(blocklist_policy="kick_two_cymbals_no_snare")

    assert vocab.token_for_mask(classes_to_mask([0, 2, 4])) is None
    assert vocab.token_for_mask(classes_to_mask([0, 1, 2])) is not None


def test_aggregate_chord_features() -> None:
    rows = np.zeros((2, 18), dtype=np.float32)
    rows[0, 0] = 0.2
    rows[1, 0] = 0.8
    rows[0, 16] = 0.4
    rows[1, 17] = 0.3

    out = aggregate_chord_features(rows, [2, 4])

    assert out.shape == (18,)
    assert out[0] == pytest.approx(0.8)
    assert out[8 + 2] == 1.0
    assert out[8 + 4] == 1.0
    assert out[16] == pytest.approx(0.4)
    assert out[17] == pytest.approx(0.3)


def test_decoder_config_rejects_structure_decoder(tmp_path: Path) -> None:
    decoder_dir = tmp_path / "decoder"
    decoder_dir.mkdir()
    (decoder_dir / "config.json").write_text(
        json.dumps({"use_structure": True, "chord_masks": [1], "vocab_size": 4})
    )
    (decoder_dir / "best.pt").write_bytes(b"not used")

    with pytest.raises(OnsetDecoderError, match="use_structure=true"):
        load_chord_decoder_bundle(
            decoder_dir,
            base_bundle=ModelBundle(model=object(), labels=PRO8_LABELS),
            device="cpu",
        )


def test_decoder_loader_missing_config_and_best_are_clean(tmp_path: Path) -> None:
    missing_config = tmp_path / "missing-config"
    missing_config.mkdir()

    with pytest.raises(OnsetDecoderError, match="Missing config.json"):
        load_chord_decoder_bundle(
            missing_config,
            base_bundle=ModelBundle(model=object(), labels=PRO8_LABELS),
            device="cpu",
        )

    missing_best = tmp_path / "missing-best"
    missing_best.mkdir()
    (missing_best / "config.json").write_text(
        json.dumps({"chord_masks": [1], "vocab_size": 4})
    )

    with pytest.raises(OnsetDecoderError, match="Missing best.pt"):
        load_chord_decoder_bundle(
            missing_best,
            base_bundle=ModelBundle(model=object(), labels=PRO8_LABELS),
            device="cpu",
        )


def test_decoder_loader_accepts_training_decoder_state(monkeypatch, tmp_path: Path) -> None:
    import torch
    import torch.nn as nn

    from audiotochart.onset_decoder_common import build_onset_conditioned_model

    class FakeEncoder(nn.Module):
        pass

    decoder_dir = tmp_path / "decoder"
    decoder_dir.mkdir()
    config = {
        "chord_masks": [1],
        "vocab_size": 4,
        "encoder_dim": 4,
        "d_model": 8,
        "n_heads": 2,
        "n_layers": 1,
        "d_ff": 16,
        "use_onset_features": True,
        "onset_feature_dim": 18,
    }
    (decoder_dir / "config.json").write_text(json.dumps(config))
    model = build_onset_conditioned_model(FakeEncoder(), config=config, vocab_size=4)
    torch.save({"decoder_state": model.decoder.state_dict(), "config": config}, decoder_dir / "best.pt")
    monkeypatch.setattr(
        "audiotochart.inference.onset_decoder._build_decoder_encoder",
        FakeEncoder,
    )

    bundle = load_chord_decoder_bundle(
        decoder_dir,
        base_bundle=ModelBundle(model=FakeEncoder(), labels=PRO8_LABELS),
        device="cpu",
    )

    assert bundle.vocab.vocab_size == 4
    assert bundle.model.training is False


def test_model_transcriber_applies_decoder_before_tom_consistency(tmp_path: Path) -> None:
    import torch

    audio = _make_wav(tmp_path)
    spec = np.zeros((4, 4, 1), dtype=np.float32)

    class _FakeModel:
        def __call__(self, x):
            logits = torch.full((x.shape[0], x.shape[1], 8), -10.0)
            logits[:, 1, 3] = 10.0
            return logits

    bundle = ModelBundle(
        model=_FakeModel(),
        labels=list(PRO8_LABELS),
        config={
            "architecture": "simple_cnn",
            "thresholds": [0.5] * 8,
            "min_peak_distance": 1,
        },
        device="cpu",
    )

    def _fake_refine(_decoder_bundle, onsets, *_args, **_kwargs):
        assert [(round(t, 2), c) for t, c, _conf in onsets] == [(0.1, 3)]
        return [(0.1, 5, 0.8)]

    def _fake_tom_consistency(onsets, *_args, **_kwargs):
        assert onsets == [(0.1, 5)]
        return onsets, {"n_reassigned": 0, "n_tom_hits": 1, "convention": []}

    with (
        patch.object(ModelTranscriber, "_ensure_loaded", return_value=bundle),
        patch.object(ModelTranscriber, "_ensure_onset_decoder_loaded", return_value=object()),
        patch("audiotochart.inference.model._compute_mel_spectrogram", return_value=(spec, 10.0)),
        patch("audiotochart.inference.onset_decoder.refine_chord_onsets", side_effect=_fake_refine),
        patch(
            "audiotochart.inference.tom_consistency.apply_tom_consistency",
            side_effect=_fake_tom_consistency,
        ),
    ):
        transcriber = ModelTranscriber(
            model_dir=tmp_path / "model",
            onset_decoder_dir=tmp_path / "decoder",
            tom_consistency=True,
        )
        hits = transcriber.transcribe(audio)

    assert [(hit.time_sec, hit.instrument) for hit in hits] == [(0.1, "tom_blue")]
