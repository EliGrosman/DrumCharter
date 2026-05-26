from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from drumcharter.training.onset_decoder import (
    ChordDecoderTrainConfig,
    run_onset_decoder_training,
)


def _write_cache(tmp_path: Path, *, n_songs: int = 4) -> Path:
    cache_dir = tmp_path / "cache"
    spec_dir = cache_dir / "spectrograms"
    label_dir = cache_dir / "labels_pro8"
    spec_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    manifest = {}
    for idx in range(n_songs):
        song_hash = f"song-{idx}"
        spec = np.zeros((120, 4), dtype=np.float32)
        labels = np.zeros((120, 8), dtype=np.float32)
        labels[10 + idx, 0] = 1.0
        np.save(spec_dir / f"{song_hash}.npy", spec)
        np.save(label_dir / f"{song_hash}.npy", labels)
        manifest[song_hash] = {
            "song_name": f"Song {idx}",
            "num_frames": 120,
            "source_archive": "rb.zip",
        }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))
    return cache_dir


def test_hybrid_cqs_checkpoint_selection_uses_hybrid_metric(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cache_dir = _write_cache(tmp_path)

    class FakeDecoder(torch.nn.Module):
        def __init__(self, vocab_size: int) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(1, vocab_size)

        def forward(self, _enc, _frames, tgt_tokens, **_kwargs):
            batch, n_tokens = tgt_tokens.shape
            x = torch.ones((batch * n_tokens, 1), device=tgt_tokens.device)
            return self.proj(x).view(batch, n_tokens, -1)

    class FakeModel(torch.nn.Module):
        def __init__(self, vocab_size: int) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(1, 1)
            self.decoder = FakeDecoder(vocab_size)

        def forward(self, _spec, frames, onset_features, tgt_tokens, tgt_key_padding_mask=None):
            return self.decoder(None, frames, tgt_tokens, onset_features=onset_features)

    def fake_build_model(_encoder, *, config, vocab_size):
        return FakeModel(vocab_size)

    def fake_load_model_bundle(_model_dir, *, device):
        return SimpleNamespace(
            model=torch.nn.Linear(1, 1).to(device),
            labels=[f"class_{idx}" for idx in range(8)],
            config={"thresholds": [0.5] * 8},
            device=device,
        )

    class FakeReport:
        def __init__(self, score: float) -> None:
            self.n_songs = 1
            self.baseline_macro_f = 0.1
            self.hybrid_macro_f = score
            self.baseline_per_class = {}
            self.hybrid_per_class = {}
            self.baseline_tom_consistency = 1.0
            self.hybrid_tom_consistency = 1.0
            self.baseline_cqs = 0.1
            self.hybrid_cqs = score
            self.baseline_cqs_components = {}
            self.hybrid_cqs_components = {}

        def as_dict(self):
            return {
                "n_songs": self.n_songs,
                "baseline_macro_f": self.baseline_macro_f,
                "hybrid_macro_f": self.hybrid_macro_f,
                "baseline_cqs": self.baseline_cqs,
                "hybrid_cqs": self.hybrid_cqs,
            }

    scores = [0.2, 0.8, 0.8]

    def fake_evaluate(*_args, **_kwargs):
        return FakeReport(scores.pop(0))

    monkeypatch.setattr(
        "drumcharter.training.onset_decoder.build_onset_conditioned_model",
        fake_build_model,
    )
    monkeypatch.setattr(
        "drumcharter.inference.checkpoint.load_model_bundle",
        fake_load_model_bundle,
    )
    monkeypatch.setattr(
        "drumcharter.training.chord_hybrid_eval.prepare_chord_hybrid_eval_songs",
        lambda *_args, **_kwargs: [object()],
    )
    monkeypatch.setattr(
        "drumcharter.training.chord_hybrid_eval.evaluate_prepared_chord_hybrid",
        fake_evaluate,
    )

    best = run_onset_decoder_training(
        ChordDecoderTrainConfig(
            cache_dir=cache_dir,
            frame_model_dir=tmp_path / "frame",
            output_dir=tmp_path / "run",
            window_frames=120,
            stride_frames=120,
            max_onsets=8,
            batch_size=2,
            num_workers=0,
            epochs=2,
            patience=5,
            selection_metric="hybrid_cqs",
            hybrid_eval_songs=1,
            device="cpu",
            amp=False,
        )
    )

    ckpt = torch.load(best, map_location="cpu", weights_only=True)
    assert ckpt["epoch"] == 2
    assert ckpt["hybrid_eval"]["hybrid_cqs"] == 0.8
    eval_report = json.loads((tmp_path / "run" / "eval_val_chord_hybrid.json").read_text())
    assert eval_report["best_epoch"] == 2
    assert eval_report["selection_metric"] == "hybrid_cqs"
