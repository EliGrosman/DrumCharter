from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from drumcharter.training.onset_precompute import (
    OnsetPrecomputeConfig,
    run_precompute_onsets,
)


def _make_cache(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "cache"
    spec_dir = cache_dir / "spectrograms"
    label_dir = cache_dir / "labels_pro8"
    spec_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    song_hash = "song-a"
    np.save(spec_dir / f"{song_hash}.npy", np.zeros((5, 4), dtype=np.float32))
    np.save(label_dir / f"{song_hash}.npy", np.zeros((5, 8), dtype=np.float32))
    (cache_dir / "manifest.json").write_text(
        json.dumps(
            {
                song_hash: {
                    "song_name": "Song A",
                    "num_frames": 5,
                    "source_archive": "rb.zip",
                }
            }
        )
    )
    return cache_dir


def test_precompute_onsets_writes_npz_schema(monkeypatch, tmp_path: Path) -> None:
    cache_dir = _make_cache(tmp_path)

    class FakeFrameModel(torch.nn.Module):
        def forward(self, x):
            logits = torch.full((x.shape[0], x.shape[1], 8), -10.0)
            logits[:, 1, 0] = 10.0
            logits[:, 2, 1] = 10.0
            logits[:, 2, 6] = 10.0
            return logits

    def fake_load_model_bundle(_model_dir, *, device):
        return SimpleNamespace(
            model=FakeFrameModel().to(device),
            labels=[f"class_{idx}" for idx in range(8)],
            config={"thresholds": [0.5] * 8},
            device=device,
        )

    monkeypatch.setattr(
        "drumcharter.inference.checkpoint.load_model_bundle",
        fake_load_model_bundle,
    )

    result = run_precompute_onsets(
        OnsetPrecomputeConfig(
            cache_dir=cache_dir,
            frame_model_dir=tmp_path / "frame-model",
            output_dir=tmp_path / "onsets",
            device="cpu",
        )
    )

    assert result.written == 1
    out_path = tmp_path / "onsets" / "song-a.npz"
    with np.load(out_path) as data:
        assert data["onset_frames"].tolist() == [1, 2, 2]
        assert data["onset_classes"].tolist() == [0, 1, 6]
        assert data["onset_features"].shape == (3, 18)
        assert int(data["num_frames"]) == 5
        assert data["onset_features"][0, 8 + 0] == 1.0
        assert data["onset_features"][1, 8 + 1] == 1.0
