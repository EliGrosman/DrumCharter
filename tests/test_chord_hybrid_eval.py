from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from audiotochart.training.chord_hybrid_eval import (
    PreparedChordHybridSong,
    evaluate_prepared_chord_hybrid,
)
from audiotochart.training.dataset import SongEntry


def test_chord_hybrid_eval_reports_baseline_and_hybrid_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "spec.npy"
    label_path = tmp_path / "labels.npy"
    spec = np.zeros((120, 4, 1), dtype=np.float32)
    labels = np.zeros((120, 8), dtype=np.float32)
    labels[10, 3] = 1.0
    labels[20, 6] = 1.0
    np.save(spec_path, spec)
    np.save(label_path, labels)

    entry = SongEntry(
        song_hash="song",
        spec_path=spec_path,
        label_path=label_path,
        song_name="Song",
        num_frames=120,
    )
    song = PreparedChordHybridSong(
        entry=entry,
        baseline_onsets=((10, 3), (20, 6)),
        onset_features=np.zeros((2, 18), dtype=np.float32),
    )

    class FakeModel:
        def eval(self):
            return self

    def fake_decode(*_args, **_kwargs):
        return [(10, 5), (20, 6)]

    monkeypatch.setattr(
        "audiotochart.training.chord_hybrid_eval.decode_chord_hybrid_onsets",
        fake_decode,
    )

    report = evaluate_prepared_chord_hybrid(
        FakeModel(),
        [song],
        device="cpu",
        window_frames=100,
        stride_frames=50,
    )

    assert report.n_songs == 1
    assert report.baseline_cqs == pytest.approx(1.0)
    assert report.hybrid_cqs < report.baseline_cqs
    assert report.baseline_per_class["y_tom"] == pytest.approx(1.0)
    assert report.hybrid_per_class["y_tom"] == pytest.approx(0.0)
    assert report.hybrid_per_class["b_tom"] == pytest.approx(0.0)
    assert report.hybrid_per_class["crash"] == pytest.approx(1.0)
