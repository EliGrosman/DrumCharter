from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drumcharter.onset_decoder_common import (
    CHORD_BOS,
    CHORD_NULL,
    build_chord_vocabulary,
    classes_to_mask,
)
from drumcharter.training.dataset import ChordConditionedDataset, DrumTranscriptionDataset, SongEntry


def _entry(tmp_path: Path, idx: int, *, frames: int = 8) -> SongEntry:
    spec_path = tmp_path / f"spec_{idx}.npy"
    label_path = tmp_path / f"label_{idx}.npy"
    np.save(spec_path, np.full((frames, 4), idx, dtype=np.float32))
    np.save(label_path, np.full((frames, 8), idx, dtype=np.float32))
    return SongEntry(
        song_hash=f"song-{idx}",
        spec_path=spec_path,
        label_path=label_path,
        song_name=f"Song {idx}",
        num_frames=frames,
    )


def _chord_entry(tmp_path: Path, labels: np.ndarray, *, name: str = "chord-song") -> SongEntry:
    spec_path = tmp_path / f"{name}_spec.npy"
    label_path = tmp_path / f"{name}_label.npy"
    np.save(spec_path, np.zeros((labels.shape[0], 4), dtype=np.float32))
    np.save(label_path, labels.astype(np.float32))
    return SongEntry(
        song_hash=name,
        spec_path=spec_path,
        label_path=label_path,
        song_name=name,
        num_frames=labels.shape[0],
    )


def test_dataset_rejects_empty_memmap_cache(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_open_entries"):
        DrumTranscriptionDataset([_entry(tmp_path, 0)], max_open_entries=0)


def test_dataset_memmap_cache_is_bounded_and_lru(tmp_path: Path) -> None:
    entries = [_entry(tmp_path, idx) for idx in range(3)]
    ds = DrumTranscriptionDataset(
        entries,
        window_frames=4,
        stride_frames=4,
        max_open_entries=2,
    )

    spec0, labels0 = ds._get_arrays(0)
    spec1, labels1 = ds._get_arrays(1)
    ds._get_arrays(0)
    spec2, labels2 = ds._get_arrays(2)

    assert list(ds._array_cache) == [0, 2]
    assert not spec0._mmap.closed
    assert not labels0._mmap.closed
    assert spec1._mmap.closed
    assert labels1._mmap.closed
    assert not spec2._mmap.closed
    assert not labels2._mmap.closed

    ds.clear_cache()
    assert spec0._mmap.closed
    assert labels0._mmap.closed
    assert spec2._mmap.closed
    assert labels2._mmap.closed


def test_chord_conditioned_dataset_groups_gt_chords(tmp_path: Path) -> None:
    labels = np.zeros((6, 8), dtype=np.float32)
    labels[1, [0, 1]] = 1.0
    labels[3, 6] = 1.0
    entry = _chord_entry(tmp_path, labels)
    vocab = build_chord_vocabulary()

    ds = ChordConditionedDataset([entry], window_frames=6, stride_frames=6, max_onsets=4)
    spec, frames, features, token_input, token_target, padding_mask = ds[0]

    assert spec.shape == (6, 4, 1)
    assert frames[:2].tolist() == [1, 3]
    assert token_input[0] == CHORD_BOS
    assert token_input[1] == vocab.token_for_mask(classes_to_mask([0, 1]))
    assert token_target[:2].tolist() == [
        vocab.token_for_mask(classes_to_mask([0, 1])),
        vocab.token_for_mask(classes_to_mask([6])),
    ]
    assert padding_mask.tolist() == [False, False, True, True]
    assert features[0, 8 + 0] == 1.0
    assert features[0, 8 + 1] == 1.0
    assert features[1, 8 + 6] == 1.0


def test_chord_conditioned_dataset_matches_predicted_onsets_and_nulls(tmp_path: Path) -> None:
    labels = np.zeros((6, 8), dtype=np.float32)
    labels[2, [0, 1]] = 1.0
    entry = _chord_entry(tmp_path, labels)
    onset_dir = tmp_path / "onsets"
    onset_dir.mkdir()
    rows = np.zeros((3, 18), dtype=np.float32)
    rows[0, 0] = 0.8
    rows[1, 1] = 0.7
    rows[2, 6] = 0.9
    np.savez(
        onset_dir / f"{entry.song_hash}.npz",
        onset_frames=np.asarray([2, 2, 4], dtype=np.int32),
        onset_classes=np.asarray([0, 1, 6], dtype=np.int32),
        onset_features=rows,
    )
    vocab = build_chord_vocabulary()

    ds = ChordConditionedDataset(
        [entry],
        window_frames=6,
        stride_frames=6,
        max_onsets=4,
        onset_dir=onset_dir,
    )
    _spec, frames, features, _token_input, token_target, padding_mask = ds[0]

    assert frames[:2].tolist() == [2, 4]
    assert token_target[:2].tolist() == [
        vocab.token_for_mask(classes_to_mask([0, 1])),
        CHORD_NULL,
    ]
    assert padding_mask.tolist() == [False, False, True, True]
    assert features[0, 0] == pytest.approx(0.8)
    assert features[0, 1] == pytest.approx(0.7)
    assert features[0, 8 + 0] == 1.0
    assert features[0, 8 + 1] == 1.0
