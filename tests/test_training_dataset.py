from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from audiotochart.training.dataset import DrumTranscriptionDataset, SongEntry


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
