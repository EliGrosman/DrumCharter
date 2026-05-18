from __future__ import annotations

import json
import logging
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MAX_OPEN_ENTRIES = 64


@dataclass(frozen=True, slots=True)
class SongEntry:
    song_hash: str
    spec_path: Path
    label_path: Path
    song_name: str
    num_frames: int
    source_archive: str = ""


_BLACKLIST_PATH = Path(__file__).resolve().parent / "blacklist.json"


def _load_blacklist() -> set[str]:
    if not _BLACKLIST_PATH.is_file():
        return set()
    try:
        data = json.loads(_BLACKLIST_PATH.read_text())
    except json.JSONDecodeError:
        return set()
    return {entry["song_hash"] for entry in data.get("songs", [])}


def _load_entries(
    cache_dir: Path,
    *,
    harmonix_only: bool = False,
) -> list[SongEntry]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    blacklist = _load_blacklist()
    if blacklist:
        log.info("Loaded blacklist of %d misaligned songs", len(blacklist))

    label_dir = cache_dir / "labels_pro8"
    spec_dir = cache_dir / "spectrograms"
    entries: list[SongEntry] = []
    n_blacklisted = 0
    n_rbn_excluded = 0

    for h, info in manifest.items():
        if h in blacklist:
            n_blacklisted += 1
            continue

        if harmonix_only:
            source = info.get("source_archive", "").upper()
            if "RBN" in source:
                n_rbn_excluded += 1
                continue

        spec_path = spec_dir / f"{h}.npy"
        label_path = label_dir / f"{h}.npy"
        if not spec_path.exists() or not label_path.exists():
            log.debug("Skipping %s — missing spec or label", info.get("song_name", h))
            continue

        entries.append(
            SongEntry(
                song_hash=h,
                spec_path=spec_path,
                label_path=label_path,
                song_name=info.get("song_name", h),
                num_frames=info["num_frames"],
                source_archive=info.get("source_archive", ""),
            )
        )

    if n_blacklisted:
        log.info("Excluded %d blacklisted (misaligned) songs", n_blacklisted)
    if n_rbn_excluded:
        log.info("Excluded %d RBN community-charted songs (harmonix_only=True)", n_rbn_excluded)

    return entries


class DrumTranscriptionDataset:
    def __init__(
        self,
        entries: list[SongEntry],
        window_frames: int = 100,
        stride_frames: int = 50,
        max_open_entries: int = DEFAULT_MAX_OPEN_ENTRIES,
    ) -> None:
        if max_open_entries < 1:
            raise ValueError("max_open_entries must be at least 1")

        self.entries = entries
        self.window_frames = window_frames
        self.stride_frames = stride_frames
        self.max_open_entries = max_open_entries

        self._index: list[tuple[int, int]] = []
        for ei, entry in enumerate(entries):
            usable = entry.num_frames - window_frames
            if usable < 0:
                continue
            for start in range(0, usable + 1, stride_frames):
                self._index.append((ei, start))

        self._array_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._index)

    @staticmethod
    def _close_array(array: np.ndarray) -> None:
        mmap_obj = getattr(array, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()

    def clear_cache(self) -> None:
        for spec, labels in self._array_cache.values():
            self._close_array(spec)
            self._close_array(labels)
        self._array_cache.clear()

    def _evict_old_arrays(self) -> None:
        while len(self._array_cache) > self.max_open_entries:
            _entry_idx, (spec, labels) = self._array_cache.popitem(last=False)
            self._close_array(spec)
            self._close_array(labels)

    def _get_arrays(self, entry_idx: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._array_cache.get(entry_idx)
        if cached is not None:
            self._array_cache.move_to_end(entry_idx)
            return cached

        entry = self.entries[entry_idx]
        arrays = (
            np.load(str(entry.spec_path), mmap_mode="r"),
            np.load(str(entry.label_path), mmap_mode="r"),
        )
        self._array_cache[entry_idx] = arrays
        self._evict_old_arrays()
        return arrays

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_array_cache"] = OrderedDict()
        return state

    def __del__(self) -> None:
        try:
            if hasattr(self, "_array_cache"):
                self.clear_cache()
        except Exception:
            pass

    def __getitem__(self, idx: int):
        entry_idx, start = self._index[idx]
        spec, labels = self._get_arrays(entry_idx)

        end = start + self.window_frames
        T = min(spec.shape[0], labels.shape[0])
        actual_end = min(end, T)
        actual_start = min(start, T - self.window_frames) if T >= self.window_frames else 0

        spec_win = np.array(spec[actual_start:actual_end])
        label_win = np.array(labels[actual_start:actual_end])

        if spec_win.shape[0] < self.window_frames:
            pad_len = self.window_frames - spec_win.shape[0]
            spec_win = np.pad(spec_win, ((0, pad_len), (0, 0)))
            label_win = np.pad(label_win, ((0, pad_len), (0, 0)))

        if spec_win.ndim == 2:
            spec_win = spec_win[:, :, np.newaxis]

        return spec_win.astype(np.float32), label_win.astype(np.float32)


def create_splits(
    entries: list[SongEntry],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[SongEntry], list[SongEntry], list[SongEntry]]:
    rng = random.Random(seed)
    shuffled = list(entries)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, round(n * train_ratio))
    n_val = max(1, round(n * val_ratio)) if n > 2 else 0

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    log.info("Split %d songs: train=%d, val=%d, test=%d", n, len(train), len(val), len(test))
    return train, val, test


def create_datasets(
    cache_dir: Path,
    *,
    window_frames: int = 100,
    stride_frames: int = 50,
    seed: int = 42,
    harmonix_only: bool = False,
    max_open_entries: int = DEFAULT_MAX_OPEN_ENTRIES,
) -> tuple[DrumTranscriptionDataset, DrumTranscriptionDataset, DrumTranscriptionDataset]:
    entries = _load_entries(cache_dir, harmonix_only=harmonix_only)
    if not entries:
        raise ValueError(f"No entries found in {cache_dir}")

    train_entries, val_entries, test_entries = create_splits(entries, seed=seed)

    train_ds = DrumTranscriptionDataset(train_entries, window_frames, stride_frames, max_open_entries)
    val_ds = DrumTranscriptionDataset(val_entries, window_frames, stride_frames, max_open_entries)
    test_ds = DrumTranscriptionDataset(test_entries, window_frames, stride_frames, max_open_entries)

    log.info("Datasets: train=%d windows, val=%d, test=%d", len(train_ds), len(val_ds), len(test_ds))
    return train_ds, val_ds, test_ds
