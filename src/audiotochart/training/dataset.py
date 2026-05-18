"""Dataset classes and factory functions for drum transcription training.

Provides memory-mapped windowed datasets for frame-level training
(DrumTranscriptionDataset) and chord-conditioned onset decoder
training (ChordConditionedDataset), along with splitting utilities.
"""

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
    """Metadata for a single preprocessed song.

    Attributes:
        song_hash: Unique 16-character hash identifying the song.
        spec_path: Path to the precomputed .npy spectrogram file.
        label_path: Path to the precomputed .npy label matrix file.
        song_name: Human-readable song name.
        num_frames: Number of time frames in the spectrogram/labels.
        source_archive: Name of the archive the song originated from.
    """

    song_hash: str
    spec_path: Path
    label_path: Path
    song_name: str
    num_frames: int
    source_archive: str = ""


_BLACKLIST_PATH = Path(__file__).resolve().parent / "blacklist.json"


def _close_numpy_handle(value: object) -> None:
    """Recursively close memory-mapped numpy handles on an array or tuple.

    Args:
        value: A numpy array or tuple of numpy arrays.
    """
    if isinstance(value, tuple):
        for item in value:
            _close_numpy_handle(item)
        return

    mmap_obj = getattr(value, "_mmap", None)
    if mmap_obj is not None:
        try:
            mmap_obj.close()
        except Exception:
            pass


def _load_blacklist() -> set[str]:
    """Load the set of blacklisted song hashes from blacklist.json.

    Returns:
        An empty set if the file is missing or malformed.
    """
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
    """Load song entries from a cache directory manifest.

    Filters out blacklisted songs and, if harmonix_only is set, excludes
    RBN community-charted songs.

    Args:
        cache_dir: Path to the cache directory containing manifest.json.
        harmonix_only: If True, exclude RBN songs.

    Returns:
        List of SongEntry objects for songs with both spec and label files.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
    """
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
    """Windowed dataset for frame-level drum transcription training.

    Loads memory-mapped spectrogram/label pairs on demand and extracts
    fixed-size windows at a given stride. Implements LRU caching over
    the memory-mapped arrays.
    """

    def __init__(
        self,
        entries: list[SongEntry],
        window_frames: int = 100,
        stride_frames: int = 50,
        max_open_entries: int = DEFAULT_MAX_OPEN_ENTRIES,
    ) -> None:
        """Initialise the frame-level dataset.

        Args:
            entries: List of SongEntry objects.
            window_frames: Number of frames per training window.
            stride_frames: Stride between consecutive windows.
            max_open_entries: Maximum number of memory-mapped arrays
                open simultaneously.

        Raises:
            ValueError: If max_open_entries is less than 1.
        """
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
        """Return the total number of windows in the dataset."""
        return len(self._index)

    @staticmethod
    def _close_array(array: np.ndarray) -> None:
        """Close the memory-mapped handle on a numpy array if present."""
        mmap_obj = getattr(array, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()

    def clear_cache(self) -> None:
        """Clear all cached arrays and close their mmap handles."""
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
        """Prepare pickle state by clearing the un-picklable array cache."""
        state = self.__dict__.copy()
        state["_array_cache"] = OrderedDict()
        return state

    def __del__(self) -> None:
        """Clean up mmap handles when the dataset is deleted."""
        try:
            if hasattr(self, "_array_cache"):
                self.clear_cache()
        except Exception:
            pass

    def __getitem__(self, idx: int):
        """Return a single training window of (spectrogram, labels).

        Pads the window if the spectrogram is shorter than the
        requested window length.

        Args:
            idx: Window index into the precomputed _index.

        Returns:
            Tuple of (spec_win, label_win) as float32 arrays.
        """
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


class ChordConditionedDataset:
    """Windowed data for chord-token onset decoder training."""

    def __init__(
        self,
        entries: list[SongEntry],
        window_frames: int = 1000,
        stride_frames: int = 500,
        max_onsets: int = 256,
        *,
        onset_dir: Path | None = None,
        tp_only: bool = False,
        tp_tolerance_frames: int = 2,
        allow_null_token: bool = True,
        blocklist_policy: str = "none",
        max_open_entries: int = DEFAULT_MAX_OPEN_ENTRIES,
    ) -> None:
        if max_open_entries < 1:
            raise ValueError("max_open_entries must be at least 1")

        from audiotochart.onset_decoder_common import build_chord_vocabulary

        self.entries = entries
        self.window_frames = window_frames
        self.stride_frames = stride_frames
        self.max_onsets = max_onsets
        self._onset_dir = onset_dir
        self._tp_only = tp_only
        self._tp_tolerance_frames = tp_tolerance_frames
        self._allow_null_token = allow_null_token
        self._chord_vocab = build_chord_vocabulary(blocklist_policy=blocklist_policy)
        self._max_open_entries = max_open_entries

        self._index: list[tuple[int, int]] = []
        for entry_idx, entry in enumerate(entries):
            usable = entry.num_frames - window_frames
            if usable < 0:
                continue
            if onset_dir is not None and not (onset_dir / f"{entry.song_hash}.npz").exists():
                continue
            for start in range(0, usable + 1, stride_frames):
                self._index.append((entry_idx, start))

        self._array_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._onset_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        """Return the total number of windows in the dataset."""
        return len(self._index)

    def clear_cache(self) -> None:
        for arrays in self._array_cache.values():
            _close_numpy_handle(arrays)
        for arrays in self._onset_cache.values():
            _close_numpy_handle(arrays)
        self._array_cache.clear()
        self._onset_cache.clear()

    def _touch(self, entry_idx: int) -> None:
        if entry_idx in self._array_cache:
            self._array_cache.move_to_end(entry_idx)
        if entry_idx in self._onset_cache:
            self._onset_cache.move_to_end(entry_idx)
        while len(set(self._array_cache) | set(self._onset_cache)) > self._max_open_entries:
            candidates = list(self._array_cache) + [
                idx for idx in self._onset_cache if idx not in self._array_cache
            ]
            evict_idx = candidates[0]
            arrays = self._array_cache.pop(evict_idx, None)
            if arrays is not None:
                _close_numpy_handle(arrays)
            onsets = self._onset_cache.pop(evict_idx, None)
            if onsets is not None:
                _close_numpy_handle(onsets)

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
        self._touch(entry_idx)
        return arrays

    def _get_onsets(self, entry_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._onset_dir is None:
            raise ValueError("onset_dir is required for predicted-onset mode")

        cached = self._onset_cache.get(entry_idx)
        if cached is not None:
            self._onset_cache.move_to_end(entry_idx)
            return cached

        from audiotochart.onset_decoder_common import fallback_onset_feature_rows

        entry = self.entries[entry_idx]
        with np.load(str(self._onset_dir / f"{entry.song_hash}.npz")) as npz:
            onset_frames = npz["onset_frames"].astype(np.int32)
            onset_classes = npz["onset_classes"].astype(np.int32)
            onset_features = (
                npz["onset_features"].astype(np.float32)
                if "onset_features" in npz.files
                else fallback_onset_feature_rows(onset_classes)
            )

        n = min(len(onset_frames), len(onset_classes), len(onset_features))
        arrays = (onset_frames[:n], onset_classes[:n], onset_features[:n])
        self._onset_cache[entry_idx] = arrays
        self._touch(entry_idx)
        return arrays

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_array_cache"] = OrderedDict()
        state["_onset_cache"] = OrderedDict()
        return state

    def __del__(self) -> None:
        try:
            if hasattr(self, "_array_cache"):
                self.clear_cache()
        except Exception:
            pass

    def _match_predicted_chords(
        self,
        pred_groups: list[tuple[int, int]],
        gt_groups: list[tuple[int, int]],
    ) -> list[int | None]:
        if not pred_groups:
            return []
        if not gt_groups:
            return [None] * len(pred_groups)

        tol = self._tp_tolerance_frames
        gt_frames = np.fromiter((f for f, _ in gt_groups), dtype=np.int64, count=len(gt_groups))
        gt_masks = np.fromiter((m for _, m in gt_groups), dtype=np.int32, count=len(gt_groups))
        gt_used = np.zeros(len(gt_groups), dtype=bool)
        assignments: list[int | None] = [None] * len(pred_groups)
        leftover: list[int] = []

        def best_match(pred_frame: int, pred_mask: int, *, same_mask_only: bool) -> int:
            lo = int(np.searchsorted(gt_frames, pred_frame - tol))
            hi = int(np.searchsorted(gt_frames, pred_frame + tol + 1))
            best = -1
            best_dist = tol + 1
            for gt_idx in range(lo, hi):
                if gt_used[gt_idx]:
                    continue
                if same_mask_only and int(gt_masks[gt_idx]) != pred_mask:
                    continue
                dist = abs(int(gt_frames[gt_idx]) - pred_frame)
                if dist < best_dist:
                    best = gt_idx
                    best_dist = dist
            return best

        for pred_idx, (pred_frame, pred_mask) in enumerate(pred_groups):
            gt_idx = best_match(pred_frame, pred_mask, same_mask_only=True)
            if gt_idx >= 0:
                gt_used[gt_idx] = True
                assignments[pred_idx] = int(gt_masks[gt_idx])
            else:
                leftover.append(pred_idx)

        for pred_idx in leftover:
            pred_frame, pred_mask = pred_groups[pred_idx]
            gt_idx = best_match(pred_frame, pred_mask, same_mask_only=False)
            if gt_idx >= 0:
                gt_used[gt_idx] = True
                assignments[pred_idx] = int(gt_masks[gt_idx])

        return assignments

    def _get_model_chord_tokens(
        self,
        entry_idx: int,
        win_start: int,
        win_end: int,
        label_win: np.ndarray,
    ) -> tuple[list[int], list[np.ndarray], list[int]]:
        from audiotochart.onset_decoder_common import (
            CHORD_NULL,
            aggregate_chord_features,
            classes_to_mask,
            labels_to_chord_events,
        )

        model_frames, model_classes, model_features = self._get_onsets(entry_idx)
        mask = (model_frames >= win_start) & (model_frames < win_end)
        win_frames = model_frames[mask] - win_start
        win_classes = model_classes[mask]
        win_features = model_features[mask]

        grouped: dict[int, dict[str, list]] = {}
        for frame, class_idx, feature_row in zip(win_frames, win_classes, win_features):
            c = int(class_idx)
            if not 0 <= c < 8:
                continue
            bucket = grouped.setdefault(int(frame), {"classes": [], "features": []})
            bucket["classes"].append(c)
            bucket["features"].append(feature_row.astype(np.float32))

        pred_groups: list[tuple[int, int]] = []
        pred_features: list[np.ndarray] = []
        for frame in sorted(grouped):
            classes = grouped[frame]["classes"]
            feature_rows = np.asarray(grouped[frame]["features"], dtype=np.float32)
            pred_mask = classes_to_mask(classes)
            pred_groups.append((frame, pred_mask))
            pred_features.append(aggregate_chord_features(feature_rows, classes))

        if not pred_groups:
            return [], [], []

        gt_groups = labels_to_chord_events(label_win)
        assignments = self._match_predicted_chords(pred_groups, gt_groups)

        frames_list: list[int] = []
        features_list: list[np.ndarray] = []
        tokens_list: list[int] = []
        for (frame, pred_mask), feature_row, matched_mask in zip(
            pred_groups,
            pred_features,
            assignments,
        ):
            if matched_mask is None:
                if self._tp_only:
                    continue
                target_token = (
                    CHORD_NULL
                    if self._allow_null_token
                    else self._chord_vocab.token_for_mask(pred_mask)
                )
            else:
                target_token = self._chord_vocab.token_for_mask(matched_mask)
            if target_token is None:
                continue
            frames_list.append(frame)
            features_list.append(feature_row)
            tokens_list.append(target_token)

        return frames_list, features_list, tokens_list

    def _get_gt_chord_tokens(
        self,
        label_win: np.ndarray,
    ) -> tuple[list[int], list[np.ndarray], list[int]]:
        from audiotochart.onset_decoder_common import (
            ONSET_FEATURE_DIM,
            aggregate_chord_features,
            labels_to_chord_events,
            mask_to_classes,
        )

        frames_list: list[int] = []
        features_list: list[np.ndarray] = []
        tokens_list: list[int] = []
        for frame, chord_mask in labels_to_chord_events(label_win):
            token = self._chord_vocab.token_for_mask(chord_mask)
            if token is None:
                continue
            classes = mask_to_classes(chord_mask)
            frames_list.append(frame)
            features_list.append(
                aggregate_chord_features(
                    np.zeros((0, ONSET_FEATURE_DIM), dtype=np.float32),
                    classes,
                )
            )
            tokens_list.append(token)
        return frames_list, features_list, tokens_list

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from audiotochart.onset_decoder_common import (
            CHORD_BOS,
            CHORD_PAD,
            ONSET_FEATURE_DIM,
        )

        entry_idx, start = self._index[idx]
        spec_full, labels_full = self._get_arrays(entry_idx)

        end = start + self.window_frames
        t_frames = min(spec_full.shape[0], labels_full.shape[0])
        actual_end = min(end, t_frames)
        actual_start = min(start, t_frames - self.window_frames) if t_frames >= self.window_frames else 0

        spec_win = np.array(spec_full[actual_start:actual_end])
        label_win = np.array(labels_full[actual_start:actual_end])

        if spec_win.shape[0] < self.window_frames:
            pad_len = self.window_frames - spec_win.shape[0]
            spec_win = np.pad(spec_win, ((0, pad_len), (0, 0)))
            label_win = np.pad(label_win, ((0, pad_len), (0, 0)))

        if spec_win.ndim == 2:
            spec_win = spec_win[:, :, np.newaxis]

        if self._onset_dir is not None:
            frames_list, features_list, tokens_list = self._get_model_chord_tokens(
                entry_idx,
                actual_start,
                actual_start + label_win.shape[0],
                label_win,
            )
        else:
            frames_list, features_list, tokens_list = self._get_gt_chord_tokens(label_win)

        n_tokens = len(tokens_list)
        seq_len = min(n_tokens, self.max_onsets)

        onset_frames = np.zeros(self.max_onsets, dtype=np.int64)
        onset_features = np.zeros((self.max_onsets, ONSET_FEATURE_DIM), dtype=np.float32)
        token_input = np.full(self.max_onsets, CHORD_PAD, dtype=np.int64)
        token_target = np.full(self.max_onsets, CHORD_PAD, dtype=np.int64)

        onset_frames[:seq_len] = frames_list[:seq_len]
        if seq_len > 0:
            onset_features[:seq_len] = np.asarray(features_list[:seq_len], dtype=np.float32)

        token_input[0] = CHORD_BOS
        if seq_len > 1:
            token_input[1:seq_len] = tokens_list[: seq_len - 1]
        token_target[:seq_len] = tokens_list[:seq_len]

        padding_mask = np.ones(self.max_onsets, dtype=bool)
        padding_mask[:seq_len] = False

        return (
            spec_win.astype(np.float32),
            onset_frames,
            onset_features,
            token_input,
            token_target,
            padding_mask,
        )


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
