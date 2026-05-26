"""Spectrogram computation and caching for drum transcription training.

Computes mel-spectrograms from audio using librosa with caching
via content-hash-based filenames to avoid recomputation.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 44100
NUM_BINS = 84

_shared_processor = None


def get_processor():
    global _shared_processor
    if _shared_processor is None:
        from adtof_pytorch.audio import AudioProcessor
        _shared_processor = AudioProcessor()
    return _shared_processor


def song_hash(song_path: Path, *, stable_key: str = "") -> str:
    key = stable_key or str(song_path.resolve())
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def load_and_mix_stems(
    stem_paths: list[Path],
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    import librosa

    if len(stem_paths) == 1:
        y, _ = librosa.load(str(stem_paths[0]), sr=sr, mono=True)
    else:
        mixed = None
        for p in stem_paths:
            y_stem, _ = librosa.load(str(p), sr=sr, mono=True)
            if mixed is None:
                mixed = np.zeros_like(y_stem)
            if len(y_stem) > len(mixed):
                mixed = np.pad(mixed, (0, len(y_stem) - len(mixed)))
            elif len(y_stem) < len(mixed):
                y_stem = np.pad(y_stem, (0, len(mixed) - len(y_stem)))
            mixed += y_stem
        y = mixed

    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak * 0.95
    return y.astype(np.float32)


def compute_spectrogram_from_array(
    audio: np.ndarray,
    output_path: Path,
    processor=None,
) -> tuple[Path, int]:
    proc = processor or get_processor()
    stft = proc.compute_stft(audio)
    filtered = proc.apply_filterbank(stft)
    features = filtered.T.astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), features)

    num_frames = features.shape[0]
    log.debug("Spectrogram %s: %s: %d frames", output_path.name, num_frames)
    return output_path, num_frames


def process_single_song(
    h: str,
    drum_stem_paths: list[str],
    midi_path: str,
    cache_dir: str,
    source_archive: str,
    song_name: str,
    has_pro: bool,
    force: bool = False,
) -> dict | None:
    from drumcharter.training.labels import onsets_to_label_matrix
    from drumcharter.training.rb_midi import NUM_CLASSES, parse_rb_drum_onsets

    cache = Path(cache_dir)
    spec_dir = cache / "spectrograms"
    spec_path = spec_dir / f"{h}.npy"

    if spec_path.exists() and not force:
        existing = np.load(str(spec_path), mmap_mode="r")
        return {
            "hash": h,
            "spec_path": str(spec_path),
            "num_frames": existing.shape[0],
            "song_name": song_name,
            "source_archive": source_archive,
        }

    audio = load_and_mix_stems([Path(p) for p in drum_stem_paths])
    _, num_frames = compute_spectrogram_from_array(audio, spec_path)
    del audio

    midi = Path(midi_path)
    onsets = parse_rb_drum_onsets(midi)

    if has_pro:
        label_dir = cache / "labels_pro8"
        label_dir.mkdir(parents=True, exist_ok=True)
        label_path = label_dir / f"{h}.npy"
        if not label_path.exists() or force:
            mat = onsets_to_label_matrix(onsets, num_frames, num_classes=NUM_CLASSES)
            np.save(str(label_path), mat)
    else:
        log.debug("Skipping pro8 labels for %s; no pro drum markers found", song_name)

    return {
        "hash": h,
        "spec_path": str(spec_path),
        "num_frames": num_frames,
        "song_name": song_name,
        "source_archive": source_archive,
    }
