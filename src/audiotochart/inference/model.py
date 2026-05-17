from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from audiotochart.chart.drum_vocab import PRO8_LABELS
from audiotochart.drums import DrumHit
from audiotochart.inference.checkpoint import (
    ModelBundle,
    PRO8_ARCHITECTURE,
    PRO8_VARIANT,
    load_model_bundle,
)

log = logging.getLogger(__name__)

_HAS_LIBROSA: bool
try:
    import librosa as _librosa

    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


class ModelTranscriberError(RuntimeError):
    """Raised when model transcription fails."""


class ModelTranscriber:
    """Transcribe drum audio using a model loaded from a local model directory.

    The model directory must contain ``config.json``, weights (``weights.pt``
    or ``best.pt``), and optionally ``labels.json``
    (see :func:`~audiotochart.inference.checkpoint.load_model_bundle`).

    The bundle is loaded lazily on the first call to :meth:`transcribe`.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        device: str | None = None,
        tom_consistency: bool = False,
    ) -> None:
        self._model_dir = Path(model_dir) if model_dir else None
        self._device = device or "cpu"
        self._tom_consistency = tom_consistency
        self._bundle: ModelBundle | None = None

    def _ensure_loaded(self) -> ModelBundle:
        if self._bundle is not None:
            return self._bundle
        if self._model_dir is None:
            raise ModelTranscriberError(
                "ModelTranscriber has no model_dir configured"
            )
        self._bundle = load_model_bundle(self._model_dir, device=self._device)
        return self._bundle

    def transcribe(self, audio_path: Path) -> list[DrumHit]:
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        bundle = self._ensure_loaded()
        config = bundle.config
        device = bundle.device
        labels = bundle.labels
        model = bundle.model
        num_classes = len(labels)

        architecture: str = config.get("architecture", "") or ""
        variant: str = config.get("variant", "") or ""

        thresholds: list[float] = config.get("thresholds", [0.5] * len(labels))
        min_peak_distance: int = config.get("min_peak_distance", 5)
        chunk_frames: int = config.get("chunk_frames", 2000)

        import torch

        is_adtof = architecture == PRO8_ARCHITECTURE or variant == PRO8_VARIANT

        if is_adtof:
            spec, fps = _compute_adtof_spectrogram(audio_path)
        else:
            if not _HAS_LIBROSA:
                raise ImportError(
                    "ModelTranscriber requires librosa. Install the 'audio' extra: "
                    "uv sync --extra audio"
                )
            spec, fps = _compute_mel_spectrogram(audio_path, config)

        T = spec.shape[0]
        log.info("Spectrogram: %d frames (%.1fs)", T, T / fps)

        if T == 0:
            log.warning("Empty spectrogram — no audio content")
            return []

        acts_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, T, chunk_frames):
                chunk = spec[start : start + chunk_frames]
                x = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
                logits = model(x)
                probs = torch.sigmoid(logits)[0].cpu().numpy()
                acts_chunks.append(probs)

        acts = (
            np.concatenate(acts_chunks, axis=0)
            if acts_chunks
            else np.zeros((0, num_classes), dtype=np.float32)
        )
        num_actual = acts.shape[1]

        if len(thresholds) != num_actual:
            if is_adtof:
                raise ModelTranscriberError(
                    "Number of thresholds "
                    f"({len(thresholds)}) doesn't match model outputs ({num_actual})"
                )
            log.warning(
                "Number of thresholds (%d) doesn't match model outputs (%d); padding with 0.5",
                len(thresholds),
                num_actual,
            )
            thresholds = (thresholds + [0.5] * num_actual)[:num_actual]

        confidence_gates: list[float | None] = config.get(
            "confidence_gates",
            [None] * num_actual,
        )
        if len(confidence_gates) != num_actual:
            if is_adtof:
                raise ModelTranscriberError(
                    "Number of confidence_gates "
                    f"({len(confidence_gates)}) doesn't match model outputs ({num_actual})"
                )
            log.warning(
                "Number of confidence_gates (%d) doesn't match model outputs (%d); padding with None",
                len(confidence_gates),
                num_actual,
            )
            confidence_gates = (confidence_gates + [None] * num_actual)[:num_actual]

        gated_classes: set[int] = set()
        for c in range(num_actual):
            gate = confidence_gates[c]
            if gate is not None:
                max_act = float(acts[:, c].max()) if acts.shape[0] > 0 else 0.0
                if max_act < gate:
                    label = labels[c] if c < len(labels) else f"class_{c}"
                    log.info("[model] gating %s on this song (max=%.3f < %.3f)", label, max_act, gate)
                    gated_classes.add(c)

        # --- Peak-pick per class. ---
        if is_adtof:
            onsets = _pick_peaks_original(acts, thresholds, fps)
        else:
            onsets = _pick_peaks_simple(acts, thresholds, min_peak_distance, fps, num_classes=len(labels))

        # Remove onsets for gated classes.
        onsets = [(t, c, conf) for t, c, conf in onsets if c not in gated_classes]
        log.info("Picked %d onsets across %d classes", len(onsets), len(labels))

        # --- Tom consistency post-processing. ---
        if self._tom_consistency and _is_pro8_output(labels, variant) and onsets:
            from audiotochart.inference.tom_consistency import apply_tom_consistency

            tc_input = [(t, c) for t, c, _ in onsets]
            tc_result, tc_stats = apply_tom_consistency(tc_input, acts, fps=fps)
            n_reassigned = tc_stats.get("n_reassigned", 0)
            if n_reassigned:
                log.info(
                    "[model] tom consistency: %d/%d tom hits reassigned (convention=%s)",
                    n_reassigned,
                    tc_stats.get("n_tom_hits", 0),
                    tc_stats.get("convention", []),
                )
            # Rebuild onsets preserving original confidence values.
            # For reassigned toms, use the activation of the new class at that frame.
            confidence_map: dict[tuple[float, int], float] = {}
            for t, c, conf in onsets:
                confidence_map[(t, c)] = conf
            onsets = []
            for t, c in tc_result:
                conf = confidence_map.get((t, c))
                if conf is None:
                    frame = min(int(round(t * fps)), acts.shape[0] - 1)
                    conf = float(acts[frame, c]) if acts.shape[0] > 0 else 0.5
                onsets.append((t, c, conf))
        elif self._tom_consistency and onsets:
            log.debug(
                "Skipping tom consistency because model output is not recognized as pro8"
            )

        hits: list[DrumHit] = []
        for time_sec, cls, confidence in onsets:
            instrument = labels[cls] if cls < len(labels) else f"class_{cls}"
            hits.append(
                DrumHit(
                    time_sec=time_sec,
                    instrument=instrument,
                    confidence=confidence,
                )
            )

        if not hits:
            log.warning("No hits produced — thresholds may be too strict")

        return hits


def _is_pro8_output(labels: list[str], variant: str) -> bool:
    return variant == PRO8_VARIANT or labels == PRO8_LABELS


def _compute_adtof_spectrogram(audio_path: Path) -> tuple[np.ndarray, float]:
    """Compute the fine-tuned model spectrogram with the training-time path."""
    import librosa
    from adtof_pytorch.audio import AudioProcessor

    audio, _ = librosa.load(str(audio_path), sr=44100, mono=True)
    peak = np.abs(audio).max() if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * 0.95
    audio = audio.astype(np.float32, copy=False)

    processor = AudioProcessor()
    stft = processor.compute_stft(audio)
    spec = processor.apply_filterbank(stft).T.astype(np.float32)
    spec = spec[:, :, np.newaxis]
    fps = float(processor.fps)
    log.info("ADTOF spectrogram: %d frames, %d bins", spec.shape[0], spec.shape[1])
    return spec, fps


def _compute_mel_spectrogram(audio_path: Path, config: dict) -> tuple[np.ndarray, float]:
    """Compute a standard mel spectrogram via librosa."""
    sr: int = config.get("sample_rate", 22050)
    n_mels: int = config.get("n_mels", 84)
    n_fft: int = config.get("n_fft", 1024)
    hop_length: int = config.get("hop_length", 512)
    fmin: float = config.get("fmin", 20.0)
    fmax: float = config.get("fmax", 8000.0)
    fps: float = float(sr) / float(hop_length)

    log.info("Loading audio %s (sr=%d)", audio_path, sr)
    audio, _ = _librosa.load(str(audio_path), sr=sr)

    log.info("Computing mel spectrogram (n_mels=%d, hop=%d)", n_mels, hop_length)
    raw_spec = _librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
    )
    spec = raw_spec.T.astype(np.float32)[:, :, np.newaxis]
    return spec, fps


def _pick_peaks_original(acts: np.ndarray, thresholds: list[float], fps: float) -> list[tuple[float, int, float]]:
    """Peak picker matching the original CloneHero-ChartGen ``pick_peaks``.

    A frame is a peak if it is strictly greater than its left neighbour,
    strictly greater than (or equal to) its right neighbour, and exceeds
    the per-class threshold.  This is the exact logic from
    ``chartgen.training.thresholds.pick_peaks``.
    """
    num_classes = acts.shape[1]
    onsets: list[tuple[float, int, float]] = []
    for c in range(num_classes):
        probs = acts[:, c]
        n = len(probs)
        if n == 0:
            continue
        thr = thresholds[c] if c < len(thresholds) else 0.5
        left = np.concatenate(([-np.inf], probs[:-1]))
        right = np.concatenate((probs[1:], [-np.inf]))
        is_peak = (probs > left) & (probs >= right) & (probs >= thr)
        peak_frames = np.flatnonzero(is_peak).astype(np.int64)
        for frame in peak_frames:
            onsets.append((float(frame) / fps, c, float(probs[frame])))
    onsets.sort(key=lambda x: x[0])
    return onsets


def _pick_peaks_simple(
    acts: np.ndarray,
    thresholds: list[float],
    min_distance: int,
    fps: float,
    num_classes: int | None = None,
) -> list[tuple[float, int, float]]:
    """Simple peak picker for non-ADTOF models."""
    n_out = acts.shape[1]
    n_track = num_classes if num_classes is not None else n_out
    onsets: list[tuple[float, int, float]] = []
    for c in range(min(n_track, n_out)):
        probs = acts[:, c]
        n = len(probs)
        if n == 0:
            continue
        thr = thresholds[c] if c < len(thresholds) else 0.5
        mask = probs > thr
        i = 0
        while i < n:
            if not mask[i]:
                i += 1
                continue
            right = min(n, i + min_distance + 1)
            local_max_idx = int(i + np.argmax(probs[i:right]))
            if mask[local_max_idx]:
                onsets.append(
                    (
                        float(local_max_idx) / fps,
                        c,
                        float(probs[local_max_idx]),
                    )
                )
            i = local_max_idx + min_distance
    onsets.sort(key=lambda x: x[0])
    return onsets
