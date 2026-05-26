"""Drum source separation using Demucs.

Wraps the Demucs library to isolate the drum stem from a mixed audio file.
"""

from __future__ import annotations

import logging
from pathlib import Path

from drumcharter.device import resolve_torch_device

log = logging.getLogger(__name__)


class SeparationError(RuntimeError):
    """Raised when Demucs drum separation fails."""


def isolate_drums(
    audio_path: Path,
    out_wav: Path,
    *,
    device: str | None = None,
    progress: bool = True,
    model_name: str = "htdemucs",
) -> Path:
    """Write a stereo WAV of the isolated drum stem.

    Parameters
    ----------
    audio_path:
        Input audio file (WAV, MP3, FLAC, etc. — anything ffmpeg can read).
    out_wav:
        Destination path for the isolated drums WAV.
    device:
        PyTorch device string (``"auto"``, ``"cuda"``, or ``"cpu"``).
        ``"auto"`` and ``None`` use CUDA when available, otherwise CPU.
    progress:
        Show a tqdm progress bar during separation.
    model_name:
        Demucs pre-trained model name.  ``"htdemucs"`` is recommended.

    Returns
    -------
    Path
        The *out_wav* path (confirmed written).

    Raises
    ------
    SeparationError
        If Demucs fails for any reason (missing model, OOM, corrupt audio).
    FileNotFoundError
        If *audio_path* does not exist.
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        import soundfile as sf
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
        from demucs.separate import load_track
    except ImportError as exc:
        raise SeparationError(
            "Demucs (drum separation) not found. "
            "Install with: uv sync --extra ai"
        ) from exc

    device = resolve_torch_device(device, purpose="Demucs separation")

    log.info("Loading Demucs model %r on %s", model_name, device)
    try:
        model = get_model(name=model_name)
    except Exception as exc:
        raise SeparationError(f"Failed to load Demucs model {model_name!r}: {exc}") from exc

    model.cpu()
    model.eval()

    if "drums" not in model.sources:
        raise SeparationError(
            f"Model {model_name!r} has no drums stem (available: {model.sources})"
        )

    try:
        wav = load_track(audio_path, model.audio_channels, model.samplerate)
    except Exception as exc:
        raise SeparationError(f"Failed to load audio {audio_path}: {exc}") from exc

    ref = wav.mean(0)
    wav_norm = (wav - ref.mean()) / (ref.std() + 1e-8)

    log.info("Running Demucs separation (device=%s)...", device)
    try:
        sources = apply_model(
            model,
            wav_norm[None],
            device=device,
            shifts=1,
            split=True,
            overlap=0.25,
            progress=progress,
            num_workers=0,
            segment=None,
        )[0]
    except Exception as exc:
        raise SeparationError(f"Demucs separation failed: {exc}") from exc

    sources = sources * (ref.std() + 1e-8) + ref.mean()

    idx = model.sources.index("drums")
    drums = sources[idx]

    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    drums_np = drums.cpu().clamp(-1.0, 1.0).numpy().T
    sf.write(str(out_wav), drums_np, model.samplerate, subtype="PCM_16")
    log.info("Drum stem saved to %s", out_wav)
    return out_wav
