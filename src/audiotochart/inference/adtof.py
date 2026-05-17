from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from audiotochart.chart.midi import iter_drum_midi_hits
from audiotochart.drums import DrumHit

log = logging.getLogger(__name__)


class TranscriptionError(RuntimeError):
    """Raised when ADTOF drum transcription fails."""


def _transcribe_drums_to_midi(
    drums_wav: Path,
    midi_out: Path,
    *,
    device: str = "cuda",
) -> Path:
    try:
        import torch
        from adtof_pytorch import transcribe_to_midi
    except ImportError as exc:
        raise ImportError(
            "ADTOF backend requires: uv sync --extra ai"
        ) from exc

    drums_wav = Path(drums_wav)
    if not drums_wav.is_file():
        raise FileNotFoundError(f"Drum audio not found: {drums_wav}")

    if device == "cuda" and not torch.cuda.is_available():
        log.info("CUDA not available, falling back to CPU")
        device = "cpu"

    midi_out = Path(midi_out)
    midi_out.parent.mkdir(parents=True, exist_ok=True)

    log.info("Transcribing drums (device=%s): %s -> %s", device, drums_wav, midi_out)
    try:
        transcribe_to_midi(str(drums_wav), str(midi_out), device=device)
    except Exception as exc:
        raise TranscriptionError(f"ADTOF transcription failed: {exc}") from exc

    if not midi_out.is_file():
        raise TranscriptionError(
            f"ADTOF did not produce output MIDI file at {midi_out}"
        )

    if midi_out.stat().st_size == 0:
        raise TranscriptionError(
            "ADTOF produced an empty MIDI file — no drum hits detected"
        )

    log.info("Drum MIDI written to %s", midi_out)
    return midi_out


class AdtofTranscriber:
    def transcribe(self, audio_path: Path) -> list[DrumHit]:
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            midi_path = Path(f.name)

        try:
            _transcribe_drums_to_midi(audio_path, midi_path)
            return iter_drum_midi_hits(midi_path)
        finally:
            midi_path.unlink(missing_ok=True)
