"""Chart generation pipeline.

Orchestrates audio loading, optional drum separation, model transcribing,
chart document generation, and Clone Hero song folder creation.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from drumcharter.audio import get_audio_duration_sec
from drumcharter.chart.convert import hits_to_chart_document
from drumcharter.chart.difficulty import generate_difficulties
from drumcharter.chart.format import DrumDifficulty, SongMetadata, write_chart_file
from drumcharter.chart.midi import midi_to_chart_document
from drumcharter.chart.songini import SongIni, write_song_ini
from drumcharter.inference.base import DrumTranscriber
from drumcharter.tempo import TempoError, detect_beat_grid

logger = logging.getLogger(__name__)

# Stage IDS used by callback
STAGE_SEPARATE = "separate"
STAGE_CHART = "chart"
STAGE_OUTPUT = "output"

STAGES = [
    (STAGE_SEPARATE, "Isolating Drums"),
    (STAGE_CHART, "Generating Drum Chart"),
    (STAGE_OUTPUT, "Writing Clone Hero Song Folder"),
]

ProgressCallback = Callable[[str, str], None]

def _safe_folder_name(s: str) -> str:
    """Sanitise a string for use as a filesystem directory name.

    Args:
        s: The input string to sanitise.

    Returns:
        A filesystem-safe directory name with slashes and backslashes
        replaced by hyphens and excess whitespace collapsed.
    """
    return " ".join(s.replace("/", "-").replace("\\", "-").split())
def _stream_filename(source_audio: Path) -> str:
    """Choose a ``song.*`` filename matching the source audio format.

    Args:
        source_audio: Path to the source audio file.

    Returns:
        A ``song`` filename with the same extension, or ``song.wav``
        for unsupported formats.
    """
    ext = source_audio.suffix.lower()
    if ext in (".ogg", ".wav", ".mp3", ".opus", ".flac"):
        return f"song{ext}"
    return "song.wav"

def generate_drum_chart_folder(
    *,
    source_audio: Path,
    output_parent: Path,
    song_name: str,
    artist_name: str,
    charter: str = "DrumCharter (AI)",
    bpm: float | None = None,
    resolution: int = 192,
    from_midi: Path | None = None,
    quantize_divisor: int | None = None,
    transcriber: DrumTranscriber | None = None,
    on_progress: ProgressCallback | None = None,
    separate_drums: bool = False,
    device: str | None = None,
    keep_workdir: bool = False,
) -> Path:
    """Create a Clone Hero song folder with notes.chart, song.ini, and audio.

    Orchestrates the full chart generation pipeline: tempo detection,
    optional drum separation, model transcribing (or MIDI import),
    chart document generation, and output folder creation.

    Args:
        source_audio: Path to the input audio file.
        output_parent: Parent directory for the output song folder.
        song_name: Name of the song.
        artist_name: Name of the artist.
        charter: Name of the chart creator. Defaults to ``"DrumCharter (AI)"``.
        bpm: Beats per minute. Auto-detected if None.
        resolution: Chart resolution in ticks per quarter note. Defaults to 192.
        from_midi: Optional path to a MIDI file to use instead of transcribing.
        quantize_divisor: Quantization grid divisor (e.g. 16 for sixteenth notes).
        transcriber: Optional DrumTranscriber instance. Uses FakeTranscriber if None.
        on_progress: Optional callback for progress updates.
        separate_drums: If True, run drum source separation via Demucs.
        device: PyTorch device string for separation.
        keep_workdir: If True, keep the temporary separation directory.

    Returns:
        Path to the generated Clone Hero song folder.

    Raises:
        FileNotFoundError: If source_audio or from_midi is not found.
    """

    source_audio = Path(source_audio)
    if not source_audio.is_file():
        raise FileNotFoundError(f"Source audio not found: {source_audio}")
    if from_midi is not None:
        from_midi = Path(from_midi)
        if not from_midi.is_file():
            raise FileNotFoundError(f"MIDI file not found: {from_midi}")
    
    def _notify(stage: str, event: str) -> None:
        if on_progress is not None:
            on_progress(stage, event)

    logger.info("Generating drum chart for %s", source_audio.name)
    _notify(STAGE_CHART, "start")

    duration_sec = get_audio_duration_sec(source_audio)
    logger.info("Audio duration: %.2f s", duration_sec)

    beat_times: list[float] | None = None

    # Auto-detect tempo and beat positions if not provided.
    if bpm is None:
        try:
            beat_grid = detect_beat_grid(source_audio)
            bpm = beat_grid.bpm
            beat_times = [float(time) for time in beat_grid.beat_times]
            logger.info("Detected tempo: %.2f BPM (%d beats)", bpm, len(beat_grid.beat_times))
        except TempoError as e:
            logger.warning("Tempo detection failed: %s. Using default 120 BPM.", e)
            bpm = 120.0

    stream_name = _stream_filename(source_audio)

    # Optional drum source separation
    transcribe_audio = source_audio
    _tmp_dir: Path | None = None
    try:
        if separate_drums:
            from drumcharter.separation import isolate_drums

            _notify(STAGE_SEPARATE, "start")
            _tmp_dir = Path(tempfile.mkdtemp(prefix="drumcharter-sep-"))
            drum_wav = _tmp_dir / "drums.wav"
            logger.info("Isolating drums via Demucs...")
            isolate_drums(source_audio, drum_wav, device=device, progress=False)
            transcribe_audio = drum_wav
            _notify(STAGE_SEPARATE, "done")
            logger.info("Drum stem ready at %s", drum_wav)

        meta = SongMetadata(
            name=song_name,
            artist=artist_name,
            charter=charter,
            resolution=resolution,
            offset=0.0,
            music_stream=stream_name,
        )
        if from_midi is None:
            if transcriber is None:
                from drumcharter.inference.fake import FakeTranscriber
                transcriber = FakeTranscriber()
            hits = transcriber.transcribe(transcribe_audio)
            logger.info("Transcriber returned %d drum hits", len(hits))
            doc = hits_to_chart_document(
                hits,
                song=meta,
                bpm=bpm,
                resolution=resolution,
                beat_times=beat_times,
                quantize_divisor=quantize_divisor,
            )
        else:
            logger.info("Using MIDI drum transcription: %s", from_midi)
            doc = midi_to_chart_document(
                from_midi,
                song=meta,
                bpm=bpm,
                resolution=resolution,
                beat_times=beat_times,
                quantize_divisor=quantize_divisor,
            )
        expert_notes = doc.drums.get(DrumDifficulty.EXPERT, [])
        if not expert_notes:
            logger.warning("No drum notes were generated; the chart will be empty.")
        generate_difficulties(doc)
        _notify(STAGE_CHART, "done")

        logger.info("Writing Clone Hero song folder")
        _notify(STAGE_OUTPUT, "start")
        folder = output_parent / _safe_folder_name(f"{artist_name} - {song_name}")
        folder.mkdir(parents=True, exist_ok=True)

        write_chart_file(doc, folder / "notes.chart")
        write_song_ini(
            SongIni(
                name=song_name,
                artist=artist_name,
                charter=charter,
                diff_drums=4,
                song_length=int(duration_sec * 1000),
            ),
            folder / "song.ini",
        )
        shutil.copy2(source_audio, folder / stream_name)
        _notify(STAGE_OUTPUT, "done")

        logger.info("Chart folder generated at %s", folder)
        return folder
    finally:
        # Clean up separation workdir unless --keep-workdir, even on failure.
        if _tmp_dir is not None and not keep_workdir:
            shutil.rmtree(_tmp_dir, ignore_errors=True)
            logger.info("Cleaned up separation workdir %s", _tmp_dir)
