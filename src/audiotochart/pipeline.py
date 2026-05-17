from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable

from audiotochart.audio import get_audio_duration_sec
from audiotochart.chart.convert import hits_to_chart_document
from audiotochart.chart.difficulty import generate_difficulties
from audiotochart.chart.format import DrumDifficulty, SongMetadata, write_chart_file
from audiotochart.chart.midi import midi_to_chart_document
from audiotochart.chart.songini import SongIni, write_song_ini
from audiotochart.inference.base import DrumTranscriber
from audiotochart.tempo import TempoError, detect_beat_grid

logger = logging.getLogger(__name__)

# Stage IDS used by callback
STAGE_CHART = "chart"
STAGE_OUTPUT = "output"

STAGES = [
    (STAGE_CHART, "Generating Drum Chart"),
    (STAGE_OUTPUT, "Writing Clone Hero Song Folder"),
]

ProgressCallback = Callable[[str, str], None]

def _safe_folder_name(s: str) -> str:
    """Sanitise a string for use as a filesystem directory name."""
    return " ".join(s.replace("/", "-").replace("\\", "-").split())

def _stream_filename(source_audio: Path) -> str:
    """Choose a ``song.*`` filename matching the source audio format."""
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
    charter: str = "AudioToChart (AI)",
    bpm: float | None = None,
    resolution: int = 192,
    from_midi: Path | None = None,
    quantize_divisor: int | None = None,
    transcriber: DrumTranscriber | None = None,
    on_progress: ProgressCallback | None = None
) -> Path:
    """Create a Clone Hero song folder with ``notes.chart``, ``song.ini``, and audio"""
    
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
            from audiotochart.inference.fake import FakeTranscriber
            transcriber = FakeTranscriber()
        hits = transcriber.transcribe(source_audio)
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
