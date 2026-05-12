from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable

from audiotochart.chart.format import DrumDifficulty, SongMetadata, write_chart_file
from audiotochart.chart.fake import create_fake_drum_chart
from audiotochart.chart.songini import SongIni, write_song_ini

logger = logging.getLogger(__name__)

# Stage IDS used by callback
STAGE_CHART = "chart"
STAGE_OUTPUT = "output"

STAGES = [
    (STAGE_CHART, "Generating Fake Drum Chart"),
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
    bpm: float = 120.0,
    measures: int = 16,
    resolution: int = 192,
    on_progress: ProgressCallback | None = None
) -> Path:
    """Create a Clone Hero song folder with ``notes.chart``, ``song.ini``, and audio"""
    
    source_audio = Path(source_audio)
    if not source_audio.is_file():
        raise FileNotFoundError(f"Source audio not found: {source_audio}")
    
    def _notify(stage: str, event: str) -> None:
        if on_progress is not None:
            on_progress(stage, event)

    logger.info("Generating fake drum chart for %s", source_audio.name)
    _notify(STAGE_CHART, "start")
    stream_name = _stream_filename(source_audio)
    meta = SongMetadata(
        name=song_name,
        artist=artist_name,
        charter=charter,
        resolution=resolution,
        offset=0.0,
        music_stream=stream_name,
    )
    doc = create_fake_drum_chart(song=meta, bpm=bpm, measures=measures)
    expert_notes = doc.drums.get(DrumDifficulty.EXPERT, [])
    if not expert_notes:
        logger.warning("No drum notes were generated; the chart will be empty.")
    _notify(STAGE_CHART, "done")

    logger.info("Writing Clone Hero song folder")
    _notify(STAGE_OUTPUT, "start")
    folder = output_parent / _safe_folder_name(f"{artist_name} - {song_name}")
    folder.mkdir(parents=True, exist_ok=True)

    write_chart_file(doc, folder / "notes.chart")
    write_song_ini(
        SongIni(name=song_name, artist=artist_name, charter=charter, diff_drums=4),
        folder / "song.ini",
    )
    shutil.copy2(source_audio, folder / stream_name)
    _notify(STAGE_OUTPUT, "done")

    logger.info("Chart folder generated at %s", folder)
    return folder
