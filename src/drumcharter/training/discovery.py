"""Discovery of Rock Band song directories from extracted archives.

Scans directory trees for notes.mid files and associates each with its
drum audio stems (drums.ogg or drums_*.ogg).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

_DRUM_SINGLE = "drums.ogg"
_DRUM_SPLIT_PATTERN = "drums_*.ogg"
_NOTES_MID = "notes.mid"


@dataclass
class RBSong:
    """Metadata for a single discovered Rock Band song.

    Attributes:
        path: Path to the song directory.
        midi_path: Path to notes.mid.
        drum_audio_paths: Paths to drum stem audio files.
        has_full_mix: Whether a full mix (song.ogg) is present.
        song_name: Display name for the song (defaults to directory name).
        source_archive: Name of the source archive file.
    """
    """Metadata for a single discovered Rock Band song.

    Attributes:
        path: Path to the song directory.
        midi_path: Path to notes.mid.
        drum_audio_paths: Paths to drum stem audio files.
        has_full_mix: Whether a full mix (song.ogg) is present.
        song_name: Display name for the song (defaults to directory name).
        source_archive: Name of the source archive file.
    """
    path: Path
    midi_path: Path
    drum_audio_paths: list[Path] = field(default_factory=list)
    has_full_mix: bool = False
    song_name: str = ""
    source_archive: str = ""

    def __post_init__(self) -> None:
        if not self.song_name:
            self.song_name = self.path.name


def discover_songs(root: Path) -> list[RBSong]:
    """Recursively discover all Rock Band songs under a root directory.

    Args:
        root: The root directory to search.

    Returns:
        A list of RBSong objects with valid drum audio.
    
    Raises:
        NotADirectoryError: If root is not a valid directory.
    """

    songs: list[RBSong] = []
    for midi_path in sorted(root.rglob(_NOTES_MID)):
        song_dir = midi_path.parent
        drum_paths = _find_drum_audio(song_dir)
        if not drum_paths:
            log.debug("Skipping %s — no drum audio found", song_dir.name)
            continue

        songs.append(
            RBSong(
                path=song_dir,
                midi_path=midi_path,
                drum_audio_paths=drum_paths,
                has_full_mix=(song_dir / "song.ogg").is_file(),
            )
        )

    log.info("Discovered %d songs with drum audio in %s", len(songs), root)
    return songs


def _find_drum_audio(song_dir: Path) -> list[Path]:
    """Find drum audio files in a song directory.

    Checks for a single drums.ogg first, then falls back to drums_*.ogg.
    """
    single = song_dir / _DRUM_SINGLE
    if single.is_file():
        return [single]
    splits = sorted(song_dir.glob(_DRUM_SPLIT_PATTERN))
    return splits

