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
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

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
    single = song_dir / _DRUM_SINGLE
    if single.is_file():
        return [single]
    splits = sorted(song_dir.glob(_DRUM_SPLIT_PATTERN))
    return splits

