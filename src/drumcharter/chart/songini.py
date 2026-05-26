"""Clone Hero ``song.ini`` generation.

Provides the :class:`SongIni` dataclass and a writer for the INI-format
metadata file used by Clone Hero.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _escape_ini_value(value: str) -> str:
    """Minimal escaping for values that may contain special characters.

    Replaces all line-break variants with spaces.

    Args:
        value: Raw string value.

    Returns:
        Escaped string safe for INI files.
    """
    return value.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


@dataclass
class SongIni:
    """Metadata stored in ``song.ini``. Only ``name`` is required by the game.

    Attributes:
        name: Song title (required).
        artist: Artist name.
        album: Album name.
        genre: Genre string.
        year: Release year.
        charter: Name of the chart creator.
        diff_drums: Drum difficulty rating (0-6).
        diff_drums_real: Pro drums difficulty rating (0-6).
        song_length: Song length in milliseconds.
        preview_start_time: Preview start position in milliseconds.
        loading_phrase: Loading screen text.
    """

    name: str
    artist: str | None = None
    album: str | None = None
    genre: str | None = None
    year: int | None = None
    charter: str | None = None
    diff_drums: int | None = None
    diff_drums_real: int | None = None
    song_length: int | None = None
    preview_start_time: int | None = None
    loading_phrase: str | None = None

    def to_lines(self) -> list[str]:
        """Generate the ``[Song]`` section as a list of INI-formatted lines.

        Omits fields set to None.

        Returns:
            Lines including the section header and key = value pairs.
        """
        lines: list[str] = ["[Song]"]
        pairs: list[tuple[str, Any]] = [
            ("name", self.name),
            ("artist", self.artist),
            ("album", self.album),
            ("genre", self.genre),
            ("year", self.year),
            ("charter", self.charter),
            ("diff_drums", self.diff_drums),
            ("diff_drums_real", self.diff_drums_real),
            ("song_length", self.song_length),
            ("preview_start_time", self.preview_start_time),
            ("loading_phrase", self.loading_phrase),
        ]
        for key, val in pairs:
            if val is None:
                continue
            if isinstance(val, str):
                lines.append(f"{key} = {_escape_ini_value(val)}")
            else:
                lines.append(f"{key} = {val}")
        return lines


def write_song_ini(ini: SongIni, path: str | Path) -> None:
    """Write ``song.ini`` as UTF-8 with LF newlines.

    Args:
        ini: The :class:`SongIni` instance to write.
        path: Destination file path.
    """
    text = "\n".join(ini.to_lines()) + "\n"
    Path(path).write_text(text, encoding="utf-8")
