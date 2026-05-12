from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _escape_ini_value(value: str) -> str:
    """Minimal escaping for values that may contain special characters."""
    return value.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


@dataclass
class SongIni:
    """Metadata stored in `song.ini`. Only `name` is required by the game."""

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
    """Write `song.ini` as UTF-8 with LF newlines."""
    text = "\n".join(ini.to_lines()) + "\n"
    Path(path).write_text(text, encoding="utf-8")
