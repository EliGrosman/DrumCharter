from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

def bpm_to_chart_integer(bpm: float) -> int:
    """Encode BPM as the integer used after `B` in `[SyncTrack]` (thousandths).

    Example: 120.0 -> 120000 meaning 120.000 BPM.
    """
    return int(round(bpm * 1000))

def chart_integer_to_bpm(value: int) -> float:
    """Decode the `B` sync value back to BPM."""
    return value / 1000.0

def _escape_quoted_value(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')

def _format_offset(offset: float) -> str:
    if offset == int(offset):
        return str(int(offset))
    return repr(float(offset))

@dataclass(frozen=True)
class SongMetadata:
    """Fields for the `[Song]` section."""
    
    name: str
    artist: str
    charter: str
    resolution: int = 192
    offset: float = 0.0
    music_stream: str = "song.ogg"
    album: str | None = None
    genre: str | None = None
    year: int | None = None
    
    def song_section_lines(self) -> list[str]:
        lines = [
            "[Song]",
            "{",
            f'  Name = "{_escape_quoted_value(self.name)}"',
            f'  Artist = "{_escape_quoted_value(self.artist)}"',
            f'  Charter = "{_escape_quoted_value(self.charter)}"',
            f"  Resolution = {self.resolution}",
            f"  Offset = {_format_offset(self.offset)}",
            f'  MusicStream = "{_escape_quoted_value(self.music_stream)}"',
        ]
        if self.album is not None:
            lines.append(f'  Album = "{_escape_quoted_value(self.album)}"')
        if self.genre is not None:
            lines.append(f'  Genre = "{_escape_quoted_value(self.genre)}"')
        if self.year is not None:
            lines.append(f"  Year = {self.year}")
        lines.append("}")
        return lines
    
class DrumDifficulty(str, Enum):
    EXPERT = "Expert"
    HARD = "Hard"
    MEDIUM = "Medium"
    EASY = "Easy"
    
    @property
    def section_name(self) -> str:
        return f"{self.value}Drums"

@dataclass(frozen=True)
class SyncTrackEvent:
    """One line inside `[SyncTrack] (e.g. ``0 = B 120000``)"""
    
    tick: int
    # Raw rich hand side after "= " e.g. "TS 4", "B 120000"
    payload: str
    
    def line(self) -> str:
        return f"  {self.tick} = {self.payload}"

@dataclass(frozen=True)
class SectionEvent:
    """Text event in `[Events]` (e.g. section markers)"""
    
    tick: int
    text: str
    
    def line(self) -> str:
        escaped = _escape_quoted_value(self.text)
        return f'  {self.tick} = E "{escaped}"'
    
@dataclass(frozen=True)
class DrumNote:
    """A single drum note: ``tick = N <note> <length>``"""
    
    tick: int
    note: int
    length: int = 0
    
    def line(self) -> str:
        return f"  {self.tick} = N {self.note} {self.length}"
    
@dataclass
class ChartDocument:
    """Complete in memory `.chart` file"""
    
    song: SongMetadata
    sync: list[SyncTrackEvent] = field(default_factory=list)
    events: list[SectionEvent] = field(default_factory=list)
    drums: dict[DrumDifficulty, list[DrumNote]] = field(default_factory=dict)
    
def _sorted_unique_sync(events: Iterable[SyncTrackEvent]) -> list[SyncTrackEvent]:
    return sorted(events, key=lambda e: (e.tick, e.payload))


def _sorted_unique_drums(notes: Iterable[DrumNote]) -> list[DrumNote]:
    seen: set[tuple[int, int, int]] = set()
    out: list[DrumNote] = []
    for n in sorted(notes, key=lambda n: (n.tick, n.note, n.length)):
        key = (n.tick, n.note, n.length)
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


def _format_block(lines: list[str]) -> str:
    return "\n".join(lines) + "\n"

def write_chart(doc: ChartDocument) -> str:
    """Render a `.chart` file body"""
    parts: list[str] = []
    
    parts.append(_format_block(doc.song.song_section_lines()))
    
    sync_lines = ["[SyncTrack]", "{", *[e.line() for e in _sorted_unique_sync(doc.sync)], "}"]
    parts.append(_format_block(sync_lines))
    
    ev_lines = ["[Events]", "{", *[e.line() for e in sorted(doc.events, key=lambda e: (e.tick, e.text))], "}"]
    parts.append(_format_block(ev_lines))
    
    for diff in (DrumDifficulty.EXPERT, DrumDifficulty.HARD, DrumDifficulty.MEDIUM, DrumDifficulty.EASY):
        notes = _sorted_unique_drums(doc.drums.get(diff) or [])
        sec = [f"[{diff.section_name}]", "{"]
        sec.extend(n.line() for n in notes)
        sec.append("}")
        parts.append(_format_block(sec))
    
    return "".join(parts)

def write_chart_file(doc: ChartDocument, path: str | Path) -> None:
    """Write UTF-8 `.chart`"""
    p = Path(path)
    p.write_text(write_chart(doc), encoding="utf-8")
