"""Clone Hero .chart file format structures and serialisation.

Provides dataclasses for SongMetadata, SyncTrackEvent, SectionEvent,
DrumNote, and ChartDocument, along with render and write helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

def bpm_to_chart_integer(bpm: float) -> int:
    """Encode BPM as the integer used after ``B`` in ``[SyncTrack]`` (thousandths).

    Example: 120.0 -> 120000 meaning 120.000 BPM.

    Args:
        bpm: Beats per minute.

    Returns:
        Integer encoding of BPM * 1000, rounded.
    """
    return int(round(bpm * 1000))

def chart_integer_to_bpm(value: int) -> float:
    """Decode the ``B`` sync value back to BPM.

    Args:
        value: Integer from the chart file (e.g. 120000).

    Returns:
        BPM as a float (e.g. 120.0).
    """
    return value / 1000.0

def _escape_quoted_value(text: str) -> str:
    """Escape backslashes and double-quotes in a chart file string value."""
    return text.replace("\\", "\\\\").replace('"', '\\"')

def _format_offset(offset: float) -> str:
    """Format an offset value for the ``[Song]`` section.

    Returns an integer string if the offset has no fractional part.
    """
    if offset == int(offset):
        return str(int(offset))
    return repr(float(offset))

@dataclass(frozen=True)
class SongMetadata:
    """Fields for the ``[Song]`` section of a ``.chart`` file.

    Attributes:
        name: Song title.
        artist: Artist name.
        charter: Name of the chart creator.
        resolution: Ticks per quarter note. Defaults to 192.
        offset: Audio offset in seconds. Defaults to 0.0.
        music_stream: Audio filename. Defaults to ``"song.ogg"``.
        album: Album name (optional).
        genre: Genre string (optional).
        year: Release year (optional).
    """
    
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
        """Generate the ``[Song]`` section as a list of lines.

        Returns:
            Lines including the section header, braces, and key-value pairs.
        """
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
    """Difficulty levels supported for pro drums."""

    EXPERT = "Expert"
    HARD = "Hard"
    MEDIUM = "Medium"
    EASY = "Easy"

    @property
    def section_name(self) -> str:
        """Return the ``.chart`` section header for this difficulty (e.g. ``ExpertDrums``)."""
        return f"{self.value}Drums"

@dataclass(frozen=True)
class SyncTrackEvent:
    """One line inside `[SyncTrack] (e.g. ``0 = B 120000``)"""
    
    tick: int
    # Raw rich hand side after "= " e.g. "TS 4", "B 120000"
    payload: str
    
    def line(self) -> str:
        """Render the sync track event as a ``.chart`` file line.

        Returns:
            A string like ``"  0 = B 120000"``.
        """
        return f"  {self.tick} = {self.payload}"

@dataclass(frozen=True)
class SectionEvent:
    """Text event in ``[Events]`` (e.g. section markers)."""

    tick: int
    text: str

    def line(self) -> str:
        """Render the section event as a ``.chart`` file line.

        Returns:
            A string like ``"  0 = E \"verse\""``.
        """
        escaped = _escape_quoted_value(self.text)
        return f'  {self.tick} = E "{escaped}"'

@dataclass(frozen=True)
class DrumNote:
    """A single drum note: ``tick = N <note> <length>``.

    Attributes:
        tick: Position in ticks.
        note: MIDI note number for the drum pad.
        length: Note sustain length in ticks (0 for normal notes).
    """

    tick: int
    note: int
    length: int = 0

    def line(self) -> str:
        """Render the drum note as a ``.chart`` file line.

        Returns:
            A string like ``"  0 = N 0 0"``.
        """
        return f"  {self.tick} = N {self.note} {self.length}"

@dataclass
class ChartDocument:
    """Complete in-memory ``.chart`` file representation.

    Attributes:
        song: Metadata for the ``[Song]`` section.
        sync: List of sync track events.
        events: List of section/event markers.
        drums: Mapping from difficulty to list of drum notes.
    """

    song: SongMetadata
    sync: list[SyncTrackEvent] = field(default_factory=list)
    events: list[SectionEvent] = field(default_factory=list)
    drums: dict[DrumDifficulty, list[DrumNote]] = field(default_factory=dict)

def _sorted_unique_sync(events: Iterable[SyncTrackEvent]) -> list[SyncTrackEvent]:
    """Return sync events sorted by tick then payload, deduplicated by sort order.

    Args:
        events: Sync track events to sort.

    Returns:
        Sorted list of sync track events.
    """
    return sorted(events, key=lambda e: (e.tick, e.payload))


def _sorted_unique_drums(notes: Iterable[DrumNote]) -> list[DrumNote]:
    """Return drum notes sorted and deduplicated by (tick, note, length).

    Args:
        notes: Drum notes to sort and deduplicate.

    Returns:
        Sorted list with duplicates removed.
    """
    seen: set[tuple[int, int, int]] = set()
    out: list[DrumNote] = []
    for n in sorted(notes, key=lambda n: (n.tick, n.note, n.length)):
        key = (n.tick, n.note, n.length)
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


def _format_block(lines: list[str]) -> str:
    """Join a list of lines into a newline-terminated block.

    Args:
        lines: Lines to join.

    Returns:
        A single string with newlines appended.
    """
    return "\n".join(lines) + "\n"

def write_chart(doc: ChartDocument) -> str:
    """Render a complete ``.chart`` file body as a string.

    Produces sections for Song, SyncTrack, Events, and all drum difficulties.

    Args:
        doc: The chart document to render.

    Returns:
        The full ``.chart`` file contents.
    """
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
    """Write a ``.chart`` file to disk as UTF-8.

    Args:
        doc: The chart document to write.
        path: Destination file path.
    """
    p = Path(path)
    p.write_text(write_chart(doc), encoding="utf-8")
