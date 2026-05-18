"""Tests for ``.chart`` generation and ``song.ini`` writing."""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import chparse
import pytest

from audiotochart.chart import (
    ChartDocument,
    DrumDifficulty,
    DrumNote,
    SectionEvent,
    SongIni,
    SongMetadata,
    SyncTrackEvent,
    bpm_to_chart_integer,
    chart_integer_to_bpm,
    write_chart,
    write_chart_file,
    write_song_ini,
)


# ---------------------------------------------------------------------------
# BPM encoding
# ---------------------------------------------------------------------------

def test_bpm_round_trip() -> None:
    assert bpm_to_chart_integer(120.0) == 120_000
    assert chart_integer_to_bpm(120_000) == pytest.approx(120.0)
    assert bpm_to_chart_integer(125.5) == 125_500


def test_bpm_fractional_precision() -> None:
    assert bpm_to_chart_integer(99.999) == 99_999
    assert bpm_to_chart_integer(200.001) == 200_001


def test_bpm_zero_and_high() -> None:
    assert bpm_to_chart_integer(0.0) == 0
    assert bpm_to_chart_integer(999.0) == 999_000


# ---------------------------------------------------------------------------
# .chart writer
# ---------------------------------------------------------------------------

def test_write_chart_matches_design_shape() -> None:
    doc = ChartDocument(
        song=SongMetadata(
            name="Enter Sandman",
            artist="Metallica",
            charter="AudioToChart (AI)",
            music_stream="song.ogg",
        ),
        sync=[
            SyncTrackEvent(0, "TS 4"),
            SyncTrackEvent(0, "B 120000"),
            SyncTrackEvent(3072, "B 125000"),
        ],
        events=[
            SectionEvent(768, "section Intro"),
            SectionEvent(3072, "section Verse 1"),
        ],
        drums={
            DrumDifficulty.EXPERT: [
                DrumNote(768, 0, 0),
                DrumNote(768, 2, 0),
                DrumNote(768, 66, 0),
                DrumNote(960, 1, 0),
                DrumNote(960, 2, 0),
                DrumNote(960, 66, 0),
            ]
        },
    )
    text = write_chart(doc)
    assert '[Song]\n{\n  Name = "Enter Sandman"' in text
    assert '  MusicStream = "song.ogg"' in text
    assert "0 = TS 4" in text and "0 = B 120000" in text
    assert "3072 = B 125000" in text
    assert '768 = E "section Intro"' in text
    assert "[ExpertDrums]" in text and "768 = N 0 0" in text
    assert "[HardDrums]" in text


def test_chparse_round_trip_metadata_and_drums() -> None:
    doc = ChartDocument(
        song=SongMetadata(
            name="RoundTrip",
            artist="The Tests",
            charter="pytest",
            resolution=192,
            offset=-0.012,
            music_stream="audio.ogg",
        ),
        sync=[SyncTrackEvent(0, "TS 4"), SyncTrackEvent(0, "B 90000")],
        events=[SectionEvent(192, "section A")],
        drums={
            DrumDifficulty.EXPERT: [
                DrumNote(384, 1, 0),
                DrumNote(384, 66, 0),
            ],
            DrumDifficulty.HARD: [DrumNote(768, 0, 0)],
        },
    )
    chart = chparse.load(StringIO(write_chart(doc)))
    assert chart.Name == "RoundTrip"
    assert chart.Artist == "The Tests"
    assert chart.Charter == "pytest"
    assert chart.Resolution == 192
    assert float(chart.Offset) == pytest.approx(-0.012)
    assert chart.MusicStream == "audio.ogg"

    expert = chart.instruments[chparse.EXPERT][chparse.DRUMS]
    assert len(expert) == 1
    n0 = expert[0]
    assert n0.time == 384 and n0.fret == 1
    # Cymbal modifier 66 is merged into the pad note by chparse.
    assert any(getattr(f, "value", f) == 66 for f in n0.flags)

    hard = chart.instruments[chparse.HARD][chparse.DRUMS]
    assert [(n.time, n.fret) for n in hard] == [(768, 0)]


def test_song_ini_writes_expected_keys(tmp_path: Path) -> None:
    path = tmp_path / "song.ini"
    write_song_ini(
        SongIni(
            name="Test Song",
            artist="Artist",
            charter="Charter",
            year=2024,
            diff_drums=5,
        ),
        path,
    )
    body = path.read_text(encoding="utf-8")
    assert body.startswith("[Song]\n")
    assert "name = Test Song" in body
    assert "artist = Artist" in body
    assert "charter = Charter" in body
    assert "year = 2024" in body
    assert "diff_drums = 5" in body


def test_write_chart_file_round_trip_file(tmp_path: Path) -> None:
    doc = ChartDocument(
        song=SongMetadata(name="File", artist="A", charter="B"),
        sync=[SyncTrackEvent(0, "TS 4"), SyncTrackEvent(0, "B 120000")],
        drums={DrumDifficulty.EXPERT: [DrumNote(0, 0, 0)]},
    )
    out = tmp_path / "notes.chart"
    write_chart_file(doc, out)
    chart = chparse.load(out.open(encoding="utf-8"))
    assert chart.Name == "File"
    assert len(list(chart.instruments[chparse.EXPERT][chparse.DRUMS])) >= 1


def test_empty_difficulty_sections_emitted() -> None:
    text = write_chart(
        ChartDocument(song=SongMetadata(name="X", artist="Y", charter="Z"), sync=[SyncTrackEvent(0, "B 60000")])
    )
    assert re.search(r"\[HardDrums\]\s*\{\s*\}", text)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_special_characters_in_metadata() -> None:
    """Quotes and backslashes in song names must be escaped."""
    doc = ChartDocument(
        song=SongMetadata(
            name='He said "hello"',
            artist="AC\\DC",
            charter="Test",
        ),
        sync=[SyncTrackEvent(0, "TS 4"), SyncTrackEvent(0, "B 120000")],
    )
    text = write_chart(doc)
    assert r'He said \"hello\"' in text
    assert r"AC\\DC" in text


def test_all_four_difficulty_sections_present() -> None:
    """Even with no notes, all difficulty sections must appear."""
    doc = ChartDocument(
        song=SongMetadata(name="Empty", artist="Nobody", charter="Test"),
        sync=[SyncTrackEvent(0, "B 120000")],
    )
    text = write_chart(doc)
    for diff in ("Expert", "Hard", "Medium", "Easy"):
        assert f"[{diff}Drums]" in text


def test_notes_sorted_by_tick_then_note() -> None:
    """Notes should be emitted sorted even if given out of order."""
    doc = ChartDocument(
        song=SongMetadata(name="Sort", artist="A", charter="B"),
        sync=[SyncTrackEvent(0, "B 120000")],
        drums={
            DrumDifficulty.EXPERT: [
                DrumNote(960, 1, 0),
                DrumNote(384, 0, 0),
                DrumNote(384, 2, 0),
                DrumNote(960, 0, 0),
            ]
        },
    )
    text = write_chart(doc)
    expert_section = text.split("[ExpertDrums]")[1].split("}")[0]
    lines = [line.strip() for line in expert_section.strip().splitlines() if "= N" in line]
    ticks = [int(line.split("=")[0].strip()) for line in lines]
    assert ticks == sorted(ticks)


def test_duplicate_notes_deduplicated() -> None:
    """Identical notes at the same tick should appear only once."""
    doc = ChartDocument(
        song=SongMetadata(name="Dupe", artist="A", charter="B"),
        sync=[SyncTrackEvent(0, "B 120000")],
        drums={
            DrumDifficulty.EXPERT: [
                DrumNote(192, 0, 0),
                DrumNote(192, 0, 0),
                DrumNote(192, 1, 0),
            ]
        },
    )
    text = write_chart(doc)
    expert_section = text.split("[ExpertDrums]")[1].split("}")[0]
    note_lines = [line.strip() for line in expert_section.strip().splitlines() if "= N" in line]
    # Should have 2 unique notes, not 3
    assert len(note_lines) == 2


def test_song_ini_minimal() -> None:
    """Only the name field is required."""
    ini = SongIni(name="Minimal")
    lines = ini.to_lines()
    assert lines[0] == "[Song]"
    assert "name = Minimal" in lines
    assert len(lines) == 2  # [Song] + name


def test_song_ini_newlines_in_values() -> None:
    """Newlines in values must be stripped."""
    ini = SongIni(name="Line\nBreak", loading_phrase="Hello\r\nWorld")
    lines = ini.to_lines()
    name_line = [line for line in lines if line.startswith("name")][0]
    assert "\n" not in name_line
    phrase_line = [line for line in lines if line.startswith("loading_phrase")][0]
    assert "\n" not in phrase_line
    assert "\r" not in phrase_line


def test_cymbal_modifier_at_same_tick() -> None:
    """Cymbal modifier and pad note at same tick both appear in output."""
    doc = ChartDocument(
        song=SongMetadata(name="Cym", artist="A", charter="B"),
        sync=[SyncTrackEvent(0, "B 120000")],
        drums={
            DrumDifficulty.EXPERT: [
                DrumNote(0, 2, 0),   # Yellow pad
                DrumNote(0, 66, 0),  # Yellow cymbal modifier
            ]
        },
    )
    text = write_chart(doc)
    assert "0 = N 2 0" in text
    assert "0 = N 66 0" in text


def test_optional_metadata_fields() -> None:
    """Album, genre, year appear when set; absent when None."""
    doc = ChartDocument(
        song=SongMetadata(
            name="Full",
            artist="Band",
            charter="C",
            album="The Album",
            genre="Rock",
            year=2024,
        ),
        sync=[SyncTrackEvent(0, "B 120000")],
    )
    text = write_chart(doc)
    assert 'Album = "The Album"' in text
    assert 'Genre = "Rock"' in text
    assert "Year = 2024" in text

    doc_minimal = ChartDocument(
        song=SongMetadata(name="Min", artist="A", charter="B"),
        sync=[SyncTrackEvent(0, "B 120000")],
    )
    text_min = write_chart(doc_minimal)
    assert "Album" not in text_min
    assert "Genre" not in text_min
    assert "Year" not in text_min
