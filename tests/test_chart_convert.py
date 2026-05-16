"""Tests for the DrumHit conversion pipeline."""

from __future__ import annotations

import pytest

from audiotochart.chart import ChartDocument, DrumDifficulty, DrumNote, SectionEvent, SyncTrackEvent, write_chart
from audiotochart.chart.convert import (
    INSTRUMENT_MAP,
    hits_to_chart_document,
    seconds_to_tick,
)
from audiotochart.chart.fake import create_fake_drum_chart
from audiotochart.chart.format import SongMetadata
from audiotochart.drums import DrumHit


# ---------------------------------------------------------------------------
# seconds_to_tick
# ---------------------------------------------------------------------------


def test_seconds_to_tick_basic() -> None:
    assert seconds_to_tick(0.5, bpm=120, resolution=192) == 192


def test_seconds_to_tick_zero() -> None:
    assert seconds_to_tick(0.0, bpm=120, resolution=192) == 0


def test_seconds_to_tick_negative() -> None:
    assert seconds_to_tick(-1.0, bpm=120, resolution=192) == 0


# ---------------------------------------------------------------------------
# INSTRUMENT_MAP
# ---------------------------------------------------------------------------


def test_hihat_creates_cymbal_modifier() -> None:
    note, cymbal = INSTRUMENT_MAP["hihat"]
    assert note == 2
    assert cymbal == 66


def test_unknown_instrument_raises() -> None:
    with pytest.raises(ValueError, match="Unknown instrument"):
        hits_to_chart_document(
            [DrumHit(0.0, "xylophone")],
            song=SongMetadata(name="X", artist="Y", charter="Z"),
            bpm=120.0,
        )


# ---------------------------------------------------------------------------
# hits_to_chart_document
# ---------------------------------------------------------------------------


def test_duplicate_hits_deduplicated() -> None:
    hits = [
        DrumHit(0.0, "kick"),
        DrumHit(0.0, "kick"),
    ]
    doc = hits_to_chart_document(
        hits,
        song=SongMetadata(name="D", artist="E", charter="F"),
        bpm=120.0,
    )
    expert = doc.drums.get(DrumDifficulty.EXPERT, [])
    kick_notes = [n for n in expert if n.note == 0]
    assert len(kick_notes) == 1


def test_hits_produce_sync_and_events() -> None:
    hits = [DrumHit(0.0, "kick")]
    doc = hits_to_chart_document(
        hits,
        song=SongMetadata(name="S", artist="A", charter="C"),
        bpm=120.0,
    )
    assert any(e.payload == "TS 4" for e in doc.sync)
    assert any(e.payload.startswith("B ") for e in doc.sync)
    assert any(isinstance(e, SectionEvent) and "Intro" in e.text for e in doc.events)


def test_multiple_instruments_produce_correct_notes() -> None:
    hits = [
        DrumHit(0.0, "kick"),
        DrumHit(0.0, "snare"),
        DrumHit(0.0, "hihat"),
    ]
    doc = hits_to_chart_document(
        hits,
        song=SongMetadata(name="M", artist="N", charter="O"),
        bpm=120.0,
    )
    expert = doc.drums.get(DrumDifficulty.EXPERT, [])
    notes = {(n.tick, n.note) for n in expert}
    assert (0, 0) in notes   # kick
    assert (0, 1) in notes   # snare
    assert (0, 2) in notes   # hihat pad
    assert (0, 66) in notes  # hihat cymbal


# ---------------------------------------------------------------------------
# Fake chart integration
# ---------------------------------------------------------------------------


def test_fake_chart_still_contains_kick_snare_hihat(tmp_path) -> None:
    doc = create_fake_drum_chart(
        song=SongMetadata(name="Test", artist="Art", charter="Ch", resolution=192),
        bpm=120.0,
        measures=2,
    )
    text = write_chart(doc)
    assert "0 = N 0 0" in text   # kick
    assert "0 = N 1 0" in text   # snare
    assert "0 = N 2 0" in text   # hihat pad
    assert "0 = N 66 0" in text  # hihat cymbal
