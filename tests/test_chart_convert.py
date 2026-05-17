"""Tests for the DrumHit conversion pipeline."""

from __future__ import annotations

import pytest

from audiotochart.chart import DrumDifficulty
from audiotochart.chart.convert import (
    INSTRUMENT_MAP,
    build_beat_tempo_map,
    build_sync_track_from_beats,
    hits_to_chart_document,
    seconds_to_tick,
    seconds_to_tick_tempo_map,
    tick_to_seconds_tempo_map,
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


def test_seconds_to_tick_tempo_map_handles_tempo_change() -> None:
    tempo_map = build_beat_tempo_map([0.5, 1.0, 1.4], resolution=192)
    assert tempo_map is not None
    assert seconds_to_tick_tempo_map(0.0, tempo_map) == 0
    assert seconds_to_tick_tempo_map(0.5, tempo_map) == 192
    assert seconds_to_tick_tempo_map(1.0, tempo_map) == 384
    assert seconds_to_tick_tempo_map(1.2, tempo_map) == 480
    assert tick_to_seconds_tempo_map(480, tempo_map) == pytest.approx(1.2)


def test_build_sync_track_from_beats_emits_tempo_changes() -> None:
    sync = build_sync_track_from_beats([0.5, 1.0, 1.4], resolution=192)
    assert [event.payload for event in sync[:2]] == ["TS 4", "B 120000"]
    assert any(event.tick == 384 and event.payload == "B 150000" for event in sync)


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


def test_all_pro8_pad_notes() -> None:
    """Every pro8 instrument maps to the correct Clone Hero pad number."""
    pro8_pads: list[tuple[str, set[int]]] = [
        ("kick", {0}),
        ("snare", {1}),
        ("hihat", {2}),
        ("tom_yellow", {2}),
        ("ride", {3}),
        ("tom_blue", {3}),
        ("crash", {4}),
        ("tom_green", {4}),
    ]
    for instrument, expected_pads in pro8_pads:
        doc = hits_to_chart_document(
            [DrumHit(0.0, instrument)],
            song=SongMetadata(name="P", artist="Q", charter="R"),
            bpm=120.0,
        )
        expert = doc.drums.get(DrumDifficulty.EXPERT, [])
        pad_notes = {n.note for n in expert if n.note <= 4}
        assert pad_notes == expected_pads, (
            f"{instrument}: expected pads {expected_pads}, got {pad_notes}"
        )


def test_cymbal_instruments_emit_modifiers() -> None:
    """hihat, ride, and crash produce cymbal modifier notes (66, 67, 68)."""
    cymbal_cases: list[tuple[str, set[int]]] = [
        ("hihat", {66}),
        ("ride", {67}),
        ("crash", {68}),
    ]
    for instrument, expected_cymbals in cymbal_cases:
        doc = hits_to_chart_document(
            [DrumHit(0.0, instrument)],
            song=SongMetadata(name="C", artist="D", charter="E"),
            bpm=120.0,
        )
        expert = doc.drums.get(DrumDifficulty.EXPERT, [])
        cymbal_notes = {n.note for n in expert if n.note >= 66}
        assert cymbal_notes == expected_cymbals, (
            f"{instrument}: expected cymbals {expected_cymbals}, got {cymbal_notes}"
        )


def test_tom_instruments_no_cymbal_modifiers() -> None:
    """tom_yellow, tom_blue, tom_green produce pad-only notes (no cymbal)."""
    for instrument in ("tom_yellow", "tom_blue", "tom_green"):
        doc = hits_to_chart_document(
            [DrumHit(0.0, instrument)],
            song=SongMetadata(name="T", artist="U", charter="V"),
            bpm=120.0,
        )
        expert = doc.drums.get(DrumDifficulty.EXPERT, [])
        cymbal_notes = {n.note for n in expert if n.note >= 66}
        assert cymbal_notes == set(), (
            f"{instrument}: should have no cymbal modifiers, got {cymbal_notes}"
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
    assert len(doc.events) == 0


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


def test_hits_to_chart_document_uses_variable_beat_map_without_quantizing() -> None:
    doc = hits_to_chart_document(
        [
            DrumHit(0.5, "kick"),
            DrumHit(1.2, "snare"),
        ],
        song=SongMetadata(name="Tempo", artist="Tests", charter="pytest"),
        bpm=120.0,
        beat_times=[0.5, 1.0, 1.4],
    )

    assert any(event.tick == 384 and event.payload == "B 150000" for event in doc.sync)
    expert = doc.drums.get(DrumDifficulty.EXPERT, [])
    assert sorted((note.tick, note.note) for note in expert) == [(192, 0), (480, 1)]


def test_quantize_is_opt_in() -> None:
    song = SongMetadata(name="Quantize", artist="Tests", charter="pytest")
    beats = [0.0, 0.5, 1.0]

    unsnapped = hits_to_chart_document(
        [DrumHit(0.51, "kick")],
        song=song,
        bpm=120.0,
        beat_times=beats,
    )
    snapped = hits_to_chart_document(
        [DrumHit(0.51, "kick")],
        song=song,
        bpm=120.0,
        beat_times=beats,
        quantize_divisor=16,
    )

    assert unsnapped.drums[DrumDifficulty.EXPERT][0].tick == 196
    assert snapped.drums[DrumDifficulty.EXPERT][0].tick == 192


def test_uneven_beat_quantization_snaps_to_local_grid() -> None:
    """Regression test: notes snap against local beat intervals, not average BPM.

    Uneven beats [0.0, 1.0, 1.8, 2.4] produce inter-beat intervals of 1.0s and
    0.8s. Sixteenth-note subdivisions of the 0.8s interval fall at 1.0, 1.2,
    1.4, 1.6 — which differ from a constant 0.25s division. A hit at 1.21s
    snapping to 1.2 (tick 240) proves the grid used actual beat positions.
    """
    song = SongMetadata(name="Uneven", artist="Tests", charter="pytest")
    uneven_beats = [0.0, 1.0, 1.8, 2.4]
    resolution = 192

    # Hit 10ms past a 16th note: 0.51 → snaps to 0.5 → tick 96
    hit_near_grid = DrumHit(0.51, "kick")
    # Hit 10ms past a 16th note in the faster (0.8s) interval: 1.21 → snaps to 1.2 → tick 240
    hit_near_fast_grid = DrumHit(1.21, "snare")
    # Hit 100ms from nearest grid (0.25): too far, stays unsnapped at 0.15 → tick 29
    hit_off_grid = DrumHit(0.15, "hihat")

    doc = hits_to_chart_document(
        [hit_near_grid, hit_near_fast_grid, hit_off_grid],
        song=song,
        bpm=60.0,
        resolution=resolution,
        beat_times=uneven_beats,
        quantize_divisor=16,
    )

    expert = doc.drums[DrumDifficulty.EXPERT]
    notes = {(n.tick, n.note) for n in expert}

    assert (96, 0) in notes, "0.51s kick should snap to tick 96 (local 16th of 1.0s interval)"
    assert (240, 1) in notes, "1.21s snare should snap to tick 240 (local 16th of 0.8s interval)"
    assert (29, 66) in notes, "0.15s hihat should stay at tick 29 (too far from grid)"
    assert (29, 2) in notes, "0.15s hihat pad should also be at tick 29"


# ---------------------------------------------------------------------------
# Fake chart integration
# ---------------------------------------------------------------------------


def test_fake_chart_still_contains_kick_snare_hihat() -> None:
    doc = create_fake_drum_chart(
        song=SongMetadata(name="Test", artist="Art", charter="Ch", resolution=192),
        duration_sec=8.0,
        bpm=120.0,
    )
    expert = doc.drums.get(DrumDifficulty.EXPERT, [])
    note_numbers = {note.note for note in expert}
    assert 0 in note_numbers   # kick
    assert 1 in note_numbers   # snare
    assert 2 in note_numbers   # hihat pad
    assert 66 in note_numbers  # hihat cymbal
