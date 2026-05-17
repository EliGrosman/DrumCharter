"""Tests for lower drum difficulty generation."""

from __future__ import annotations

from audiotochart.chart.difficulty import (
    CYMBAL_YELLOW,
    KICK,
    SNARE,
    YELLOW_PAD,
    generate_difficulties,
)
from audiotochart.chart.format import ChartDocument, DrumDifficulty, DrumNote, SongMetadata

RESOLUTION = 192


def _doc(expert: list[DrumNote]) -> ChartDocument:
    return ChartDocument(
        song=SongMetadata(
            name="Difficulty",
            artist="Tests",
            charter="pytest",
            resolution=RESOLUTION,
        ),
        drums={DrumDifficulty.EXPERT: expert},
    )


def test_generate_difficulties_populates_lower_sections() -> None:
    doc = _doc(
        [
            DrumNote(0, KICK),
            DrumNote(RESOLUTION, SNARE),
            DrumNote(RESOLUTION * 2, KICK),
            DrumNote(RESOLUTION * 3, SNARE),
            DrumNote(RESOLUTION * 4, YELLOW_PAD),
            DrumNote(RESOLUTION * 4, CYMBAL_YELLOW),
        ]
    )

    generate_difficulties(doc)

    assert doc.drums[DrumDifficulty.HARD]
    assert doc.drums[DrumDifficulty.MEDIUM]
    assert doc.drums[DrumDifficulty.EASY]
    assert len(doc.drums[DrumDifficulty.HARD]) <= len(doc.drums[DrumDifficulty.EXPERT])
    assert len(doc.drums[DrumDifficulty.MEDIUM]) <= len(doc.drums[DrumDifficulty.HARD])
    assert len(doc.drums[DrumDifficulty.EASY]) <= len(doc.drums[DrumDifficulty.MEDIUM])


def test_generate_difficulties_skips_empty_expert() -> None:
    doc = _doc([])

    generate_difficulties(doc)

    assert DrumDifficulty.HARD not in doc.drums
    assert DrumDifficulty.MEDIUM not in doc.drums
    assert DrumDifficulty.EASY not in doc.drums


def test_hard_preserves_cymbal_modifier_when_pad_survives() -> None:
    doc = _doc(
        [
            DrumNote(0, YELLOW_PAD),
            DrumNote(0, CYMBAL_YELLOW),
        ]
    )

    generate_difficulties(doc)

    hard_notes = {(note.tick, note.note) for note in doc.drums[DrumDifficulty.HARD]}
    assert hard_notes == {(0, YELLOW_PAD), (0, CYMBAL_YELLOW)}


def test_hard_removes_cymbal_modifier_when_pad_is_thinned() -> None:
    doc = _doc(
        [
            DrumNote(0, YELLOW_PAD),
            DrumNote(0, CYMBAL_YELLOW),
            DrumNote(RESOLUTION // 16, YELLOW_PAD),
            DrumNote(RESOLUTION // 16, CYMBAL_YELLOW),
        ]
    )

    generate_difficulties(doc)

    hard_notes = {(note.tick, note.note) for note in doc.drums[DrumDifficulty.HARD]}
    assert hard_notes == {(0, YELLOW_PAD), (0, CYMBAL_YELLOW)}


def test_medium_preserves_pro_cymbal_modifier_from_yellow_hihat() -> None:
    doc = _doc(
        [
            DrumNote(0, YELLOW_PAD),
            DrumNote(0, CYMBAL_YELLOW),
        ]
    )

    generate_difficulties(doc)

    medium_notes = {(note.tick, note.note) for note in doc.drums[DrumDifficulty.MEDIUM]}
    assert medium_notes == {(0, YELLOW_PAD), (0, CYMBAL_YELLOW)}


def test_easy_preserves_sparse_pro_hihat_cymbal_modifier() -> None:
    doc = _doc(
        [
            DrumNote(0, KICK),
            DrumNote(0, YELLOW_PAD),
            DrumNote(0, CYMBAL_YELLOW),
            DrumNote(RESOLUTION, SNARE),
            DrumNote(RESOLUTION, YELLOW_PAD),
            DrumNote(RESOLUTION, CYMBAL_YELLOW),
            DrumNote(RESOLUTION * 2, YELLOW_PAD),
            DrumNote(RESOLUTION * 2, CYMBAL_YELLOW),
        ]
    )

    generate_difficulties(doc)

    easy_notes = {(note.tick, note.note) for note in doc.drums[DrumDifficulty.EASY]}
    assert (0, YELLOW_PAD) in easy_notes
    assert (0, CYMBAL_YELLOW) in easy_notes
    assert (RESOLUTION, YELLOW_PAD) in easy_notes
    assert (RESOLUTION, CYMBAL_YELLOW) in easy_notes
