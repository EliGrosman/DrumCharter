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
