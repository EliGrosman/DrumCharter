"""Tests for General MIDI drum conversion."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

import drumcharter.pipeline as pipeline
from drumcharter.chart.format import DrumDifficulty, SongMetadata
from drumcharter.chart.midi import iter_drum_midi_hits, midi_to_chart_document

pretty_midi = pytest.importorskip("pretty_midi")


def _make_wav(tmp_path: Path, name: str, duration_sec: float, sample_rate: int = 44100) -> Path:
    path = tmp_path / name
    num_samples = int(duration_sec * sample_rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_samples)
    return path


def _write_midi(
    path: Path,
    *,
    drum_pitches: list[int],
    non_drum_pitches: list[int] | None = None,
    drum_track_name: str = "drums",
    is_drum: bool = True,
) -> Path:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)

    drums = pretty_midi.Instrument(program=0, is_drum=is_drum, name=drum_track_name)
    for index, pitch in enumerate(drum_pitches):
        start = index * 0.5
        drums.notes.append(
            pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=start + 0.1)
        )
    midi.instruments.append(drums)

    if non_drum_pitches:
        piano = pretty_midi.Instrument(program=0, is_drum=False, name="piano")
        for index, pitch in enumerate(non_drum_pitches):
            start = index * 0.5
            piano.notes.append(
                pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=start + 0.1)
            )
        midi.instruments.append(piano)

    midi.write(str(path))
    return path


def _write_chart_midi_chord(path: Path, pitches: list[int]) -> Path:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    drums = pretty_midi.Instrument(program=0, is_drum=False, name="PART DRUMS")
    for pitch in pitches:
        drums.notes.append(
            pretty_midi.Note(velocity=100, pitch=pitch, start=0.0, end=0.1)
        )
    midi.instruments.append(drums)
    midi.write(str(path))
    return path


def _expert_note_numbers(path: Path) -> set[int]:
    doc = midi_to_chart_document(
        path,
        song=SongMetadata(name="MIDI", artist="Tests", charter="pytest"),
        bpm=120.0,
    )
    return {note.note for note in doc.drums[DrumDifficulty.EXPERT]}


def test_kick_midi_pitch_becomes_clone_hero_kick(tmp_path: Path) -> None:
    midi_path = _write_midi(tmp_path / "kick.mid", drum_pitches=[36])

    assert 0 in _expert_note_numbers(midi_path)


def test_hihat_midi_pitch_becomes_pad_and_cymbal_modifier(tmp_path: Path) -> None:
    midi_path = _write_midi(tmp_path / "hihat.mid", drum_pitches=[42])

    notes = _expert_note_numbers(midi_path)
    assert 2 in notes
    assert 66 in notes


def test_non_drum_midi_instruments_are_ignored(tmp_path: Path) -> None:
    midi_path = _write_midi(tmp_path / "non_drum.mid", drum_pitches=[], non_drum_pitches=[36])

    assert iter_drum_midi_hits(midi_path) == []


def test_unmapped_midi_pitches_are_skipped(tmp_path: Path) -> None:
    midi_path = _write_midi(tmp_path / "unmapped.mid", drum_pitches=[37])

    assert iter_drum_midi_hits(midi_path) == []


def test_chart_midi_part_drums_track_is_read_even_when_not_marked_drum(tmp_path: Path) -> None:
    midi_path = _write_midi(
        tmp_path / "chart.mid",
        drum_pitches=[96, 98],
        drum_track_name="PART DRUMS",
        is_drum=False,
    )

    notes = _expert_note_numbers(midi_path)
    assert 0 in notes
    assert 2 in notes
    assert 66 in notes


def test_chart_midi_tom_markers_override_default_cymbal_lanes(tmp_path: Path) -> None:
    midi_path = _write_chart_midi_chord(tmp_path / "chart_tom.mid", pitches=[98, 110])

    notes = _expert_note_numbers(midi_path)
    assert 2 in notes
    assert 66 not in notes


def test_pipeline_can_generate_folder_from_midi_drums(tmp_path: Path) -> None:
    audio_path = _make_wav(tmp_path, "song.wav", duration_sec=2.0)
    midi_path = _write_midi(tmp_path / "drums.mid", drum_pitches=[36, 42])

    folder = pipeline.generate_drum_chart_folder(
        source_audio=audio_path,
        output_parent=tmp_path / "out",
        song_name="Song",
        artist_name="Artist",
        bpm=120.0,
        from_midi=midi_path,
    )

    chart = (folder / "notes.chart").read_text(encoding="utf-8")
    assert "0 = N 0 0" in chart
    assert "192 = N 2 0" in chart
    assert "192 = N 66 0" in chart
