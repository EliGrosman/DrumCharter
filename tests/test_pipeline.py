"""Pipeline regression tests for the first self-contained generator."""

from __future__ import annotations

from pathlib import Path

import audiotochart.pipeline as pipeline


def test_generate_drum_chart_folder_writes_clone_hero_song_folder(tmp_path: Path) -> None:
    source_audio = tmp_path / "song.wav"
    source_audio.write_bytes(b"RIFF")

    folder = pipeline.generate_drum_chart_folder(
        source_audio=source_audio,
        output_parent=tmp_path / "out",
        song_name="Song",
        artist_name="Artist",
        bpm=128.0,
    )

    assert folder == tmp_path / "out" / "Artist - Song"
    assert (folder / "song.wav").read_bytes() == b"RIFF"
    assert (folder / "song.ini").read_text(encoding="utf-8") == (
        "[Song]\n"
        "name = Song\n"
        "artist = Artist\n"
        "charter = AudioToChart (AI)\n"
        "diff_drums = 4\n"
    )

    chart = (folder / "notes.chart").read_text(encoding="utf-8")
    assert 'Name = "Song"' in chart
    assert 'Artist = "Artist"' in chart
    assert 'MusicStream = "song.wav"' in chart
    assert "0 = B 128000" in chart
    assert "[ExpertDrums]" in chart
    assert "0 = N 0 0" in chart
    assert "0 = N 2 0" in chart
    assert "0 = N 66 0" in chart
