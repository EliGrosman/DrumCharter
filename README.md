# AudioToChart

AudioToChart is a learning-first CLI for generating Clone Hero drum chart
folders from local audio files.

The current version writes a valid song folder with a deterministic fake drum chart, 
optional MIDI-driven drum notes, tempo/beat-grid detection, post-processing, 
and generated lower difficulties. Audio inference backends have not been added yet.

## Usage

```bash
uv run audiotochart generate ./song.wav --song "Test Song" --artist "Test Artist" -o ./out
```

For development, you can drive the drum chart from a General MIDI drum file:

```bash
uv run audiotochart generate ./song.wav --from-midi ./drums.mid --song "Test Song" --artist "Test Artist" -o ./out
```

The output folder contains:

```text
Test Artist - Test Song/
  notes.chart
  song.ini
  song.wav
```

## Development

```bash
uv run pytest
```
