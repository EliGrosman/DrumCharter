# AudioToChart

AudioToChart is a learning-first CLI for generating Clone Hero drum chart
folders from local audio files.

The current version is the first milestone: it writes a valid song folder with
a deterministic fake Expert drum chart. This proves the packaging, CLI, chart
format, and folder output before adding audio inference.

## Usage

```bash
uv run audiotochart generate ./song.wav --song "Test Song" --artist "Test Artist" -o ./out
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