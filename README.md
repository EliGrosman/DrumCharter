# AudioToChart

AudioToChart generates Clone Hero drum chart folders from any audio file.
For best quality, use the fine-tuned model backend with automatic drum separation.

## Quickstart

```bash
uv sync --extra ai
uv run audiotochart generate ./song.wav --song "Song Name" --artist "Artist" -o ./out
```

The default model directory is `models/finetuned`. This folder is gitignored
while model-weight release details are being finalized, so place a local model
bundle there before running the model backend.

An optional chord onset decoder can live at `models/onset_decoder`. That
directory is also gitignored and local-only for now. If it exists, the model
backend uses it automatically to refine, reject, or expand same-frame chord
events after baseline peak picking. Pass `--no-onset-decoder` to force baseline
model output.

Drum isolation via Demucs runs automatically for the `model` backend. No GPU is
required; Demucs works on CPU, just slower.

If Demucs is unavailable or too slow, skip it:
```bash
uv run audiotochart generate ./song.wav --song "Song Name" --artist "Artist" --no-separate-drums -o ./out
```

## Output

The output folder contains a song folder ready for Clone Hero:

```text
Artist - Song Name/
  notes.chart
  song.ini
  song.wav
```

Copy that generated song folder into your Clone Hero songs directory.

## Documentation

Detailed documentation is available under `docs/`:

- **[docs/architecture.md](docs/architecture.md)** — system architecture and code organization
- **[docs/model.md](docs/model.md)** — model architecture, two-phase training, loss functions, onset decoder, tom consistency
- **[docs/training.md](docs/training.md)** — full training pipeline: data preparation, frame-level training, chord decoder training, evaluation

## Backend Quick Reference

| Backend | Description | Separation default |
|---------|-------------|-------------------|
| `model` | Fine-tuned frame model from CloneHero-ChartGen | Yes (Demucs) |
| `adtof` | Stock ADTOF drum transcription | No |
| `fake` | Deterministic 16-bar rock backbeat (dev/testing) | No |

## Model Bundle

AudioToChart looks for the local fine-tuned model at:

```text
models/finetuned/
  config.json
  best.pt
  thresholds.json
  labels.json
```

`best.pt` may also be named `weights.pt`. `labels.json` is optional for
CloneHero-ChartGen `pro8` model directories because the eight labels are known:
`kick`, `snare`, `hihat`, `tom_yellow`, `ride`, `tom_blue`, `crash`,
`tom_green`.

Original `phase3_harmonix_b`-style run directories load without editing, but
for this standalone repo the standard location is `models/finetuned`.

## Optional Chord Onset Decoder

AudioToChart currently supports the CloneHero-ChartGen chord decoder path, such
as a `runs/chord_decoder_phase3b_physical_v1`-style bundle copied to:

```text
models/onset_decoder/
  config.json
  best.pt
```

The decoder is separate from model-weight release planning. Absence of
`models/onset_decoder` is normal and does not block generation. An explicit
`--onset-decoder-dir PATH` must contain a supported chord decoder bundle or the
command fails with a clear error. Structure-conditioned decoder configs
(`use_structure: true`) are not supported in this first standalone port.

## Development

```bash
uv sync
uv run pytest
```

## Advanced Usage

For production-quality charts:

```bash
uv run audiotochart generate ./song.wav \
    --song "Song Name" --artist "Artist Name" \
    --backend model --model-dir models/finetuned \
    --onset-decoder-dir models/onset_decoder \
    --device auto --quantize 1/16 --tom-consistency -o ./out
```

- `--device auto` uses CUDA when available, otherwise CPU.
- `--quantize` controls grid snap; `1/16` is a good default; `none` disables snapping.
- `--tom-consistency` attempts to fix improbable tom patterns.
- `--no-onset-decoder` skips the optional chord decoder even when configured.
- `--backend adtof` uses the stock ADTOF model (no fine-tuning, no separation needed).

### Fast Path (no AI dependencies)

```bash
uv run audiotochart generate ./song.wav --song "Test" --artist "Test" --backend fake -o ./out
```

The `fake` backend generates a deterministic rock backbeat without any AI
dependencies. Useful for testing Clone Hero output format with no install friction.

For development, you can drive the drum chart from a General MIDI drum file:

```bash
uv run audiotochart generate ./song.wav --from-midi ./drums.mid --song "Test Song" --artist "Test Artist" -o ./out
```

## Troubleshooting

- **Demucs not found (model backend):** Install with `uv sync --extra ai` or use `--no-separate-drums` to skip separation (quality will be lower).
- **CUDA unavailable:** Use `--device auto` (default) to fall back to CPU. Or omit `--device` entirely.
- **Slow CPU separation:** Demucs on CPU is slow. Either wait longer, use `--no-separate-drums`, or install CUDA drivers and `--device cuda`.
- **Empty chart:** The model may not have detected enough drum hits. Try lowering thresholds in the model directory, or use `--backend fake` to verify the output pipeline works.
- **Tempo detection failed:** Pass `--bpm 120` or another known BPM to bypass automatic beat detection.

## Training

AudioToChart includes a full training pipeline for fine-tuning the ADTOF Frame RNN and training the chord onset decoder. See **[docs/training.md](docs/training.md)** for detailed instructions.

## Training

AudioToChart includes a full training pipeline for fine-tuning the ADTOF Frame RNN and training the chord onset decoder. See **[docs/training.md](docs/training.md)** for detailed instructions.

## Not Included

This standalone repo currently excludes training, dataset prep, evaluation
dashboards, and non-chord or structure-conditioned onset decoder variants.
