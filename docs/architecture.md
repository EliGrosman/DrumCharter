# Architecture

DrumCharter converts drum audio into Clone Hero `.chart` files. The system is organised into four layers: **pipeline**, **inference**, **chart formatting**, and **training**.

## Pipeline Layer

The entry point is `drumcharter generate`, defined in `cli.py`. It resolves the selected backend and calls into `pipeline.py`, which orchestrates:

1. **Audio loading** (`audio.py`) — reads duration via librosa or WAV header.
2. **Tempo detection** (`tempo.py`) — librosa-based beat tracking; supports manual BPM override.
3. **Drum separation** (`separation.py`, optional) — Demucs `htdemucs` model extracts the drum stem.
4. **Transcription** — delegates to the chosen inference backend.
5. **Chart document generation** (`chart/convert.py`) — maps hits to chart notes, applies quantization and simultaneous-hit limiting, emits a `ChartDocument`.
6. **Output writing** — renders `notes.chart`, `song.ini`, and copies audio into a Clone Hero song folder.

Progress is reported through an optional callback so the CLI can show a Rich status spinner.

## Inference Layer

Defined in `inference/`. The `DrumTranscriber` protocol (`base.py`) is satisfied by three backends:

- **FakeTranscriber** (`fake.py`) — developer-only; generates a deterministic rock beat.
- **AdtofTranscriber** (`adtof.py`) — wraps `adtof_pytorch` to produce a MIDI file, then reads back hits via `chart/midi.py`.
- **ModelTranscriber** (`model.py`) — loads a finetuned ADTOF Frame RNN plus optional chord onset decoder.

### Model Backend (model.py)

1. Loads a `ModelBundle` via `checkpoint.py` (config, weights, labels, thresholds).
2. Computes logits over the full audio.
3. Applies threshold-based peak-picking to produce candidate onsets.
4. If an onset decoder directory is configured, runs the chord decoder (`onset_decoder.py`) which uses a transformer conditioned on onset features to predict chord tokens.
5. Applies tom consistency (`tom_consistency.py`) to reassign low-confidence tom hits to the song's dominant convention.

The ADTOF model is builtin `adtof_model.py` using `adtof_pytorch`'s Frame RNN with a custom output head.

## Chart Formatting Layer

Defined in `chart/`. All conversions flow through `hits_to_chart_document` in `convert.py`:

- **Instrument mapping** — each `DrumHit` instrument (e.g. `"kick"`, `"snare"`) is mapped to a chart note + optional cymbal modifier via `drum_vocab.py`.
- **Tick conversion** — seconds are converted to chart ticks using either constant BPM (`seconds_to_tick`) or a variable-tempo beat map (`seconds_to_tick_tempo_map`).
- **Quantization** — hits are snapped to a musical grid (e.g. 16th notes) via `snap_hits_to_grid` in `postprocess.py`.
- **Simultaneous hit limiting** — `limit_simultaneous_hits` enforces the two-hand-lane limit.
- **Difficulty generation** — `difficulty.py` derives Hard, Medium, and Easy from Expert by thinning notes.
- **File writing** — `format.py` renders the `[Song]`, `[SyncTrack]`, `[Events]`, and difficulty sections into the `.chart` format.

MIDI input is supported through `midi.py`, which maps both General MIDI pitches and Clone Hero chart MIDI note numbers to project instruments.

## Training Layer

See [training.md](training.md) for full details. The training pipeline:

1. Discovers and extracts Rock Band song archives (`discovery.py`, `extract.py`).
2. Parses drum MIDI notes into instrument labels (`rb_midi.py`).
3. Computes mel-spectrograms (`spectrogram.py`spectrogram.py`) and binary label matrices (`labels.py`).
4. Trains a frame-level ADTOF model (`train.py`, `model.py`) with two-phase warmup/finetuning.
5. Optionally precomputes candidate onsets (`onset_precompute.py`) for chord-decoder training (`onset_decoder.py` in training/).

## Configuration

User settings are stored in `~/.config/drumcharter/config.json` and managed by `config.py`. The CLI supports both command-line flags and an interactive prompt mode.

## File Layout

```
src/drumcharter/
├── cli.py              # Click entry point
├── pipeline.py         # Chart generation orchestration
├── audio.py            / Audio I/O utilities
├── tempo.py            # Beat tracking
├── separation.py       # Demucs drum isolation
├── device.py           # PyTorch device resolution
├── download.py         # YouTube audio download
├── drums.py            # DrumHit dataclass
├── postprocess.py      # Quantisation, simultaneous-hit limiting
├── config.py           # User config manager
├── adtof_model.py      # ADTOF model builder
├── chart/              # Chart format and conversion
│   ├── convert.py      # hits -> ChartDocument
│   ├── difficulty.py   # Difficulty derivation
│   ├── format.py       # .chart file serialisation
│   ├── midi.py              # MIDI input
│   ├── songini.py      # song.ini output
│   └── drum_vocab.py   # Note/instrument mappings
├── inference/          # Transcription backends
│   ├── model.py        # Model-based transcriber
│   ├── checkpoint.py   # Model bundle loading
│   ├── onset_decoder.py # Chord onset decoder
│   ├── adtof.py        # ADTOF wrapper
│   ├── fake.py         # Fake/ test backend
│   ├── tom_consistency.py
│   └── base.py         # DrumTranscriber protocol
└── training/           # Model training pipeline
    ├── train.py        # Training loop
    ├── dataset.py      # PyTorch datasets
    ├── model.py        # Model construction
    ├── config.py       # TrainConfig
    ├── discovery.py    # Song discovery
    ├── extract.py      # Archive extraction
    ├── rb_midi.py      # MIDI parsing
    ├── spectrogram.py  # Spectrogram computation
    ├── labels.py       # Label matrix construction
    ├── losses.py       # Loss functions
    ├── evaluate.py     # Evaluation metrics
    ├── thresholds.py   # Peak-picking/optimisation
    ├── pipeline.py     # Data preparation pipeline
    ├── onset_decoder.py # Chord decoder training
    ├── onset_precompute.py # Onset precomputation
    └── chord_hybrid_eval.py # Hybrid eval
```