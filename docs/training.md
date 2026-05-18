# Training

This document covers the full model training pipeline for AudioToChart, from data discovery through model evaluation.

## Overview

The training system (`src/audiotochart/training/`) supports two complementary training regimes:

1. **Frame-level training** — trains the ADTOF Frame RNN to produce per-frame activations for 8 drum classes.
2. **Chord onset decoder training** — trains a transformer-based decoder to predict chord tokens from frame-model candidate onsets.

Both use data derived from Rock Band song archives (`.7z` files containing MIDI charts and audio stems).

## Prerequisites

```bash
uv sync --extra ai --extra training
```

## Data Preparation

### Pipeline

The data preparation pipeline (`pipeline.py`) discovers, extracts, and preprocesses Rock Band songs:

```
discovery -> extraction -> spectrogram computation -> label generation -> caching
```

### Discovery (`discovery.py`)

Scans directory trees for `notes.mid` files and associates each with its drum audio stems (`drums.ogg` or `drums_*.ogg`). Returns a list of `RBSong` objects.

### Extraction (`extract.py`)

Finds `.7z` archives in source directories and selectively extracts:
- `notes.mid` — the drum MIDI chart
- `drums.ogg` — drum audio stem
- `song.ogg` — full song mix (optional)

Extracted files are cached under `~/.cache/audiotochart/training/` with content-hash-based subdirectories to avoid redundant extraction.

### Spectrogram Computation (`spectrogram.py`)

Computes mel-spectrograms from drum audio:
- Sample rate: 22050 Hz
- FFT window: 1024, hop: 512
- Mel bands: 128
- Output is cached as `.npy` files keyed by audio content hash.

### MIDI Parsing (`rb_midi.py`)

Parses Rock Band `notes.mid` files to extract drum onsets. Supports the 8-class pro drum mapping:
- Note 0: kick (default), note 110: yellow-tom marker
- Note 1: snare
- Note 2: yellow pad / hi-hat (cymbal if marker 110 present)
- Note 3: blue pad / ride (cymbal if marker 111 present)
- Note 4: green pad / crash (cymbal if marker 112 present)

Returns a list of `DrumOnset` objects with timing and class information.

### Label Generation (`labels.py`)

Converts `DrumOnset` lists into binary frame-level arrays of shape `(num_frames, 8)` at 100 FPS. Each frame marks which drum class(es) are active.

### Dataset Classes (`dataset.py`)

Provides PyTorch `Dataset` implementations:

- **DrumTranscriptionDataset** — for frame-level training. Loads spectrogram windows and corresponding label matrices. Supports:
  - Random window sampling with configurable stride overlap
  - Optional skip of harmonix-only songs
  - Song-level indexing for evaluation
- **ChordConditionedDataset** — for onset decoder training. Loads spectrogram windows, frame-model features, candidate onset positions, and target chord token sequences.

`create_datasets()` and `create_splits()` split songs into train/val/test using a random 80/10/10 split seeded for reproducibility.

## Frame-Level Training

### CLI

```bash
audiotochart train prepare <data_dirs...> --cache-dir ~/.cache/audiotochart/training
audiotochart train frame --cache-dir ... --output-dir ./runs/frame_run [options]
```

### Training Loop (`train.py`)

The `run_training()` function implements the two-phase workflow:

**Phase A — Warmup:**
- All layers trainable
- AdamW, cosine LR schedule
- BCE loss with class weighting

**Phase B — Fine-tuning:**
- CNN frozen, RNN layers and head trainable
- 3-tier learning rates (RNN, CNN_early, CNN_late, head)
- Early stopping based on validation loss

### Configuration (`config.py`)

The `TrainConfig` dataclass exposes all hyperparameters:
- `window_frames` / `stride_frames`: training window dimensions
- `batch_size`, `num_workers`
- `warmup_epochs`, `finetune_epochs`, `patience`
- Learning rates for each phase and parameter group
- `harmonix_only`: restrict to Harmonix-charted songs (exclude RBN)
- `amp`: mixed-precision training
- Loss function selection
- Class weights

### Evaluation (`evaluate.py`)

Performs song-level evaluation:
1. Runs the model over full songs (chunked to avoid OOM)
2. Applies peak-picking (from `thresholds.py`)
3. Computes F-measure at multiple timing tolerances (25ms, 50ms, 75ms, 100ms)
4. Generates per-class and macro-averaged metrics
5. Outputs JSON reports and a human-readable table

### Threshold Optimisation (`thresholds.py`)

`optimize_thresholds()` finds per-class activation thresholds that maximise F-measure on validation data, using Nelder-Mead search over the [0, 1] range. These thresholds are saved to `thresholds.json` in the model bundle.

## Chord Onset Decoder Training

### Precomputation

```bash
audiotochart train precompute-onsets --cache-dir ... --frame-model <dir> --output-dir ./onsets
```

Runs the trained frame model over training audio to extract candidate onset positions, frame-level activations, and encoder features. Results are saved as `.npz` files.

### Training

```bash
audiotochart train onset-decoder --cache-dir ... --frame-model <dir> --output-dir ./od_run [options]
```

The chord decoder (`training/onset_decoder.py`) is a transformer trained to predict chord token sequences from onset-conditioned encoder features. Key hyperparameters:
- `d_model`, `n_heads`, `n_layers`, `d_ff`: transformer dimensions
- `max_onsets`: max candidate onsets per window
- `tp_only`: only train on matched onset groups
- `use_null_token`: include NULL token for unmatched candidates
- `blocklist_policy`: filter illegal chord combinations
- `selection_metric`: checkpoint selection (val_loss, hybrid_macro_f, hybrid_cqs)

### Hybrid Evaluation (`chord_hybrid_eval.py`)

During training, the hybrid evaluation runs the full inference pipeline (frame model + decoder) on a subset of validation songs at regular intervals, computing Chart Quality Score (CQS) for checkpoint selection.

## Chart Quality Score (`cqs.py`)

CQS is a precision-oriented metric that measures transcription quality along three axes:

- **Coverage**: proportion of ground-truth chart notes that are matched by predicted hits.
- **Identity**: whether each matched note has the correct drum class.
- **Precision**: proportion of predicted hits that correspond to real ground-truth notes.

CQS correlates well with human perception of chart quality and is used for model selection in chord decoder training.
