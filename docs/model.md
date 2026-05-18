# Model Architecture

AudioToChart uses a two-stage modelling approach: a **frame-level CNN-RNN** produces per-frame activations, and an optional **chord onset decoder** refines these into clean chord predictions.

## Frame-Level Model: ADTOF Frame RNN

Built and wrapped in `adtof_model.py` using the `adtof_pytorch` library.

### Architecture

1. **CNN backbone** — 6 convolutional blocks that process mel-spectrogram frames as a 2D image (time x frequency). Each block doubles channels while downsampling frequency. The final CNN output is permuted and flattened per time step.
2. **Context layer** — optional 1D convolution that mixes adjacent time frames.
3. **GRU layers** — 3 bidirectional GRU layers with 256 hidden units each.
4. **Output head** — a linear layer projecting GRU outputs to 8 drum classes, replacing the original ADTOF output layer.

The default ADTOF weights are loaded and the CNN can be frozen during fine-tuning to preserve general feature representations.

### Input

Mel-spectrograms computed with:
- 1024 FFT window, 512 hop length
- 128 mel bands
- Sample rate: 22050 Hz
- Normalised to zero mean, unit variance

### Output

8-class activations per frame (100 FPS):
0. Kick
1. Snare
2. Hi-hat
3. Yellow tom
4. Ride
5. Blue tom
6. Crash
7. Green tom (floor tom)

## Two-Phase Training

Training is done in two phases (see `train.py` and `config.py`):

**Phase A — Warmup** (default 5 epochs):
- All layers trainable
- AdamW, learning rate 1e-3
- Cosine annealing to 1e-4
- BCE loss with class weighting

**Phase B — Fine-tuning** (default 30 epochs):
- CNN backbone frozen (`freeze_adtof_cnn`)
- RNN layers and output head trainable
- 3-tier learning rates: RNN 1e-4, CNN (early) 5e-5, CNN (late) 1e-5, head 1e-3
- Cosine annealing to 0
- Early stopping with patience 8

### Loss Functions

Defined in `losses.py`:

- **BCE** (`rhythm_game_bce`) — standard binary cross-entropy with optional Gaussian temporal smoothing of targets to allow timing tolerance.
- **Focal loss** (`rhythm_game_focal`) — down-weights easy examples to focus on hard-to-classify frames.
- **Dice loss** — optimises overlap between predictions and targets; useful for class imbalance.
- **Combined** — `rhythm_game_focal_dice` combines focal BCE + dice + L2 penalty.

## Onset Decoder

The chord onset decoder (used in both `inference/onset_decoder.py` and `training/onset_decoder.py`) is a transformer-based model that predicts chord tokens conditioned on candidate onsets from the frame model.

### Architecture

- **Encoder** — the frame-level ADTOF RNN (shared/frozen) produces per-frame features.
- **Transformer decoder** — lightweight decoder with configurable layers (default 4), attention heads (4), d_model (128), d_ff (512).
- **Output** — classification over `NUM_CHORD_CLASSES` chord tokens (including BOS, EOS, and NULL/reject tokens for unmatched onsets).

### Token Vocabulary

Each chord token encodes the set of simultaneously-played drum classes at a single onset. The vocabulary includes all legal drum class combinations, plus special tokens:

- BOS (beginning of sequence)
- EOS (end of sequence)
- NULL — used during training for unmatched frame-model onsets that don't correspond to any ground-truth hit.

### Greedy Decoding

During inference (`_greedy_decode_chords`), the decoder starts from the BOS token and iteratively predicts the next chord token, using the predicted onset positions as positional conditioning.

## Tom Consistency Post-Processing

`tom_consistency.py` applies a song-level correction to tom drum predictions:

1. **Anchor identification** — tom hits with activation >= 0.7 are anchors.
2. **Convention detection** — anchor classes comprising >= 15% of all anchors form the song's tom convention.
3. **Reassignment** — low-confidence (activation < 0.5) non-convention tom hits outside of fill regions are reassigned to the convention's highest-activation class at that frame.
4. **Fills** — rapid sequences of >= 4 consecutive tom hits with < 0.18 s spacing are preserved as-is.

This prevents the model from inconsistently flipping tom assignments mid-song (e.g. high tom vs. floor tom on repeated patterns).

## Checkpoint Format

Model checkpoints are stored as directories containing:
- `config.json` — model configuration and training hyperparameters
- `best.pt` or `weights.pt` — PyTorch state dict
- `labels.json` — class label strings
- `thresholds.json` (optional) — per-class activation thresholds from optimisation

The `ModelBundle` dataclass (`checkpoint.py`) loads and validates these files, returning a ready-to-use model on the appropriate device.