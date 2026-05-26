"""Shared constants and utilities for chord onset decoding.

This module defines the drum-class vocabulary, chord mask encoding, physical
validity constraints, and feature-aggregation helpers used by both the
inference and training onset decoders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

ONSET_FEATURE_DIM = 18

NUM_CHORD_CLASSES = 8
CHORD_PAD = 0
CHORD_BOS = 1
CHORD_NULL = 2
CHORD_FIRST = 3

KICK_CLASS = 0
SNARE_CLASS = 1
HIHAT_CLASS = 2
Y_TOM_CLASS = 3
RIDE_CLASS = 4
B_TOM_CLASS = 5
CRASH_CLASS = 6
F_TOM_CLASS = 7

CLASS_SYMBOLS = ("K", "R", "Yc", "Y", "Bc", "B", "Gc", "G")
HAND_CLASSES = frozenset(
    {
        SNARE_CLASS,
        HIHAT_CLASS,
        Y_TOM_CLASS,
        RIDE_CLASS,
        B_TOM_CLASS,
        CRASH_CLASS,
        F_TOM_CLASS,
    }
)
CYMBAL_CLASSES = frozenset({HIHAT_CLASS, RIDE_CLASS, CRASH_CLASS})
LANE_BY_CLASS = {
    SNARE_CLASS: "R",
    HIHAT_CLASS: "Y",
    Y_TOM_CLASS: "Y",
    RIDE_CLASS: "B",
    B_TOM_CLASS: "B",
    CRASH_CLASS: "G",
    F_TOM_CLASS: "G",
}


@dataclass(frozen=True, slots=True)
class ChordVocabulary:
    """Mapping between chord masks and decoder token IDs.

    A chord mask is a bitmask over the 8 drum classes. This class provides
    bidirectional conversion between masks, token indices, and human-readable
    names, and exposes the blocklist policy used to filter physically invalid
    chords.

    Attributes:
        masks: Immutable tuple of valid chord masks (bitmasks).
        blocklist_policy: The blocklist policy used to filter this vocabulary.

    Example:
        >>> vocab = build_chord_vocabulary()
        >>> vocab.mask_to_token[3]  # Kick + Snare mask -> token ID
        3
    """

    masks: tuple[int, ...]
    blocklist_policy: str = "none"

    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary including PAD, BOS, and NULL tokens."""
        return CHORD_FIRST + len(self.masks)

    @property
    def mask_to_token(self) -> dict[int, int]:
        """Map from chord mask (int) to decoder token index."""
        return {mask: CHORD_FIRST + idx for idx, mask in enumerate(self.masks)}

    @property
    def token_to_mask(self) -> dict[int, int]:
        """Map from decoder token index to chord mask (int)."""
        return {CHORD_FIRST + idx: mask for idx, mask in enumerate(self.masks)}

    @property
    def token_names(self) -> list[str]:
        """Human-readable names for every token, starting with PAD, BOS, NULL."""
        return ["PAD", "BOS", "NULL"] + [mask_to_name(mask) for mask in self.masks]

    def token_for_mask(self, mask: int) -> int | None:
        """Return the token index for a chord mask, or None if invalid.

        Args:
            mask: The chord bitmask to look up.

        Returns:
            The token index, or ``None`` if the mask is not in this vocabulary.
        """
        return self.mask_to_token.get(int(mask))

    def mask_for_token(self, token: int) -> int | None:
        """Return the chord mask for a token index, or None if invalid.

        Args:
            token: The decoder token index to look up.

        Returns:
            The chord bitmask, or ``None`` if the token is not in this vocabulary.
        """
        return self.token_to_mask.get(int(token))


def build_onset_feature_rows(
    activations: np.ndarray,
    onset_frames: Sequence[int],
    onset_classes: Sequence[int],
    *,
    thresholds: Sequence[float] | None = None,
) -> np.ndarray:
    """Build 18-column per-onset feature rows for decoder conditioning.

    For each onset, the function produces a row containing:
    - Columns 0-7: chord activation scores from the frame-level model.
    - Columns 8-15: one-hot encoding of the onset drum class.
    - Column 16: the activation score for the onset class.
    - Column 17: the activation score minus the per-class threshold.

    Args:
        activations: Frame-level activation array of shape ``(T, num_classes)``.
        onset_frames: Frame indices where onsets occur.
        onset_classes: Drum class indices for each onset onset.
        thresholds: Per-class threshold values used for column 17.

    Returns:
        An array of shape ``(n_onsets, 18)`` with the computed features.
    """

    n_onsets = min(len(onset_frames), len(onset_classes))
    out = np.zeros((n_onsets, ONSET_FEATURE_DIM), dtype=np.float32)
    if n_onsets == 0:
        return out

    num_classes = activations.shape[1] if activations.ndim == 2 else 0
    for idx, (frame, class_idx) in enumerate(
        zip(onset_frames[:n_onsets], onset_classes[:n_onsets])
    ):
        safe_frame = (
            int(np.clip(int(frame), 0, max(0, activations.shape[0] - 1)))
            if activations.size
            else 0
        )
        if activations.ndim == 2 and activations.shape[0] > 0:
            row = activations[safe_frame]
            take = min(NUM_CHORD_CLASSES, row.shape[0])
            out[idx, :take] = row[:take]

        c = int(class_idx)
        if 0 <= c < NUM_CHORD_CLASSES:
            out[idx, NUM_CHORD_CLASSES + c] = 1.0
            if activations.ndim == 2 and activations.shape[0] > 0 and c < num_classes:
                score = float(activations[safe_frame, c])
                out[idx, 16] = score
                threshold = 0.0
                if thresholds is not None and c < len(thresholds):
                    threshold = float(thresholds[c])
                out[idx, 17] = score - threshold
        elif activations.ndim == 2 and activations.shape[0] > 0:
            out[idx, 16] = float(np.max(activations[safe_frame]))
    return out


def fallback_onset_feature_rows(onset_classes: Sequence[int]) -> np.ndarray:
    """Build feature rows with only the one-hot class encoding (no activations).

    Used as a fallback when no frame-level activations are available.

    Args:
        onset_classes: Drum class indices for each onset.

    Returns:
        An array of shape ``(n_onsets, 18)`` with only the one-hot class columns set.
    """


def classes_to_mask(classes: Iterable[int]) -> int:
    """Convert an iterable of drum class indices to a chord bitmask.

    Args:
        classes: Drum class indices to combine into a single mask.

    Returns:
        A bitmask with bits set for each class in *classes*.
    """
    mask = 0
    for class_idx in classes:
        c = int(class_idx)
        if 0 <= c < NUM_CHORD_CLASSES:
            mask |= 1 << c
    return mask


def mask_to_classes(mask: int) -> list[int]:
    """Decode a chord bitmask into a list of drum class indices.

    Args:
        mask: A chord bitmask.

    Returns:
        List of drum class indices represented by the mask.
    """
    return [idx for idx in range(NUM_CHORD_CLASSES) if int(mask) & (1 << idx)]


def mask_to_name(mask: int) -> str:
    """Convert a chord bitmask to a human-readable symbol string.

    Args:
        mask: A chord bitmask.

    Returns:
        A string like "KR" (kick+snare) or "YcB" (yellow cymbal+blue cymbal).
        Returns "NULL" for mask value 0.
    """
    mask = int(mask)
    if mask == 0:
        return "NULL"
    return "".join(
        CLASS_SYMBOLS[idx]
        for idx in range(NUM_CHORD_CLASSES)
        if mask & (1 << idx)
    )


def physical_invalid_reason(mask: int) -> str | None:
    """Check why a chord mask is physically impossible, or None if valid.

    A chord is invalid if:
    - Two hand classes share the same lane (e.g. snare + hi-hat both use yellow).
    - More than two hand classes are present simultaneously.

    Args:
        mask: A chord bitmask to validate.

    Returns:
        A string describing the reason ("lane_conflict", "three_hands"), or None.
    """
    hand_classes = [idx for idx in HAND_CLASSES if int(mask) & (1 << idx)]
    lanes = [LANE_BY_CLASS[idx] for idx in hand_classes]
    if len(lanes) != len(set(lanes)):
        return "lane_conflict"
    if len(hand_classes) > 2:
        return "three_hands"
    return None


def is_physically_valid(mask: int) -> bool:
    """Check if a chord mask represents a physically playable chord.

    Args:
        mask: A chord bitmask to validate.

    Returns:
        True if the mask is non-zero and has no physical impossibilities.
    """
    return int(mask) > 0 and physical_invalid_reason(mask) is None


def is_blocklisted(mask: int, policy: str = "none") -> bool:
    """Check if a chord mask is blocklisted under the given policy.

    Args:
        mask: A chord bitmask to check.
        policy: The blocklist policy name. Supported: "none",
            "kick_two_cymbals_no_snare".

    Returns:
        True if the mask is blocklisted under the given policy.

    Raises:
        ValueError: If an unknown policy name is provided.
    """
    policy = policy or "none"
    if policy == "none":
        return False
    if policy == "kick_two_cymbals_no_snare":
        has_kick = bool(int(mask) & (1 << KICK_CLASS))
        has_snare = bool(int(mask) & (1 << SNARE_CLASS))
        cymbal_count = sum(
            1 for class_idx in CYMBAL_CLASSES if int(mask) & (1 << class_idx)
        )
        return has_kick and cymbal_count >= 2 and not has_snare
    raise ValueError(f"Unknown chord blocklist policy: {policy}")


def build_chord_vocabulary(*, blocklist_policy: str = "none") -> ChordVocabulary:
    """Build a chord vocabulary from physically valid and unblocklisted masks.

    Generates all 2^8 - 1 non-zero masks, filters out those that are
    physically impossible (via :func:`is_physically_valid`) and those
    matching the blocklist policy.

    Args:
        blocklist_policy: The blocklist policy to apply. Default "none" (no filtering).

    Returns:
        A :class:`ChordVocabulary` containing all valid chord masks.
    """
    masks = tuple(
        mask
        for mask in range(1, 1 << NUM_CHORD_CLASSES)
        if is_physically_valid(mask) and not is_blocklisted(mask, blocklist_policy)
    )
    return ChordVocabulary(masks=masks, blocklist_policy=blocklist_policy)


DEFAULT_CHORD_VOCAB = build_chord_vocabulary()
"""Default vocabulary built with no blocklist policy."""

CHORD_VOCAB_SIZE = DEFAULT_CHORD_VOCAB.vocab_size
"""Total vocabulary size including PAD, BOS, NULL tokens."""


def row_to_mask(row: np.ndarray, *, threshold: float = 0.5) -> int:
    """Convert a single activation row to a chord mask using a threshold.

    Args:
        row: A 1-D activation array (at least NUM_CHORD_CLASSES elements).
        threshold: Minimum activation to count as an onset.

    Returns:
        A chord bitmask with bits set for classes exceeding the threshold.
    """
    return classes_to_mask(np.flatnonzero(row[:NUM_CHORD_CLASSES] > threshold))


def labels_to_chord_events(
    labels: np.ndarray,
    *,
    threshold: float = 0.5,
) -> list[tuple[int, int]]:
    """Convert a frame-level label tensor to a list of (frame, mask) events.

    Args:
        labels: A 2-D array of shape ``(T, NUM_CHORD_CLASSES)`` with activation
            values for each drum class at each frame.
        threshold: Minimum activation to produce a chord event.

    Returns:
        A list of ``(frame_index, chord_mask)`` tuples for frames where at
        least one class exceeds the threshold.

    Raises:
        ValueError: If *labels* does not have shape ``(T, NUM_CHORD_CLASSES)``.
    """
    if labels.ndim != 2 or labels.shape[1] != NUM_CHORD_CLASSES:
        raise ValueError(
            f"Expected labels shape [T, {NUM_CHORD_CLASSES}], got {labels.shape}"
        )

    events: list[tuple[int, int]] = []
    for frame in np.where(labels.any(axis=1))[0]:
        mask = row_to_mask(labels[int(frame)], threshold=threshold)
        if mask:
            events.append((int(frame), mask))
    return events


def aggregate_chord_features(
    feature_rows: np.ndarray,
    classes: Sequence[int],
) -> np.ndarray:
    """Aggregate same-frame event feature rows into one chord feature row.

    Takes multiple onset feature rows that share the same frame and combines
    them by taking the maximum activation for each column, then setting the
    one-hot class columns for all classes in *classes*.

    Args:
        feature_rows: An array of per-onset feature rows (shape ``(N, 18)``
            or ``(1, 18)``).
        classes: Drum class indices whose one-hot columns to set.

    Returns:
        A single 1-D feature array of length 18.
    """

    out = np.zeros(ONSET_FEATURE_DIM, dtype=np.float32)
    if feature_rows.size:
        rows = np.asarray(feature_rows, dtype=np.float32)
        if rows.ndim == 1:
            rows = rows[np.newaxis, :]
        take = min(NUM_CHORD_CLASSES, rows.shape[1])
        out[:take] = np.max(rows[:, :take], axis=0)
        if rows.shape[1] > 16:
            out[16] = float(np.max(rows[:, 16]))
        if rows.shape[1] > 17:
            out[17] = float(np.max(rows[:, 17]))

    for class_idx in classes:
        c = int(class_idx)
        if 0 <= c < NUM_CHORD_CLASSES:
            out[NUM_CHORD_CLASSES + c] = 1.0
    return out


def build_onset_conditioned_model(
    encoder: object,
    *,
    config: dict,
    vocab_size: int,
) -> object:
    """Build a chord onset decoder model conditioned on encoder features.

    Constructs a two-part model:
    1. An ``OnsetConditionedDecoder`` — a transformer-decoder that takes
       encoder features, onset frame indices, and optional onset features as
       input and outputs chord token logits.
    2. An ``OnsetConditionedModel`` — wraps the external *encoder* and the
       decoder, freezing encoder parameters during training.

    Args:
        encoder: The pre-trained frame-level encoder model (frozen during
            decoding). Its output dimension should match *config*["encoder_dim"].
        config: Configuration dict with keys:
            - ``d_model`` (int): Transformer hidden dimension. Default 128.
            - ``n_heads`` (int): Number of attention heads. Default 4.
            - ``n_layers`` (int): Number of decoder layers. Default 4.
            - ``d_ff`` (int): Feed-forward hidden dimension. Default 512.
            - ``max_frames`` (int): Maximum positional encoding frames. Default 1024.
            - ``encoder_dim`` (int): Encoder output dimension. Default 120.
            - ``dropout`` (float): Dropout rate. Default 0.0.
            - ``use_onset_features`` (bool): Whether to include onset features.
            - ``onset_feature_dim`` (int): Onset feature dimension. Default 18.
        vocab_size: Size of the chord token vocabulary.

    Returns:
        An ``OnsetConditionedModel`` instance ready for training or inference.
    """
    import torch
    import torch.nn as nn

    class OnsetConditionedDecoder(nn.Module):
        """Transformer decoder for chord onset prediction.

        Takes encoder features and onset frame indices as conditioning, then
        autoregressively predicts chord tokens using a causal transformer
        decoder with optional onset feature injection.
        """

        def __init__(
            self,
            *,
            vocab_size: int,
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 4,
            d_ff: int = 512,
            max_frames: int = 1024,
            encoder_dim: int = 120,
            dropout: float = 0.0,
            use_onset_features: bool = False,
            onset_feature_dim: int = ONSET_FEATURE_DIM,
        ) -> None:
            super().__init__()
            self.d_model = d_model
            self.vocab_size = vocab_size
            self.use_structure = False
            self.use_onset_features = use_onset_features
            self.onset_feature_dim = onset_feature_dim

            self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.frame_pos_emb = nn.Embedding(max_frames, d_model)
            self.onset_proj = nn.Linear(encoder_dim, d_model)

            if use_onset_features:
                self.onset_feature_proj = nn.Sequential(
                    nn.Linear(onset_feature_dim, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, d_model),
                )
                self.onset_feature_norm = nn.LayerNorm(d_model)
            else:
                self.onset_feature_proj = None
                self.onset_feature_norm = None

            self.encoder_proj = nn.Linear(encoder_dim, d_model)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=n_layers,
            )
            self.output_proj = nn.Linear(d_model, vocab_size)
            self._init_weights()

        def _init_weights(self) -> None:
            """Initialize all module weights using standard PyTorch init."""
            nn.init.normal_(self.token_emb.weight, std=0.02)
            with torch.no_grad():
                self.token_emb.weight[0].zero_()
            nn.init.normal_(self.frame_pos_emb.weight, std=0.02)
            nn.init.xavier_uniform_(self.onset_proj.weight)
            nn.init.zeros_(self.onset_proj.bias)
            if self.use_onset_features and self.onset_feature_proj is not None:
                for module in self.onset_feature_proj:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        nn.init.zeros_(module.bias)
            nn.init.xavier_uniform_(self.encoder_proj.weight)
            nn.init.zeros_(self.encoder_proj.bias)
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

        def forward(
            self,
            encoder_features,
            onset_frames,
            tgt_tokens,
            tgt_mask=None,
            tgt_key_padding_mask=None,
            onset_features=None,
        ):
            _batch, n_tokens = tgt_tokens.shape
            memory = self.encoder_proj(encoder_features)

            t_enc = encoder_features.shape[1]
            safe_frames = onset_frames.clamp(0, t_enc - 1)
            idx = safe_frames.unsqueeze(-1).expand(-1, -1, encoder_features.shape[2])
            onset_enc = encoder_features.gather(1, idx)
            onset_repr = self.onset_proj(onset_enc)

            safe_pos = onset_frames.clamp(0, self.frame_pos_emb.num_embeddings - 1)
            frame_pos = self.frame_pos_emb(safe_pos)
            tok = self.token_emb(tgt_tokens)
            x = tok + onset_repr + frame_pos

            if self.use_onset_features:
                if onset_features is None:
                    raise ValueError(
                        "Decoder built with use_onset_features=True but "
                        "onset_features=None was passed."
                    )
                feat_proj = self.onset_feature_proj(onset_features)
                x = x + self.onset_feature_norm(feat_proj)

            if tgt_mask is None:
                tgt_mask = _causal_bool_mask(n_tokens, tgt_tokens.device)

            out = self.transformer_decoder(
                tgt=x,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            return self.output_proj(out)

    class OnsetConditionedModel(nn.Module):
        """Full onset-conditioned model wrapping an encoder and decoder.

        The encoder is frozen (no gradient updates) and used only to extract
        features. The decoder is trained to predict chord tokens autoregressively.
        """

        def __init__(self, encoder: object, decoder: OnsetConditionedDecoder) -> None:
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            for param in self.encoder.parameters():
                param.requires_grad = False

        def encode(self, spec):
            """Extract encoder features from a spectrogram (no gradient).

            Args:
                spec: Input spectrogram tensor.

            Returns:
                Encoder feature tensor.
            """
            with torch.no_grad():
                return _forward_encoder_features(self.encoder, spec)

        def forward(
            self,
            spec,
            onset_frames,
            onset_features,
            tgt_tokens,
            tgt_key_padding_mask=None,
        ):
            """Forward pass through the full onset-conditioned model.

            Args:
                spec: Input spectrogram tensor.
                onset_frames: Frame indices of onsets.
                onset_features: Per-onset feature rows.
                tgt_tokens: Target chord token sequence.
                tgt_key_padding_mask: Optional mask for padded tokens.

            Returns:
                Chord token logits from the decoder.
            """
            enc_features = self.encode(spec)
            return self.decoder(
                enc_features,
                onset_frames,
                tgt_tokens,
                tgt_key_padding_mask=tgt_key_padding_mask,
                onset_features=onset_features,
            )

    decoder = OnsetConditionedDecoder(
        vocab_size=vocab_size,
        d_model=config.get("d_model", 128),
        n_heads=config.get("n_heads", 4),
        n_layers=config.get("n_layers", 4),
        d_ff=config.get("d_ff", 512),
        max_frames=config.get("max_frames", 1024),
        encoder_dim=config.get("encoder_dim", 120),
        dropout=config.get("dropout", 0.0),
        use_onset_features=config.get("use_onset_features", False),
        onset_feature_dim=config.get("onset_feature_dim", ONSET_FEATURE_DIM),
    )
    return OnsetConditionedModel(encoder, decoder)


def _causal_bool_mask(length: int, device: object) -> object:
    """Create a causal (upper-triangular) boolean attention mask.

    Args:
        length: The sequence length of the mask.
        device: The torch device to place the mask on.

    Returns:
        A ``(length, length)`` boolean tensor with ``True`` above the diagonal.
    """
    import torch

    return torch.triu(
        torch.ones((length, length), dtype=torch.bool, device=device),
        diagonal=1,
    )


def _forward_encoder_features(model: object, x: object) -> object:
    """Extract features from the encoder without computing gradients.

    Args:
        model: The encoder model to run.
        x: Input tensor.

    Returns:
        Encoder feature tensor.
    """
    from drumcharter.adtof_model import forward_adtof_features

    return forward_adtof_features(model, x)
