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
    """Mapping between chord masks and decoder token IDs."""

    masks: tuple[int, ...]
    blocklist_policy: str = "none"

    @property
    def vocab_size(self) -> int:
        return CHORD_FIRST + len(self.masks)

    @property
    def mask_to_token(self) -> dict[int, int]:
        return {mask: CHORD_FIRST + idx for idx, mask in enumerate(self.masks)}

    @property
    def token_to_mask(self) -> dict[int, int]:
        return {CHORD_FIRST + idx: mask for idx, mask in enumerate(self.masks)}

    @property
    def token_names(self) -> list[str]:
        return ["PAD", "BOS", "NULL"] + [mask_to_name(mask) for mask in self.masks]

    def token_for_mask(self, mask: int) -> int | None:
        return self.mask_to_token.get(int(mask))

    def mask_for_token(self, token: int) -> int | None:
        return self.token_to_mask.get(int(token))


def build_onset_feature_rows(
    activations: np.ndarray,
    onset_frames: Sequence[int],
    onset_classes: Sequence[int],
    *,
    thresholds: Sequence[float] | None = None,
) -> np.ndarray:
    """Build 18-column per-onset feature rows for decoder conditioning."""

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
    rows = np.zeros((len(onset_classes), ONSET_FEATURE_DIM), dtype=np.float32)
    for idx, class_idx in enumerate(onset_classes):
        c = int(class_idx)
        if 0 <= c < NUM_CHORD_CLASSES:
            rows[idx, NUM_CHORD_CLASSES + c] = 1.0
    return rows


def classes_to_mask(classes: Iterable[int]) -> int:
    mask = 0
    for class_idx in classes:
        c = int(class_idx)
        if 0 <= c < NUM_CHORD_CLASSES:
            mask |= 1 << c
    return mask


def mask_to_classes(mask: int) -> list[int]:
    return [idx for idx in range(NUM_CHORD_CLASSES) if int(mask) & (1 << idx)]


def mask_to_name(mask: int) -> str:
    mask = int(mask)
    if mask == 0:
        return "NULL"
    return "".join(
        CLASS_SYMBOLS[idx]
        for idx in range(NUM_CHORD_CLASSES)
        if mask & (1 << idx)
    )


def physical_invalid_reason(mask: int) -> str | None:
    hand_classes = [idx for idx in HAND_CLASSES if int(mask) & (1 << idx)]
    lanes = [LANE_BY_CLASS[idx] for idx in hand_classes]
    if len(lanes) != len(set(lanes)):
        return "lane_conflict"
    if len(hand_classes) > 2:
        return "three_hands"
    return None


def is_physically_valid(mask: int) -> bool:
    return int(mask) > 0 and physical_invalid_reason(mask) is None


def is_blocklisted(mask: int, policy: str = "none") -> bool:
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
    masks = tuple(
        mask
        for mask in range(1, 1 << NUM_CHORD_CLASSES)
        if is_physically_valid(mask) and not is_blocklisted(mask, blocklist_policy)
    )
    return ChordVocabulary(masks=masks, blocklist_policy=blocklist_policy)


DEFAULT_CHORD_VOCAB = build_chord_vocabulary()
CHORD_VOCAB_SIZE = DEFAULT_CHORD_VOCAB.vocab_size


def row_to_mask(row: np.ndarray, *, threshold: float = 0.5) -> int:
    return classes_to_mask(np.flatnonzero(row[:NUM_CHORD_CLASSES] > threshold))


def labels_to_chord_events(
    labels: np.ndarray,
    *,
    threshold: float = 0.5,
) -> list[tuple[int, int]]:
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
    """Aggregate same-frame event feature rows into one chord feature row."""

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
    import torch
    import torch.nn as nn

    class OnsetConditionedDecoder(nn.Module):
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
        def __init__(self, encoder: object, decoder: OnsetConditionedDecoder) -> None:
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            for param in self.encoder.parameters():
                param.requires_grad = False

        def encode(self, spec):
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
    import torch

    return torch.triu(
        torch.ones((length, length), dtype=torch.bool, device=device),
        diagonal=1,
    )


def _forward_encoder_features(model: object, x: object) -> object:
    from audiotochart.adtof_model import forward_adtof_features

    return forward_adtof_features(model, x)
