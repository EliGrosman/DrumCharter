from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from audiotochart.inference.checkpoint import (
    ModelBundle,
    PRO8_ARCHITECTURE,
    _build_model_for_architecture,
)

log = logging.getLogger(__name__)

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


class OnsetDecoderError(RuntimeError):
    """Raised when an onset decoder bundle cannot be used."""


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


@dataclass(slots=True)
class ChordDecoderBundle:
    model: object
    config: dict
    vocab: ChordVocabulary
    device: str
    source_dir: Path


def build_onset_feature_rows(
    activations: np.ndarray,
    onset_frames: Sequence[int],
    onset_classes: Sequence[int],
    *,
    thresholds: Sequence[float] | None = None,
) -> np.ndarray:
    """Build 18-column per-onset feature rows for decoder conditioning.

    Layout:
    - 0:8   per-class activations at the onset frame
    - 8:16  one-hot baseline-picked class for this onset event
    - 16    picked class score
    - 17    picked class threshold margin, or 0 if thresholds are absent
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


def load_chord_decoder_bundle(
    decoder_dir: Path,
    *,
    base_bundle: ModelBundle,
    device: str,
) -> ChordDecoderBundle:
    """Load a standalone chord decoder bundle.

    Supports checkpoints with a full ``model_state`` first, then falls back to
    ``decoder_state`` using the already-loaded base encoder.
    """

    import torch

    decoder_dir = Path(decoder_dir)
    if not decoder_dir.is_dir():
        raise OnsetDecoderError(f"Onset decoder directory not found: {decoder_dir}")

    cfg_path = decoder_dir / "config.json"
    if not cfg_path.is_file():
        raise OnsetDecoderError(f"Missing config.json in onset decoder directory: {decoder_dir}")
    config = json.loads(cfg_path.read_text(encoding="utf-8"))

    if config.get("use_structure"):
        raise OnsetDecoderError(
            "Onset decoder config has use_structure=true, which is not supported "
            "by AudioToChart v1 live transcription"
        )

    if "chord_masks" not in config:
        raise OnsetDecoderError(
            "Unsupported onset decoder bundle: only chord decoder configs with "
            "'chord_masks' are supported"
        )

    weights_path = decoder_dir / "best.pt"
    if not weights_path.is_file():
        raise OnsetDecoderError(f"Missing best.pt in onset decoder directory: {decoder_dir}")

    raw_masks = config["chord_masks"]
    if not isinstance(raw_masks, list) or not all(isinstance(m, int) for m in raw_masks):
        raise OnsetDecoderError("onset decoder config field 'chord_masks' must be a list of integers")
    vocab = ChordVocabulary(
        masks=tuple(int(mask) for mask in raw_masks),
        blocklist_policy=config.get("blocklist_policy", "none"),
    )
    configured_vocab_size = config.get("vocab_size")
    if configured_vocab_size is not None and int(configured_vocab_size) != vocab.vocab_size:
        raise OnsetDecoderError(
            "onset decoder config vocab_size "
            f"({configured_vocab_size}) does not match chord_masks ({vocab.vocab_size})"
        )

    model = _build_onset_conditioned_model(
        _build_decoder_encoder(),
        config=config,
        vocab_size=vocab.vocab_size,
    )

    ckpt = torch.load(str(weights_path), map_location=device, weights_only=True)
    load_error: Exception | None = None
    full_state = None
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        full_state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and _looks_like_full_model_state(ckpt):
        full_state = ckpt

    if full_state is not None:
        try:
            model.load_state_dict(full_state, strict=True)
        except Exception as exc:
            load_error = exc
        else:
            model.to(device)
            model.eval()
            log.info("Loaded chord onset decoder full model_state from %s", weights_path)
            return ChordDecoderBundle(
                model=model,
                config=config,
                vocab=vocab,
                device=device,
                source_dir=decoder_dir,
            )

    if isinstance(ckpt, dict) and "decoder_state" in ckpt:
        fallback_model = _build_onset_conditioned_model(
            base_bundle.model,
            config=config,
            vocab_size=vocab.vocab_size,
        )
        try:
            fallback_model.decoder.load_state_dict(ckpt["decoder_state"], strict=True)
        except Exception as exc:
            raise OnsetDecoderError(
                f"Failed to load onset decoder checkpoint {weights_path}: {exc}"
            ) from exc
        fallback_model.to(device)
        fallback_model.eval()
        log.info("Loaded chord onset decoder decoder_state from %s", weights_path)
        return ChordDecoderBundle(
            model=fallback_model,
            config=config,
            vocab=vocab,
            device=device,
            source_dir=decoder_dir,
        )

    if load_error is not None:
        raise OnsetDecoderError(
            f"Failed to load onset decoder full model_state from {weights_path}: {load_error}"
        ) from load_error
    raise OnsetDecoderError(
        f"Onset decoder checkpoint {weights_path} must contain model_state or decoder_state"
    )


def refine_chord_onsets(
    decoder_bundle: ChordDecoderBundle,
    onsets: list[tuple[float, int, float]],
    acts: np.ndarray,
    spec: np.ndarray,
    *,
    fps: float,
    thresholds: Sequence[float],
) -> list[tuple[float, int, float]]:
    """Apply the chord decoder and return refined ``(time, class, confidence)`` onsets."""

    if not onsets:
        return []

    T = spec.shape[0]
    if T == 0:
        return onsets

    onset_frames = [min(int(round(t * fps)), T - 1) for t, _c, _conf in onsets]
    onset_classes = [int(c) for _t, c, _conf in onsets]
    onset_features = build_onset_feature_rows(
        acts,
        onset_frames,
        onset_classes,
        thresholds=thresholds,
    )
    baseline_onsets = list(zip(onset_frames, onset_classes))

    refined = decode_chord_hybrid_onsets(
        decoder_bundle.model,
        baseline_onsets=baseline_onsets,
        onset_features=onset_features,
        spec=spec,
        device=decoder_bundle.device,
        window_frames=decoder_bundle.config.get("window_frames", 1000),
        stride_frames=decoder_bundle.config.get("stride_frames", 500),
        max_onsets=decoder_bundle.config.get("max_onsets", 256),
        vocab=decoder_bundle.vocab,
    )

    confidence_map = {
        (frame, cls): conf
        for frame, cls, (_time, _class_idx, conf) in zip(
            onset_frames,
            onset_classes,
            onsets,
        )
    }
    out: list[tuple[float, int, float]] = []
    for frame, cls in refined:
        confidence = confidence_map.get((frame, cls))
        if confidence is None:
            safe_frame = min(max(int(frame), 0), acts.shape[0] - 1)
            confidence = (
                float(acts[safe_frame, cls])
                if acts.shape[0] > 0 and 0 <= cls < acts.shape[1]
                else 0.5
            )
        out.append((float(frame) / fps, int(cls), float(confidence)))
    out.sort(key=lambda item: (item[0], item[1]))
    return out


def decode_chord_hybrid_onsets(
    model: object,
    baseline_onsets: list[tuple[int, int]],
    onset_features: np.ndarray,
    spec: np.ndarray,
    *,
    device: str,
    window_frames: int = 1000,
    stride_frames: int = 500,
    max_onsets: int = 256,
    vocab: ChordVocabulary | None = None,
) -> list[tuple[int, int]]:
    """Run a chord decoder over baseline onset timings and expand tokens to events."""

    if not baseline_onsets:
        return []

    import torch

    vocab = vocab or build_chord_vocabulary()
    torch_device = torch.device(device)
    t_frames = spec.shape[0]
    decoder_predictions: dict[int, tuple[int, int]] = {}

    for win_start in range(0, t_frames, stride_frames):
        win_end = min(win_start + window_frames, t_frames)
        if win_end - win_start < 100:
            break

        win_indices = [
            idx
            for idx, (frame, _class_idx) in enumerate(baseline_onsets)
            if win_start <= frame < win_end
        ]
        if not win_indices:
            continue

        group_frames, group_features = _group_window_onsets(
            baseline_onsets,
            onset_features,
            win_indices,
            win_start=win_start,
        )
        if not group_frames:
            continue

        chunk = spec[win_start:win_end]
        if chunk.shape[0] < window_frames:
            chunk = np.pad(
                chunk,
                ((0, window_frames - chunk.shape[0]), (0, 0), (0, 0)),
            )

        x = torch.from_numpy(chunk).float().unsqueeze(0).to(torch_device)
        enc = model.encode(x)

        pred_tokens = _greedy_decode_chords(
            model.decoder,
            enc,
            group_frames,
            device=torch_device,
            max_onsets=max_onsets,
            onset_features=torch.from_numpy(group_features.astype(np.float32))
            .unsqueeze(0)
            .to(torch_device),
        )

        win_center = win_start + window_frames // 2
        for local_idx, local_frame in enumerate(group_frames):
            if local_idx >= len(pred_tokens):
                break
            global_frame = win_start + int(local_frame)
            dist = abs(global_frame - win_center)
            prev = decoder_predictions.get(global_frame)
            if prev is None or dist < prev[1]:
                decoder_predictions[global_frame] = (int(pred_tokens[local_idx]), dist)

    hybrid_onsets: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for frame in sorted(decoder_predictions):
        token = decoder_predictions[frame][0]
        if token == CHORD_NULL:
            continue
        mask = vocab.mask_for_token(token)
        if mask is None:
            continue
        for class_idx in mask_to_classes(mask):
            key = (int(frame), int(class_idx))
            if key in seen:
                continue
            seen.add(key)
            hybrid_onsets.append(key)

    hybrid_onsets.sort()
    return hybrid_onsets


def _group_window_onsets(
    baseline_onsets: list[tuple[int, int]],
    onset_features: np.ndarray,
    win_indices: list[int],
    *,
    win_start: int,
) -> tuple[list[int], np.ndarray]:
    grouped: dict[int, dict[str, list]] = {}
    for global_idx in win_indices:
        frame, class_idx = baseline_onsets[global_idx]
        local_frame = int(frame) - win_start
        bucket = grouped.setdefault(local_frame, {"classes": [], "features": []})
        bucket["classes"].append(int(class_idx))
        if global_idx < len(onset_features):
            bucket["features"].append(onset_features[global_idx])

    frames: list[int] = []
    features: list[np.ndarray] = []
    for local_frame in sorted(grouped):
        classes = grouped[local_frame]["classes"]
        feature_rows = np.asarray(grouped[local_frame]["features"], dtype=np.float32)
        frames.append(local_frame)
        features.append(aggregate_chord_features(feature_rows, classes))

    return frames, np.asarray(features, dtype=np.float32)


def _greedy_decode_chords(
    decoder: object,
    encoder_features: object,
    onset_frames: list[int],
    *,
    device: object,
    max_onsets: int,
    onset_features: object | None = None,
) -> list[int]:
    import torch

    n_onsets = min(len(onset_frames), max_onsets)
    if n_onsets == 0:
        return []

    frames_t = torch.tensor([onset_frames[:n_onsets]], dtype=torch.long, device=device)
    tokens = [CHORD_BOS]
    for _step in range(n_onsets):
        tgt = torch.tensor([tokens], dtype=torch.long, device=device)
        cur_frames = frames_t[:, : len(tokens)]
        if cur_frames.shape[1] < len(tokens):
            cur_frames = torch.cat(
                [
                    cur_frames[:, :1].expand(-1, len(tokens) - cur_frames.shape[1]),
                    cur_frames,
                ],
                dim=1,
            )
        logits = decoder(
            encoder_features,
            cur_frames,
            tgt,
            onset_features=onset_features[:, : len(tokens)]
            if onset_features is not None
            else None,
        )
        tokens.append(int(logits[0, -1].argmax().item()))
    return tokens[1:]


def _looks_like_full_model_state(state: dict) -> bool:
    return any(isinstance(key, str) and key.startswith("encoder.") for key in state) and any(
        isinstance(key, str) and key.startswith("decoder.") for key in state
    )


def _build_decoder_encoder() -> object:
    return _build_model_for_architecture(PRO8_ARCHITECTURE, {"num_classes": NUM_CHORD_CLASSES})


def _build_onset_conditioned_model(
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
            if self.use_onset_features:
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
            _B, N = tgt_tokens.shape
            memory = self.encoder_proj(encoder_features)

            T_enc = encoder_features.shape[1]
            safe_frames = onset_frames.clamp(0, T_enc - 1)
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
                tgt_mask = _causal_bool_mask(N, tgt_tokens.device)

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
        dropout=0.0,
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
