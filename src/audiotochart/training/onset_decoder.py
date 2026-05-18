from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from audiotochart.onset_decoder_common import (
    CHORD_NULL,
    CHORD_PAD,
    ONSET_FEATURE_DIM,
    build_chord_vocabulary,
    build_onset_conditioned_model,
)
from audiotochart.training.dataset import ChordConditionedDataset, _load_entries, create_splits
from audiotochart.training.model import count_parameters

log = logging.getLogger(__name__)


@dataclass
class ChordDecoderTrainConfig:
    cache_dir: Path
    frame_model_dir: Path
    output_dir: Path

    window_frames: int = 1000
    stride_frames: int = 500
    max_onsets: int = 256
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42
    harmonix_only: bool = False
    onset_dir: Path | None = None
    tp_only: bool = False
    use_null_token: bool = True
    blocklist_policy: str = "none"

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_frames: int = 1024
    encoder_dim: int = 120

    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    patience: int = 5
    label_smoothing: float = 0.1
    selection_metric: str = "hybrid_cqs"
    hybrid_eval_songs: int = 50
    hybrid_eval_every: int = 1

    device: str = "cuda"
    amp: bool = True


def _config_payload(cfg: ChordDecoderTrainConfig) -> dict:
    vocab = build_chord_vocabulary(blocklist_policy=cfg.blocklist_policy)
    payload = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(cfg).items()
        },
        "variant": "pro8",
        "vocab_size": vocab.vocab_size,
        "chord_masks": list(vocab.masks),
        "use_onset_features": True,
        "onset_feature_dim": ONSET_FEATURE_DIM,
        "use_structure": False,
    }
    return payload


def _collate_chord(batch):
    specs = torch.from_numpy(np.stack([b[0] for b in batch]))
    frames = torch.from_numpy(np.stack([b[1] for b in batch]))
    onset_features = torch.from_numpy(np.stack([b[2] for b in batch]))
    token_input = torch.from_numpy(np.stack([b[3] for b in batch]))
    token_target = torch.from_numpy(np.stack([b[4] for b in batch]))
    padding_mask = torch.from_numpy(np.stack([b[5] for b in batch]))
    return specs, frames, onset_features, token_input, token_target, padding_mask


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _selection_is_better(metric: str, current: float, best: float) -> bool:
    if metric == "val_loss":
        return current < best
    if metric in {"hybrid_macro_f", "hybrid_cqs"}:
        return current > best
    raise ValueError(f"Unsupported selection metric: {metric}")


@torch.no_grad()
def validate_chord_decoder(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0
    pred_null = 0

    for spec, frames, onset_features, token_input, token_target, padding_mask in val_loader:
        spec = spec.to(device, non_blocking=True)
        frames = frames.to(device, non_blocking=True)
        onset_features = onset_features.to(device, non_blocking=True)
        token_input = token_input.to(device, non_blocking=True)
        token_target = token_target.to(device, non_blocking=True)
        padding_mask = padding_mask.to(device, non_blocking=True)

        logits = model(
            spec,
            frames,
            onset_features,
            token_input,
            tgt_key_padding_mask=padding_mask,
        )
        non_pad = ~padding_mask
        non_pad_flat = non_pad.view(-1)
        if bool(non_pad_flat.any().item()):
            logits_valid = logits.float().view(-1, logits.size(-1))[non_pad_flat]
            targets_valid = token_target.view(-1)[non_pad_flat]
            loss = criterion(logits_valid, targets_valid)
            total_loss += float(loss.item())
            n_batches += 1

            preds = logits.argmax(dim=-1)
            correct += (preds[non_pad] == token_target[non_pad]).sum().item()
            pred_null += (preds[non_pad] == CHORD_NULL).sum().item()
            total += non_pad.sum().item()

    return (
        total_loss / max(1, n_batches),
        correct / max(1, total),
        pred_null / max(1, total),
    )


def run_onset_decoder_training(cfg: ChordDecoderTrainConfig) -> Path:
    from audiotochart.inference.checkpoint import load_model_bundle
    from audiotochart.training.chord_hybrid_eval import (
        evaluate_prepared_chord_hybrid,
        hybrid_selection_value,
        prepare_chord_hybrid_eval_songs,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.selection_metric not in {"val_loss", "hybrid_macro_f", "hybrid_cqs"}:
        raise ValueError(f"Unsupported selection metric: {cfg.selection_metric}")

    config_payload = _config_payload(cfg)
    (cfg.output_dir / "config.json").write_text(json.dumps(config_payload, indent=2))
    log_path = cfg.output_dir / "train_log.jsonl"
    best_path = cfg.output_dir / "best.pt"
    last_path = cfg.output_dir / "last.pt"

    _seed_everything(cfg.seed)
    torch.backends.cudnn.benchmark = True

    if cfg.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available; falling back to CPU")
        cfg.device = "cpu"
    device = torch.device(cfg.device)
    use_amp = cfg.amp and device.type == "cuda"

    entries = _load_entries(cfg.cache_dir, harmonix_only=cfg.harmonix_only)
    if not entries:
        raise ValueError(f"No entries found in cache: {cfg.cache_dir}")
    train_entries, val_entries, test_entries = create_splits(entries, seed=cfg.seed)

    train_ds = ChordConditionedDataset(
        train_entries,
        window_frames=cfg.window_frames,
        stride_frames=cfg.stride_frames,
        max_onsets=cfg.max_onsets,
        onset_dir=cfg.onset_dir,
        tp_only=cfg.tp_only,
        allow_null_token=cfg.use_null_token,
        blocklist_policy=cfg.blocklist_policy,
    )
    val_ds = ChordConditionedDataset(
        val_entries,
        window_frames=cfg.window_frames,
        stride_frames=cfg.stride_frames,
        max_onsets=cfg.max_onsets,
        onset_dir=cfg.onset_dir,
        tp_only=cfg.tp_only,
        allow_null_token=cfg.use_null_token,
        blocklist_policy=cfg.blocklist_policy,
    )
    if len(train_ds) == 0:
        raise ValueError("No training windows found for onset decoder")

    log.info(
        "Onset decoder datasets: train=%d windows (%d songs), val=%d windows (%d songs), test=%d songs",
        len(train_ds),
        len(train_entries),
        len(val_ds),
        len(val_entries),
        len(test_entries),
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=_collate_chord,
        pin_memory=pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=_collate_chord,
        pin_memory=pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )

    base_bundle = load_model_bundle(cfg.frame_model_dir, device=cfg.device)
    vocab = build_chord_vocabulary(blocklist_policy=cfg.blocklist_policy)
    model = build_onset_conditioned_model(
        base_bundle.model,
        config=config_payload,
        vocab_size=vocab.vocab_size,
    ).to(device)
    total_params, trainable_params = count_parameters(model)
    log.info("Onset decoder model: %d total params, %d trainable", total_params, trainable_params)

    prepared_hybrid_songs = []
    if cfg.selection_metric != "val_loss":
        if cfg.hybrid_eval_songs <= 0:
            raise ValueError(
                "Hybrid checkpoint selection requires hybrid_eval_songs > 0; "
                "use selection_metric='val_loss' to skip hybrid validation"
            )
        thresholds = base_bundle.config.get("thresholds")
        if thresholds is None:
            raise FileNotFoundError(
                "Hybrid checkpoint selection requires thresholds in the frame model "
                f"bundle. Expected thresholds.json next to {cfg.frame_model_dir}."
            )
        confidence_gates = base_bundle.config.get("confidence_gates")
        prepared_hybrid_songs = prepare_chord_hybrid_eval_songs(
            base_bundle.model,
            val_entries,
            thresholds=list(thresholds),
            confidence_gates=list(confidence_gates) if confidence_gates is not None else None,
            device=cfg.device,
            max_songs=cfg.hybrid_eval_songs,
        )
        if not prepared_hybrid_songs:
            raise ValueError("No songs prepared for hybrid onset-decoder validation")
        log.info("Prepared %d songs for chord-hybrid validation", len(prepared_hybrid_songs))

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    total_steps = max(1, cfg.epochs * max(1, len(train_loader)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = nn.CrossEntropyLoss(
        ignore_index=CHORD_PAD,
        label_smoothing=cfg.label_smoothing,
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_loss = float("inf")
    best_selection_value = float("inf") if cfg.selection_metric == "val_loss" else float("-inf")
    best_epoch = -1
    epochs_since_best = 0

    for epoch in range(1, cfg.epochs + 1):
        start_time = time.monotonic()
        model.train()
        for param in model.encoder.parameters():
            param.requires_grad = False

        total_loss = 0.0
        n_batches = 0
        for spec, frames, onset_features, token_input, token_target, padding_mask in train_loader:
            spec = spec.to(device, non_blocking=True)
            frames = frames.to(device, non_blocking=True)
            onset_features = onset_features.to(device, non_blocking=True)
            token_input = token_input.to(device, non_blocking=True)
            token_target = token_target.to(device, non_blocking=True)
            padding_mask = padding_mask.to(device, non_blocking=True)
            non_pad_flat = (~padding_mask).view(-1)
            if not bool(non_pad_flat.any().item()):
                continue

            optimizer.zero_grad(set_to_none=True)
            if use_amp and scaler is not None:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(
                        spec,
                        frames,
                        onset_features,
                        token_input,
                        tgt_key_padding_mask=padding_mask,
                    )
                    logits_valid = logits.view(-1, logits.size(-1))[non_pad_flat]
                    targets_valid = token_target.view(-1)[non_pad_flat]
                    loss = criterion(logits_valid, targets_valid)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(
                    spec,
                    frames,
                    onset_features,
                    token_input,
                    tgt_key_padding_mask=padding_mask,
                )
                logits_valid = logits.view(-1, logits.size(-1))[non_pad_flat]
                targets_valid = token_target.view(-1)[non_pad_flat]
                loss = criterion(logits_valid, targets_valid)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_loss += float(loss.item())
            n_batches += 1

        train_loss = total_loss / max(1, n_batches)
        val_loss, val_accuracy, val_null_rate = validate_chord_decoder(
            model,
            val_loader,
            criterion,
            device,
        )
        elapsed = time.monotonic() - start_time
        lr_now = scheduler.get_last_lr()[0]
        hybrid_report = None
        if prepared_hybrid_songs and epoch % max(1, cfg.hybrid_eval_every) == 0:
            hybrid_report = evaluate_prepared_chord_hybrid(
                model,
                prepared_hybrid_songs,
                device=device,
                window_frames=cfg.window_frames,
                stride_frames=cfg.stride_frames,
                max_onsets=cfg.max_onsets,
                vocab=vocab,
            )
            log.info(
                "Chord hybrid val: F %.4f -> %.4f (%+.4f)  CQS %.4f -> %.4f (%+.4f)",
                hybrid_report.baseline_macro_f,
                hybrid_report.hybrid_macro_f,
                hybrid_report.hybrid_macro_f - hybrid_report.baseline_macro_f,
                hybrid_report.baseline_cqs,
                hybrid_report.hybrid_cqs,
                hybrid_report.hybrid_cqs - hybrid_report.baseline_cqs,
            )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_null_rate": val_null_rate,
            "lr": lr_now,
            "elapsed_s": elapsed,
        }
        if hybrid_report is not None:
            record.update(
                {
                    "baseline_macro_f": hybrid_report.baseline_macro_f,
                    "hybrid_macro_f": hybrid_report.hybrid_macro_f,
                    "baseline_cqs": hybrid_report.baseline_cqs,
                    "hybrid_cqs": hybrid_report.hybrid_cqs,
                }
            )
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        log.info(
            "[onset decoder %d/%d] train_loss=%.4f val_loss=%.4f val_acc=%.3f val_null=%.3f lr=%.2e",
            epoch,
            cfg.epochs,
            train_loss,
            val_loss,
            val_accuracy,
            val_null_rate,
            lr_now,
        )

        payload = {
            "decoder_state": model.decoder.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "config": config_payload,
        }
        if hybrid_report is not None:
            payload["hybrid_eval"] = hybrid_report.as_dict()
        torch.save(payload, last_path)

        current_selection_value = None
        if cfg.selection_metric == "val_loss":
            current_selection_value = val_loss
        elif hybrid_report is not None:
            current_selection_value = hybrid_selection_value(hybrid_report, cfg.selection_metric)

        improved = (
            current_selection_value is not None
            and _selection_is_better(
                cfg.selection_metric,
                float(current_selection_value),
                best_selection_value,
            )
        )
        if improved:
            best_val_loss = val_loss
            best_selection_value = float(current_selection_value)
            best_epoch = epoch
            epochs_since_best = 0
            torch.save(payload, best_path)
            log.info(
                "New best onset decoder at epoch %d (%s=%.4f)",
                epoch,
                cfg.selection_metric,
                best_selection_value,
            )
        elif current_selection_value is not None:
            epochs_since_best += 1

        if epochs_since_best >= cfg.patience:
            log.info("Early stopping at epoch %d; best epoch was %d", epoch, best_epoch)
            break

    if not best_path.exists():
        log.warning("No best.pt saved; using last.pt")
        return last_path
    if prepared_hybrid_songs:
        best_ckpt = torch.load(str(best_path), map_location=device, weights_only=True)
        model.decoder.load_state_dict(best_ckpt["decoder_state"], strict=True)
        final_report = evaluate_prepared_chord_hybrid(
            model,
            prepared_hybrid_songs,
            device=device,
            window_frames=cfg.window_frames,
            stride_frames=cfg.stride_frames,
            max_onsets=cfg.max_onsets,
            vocab=vocab,
        )
        (cfg.output_dir / "eval_val_chord_hybrid.json").write_text(
            json.dumps(
                {
                    **final_report.as_dict(),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_selection_value": best_selection_value,
                    "selection_metric": cfg.selection_metric,
                },
                indent=2,
            )
        )
    return best_path
