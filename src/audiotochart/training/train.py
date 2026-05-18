from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from audiotochart.training.config import TrainConfig
from audiotochart.training.dataset import create_datasets
from audiotochart.training.evaluate import CLASS_NAMES_8, collect_song_activations, evaluate, format_report
from audiotochart.training.losses import rhythm_game_bce, rhythm_game_focal
from audiotochart.training.model import build_finetune_model, count_parameters, forward_logits
from audiotochart.training.thresholds import (
    fmeasure_with_tolerance, labels_to_frame_list, optimize_thresholds, pick_peaks,
)

log = logging.getLogger(__name__)


def _collate(batch):
    specs = np.stack([b[0] for b in batch], axis=0)
    labels = np.stack([b[1] for b in batch], axis=0)
    return torch.from_numpy(specs), torch.from_numpy(labels)


def _make_loader(ds, *, batch_size, shuffle, num_workers):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=_collate, pin_memory=True, drop_last=shuffle, persistent_workers=num_workers > 0)


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_pos_weight(cfg: TrainConfig, device):
    weights = cfg.resolved_class_weights()
    return torch.tensor([w * cfg.onset_weight for w in weights], dtype=torch.float32, device=device)


def _build_focal_alpha(cfg, device):
    weights = torch.tensor(cfg.resolved_class_weights(), dtype=torch.float32)
    alpha = torch.sqrt(weights)
    alpha = alpha / alpha.max()
    return alpha.to(device)


def _quick_val_macro_f(model, val_entries, *, num_classes, device, threshold=0.3, tolerance_frames=2):
    pairs = collect_song_activations(model, val_entries, device=device)
    f_per_class = [[] for _ in range(num_classes)]
    for acts, labels in pairs:
        for c in range(num_classes):
            picks = pick_peaks(acts[:, c], threshold)
            gt = labels_to_frame_list(labels[:, c])
            _, _, f = fmeasure_with_tolerance(picks, gt, tolerance_frames=tolerance_frames)
            f_per_class[c].append(f)
    macros = [sum(fs) / len(fs) if fs else 0.0 for fs in f_per_class]
    names = CLASS_NAMES_8[:num_classes]

    per_class = {names[i]: float(macros[i]) for i in range(num_classes)}
    macro_f = sum(per_class.values()) / len(per_class) if per_class else 0.0
    return macro_f, per_class


def _save_checkpoint(path, model, cfg, epoch, val_metric, extra=None):
    payload = {"model_state": model.state_dict(), "config": asdict(cfg), "epoch": epoch, "val_metric": val_metric}
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["config"]["cache_dir"] = str(payload["config"]["cache_dir"])
    payload["config"]["output_dir"] = str(payload["config"]["output_dir"])
    torch.save(payload, path)


def _train_one_epoch(model, loader, optimizer, scheduler, scaler, *, pos_weight, focal_alpha, timing_sigma, focal_gamma, dice_weight, loss_fn, device, use_amp, log_every=100, epoch_label=""):
    model.train()
    total_loss = 0.0
    n_batches = 0
    total_batches = len(loader)
    window_loss = 0.0
    window_count = 0
    t_start = time.monotonic()
    t_window = t_start

    for spec, labels in loader:
        spec = spec.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = forward_logits(model, spec)
                target = labels
                if logits.shape[1] != target.shape[1]:
                    T = min(logits.shape[1], target.shape[1])
                    logits_t = logits[:, :T]
                    target = target[:, :T]
                else:
                    logits_t = logits
                if loss_fn == "bce":
                    loss = rhythm_game_bce(logits_t, target, pos_weight=pos_weight, timing_sigma=timing_sigma)
                else:
                    dw = dice_weight if loss_fn == "focal+dice" else 0.0
                    loss = rhythm_game_focal(logits_t, target, alpha=focal_alpha, gamma=focal_gamma, dice_weight=dw, timing_sigma=timing_sigma)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = forward_logits(model, spec)
            target = labels
            if logits.shape[1] != target.shape[1]:
                T = min(logits.shape[1], target.shape[1])
                logits_t = logits[:, :T]
                target = target[:, :T]
            else:
                logits_t = logits
            if loss_fn == "bce":
                loss = rhythm_game_bce(logits_t, target, pos_weight=pos_weight, timing_sigma=timing_sigma)
            else:
                dw = dice_weight if loss_fn == "focal+dice" else 0.0
                loss = rhythm_game_focal(logits_t, target, alpha=focal_alpha, gamma=focal_gamma, dice_weight=dw, timing_sigma=timing_sigma)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_val = float(loss.detach().item())
        total_loss += loss_val
        window_loss += loss_val
        n_batches += 1
        window_count += 1

        if n_batches == 1 or n_batches % log_every == 0 or n_batches == total_batches:
            now = time.monotonic()
            window_dt = max(now - t_window, 1e-6)
            log.info(
                "%s step %d/%d  loss(window)=%.4f  %.1f it/s  elapsed=%.0fs",
                epoch_label, n_batches, total_batches, window_loss / max(1, window_count),
                window_count / window_dt, now - t_start,
            )
            window_loss = 0.0
            window_count = 0
            t_window = now

    return total_loss / max(1, n_batches)


def run_training(cfg: TrainConfig, *, resume_from: Path | None = None) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.output_dir / "train_log.jsonl"
    config_path = cfg.output_dir / "config.json"
    best_path = cfg.output_dir / "best.pt"
    last_path = cfg.output_dir / "last.pt"

    config_path.write_text(json.dumps({**asdict(cfg), "cache_dir": str(cfg.cache_dir), "output_dir": str(cfg.output_dir)}, indent=2))

    _seed_everything(cfg.seed)
    torch.backends.cudnn.benchmark = True

    if cfg.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available — falling back to CPU")
        cfg.device = "cpu"
    device = torch.device(cfg.device)
    use_amp = cfg.amp and device.type == "cuda"

    train_ds, val_ds, test_ds = create_datasets(
        cfg.cache_dir, window_frames=cfg.window_frames, stride_frames=cfg.stride_frames, seed=cfg.seed, harmonix_only=cfg.harmonix_only,
    )
    log.info("Datasets: train=%d windows (%d songs), val=%d songs, test=%d songs", len(train_ds), len(train_ds.entries), len(val_ds.entries), len(test_ds.entries))

    train_loader = _make_loader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    num_classes = 8
    model = build_finetune_model(num_classes=num_classes, freeze_cnn=True).to(device)
    if resume_from is not None:
        log.info("Resuming model weights from %s", resume_from)
        ckpt = torch.load(resume_from, map_location=device)
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            log.warning("Resume: missing keys=%s", missing)
        if unexpected:
            log.warning("Resume: unexpected keys=%s", unexpected)

    total, trainable = count_parameters(model)
    log.info("Model: %d total params, %d trainable (trainable CNN frozen)", total, trainable)

    pos_weight = _build_pos_weight(cfg, device)
    focal_alpha = _build_focal_alpha(cfg, device)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    tol_frames_primary = max(1, round(cfg.tolerance_ms_primary * 100 / 1000))

    best_val_f = -1.0
    best_epoch = -1
    epochs_since_best = 0

    if resume_from is not None:
        _save_checkpoint(best_path, model, cfg, epoch=0, val_metric=-1.0, extra={"phase": "resume"})
        _save_checkpoint(last_path, model, cfg, epoch=0, val_metric=-1.0, extra={"phase": "resume"})

    def _log_epoch(record):
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    steps_per_epoch = max(1, len(train_loader))

    if cfg.warmup_epochs > 0:
        log.info("Phase A: frozen-CNN warmup, %d epochs", cfg.warmup_epochs)
        optimizer = torch.optim.AdamW(
            [{"params": model.gru_layers.parameters(), "lr": cfg.warmup_lr_gru}, {"params": model.output_layer.parameters(), "lr": cfg.warmup_lr_head}],
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.warmup_epochs * steps_per_epoch)
    else:
        log.info("Phase A: skipped (warmup_epochs=0)")
        optimizer = None
        scheduler = None

    for epoch in range(cfg.warmup_epochs):
        t0 = time.monotonic()
        train_loss = _train_one_epoch(model, train_loader, optimizer, scheduler, scaler, pos_weight=pos_weight, focal_alpha=focal_alpha, timing_sigma=cfg.timing_sigma, focal_gamma=cfg.focal_gamma, dice_weight=cfg.dice_weight, loss_fn=cfg.loss_fn, device=device, use_amp=use_amp, epoch_label=f"[A {epoch+1}/{cfg.warmup_epochs}]")
        val_f, per_class = _quick_val_macro_f(model, val_ds.entries, num_classes=num_classes, device=cfg.device, tolerance_frames=tol_frames_primary)
        elapsed = time.monotonic() - t0
        log.info("[A %d/%d] train_loss=%.4f val_macroF=%.4f (%.0fs)", epoch+1, cfg.warmup_epochs, train_loss, val_f, elapsed)
        _log_epoch({"phase": "A", "epoch": epoch+1, "train_loss": train_loss, "val_macro_f": val_f, "val_per_class_f": per_class, "elapsed_s": elapsed})
        if val_f > best_val_f:
            best_val_f = val_f
            best_epoch = epoch+1
            _save_checkpoint(best_path, model, cfg, epoch=epoch+1, val_metric=val_f, extra={"phase": "A"})
        _save_checkpoint(last_path, model, cfg, epoch=epoch+1, val_metric=val_f, extra={"phase": "A"})

    # Phase B
    for p in model.cnn_blocks.parameters():
        p.requires_grad = True

    if cfg.finetune_epochs > 0:
        cnn_early = list(model.cnn_blocks[0].parameters())
        cnn_late: list = []
        for block in model.cnn_blocks[1:]:
            cnn_late.extend(block.parameters())
        gru = list(model.gru_layers.parameters())
        head = list(model.output_layer.parameters())
        if getattr(model, "context_layer", None) is not None:
            gru.extend(model.context_layer.parameters())

        optimizer = torch.optim.AdamW([
            {"params": cnn_early, "lr": cfg.finetune_lr_cnn_early},
            {"params": cnn_late, "lr": cfg.finetune_lr_cnn_late},
            {"params": gru, "lr": cfg.finetune_lr},
            {"params": head, "lr": cfg.finetune_lr_head},
        ], weight_decay=cfg.weight_decay)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=[cfg.finetune_lr_cnn_early, cfg.finetune_lr_cnn_late, cfg.finetune_lr, cfg.finetune_lr_head],
            epochs=cfg.finetune_epochs, steps_per_epoch=len(train_loader),
            pct_start=cfg.onecycle_pct_start, anneal_strategy="cos",
            div_factor=cfg.onecycle_div_factor, final_div_factor=cfg.onecycle_final_div,
        )

    for epoch in range(cfg.finetune_epochs):
        t0 = time.monotonic()
        train_loss = _train_one_epoch(model, train_loader, optimizer, scheduler, scaler, pos_weight=pos_weight, focal_alpha=focal_alpha, timing_sigma=cfg.timing_sigma, focal_gamma=cfg.focal_gamma, dice_weight=cfg.dice_weight, loss_fn=cfg.loss_fn, device=device, use_amp=use_amp, epoch_label=f"[B {epoch+1}/{cfg.finetune_epochs}]")
        val_f, per_class = _quick_val_macro_f(model, val_ds.entries, num_classes=num_classes, device=cfg.device, tolerance_frames=tol_frames_primary)
        elapsed = time.monotonic() - t0
        log.info("[B %d/%d] train_loss=%.4f val_macroF=%.4f (%.0fs)", epoch+1, cfg.finetune_epochs, train_loss, val_f, elapsed)
        _log_epoch({"phase": "B", "epoch": epoch+1, "train_loss": train_loss, "val_macro_f": val_f, "val_per_class_f": per_class, "elapsed_s": elapsed})

        improved = val_f > best_val_f
        if improved:
            best_val_f = val_f
            best_epoch = cfg.warmup_epochs + epoch + 1
        epochs_since_best = 0 if improved else epochs_since_best + 1
        _save_checkpoint(best_path if improved else last_path, model, cfg, epoch=best_epoch if improved else (cfg.warmup_epochs + epoch + 1), val_metric=val_f, extra={"phase": "B"})

        if epochs_since_best >= cfg.patience:
            log.info("Early stopping at epoch %d", cfg.warmup_epochs + epoch + 1)
            break

    # Threshold optimization
    log.info("Loading best checkpoint and optimizing thresholds on val set")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_pairs = collect_song_activations(model, val_ds.entries, device=cfg.device)
    acts_per_class = [
        np.concatenate([p[0][:, c] for p in val_pairs]) if val_pairs else np.zeros(0, dtype=np.float32)
        for c in range(num_classes)
    ]
    gt_per_class: list[np.ndarray] = []
    for c in range(num_classes):
        offsets: list[np.ndarray] = []
        running = 0
        for _, labels in val_pairs:
            track = labels_to_frame_list(labels[:, c]) + running
            offsets.append(track)
            running += labels.shape[0]
        gt_per_class.append(np.concatenate(offsets) if offsets else np.zeros(0, dtype=np.int64))

    thresholds, val_f_scores = optimize_thresholds(acts_per_class, gt_per_class, tolerance_frames=tol_frames_primary)
    (cfg.output_dir / "thresholds.json").write_text(json.dumps({"thresholds": thresholds, "val_f_scores": val_f_scores, "tolerance_frames": tol_frames_primary}, indent=2))

    # Test eval
    log.info("Evaluating on test set (%d songs)", len(test_ds.entries))
    report = evaluate(model, test_ds.entries, thresholds, fps=100, tolerance_ms_primary=cfg.tolerance_ms_primary, tolerance_ms_secondary=cfg.tolerance_ms_secondary, device=cfg.device)
    report.save(cfg.output_dir / "eval_test.json")
    log.info("\n%s", format_report(report, class_names=CLASS_NAMES_8))

    return best_path
