"""
train.py
--------
Three-stage training loop.

Usage:
    python train.py --config config.yaml [--stage 1] [--resume checkpoints/best.pt]
"""

import argparse
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

from data    import build_datasets_from_config, build_collator, compute_pos_weights
from models  import MisogynistMemeModel, CulturalMisogynistLoss, CULTURES
from utils   import compute_metrics, format_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, metrics, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics":   metrics,
    }, path)
    logger.info(f"Saved checkpoint → {path}")


def load_checkpoint(model, optimizer, path: Path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    logger.info(f"Loaded checkpoint from {path}  (epoch {ckpt['epoch']})")
    return ckpt["epoch"], ckpt.get("metrics", {})


# ── One epoch of training ────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, scheduler, scaler, loss_fn,
    device, cfg, epoch: int
):
    model.train()
    total_loss  = 0.0
    log_every   = cfg["output"]["log_every_n_steps"]
    use_fp16    = cfg["training"]["fp16"] and device.type == "cuda"
    grad_clip   = cfg["training"]["grad_clip"]

    for step, batch in enumerate(loader, 1):
        pixel_values   = batch["pixel_values"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        label_mask     = batch["label_mask"].to(device)

        optimizer.zero_grad()

        with autocast(enabled=use_fp16):
            logits = model(pixel_values, input_ids, attention_mask)
            loss   = loss_fn(logits, labels, label_mask)

        if use_fp16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

        if step % log_every == 0:
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch} | Step {step}/{len(loader)} | "
                f"loss={loss.item():.4f} | lr={lr:.2e}"
            )

    return total_loss / len(loader)


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, cfg):
    model.eval()
    use_fp16 = cfg["training"]["fp16"] and device.type == "cuda"

    total_loss  = 0.0
    all_logits  = defaultdict(list)
    all_labels  = defaultdict(list)

    for batch in loader:
        pixel_values   = batch["pixel_values"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        label_mask     = batch["label_mask"].to(device)

        with autocast(enabled=use_fp16):
            logits = model(pixel_values, input_ids, attention_mask)
            loss   = loss_fn(logits, labels, label_mask)

        total_loss += loss.item()

        # collect per-culture logits and labels (only present ones)
        for i, culture in enumerate(CULTURES):
            mask = label_mask[:, i].cpu()
            if mask.any():
                all_logits[culture].extend(logits[culture][mask.to(device)].cpu().tolist())
                all_labels[culture].extend(labels[mask, i].cpu().tolist())

    metrics = compute_metrics(all_logits, all_labels)
    metrics["val_loss"] = total_loss / len(loader)
    return metrics


# ── Stage runner ─────────────────────────────────────────────────────────────

def run_stage(
    stage: int,
    model, train_ds, val_ds,
    collator, pos_weights,
    device, cfg,
    resume_from: Path = None,
):
    t_cfg = cfg["training"]
    o_cfg = cfg["output"]
    d_cfg = cfg["data"]
    s_key = f"stage{stage}"

    n_epochs    = t_cfg[s_key]["epochs"]
    ckpt_dir    = Path(o_cfg["checkpoint_dir"]) / f"stage{stage}"
    best_metric = o_cfg["save_best_metric"]

    # ── prepare model for this stage ────────────────────────────────────────
    if stage == 1:
        logger.info("=== Stage 1: freezing encoders, training fusion + heads ===")
        model.freeze_encoders()

    elif stage == 2:
        logger.info("=== Stage 2: encoders still frozen, lower LR ===")
        model.freeze_encoders()

    elif stage == 3:
        if not t_cfg["stage3"]["enabled"]:
            logger.info("Stage 3 disabled in config — skipping.")
            return
        n = t_cfg["stage3"]["unfreeze_last_n_blocks"]
        logger.info(f"=== Stage 3: unfreezing last {n} encoder blocks ===")
        model.unfreeze_encoder_last_n(n)

    # ── optimizer + scheduler ────────────────────────────────────────────────
    param_groups = model.get_param_groups(stage, cfg)
    optimizer    = AdamW(param_groups, weight_decay=t_cfg["weight_decay"])

    train_loader = DataLoader(
        train_ds,
        batch_size=d_cfg["train_batch_size"],
        shuffle=True,
        num_workers=d_cfg["num_workers"],
        collate_fn=collator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=d_cfg["val_batch_size"],
        shuffle=False,
        num_workers=d_cfg["num_workers"],
        collate_fn=collator,
        pin_memory=True,
    )

    total_steps   = len(train_loader) * n_epochs
    warmup_steps  = int(total_steps * t_cfg["warmup_ratio"])
    scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler        = GradScaler(enabled=t_cfg["fp16"] and device.type == "cuda")

    loss_fn = CulturalMisogynistLoss(pos_weights).to(device)

    start_epoch = 1
    best_val    = -float("inf")

    if resume_from and resume_from.exists():
        start_epoch, prev_metrics = load_checkpoint(model, optimizer, resume_from, device)
        best_val = prev_metrics.get(best_metric, best_val)
        start_epoch += 1

    # ── training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, n_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            loss_fn, device, cfg, epoch,
        )

        if epoch % t_cfg.get("eval_every_n_epochs", o_cfg["eval_every_n_epochs"]) == 0:
            metrics = evaluate(model, val_loader, loss_fn, device, cfg)
            logger.info(
                f"\nEpoch {epoch}/{n_epochs}  train_loss={train_loss:.4f}  "
                f"val_loss={metrics['val_loss']:.4f}\n" + format_metrics(metrics)
            )

            current = metrics.get(best_metric, 0.0)
            if current > best_val:
                best_val = current
                save_checkpoint(model, optimizer, epoch, metrics, ckpt_dir / "best.pt")

        save_checkpoint(model, optimizer, epoch, {}, ckpt_dir / "last.pt")

    logger.info(f"Stage {stage} done. Best {best_metric}={best_val:.4f}")
    return ckpt_dir / "best.pt"


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--stage",  type=int, default=None,
                        help="Run only this stage (1/2/3). Default: run all.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── data ─────────────────────────────────────────────────────────────────
    logger.info("Loading datasets...")
    train_ds, val_ds = build_datasets_from_config(cfg)

    # compute pos_weights from the underlying DataFrames
    # (ConcatDataset wraps individual datasets; retrieve dfs from them)
    all_dfs = []
    for ds in (train_ds.datasets if hasattr(train_ds, "datasets") else [train_ds]):
        all_dfs.append(ds.df)
    import pandas as pd
    combined_df  = pd.concat(all_dfs, ignore_index=True)
    pos_weights  = compute_pos_weights(combined_df)

    # override pos_weights from config if set manually
    for culture in CULTURES:
        override = cfg["pos_weight"].get(culture)
        if override is not None:
            pos_weights[culture] = torch.tensor(float(override))

    # ── model ─────────────────────────────────────────────────────────────────
    logger.info("Building model...")
    model = MisogynistMemeModel(cfg).to(device)

    # ── collator ──────────────────────────────────────────────────────────────
    collator = build_collator(cfg)

    # ── run stages ────────────────────────────────────────────────────────────
    stages    = [args.stage] if args.stage else [1, 2, 3]
    best_ckpt = Path(args.resume) if args.resume else None

    for stage in stages:
        best_ckpt = run_stage(
            stage, model, train_ds, val_ds,
            collator, pos_weights, device, cfg,
            resume_from=best_ckpt,
        )


if __name__ == "__main__":
    main()