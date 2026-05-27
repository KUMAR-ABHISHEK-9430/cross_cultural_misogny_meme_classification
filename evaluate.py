"""
evaluate.py
-----------
Load a checkpoint and run evaluation on dev or test split.

Usage:
    python evaluate.py --config config.yaml --checkpoint checkpoints/stage2/best.pt
                       --split dev   # or test
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from data   import build_collator, load_split_csv, MisogynistMemeDataset, download_drive_folder
from models import MisogynistMemeModel, CulturalMisogynistLoss, CULTURES
from utils  import compute_metrics, format_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split",      default="dev", choices=["dev", "test"])
    parser.add_argument("--output",     default=None,
                        help="Path to save predictions JSON. Optional.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── load datasets ─────────────────────────────────────────────────────────
    data_cfg   = cfg["data"]
    cache_root = Path(data_cfg["cache_dir"])

    dfs, img_dirs = [], []
    for ds_name, folder_id in data_cfg["drive_folders"].items():
        folder  = download_drive_folder(folder_id, cache_root)
        img_dir = folder / "images" if (folder / "images").exists() else folder
        csv_p   = folder / f"{args.split}.csv"
        if not csv_p.exists():
            logger.warning(f"No {args.split}.csv in {folder} — skipping")
            continue
        df = load_split_csv(csv_p, data_cfg)
        dfs.append(df)
        img_dirs.append(img_dir)

    combined_df = pd.concat(dfs, ignore_index=True)
    dataset     = MisogynistMemeDataset(combined_df, img_dirs[0], data_cfg["image_size"])

    collator   = build_collator(cfg)
    loader     = DataLoader(
        dataset,
        batch_size=data_cfg["val_batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        collate_fn=collator,
    )

    # ── load model ────────────────────────────────────────────────────────────
    model = MisogynistMemeModel(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # dummy pos_weights (not used in eval loss)
    pos_weights = {c: torch.tensor(1.0) for c in CULTURES}
    loss_fn     = CulturalMisogynistLoss(pos_weights)

    use_fp16    = cfg["training"]["fp16"] and device.type == "cuda"
    total_loss  = 0.0
    all_logits  = defaultdict(list)
    all_labels  = defaultdict(list)
    predictions = []

    with torch.no_grad():
        for batch in loader:
            pv  = batch["pixel_values"].to(device)
            iid = batch["input_ids"].to(device)
            am  = batch["attention_mask"].to(device)
            lb  = batch["labels"].to(device)
            lm  = batch["label_mask"].to(device)

            with autocast(enabled=use_fp16):
                logits = model(pv, iid, am)
                loss   = loss_fn(logits, lb, lm)
            total_loss += loss.item()

            for i, culture in enumerate(CULTURES):
                mask = lm[:, i].cpu()
                if mask.any():
                    all_logits[culture].extend(logits[culture][mask.to(device)].cpu().tolist())
                    all_labels[culture].extend(lb[mask, i].cpu().tolist())

            # Build per-sample prediction rows
            for j, img_id in enumerate(batch["image_ids"]):
                row = {"image_id": img_id}
                for culture in CULTURES:
                    logit = logits[culture][j].item()
                    row[f"{culture}_logit"] = logit
                    row[f"{culture}_pred"]  = int(logit > 0.0)
                predictions.append(row)

    metrics = compute_metrics(all_logits, all_labels)
    metrics["val_loss"] = total_loss / len(loader)
    logger.info(f"\n{format_metrics(metrics)}")

    if args.output:
        out = {"metrics": metrics, "predictions": predictions}
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Predictions saved → {args.output}")


if __name__ == "__main__":
    main()