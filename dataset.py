"""
data/dataset.py

Loads meme images + transcripts from a CSV that has columns:
    image_id | transcriptions | indian_labels | irish_labels | chinese_labels

Each label column contains strings like "misogyny" or "not-misogyny".
Returns a dict with the raw PIL image, transcript string, and 3 binary labels.
"""

import os
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


# ── Label helpers ──────────────────────────────────────────────────────────────

def _parse_label(value: str, positive_string: str = "misogyny") -> float:
    """Convert 'misogyny' → 1.0, anything else → 0.0. Handles NaN."""
    if pd.isna(value):
        return -1.0          # -1 = missing; ignored in loss computation
    return 1.0 if str(value).strip().lower() == positive_string.lower() else 0.0


# ── Dataset ────────────────────────────────────────────────────────────────────

class MemeDataset(Dataset):
    """
    Args:
        csv_path    : path to train.csv / dev.csv
        images_dir  : directory that holds all meme images
        cfg         : the `data` + `labels` section of config.yaml
        transform   : optional torchvision / albumentations transform on PIL image
    """

    def __init__(self, csv_path: str, images_dir: str, cfg: dict,
                 transform=None):
        self.images_dir = Path(images_dir)
        self.transform  = transform
        self.cfg        = cfg

        col_map         = cfg["labels"]["csv_columns"]   # {culture: csv_col_name}
        pos_str         = cfg["labels"]["positive_string"]
        self.cultures   = cfg["labels"]["cultures"]      # ["indian","irish","chinese"]
        self.col_map    = col_map
        self.pos_str    = pos_str

        df = pd.read_csv(csv_path)
        df = df.reset_index(drop=True)

        # Add any missing label columns as NaN (so missing English labels work)
        for culture, col in col_map.items():
            if col not in df.columns:
                df[col] = float("nan")

        self.df = df

    # ── Internals ──────────────────────────────────────────────────────────

    def _find_image(self, image_id) -> Path:
        """Try .jpg then .png for a given image_id."""
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            p = self.images_dir / f"{image_id}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(
            f"No image found for id={image_id} in {self.images_dir}"
        )

    # ── Public ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # ── Image ──────────────────────────────────────────────────────────
        img_path = self._find_image(row["image_id"])
        image    = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # ── Transcript ─────────────────────────────────────────────────────
        transcript = str(row.get("transcriptions", "")).strip()

        # ── Labels (float; -1.0 = missing, skip in loss) ───────────────────
        labels = {}
        for culture, col in self.col_map.items():
            labels[culture] = torch.tensor(
                _parse_label(row.get(col, float("nan")), self.pos_str),
                dtype=torch.float32,
            )

        return {
            "image_id":   str(row["image_id"]),
            "image":      image,          # PIL Image or tensor if transform given
            "transcript": transcript,
            "labels":     labels,         # {"indian": tensor, "irish": tensor, ...}
        }

    # ── Utility ────────────────────────────────────────────────────────────

    def compute_pos_weights(self) -> dict:
        """
        Compute per-culture pos_weight = neg_count / pos_count.
        Use as pos_weight in BCEWithLogitsLoss to handle class imbalance.
        Returns {culture: float}.  Missing labels (-1) are excluded.
        """
        weights = {}
        for culture, col in self.col_map.items():
            vals = self.df[col].dropna().apply(
                lambda v: _parse_label(v, self.pos_str)
            )
            vals = vals[vals >= 0]           # drop missing
            pos  = (vals == 1).sum()
            neg  = (vals == 0).sum()
            weights[culture] = float(neg / pos) if pos > 0 else 1.0
        return weights


# ── Quick smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml, sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    root    = cfg["data"]["root"]
    ds = MemeDataset(
        csv_path   = os.path.join(root, cfg["data"]["train_csv"]),
        images_dir = os.path.join(root, cfg["data"]["images_dir"]),
        cfg        = cfg,
    )

    print(f"Dataset size: {len(ds)}")
    sample = ds[0]
    print(f"image_id : {sample['image_id']}")
    print(f"image    : {sample['image'].size}")
    print(f"transcript: {sample['transcript'][:60]}...")
    print(f"labels   : {sample['labels']}")
    print(f"pos_weights: {ds.compute_pos_weights()}")