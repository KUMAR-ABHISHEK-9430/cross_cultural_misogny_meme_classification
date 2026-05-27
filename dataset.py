"""
data/dataset.py
---------------
Handles:
  - Downloading dataset folders from Google Drive (via gdown)
  - Normalising CSV labels to a uniform schema:
        image_id | transcriptions | indian_label | irish_label | chinese_label
  - PyTorch Dataset returning (image, transcript, labels) per sample
  - Computing pos_weight per culture for BCEWithLogitsLoss
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)

# ── Label normalisation ────────────────────────────────────────────────────────

_LABEL_MAP = {
    "misogyny": 1, "misogynous": 1, "yes": 1, "1": 1, 1: 1,
    "not-misogyny": 0, "not misogyny": 0, "non-misogyny": 0, "no": 0, "0": 0, 0: 0,
}

def _normalise_label(raw) -> Optional[int]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    key = str(raw).strip().lower() if not isinstance(raw, int) else raw
    if key not in _LABEL_MAP:
        logger.warning(f"Unknown label value: {raw!r} — treating as missing")
        return None
    return _LABEL_MAP[key]


# ── Google Drive download ──────────────────────────────────────────────────────

def download_drive_folder(folder_id: str, dest: Path) -> Path:
    """
    Download a Google Drive folder to dest/<folder_id>/.
    Skips download if folder already cached and non-empty.
    """
    import gdown
    out_dir = dest / folder_id
    if out_dir.exists() and any(out_dir.iterdir()):
        logger.info(f"Folder {folder_id} already cached at {out_dir}")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Drive folder {folder_id} → {out_dir}")
    gdown.download_folder(id=folder_id, output=str(out_dir), quiet=False, use_cookies=False)
    return out_dir


# ── CSV loading + normalisation ────────────────────────────────────────────────

def load_split_csv(csv_path: Path, cfg_data: dict) -> pd.DataFrame:
    """
    Load one CSV split and normalise to the uniform schema:
        image_id | transcriptions | indian_label | irish_label | chinese_label

    Missing label columns are left as NaN — the Dataset masks them at loss time.
    """
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = df.columns.str.strip().str.lower()

    col_id  = cfg_data["col_image_id"].lower()
    col_txt = cfg_data["col_transcript"].lower()
    df = df.rename(columns={col_id: "image_id", col_txt: "transcriptions"})

    culture_cols = {
        "indian":  cfg_data["col_indian"].lower(),
        "irish":   cfg_data["col_irish"].lower(),
        "chinese": cfg_data["col_chinese"].lower(),
    }
    for culture, raw_col in culture_cols.items():
        if raw_col in df.columns:
            df[f"{culture}_label"] = df[raw_col].apply(_normalise_label)
        else:
            df[f"{culture}_label"] = np.nan   # column absent in this CSV

    keep = ["image_id", "transcriptions", "indian_label", "irish_label", "chinese_label"]
    return df[keep].reset_index(drop=True)


# ── Dataset ────────────────────────────────────────────────────────────────────

class MisogynistMemeDataset(Dataset):
    """
    Returns one sample dict per meme:

        image_id   : str
        image      : PIL.Image (RGB)  — processor handles resize/normalise
        transcript : str
        labels     : FloatTensor [3]  — (indian, irish, chinese) 0/1 or -1 if missing
        label_mask : BoolTensor  [3]  — True = present, False = skip in loss
    """

    CULTURES = ["indian", "irish", "chinese"]

    def __init__(self, df: pd.DataFrame, image_dir: Path, image_size: int = 384):
        self.df        = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def _load_image(self, image_id: str) -> Image.Image:
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = self.image_dir / f"{image_id}{ext}"
            if p.exists():
                return Image.open(p).convert("RGB")
        raise FileNotFoundError(f"No image for id={image_id!r} in {self.image_dir}")

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]

        try:
            image = self._load_image(str(row["image_id"]))
        except FileNotFoundError as e:
            logger.warning(str(e) + " — using blank image.")
            image = Image.new("RGB", (self.image_size, self.image_size), color=0)

        transcript = str(row["transcriptions"]) if pd.notna(row["transcriptions"]) else ""

        labels, label_mask = [], []
        for culture in self.CULTURES:
            raw = row[f"{culture}_label"]
            if pd.isna(raw):
                labels.append(-1.0)
                label_mask.append(False)
            else:
                labels.append(float(raw))
                label_mask.append(True)

        return {
            "image_id":   str(row["image_id"]),
            "image":      image,
            "transcript": transcript,
            "labels":     torch.tensor(labels,     dtype=torch.float32),   # [3]
            "label_mask": torch.tensor(label_mask, dtype=torch.bool),      # [3]
        }


# ── pos_weight computation ─────────────────────────────────────────────────────

def compute_pos_weights(
    df: pd.DataFrame,
    cultures: List[str] = ("indian", "irish", "chinese"),
) -> Dict[str, torch.Tensor]:
    """
    pos_weight = neg_count / pos_count per culture.
    Only rows with a present label are counted.
    """
    weights = {}
    for culture in cultures:
        col     = f"{culture}_label"
        present = df[col].dropna().astype(float)
        if len(present) == 0:
            logger.warning(f"No labels for {culture} — pos_weight=1.0")
            weights[culture] = torch.tensor(1.0)
            continue
        pos = (present == 1).sum()
        neg = (present == 0).sum()
        w   = torch.tensor(neg / pos if pos > 0 else 1.0, dtype=torch.float32)
        logger.info(f"  {culture}: pos={pos}, neg={neg}, pos_weight={w.item():.2f}")
        weights[culture] = w
    return weights


# ── Build datasets from config ─────────────────────────────────────────────────

def build_datasets_from_config(cfg: dict) -> Tuple[Dataset, Dataset]:
    """
    1. Downloads every Drive folder in cfg['data']['drive_folders'].
    2. Reads train.csv + dev.csv from each folder.
    3. Concatenates all splits and returns (train_dataset, val_dataset).
    """
    from torch.utils.data import ConcatDataset

    data_cfg   = cfg["data"]
    cache_root = Path(data_cfg["cache_dir"])
    image_size = data_cfg.get("image_size", 384)

    train_pairs, val_pairs = [], []

    for ds_name, folder_id in data_cfg["drive_folders"].items():
        folder  = download_drive_folder(folder_id, cache_root)
        img_dir = folder / "images" if (folder / "images").exists() else folder

        for split, pairs in [("train", train_pairs), ("dev", val_pairs)]:
            csv_path = folder / f"{split}.csv"
            if not csv_path.exists():
                logger.warning(f"No {split}.csv in {folder} — skipping.")
                continue
            df = load_split_csv(csv_path, data_cfg)
            logger.info(f"  {ds_name}/{split}: {len(df)} samples")
            pairs.append((df, img_dir))

    def _concat(pairs):
        datasets = [MisogynistMemeDataset(df, img_dir, image_size) for df, img_dir in pairs]
        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    return _concat(train_pairs), _concat(val_pairs)