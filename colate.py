"""
data/collate.py

Custom collate for DataLoader.
Handles:
  - Batching SigLIP pixel_values (already fixed size from processor)
  - Batching XLM-R input_ids / attention_mask (padded to batch max len)
  - Stacking per-culture label tensors
  - Passing raw strings through (image_id, transcript)
"""

from dataclasses import dataclass
from typing import List, Dict
import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class MisogynySample:
    """Typed container for a single processed sample (post-encoder-processor)."""
    image_id:        str
    pixel_values:    torch.Tensor   # [3, H, W] — from SigLIP processor
    input_ids:       torch.Tensor   # [seq_len]  — from XLM-R tokenizer
    attention_mask:  torch.Tensor   # [seq_len]
    labels:          Dict[str, torch.Tensor]   # {culture: scalar tensor}


def collate_fn(samples: List[dict]) -> dict:
    """
    Called by DataLoader with a list of raw __getitem__ dicts.

    NOTE: pixel_values and input_ids are NOT processed here.
    They are still PIL Images and raw strings at this point.
    Preprocessing happens in the model's forward() via the stored processors,
    OR you can move it here by passing the processors in — see below.

    This version handles the case where the Dataset returns:
      - image: torch.Tensor (if SigLIP processor applied in Dataset transform)
      - input_ids / attention_mask: tensors (if XLM-R tokenizer applied in Dataset)

    If you apply processors in the training loop instead, this still works —
    just batch the raw strings and PIL images as lists.
    """

    image_ids   = [s["image_id"]   for s in samples]
    transcripts = [s["transcript"] for s in samples]

    # ── Images ─────────────────────────────────────────────────────────────
    # If transform was applied in Dataset, images are tensors → stack
    # If not, images are PIL → keep as list (processor handles batching)
    if isinstance(samples[0]["image"], torch.Tensor):
        images = torch.stack([s["image"] for s in samples])   # [B, C, H, W]
    else:
        images = [s["image"] for s in samples]                # list of PIL

    # ── Labels ─────────────────────────────────────────────────────────────
    culture_keys = list(samples[0]["labels"].keys())
    labels = {
        culture: torch.stack([s["labels"][culture] for s in samples])
        for culture in culture_keys
    }   # each → [B]

    return {
        "image_ids":   image_ids,
        "images":      images,
        "transcripts": transcripts,
        "labels":      labels,
    }


class ProcessorCollate:
    """
    Drop-in replacement for collate_fn that also runs the SigLIP image
    processor and XLM-R tokenizer inside the DataLoader worker.

    Usage:
        collate = ProcessorCollate(siglip_processor, xlmr_tokenizer, cfg)
        loader  = DataLoader(dataset, collate_fn=collate, ...)
    """

    def __init__(self, siglip_processor, xlmr_tokenizer, cfg: dict):
        self.img_proc  = siglip_processor
        self.txt_tok   = xlmr_tokenizer
        self.max_img   = cfg["model"]["max_img_tokens"]
        self.max_txt   = cfg["model"]["max_txt_tokens"]

    def __call__(self, samples: List[dict]) -> dict:
        image_ids   = [s["image_id"]   for s in samples]
        transcripts = [s["transcript"] for s in samples]

        # ── SigLIP image processing ─────────────────────────────────────────
        # Returns pixel_values: [B, 3, 384, 384]
        img_inputs = self.img_proc(
            images         = [s["image"] for s in samples],
            return_tensors = "pt",
        )

        # ── XLM-R tokenization ──────────────────────────────────────────────
        txt_inputs = self.txt_tok(
            transcripts,
            padding        = "longest",
            truncation     = True,
            max_length     = self.max_txt + 2,   # +2 for [CLS]/[SEP]
            return_tensors = "pt",
        )

        # ── Labels ──────────────────────────────────────────────────────────
        culture_keys = list(samples[0]["labels"].keys())
        labels = {
            culture: torch.stack([s["labels"][culture] for s in samples])
            for culture in culture_keys
        }

        return {
            "image_ids":      image_ids,
            "pixel_values":   img_inputs["pixel_values"],     # [B, 3, H, W]
            "input_ids":      txt_inputs["input_ids"],        # [B, T]
            "attention_mask": txt_inputs["attention_mask"],   # [B, T]
            "labels":         labels,                         # {culture: [B]}
        }