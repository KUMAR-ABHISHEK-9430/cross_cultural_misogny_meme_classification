"""
data/collate.py
---------------
Custom collator that:
  - Runs SigLIP image processor on a batch of PIL images
  - Runs XLM-R tokenizer on a batch of transcript strings
  - Stacks labels and label_masks
  - Returns everything as tensors ready for the model
"""

from typing import Dict, List
import torch
from transformers import AutoProcessor, AutoTokenizer


class MemeCollator:
    """
    Args:
        siglip_processor : HuggingFace processor for SigLIP
                           (handles image resize + normalisation)
        xlmr_tokenizer   : HuggingFace tokenizer for XLM-R
        max_img_tokens   : max number of patch tokens kept from SigLIP
        max_txt_tokens   : max token length fed to XLM-R
        device           : 'cpu' — move to GPU inside the training loop
    """

    def __init__(
        self,
        siglip_processor,
        xlmr_tokenizer,
        max_img_tokens: int = 64,
        max_txt_tokens: int = 64,
    ):
        self.siglip_proc  = siglip_processor
        self.xlmr_tok     = xlmr_tokenizer
        self.max_img_tokens = max_img_tokens
        self.max_txt_tokens = max_txt_tokens

    def __call__(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        images      = [s["image"]      for s in samples]
        transcripts = [s["transcript"] for s in samples]
        image_ids   = [s["image_id"]   for s in samples]
        labels      = torch.stack([s["labels"]     for s in samples])   # [B, 3]
        label_masks = torch.stack([s["label_mask"] for s in samples])   # [B, 3]

        # ── image processing (SigLIP) ────────────────────────────────────────
        # SigLIP processor returns pixel_values of shape [B, C, H, W]
        img_inputs = self.siglip_proc(
            images=images,
            return_tensors="pt",
        )
        pixel_values = img_inputs["pixel_values"]   # [B, 3, 384, 384]

        # ── text tokenisation (XLM-R) ────────────────────────────────────────
        txt_inputs = self.xlmr_tok(
            transcripts,
            padding=True,
            truncation=True,
            max_length=self.max_txt_tokens,
            return_tensors="pt",
        )
        # input_ids: [B, N_txt], attention_mask: [B, N_txt]

        return {
            "image_ids":           image_ids,                         # List[str]
            "pixel_values":        pixel_values,                      # [B, 3, H, W]
            "input_ids":           txt_inputs["input_ids"],           # [B, N_txt]
            "attention_mask":      txt_inputs["attention_mask"],      # [B, N_txt]
            "labels":              labels,                            # [B, 3]
            "label_mask":          label_masks,                       # [B, 3]
        }


def build_collator(cfg: dict):
    """
    Build a MemeCollator from config.
    Loads the SigLIP processor and XLM-R tokenizer from HuggingFace.
    """
    model_cfg = cfg["model"]
    data_cfg  = cfg["data"]

    siglip_proc = AutoProcessor.from_pretrained(model_cfg["siglip_name"])
    xlmr_tok    = AutoTokenizer.from_pretrained(model_cfg["xlmr_name"])

    return MemeCollator(
        siglip_processor=siglip_proc,
        xlmr_tokenizer=xlmr_tok,
        max_img_tokens=model_cfg["max_img_tokens"],
        max_txt_tokens=model_cfg["max_txt_tokens"],
    )