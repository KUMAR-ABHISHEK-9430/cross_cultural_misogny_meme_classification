"""
models/encoders.py

Thin wrappers around SigLIP (vision) and XLM-R (text).
Both expose a single forward() that returns token-level embeddings:
    SigLIPEncoder  → [B, N_img, 768]
    XLMREncoder    → [B, N_txt, 768]

Freezing is done externally in train.py — encoders themselves are neutral.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoProcessor,
    SiglipVisionModel,
    XLMRobertaModel,
    AutoTokenizer,
)


# ── SigLIP Vision Encoder ──────────────────────────────────────────────────────

class SigLIPEncoder(nn.Module):
    """
    Wraps google/siglip-so400m-patch14-384.
    forward() takes pixel_values [B, 3, 384, 384] and returns
    patch embeddings [B, N_patches, hidden_size].

    N_patches = (384 / 14)^2 = 729 patches for the so400m variant.
    We cap at max_tokens by taking the first max_tokens patches
    (spatial order = top-left to bottom-right).
    """

    def __init__(self, model_name: str, max_tokens: int = 64):
        super().__init__()
        self.max_tokens = max_tokens
        self.model      = SiglipVisionModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size   # 768 for base

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, H, W]  (already preprocessed)
        Returns:
            patch_embeddings: [B, min(N_patches, max_tokens), hidden_size]
        """
        outputs = self.model(pixel_values=pixel_values)
        # last_hidden_state: [B, N_patches, hidden_size]
        embeddings = outputs.last_hidden_state
        # Cap sequence length
        embeddings = embeddings[:, :self.max_tokens, :]
        return embeddings   # [B, N, 768]

    @property
    def output_dim(self) -> int:
        return self.hidden_size


# ── XLM-R Text Encoder ────────────────────────────────────────────────────────

class XLMREncoder(nn.Module):
    """
    Wraps xlm-roberta-base.
    forward() takes input_ids + attention_mask and returns
    token embeddings [B, N_txt, hidden_size].

    We cap at max_tokens (excluding [CLS] and [SEP]).
    """

    def __init__(self, model_name: str, max_tokens: int = 64):
        super().__init__()
        self.max_tokens  = max_tokens
        self.model       = XLMRobertaModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size   # 768

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      [B, T]
            attention_mask: [B, T]
        Returns:
            token_embeddings: [B, min(T, max_tokens+2), hidden_size]
            (+2 because [CLS] and [SEP] are kept)
        """
        outputs = self.model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
        )
        # last_hidden_state: [B, T, hidden_size]
        embeddings = outputs.last_hidden_state
        # Cap length (keep [CLS] at pos 0)
        cap = self.max_tokens + 2   # +2 for special tokens
        embeddings = embeddings[:, :cap, :]
        return embeddings   # [B, T_capped, 768]

    @property
    def output_dim(self) -> int:
        return self.hidden_size


# ── Loader helpers ─────────────────────────────────────────────────────────────

def load_siglip_processor(model_name: str):
    """Returns the SigLIP image processor (handles resize + normalize)."""
    return AutoProcessor.from_pretrained(model_name)


def load_xlmr_tokenizer(model_name: str):
    """Returns the XLM-R tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)