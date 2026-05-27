"""
models/encoders.py
------------------
Thin wrappers around SigLIP and XLM-R that:
  - Load the pretrained weights
  - Expose a clean forward() returning token embeddings [B, N, 768]
  - Provide freeze() / unfreeze_last_n() helpers for staged training
"""

import logging
from typing import Optional
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    SiglipVisionModel,
    SiglipVisionConfig,
)

logger = logging.getLogger(__name__)


class SigLIPEncoder(nn.Module):
    """
    Wraps google/siglip-so400m-patch14-384.

    forward() returns patch token embeddings of shape [B, N_img, 768].
    The [CLS] token is excluded — we prepend our own CLS in the fusion block.
    """

    def __init__(self, model_name: str, max_tokens: int = 64):
        super().__init__()
        self.max_tokens = max_tokens
        self.model = SiglipVisionModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size  # 768 for so400m

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, 384, 384]

        Returns:
            patch_tokens: [B, N_img, hidden_size]  (N_img <= max_tokens)
        """
        outputs = self.model(pixel_values=pixel_values)
        # last_hidden_state: [B, num_patches, hidden_size]
        # e.g. 384px / 14px patch = 27×27 = 729 patches — we cap at max_tokens
        tokens = outputs.last_hidden_state          # [B, 729, 768]
        tokens = tokens[:, : self.max_tokens, :]   # [B, max_tokens, 768]
        return tokens

    def freeze(self):
        """Freeze all parameters."""
        for p in self.parameters():
            p.requires_grad = False
        logger.info("SigLIPEncoder: all layers frozen")

    def unfreeze_last_n(self, n: int):
        """Unfreeze the last n transformer blocks of the vision encoder."""
        blocks = self.model.vision_model.encoder.layers
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        logger.info(f"SigLIPEncoder: last {n} blocks unfrozen")


class XLMREncoder(nn.Module):
    """
    Wraps xlm-roberta-base.

    forward() returns token embeddings of shape [B, N_txt, 768].
    """

    def __init__(self, model_name: str, max_tokens: int = 64):
        super().__init__()
        self.max_tokens = max_tokens
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size  # 768

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      [B, N_txt]
            attention_mask: [B, N_txt]

        Returns:
            token_embeddings: [B, N_txt, hidden_size]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # last_hidden_state includes [CLS] at position 0
        # We return all tokens (including CLS); fusion block can use it or ignore it
        return outputs.last_hidden_state   # [B, N_txt, 768]

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        logger.info("XLMREncoder: all layers frozen")

    def unfreeze_last_n(self, n: int):
        blocks = self.model.encoder.layer
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        logger.info(f"XLMREncoder: last {n} blocks unfrozen")