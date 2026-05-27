"""
models/model.py
---------------
Full model: SigLIP + XLM-R → CrossAttentionFusion → 3 culture heads.

Each culture head outputs 1 logit.
Loss: BCEWithLogitsLoss per culture, summed.
Rows where a culture label is missing are masked out of that culture's loss.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import SigLIPEncoder, XLMREncoder
from .fusion   import CrossAttentionFusion

logger = logging.getLogger(__name__)

CULTURES = ["indian", "irish", "chinese"]


class CultureHead(nn.Module):
    """
    Simple MLP head: d_shared → 512 → 1 logit.
    Sigmoid is NOT applied here — BCEWithLogitsLoss handles it.
    """

    def __init__(self, d_shared: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_shared, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # [B]


class MisogynistMemeModel(nn.Module):
    """
    Full pipeline model.

    Args:
        cfg: full config dict (loaded from config.yaml)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]

        self.siglip  = SigLIPEncoder(m["siglip_name"], max_tokens=m["max_img_tokens"])
        self.xlmr    = XLMREncoder(m["xlmr_name"],   max_tokens=m["max_txt_tokens"])
        self.fusion  = CrossAttentionFusion(
            d_img    = self.siglip.hidden_size,
            d_txt    = self.xlmr.hidden_size,
            d_model  = m["d_model"],
            d_shared = m["d_shared"],
            n_layers = m["n_fusion_layers"],
            n_heads  = m["n_heads"],
            dropout  = m["dropout"],
        )
        # One head per culture
        self.heads = nn.ModuleDict({
            culture: CultureHead(m["d_shared"], m["dropout"])
            for culture in CULTURES
        })

    def forward(
        self,
        pixel_values:   torch.Tensor,   # [B, 3, H, W]
        input_ids:      torch.Tensor,   # [B, N_txt]
        attention_mask: torch.Tensor,   # [B, N_txt]
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict of {culture: logits [B]} for each culture
        """
        img_tokens = self.siglip(pixel_values)                      # [B, N_img, 768]
        txt_tokens = self.xlmr(input_ids, attention_mask)           # [B, N_txt, 768]
        shared     = self.fusion(img_tokens, txt_tokens, attention_mask)  # [B, 1024]

        return {culture: self.heads[culture](shared) for culture in CULTURES}

    # ── Freeze / unfreeze helpers ─────────────────────────────────────────────

    def freeze_encoders(self):
        self.siglip.freeze()
        self.xlmr.freeze()

    def unfreeze_encoder_last_n(self, n: int):
        self.siglip.unfreeze_last_n(n)
        self.xlmr.unfreeze_last_n(n)

    def get_param_groups(self, stage: int, cfg: dict) -> list:
        """
        Return optimizer param groups for the given training stage.

        Stage 1: fusion + heads only
        Stage 2: fusion + heads (lower LR)
        Stage 3: encoder last-N + fusion + heads (very low LR)
        """
        s = cfg["training"]

        if stage == 1:
            lr_f = s["stage1"]["lr_fusion"]
            lr_h = s["stage1"]["lr_heads"]
            return [
                {"params": self.fusion.parameters(),  "lr": lr_f, "name": "fusion"},
                {"params": [p for h in self.heads.values() for p in h.parameters()],
                 "lr": lr_h, "name": "heads"},
            ]

        elif stage == 2:
            lr_f = s["stage2"]["lr_fusion"]
            lr_h = s["stage2"]["lr_heads"]
            return [
                {"params": self.fusion.parameters(),  "lr": lr_f, "name": "fusion"},
                {"params": [p for h in self.heads.values() for p in h.parameters()],
                 "lr": lr_h, "name": "heads"},
            ]

        elif stage == 3:
            lr_e = s["stage3"]["lr_encoders"]
            lr_f = s["stage3"]["lr_fusion"]
            lr_h = s["stage3"]["lr_heads"]
            n    = s["stage3"]["unfreeze_last_n_blocks"]
            # only the last-n encoder blocks (already unfrozen by caller)
            enc_params = (
                list(self.siglip.model.vision_model.encoder.layers[-n:].parameters()) +
                list(self.xlmr.model.encoder.layer[-n:].parameters())
            )
            return [
                {"params": enc_params,                "lr": lr_e, "name": "encoders"},
                {"params": self.fusion.parameters(),  "lr": lr_f, "name": "fusion"},
                {"params": [p for h in self.heads.values() for p in h.parameters()],
                 "lr": lr_h, "name": "heads"},
            ]
        else:
            raise ValueError(f"Unknown stage: {stage}")


# ── Loss function ─────────────────────────────────────────────────────────────

class CulturalMisogynistLoss(nn.Module):
    """
    BCE loss summed across cultures, with per-sample masking for missing labels.

    Args:
        pos_weights: dict {culture: scalar tensor}
                     Compensates for class imbalance (neg/pos ratio).
    """

    def __init__(self, pos_weights: Dict[str, torch.Tensor]):
        super().__init__()
        self.pos_weights = pos_weights   # kept on CPU; moved to device in forward

    def forward(
        self,
        logits:     Dict[str, torch.Tensor],   # {culture: [B]}
        labels:     torch.Tensor,              # [B, 3]  — 0/1 or -1 if missing
        label_mask: torch.Tensor,              # [B, 3]  — True = present
    ) -> torch.Tensor:
        """
        Returns scalar loss (mean over present labels and batch).
        """
        total_loss  = torch.tensor(0.0, device=labels.device)
        n_valid     = 0

        for i, culture in enumerate(CULTURES):
            mask = label_mask[:, i]            # [B] bool
            if not mask.any():
                continue                       # no labels for this culture in this batch

            logit_c = logits[culture][mask]    # [n_valid]
            label_c = labels[mask, i]          # [n_valid]

            pw = self.pos_weights.get(culture, torch.tensor(1.0)).to(labels.device)

            loss_c = F.binary_cross_entropy_with_logits(
                logit_c, label_c, pos_weight=pw
            )
            total_loss = total_loss + loss_c
            n_valid   += 1

        return total_loss / max(n_valid, 1)