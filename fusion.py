"""
models/fusion.py
----------------
Cross-attention fusion module.

Architecture (Option 2 — Bidirectional + CLS, as recommended):

    [CLS] token  ──┐
    Image tokens ──┤  concat  →  Self-Attention × N layers  →  take [CLS] output
    Text tokens  ──┘

The [CLS] token is a learnable parameter.  After N transformer blocks it
summarises the joint image+text context.  We then project it to d_shared (1024).
"""

import torch
import torch.nn as nn


class FusionBlock(nn.Module):
    """One transformer encoder block used inside the fusion stack."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,    # input shape [B, N, d_model]
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:                [B, N, d_model]  (CLS + img + txt tokens)
            key_padding_mask: [B, N] bool, True = ignore that position
                              (used to mask padded text tokens)

        Returns:
            [B, N, d_model]
        """
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        # FFN with residual
        x = self.norm2(x + self.ffn(x))
        return x


class CrossAttentionFusion(nn.Module):
    """
    Full fusion module:
        1. Linear projections to align SigLIP (768) and XLM-R (768) dims.
        2. Prepend learnable [CLS] token.
        3. N transformer self-attention blocks over the concatenated sequence.
        4. Extract [CLS] output.
        5. Project to d_shared (1024) with LayerNorm.

    Args:
        d_img         : SigLIP hidden size (768)
        d_txt         : XLM-R hidden size  (768)
        d_model       : internal dim of the fusion transformer (768)
        d_shared      : output dim of the shared representation (1024)
        n_layers      : number of FusionBlocks (2-4)
        n_heads       : attention heads per block
        dropout       : dropout rate
    """

    def __init__(
        self,
        d_img:    int = 768,
        d_txt:    int = 768,
        d_model:  int = 768,
        d_shared: int = 1024,
        n_layers: int = 3,
        n_heads:  int = 8,
        dropout:  float = 0.1,
    ):
        super().__init__()

        # Trainable projections — even when encoders are frozen these learn
        self.img_proj = nn.Sequential(
            nn.Linear(d_img, d_model),
            nn.LayerNorm(d_model),
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(d_txt, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable [CLS] token  [1, 1, d_model]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Fusion transformer blocks
        self.blocks = nn.ModuleList([
            FusionBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

        # Project CLS output → shared representation
        self.to_shared = nn.Sequential(
            nn.Linear(d_model, d_shared),
            nn.LayerNorm(d_shared),
            nn.GELU(),
        )

    def forward(
        self,
        img_tokens: torch.Tensor,         # [B, N_img, d_img]
        txt_tokens: torch.Tensor,         # [B, N_txt, d_txt]
        txt_attention_mask: torch.Tensor, # [B, N_txt]  1=real, 0=pad
    ) -> torch.Tensor:
        """
        Returns:
            shared: [B, d_shared]  — the joint representation for the heads
        """
        B = img_tokens.size(0)

        # 1. Project both modalities to d_model
        img = self.img_proj(img_tokens)   # [B, N_img, d_model]
        txt = self.txt_proj(txt_tokens)   # [B, N_txt, d_model]

        # 2. Prepend [CLS]  →  [B, 1+N_img+N_txt, d_model]
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, img, txt], dim=1)

        # 3. Build key_padding_mask for MultiheadAttention
        #    True = position should be IGNORED
        #    CLS (1) and image tokens (N_img) are always valid
        img_valid  = torch.zeros(B, 1 + img_tokens.size(1),
                                 dtype=torch.bool, device=seq.device)   # [B, 1+N_img]
        txt_pad    = (txt_attention_mask == 0)                           # [B, N_txt]
        kp_mask    = torch.cat([img_valid, txt_pad], dim=1)              # [B, 1+N_img+N_txt]

        # 4. Run fusion blocks
        x = seq
        for block in self.blocks:
            x = block(x, key_padding_mask=kp_mask)

        # 5. Extract [CLS] output (position 0) and project
        cls_out = x[:, 0, :]              # [B, d_model]
        shared  = self.to_shared(cls_out) # [B, d_shared]
        return shared