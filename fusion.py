"""
models/fusion.py

Cross-attention fusion of image tokens [B, N_img, D] and text tokens [B, N_txt, D].

Design: bidirectional self-attention over concatenated sequences with a [CLS] token.
    Input  → [CLS] | image_tokens | text_tokens    shape [B, 1+N_img+N_txt, D]
    After N transformer blocks → extract CLS token  shape [B, D]
    Linear projection → shared representation       shape [B, shared_dim]

Why bidirectional self-attention (not cross-attention)?
    Both modalities attend to each other AND to themselves simultaneously.
    This lets image patches notice which text words matter, and text tokens
    notice which image regions they relate to — in every layer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Linear projection (768 → proj_dim, shared across both modalities) ─────────

class ModalityProjection(nn.Module):
    """Projects encoder output to fusion dim. One per modality."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ── Single transformer block (standard pre-norm) ───────────────────────────────

class FusionBlock(nn.Module):
    """
    Pre-norm transformer block:
        x → LayerNorm → MultiHeadAttention → residual
          → LayerNorm → FFN → residual
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x:           torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                [B, seq_len, d_model]
            key_padding_mask: [B, seq_len] bool, True = ignore (padding)
        Returns:
            x: [B, seq_len, d_model]
        """
        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(
            query            = x,
            key              = x,
            value            = x,
            key_padding_mask = key_padding_mask,
            need_weights     = False,
        )
        x = residual + attn_out

        # FFN with pre-norm
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x


# ── Full fusion module ─────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Fuses image and text token sequences via bidirectional transformer.

    Architecture:
        img_proj → project image tokens to proj_dim
        txt_proj → project text tokens to proj_dim
        [CLS] learnable token prepended
        N fusion transformer blocks
        CLS token extracted → linear → shared_dim representation

    Args:
        encoder_dim  : output dim of both encoders (768)
        proj_dim     : internal dim of fusion (768)
        shared_dim   : final output dim (1024)
        n_heads      : attention heads in each block (8)
        n_layers     : number of FusionBlock stacked (2-4)
        dropout      : dropout rate
    """

    def __init__(
        self,
        encoder_dim: int = 768,
        proj_dim:    int = 768,
        shared_dim:  int = 1024,
        n_heads:     int = 8,
        n_layers:    int = 3,
        dropout:     float = 0.1,
    ):
        super().__init__()

        # Per-modality projections (trainable even when encoders are frozen)
        self.img_proj = ModalityProjection(encoder_dim, proj_dim, dropout)
        self.txt_proj = ModalityProjection(encoder_dim, proj_dim, dropout)

        # Learnable [CLS] token: shape [1, 1, proj_dim]
        self.cls_token = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.02)

        # Transformer fusion blocks
        self.blocks = nn.ModuleList([
            FusionBlock(proj_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(proj_dim)

        # Project CLS output → shared representation
        self.head_proj = nn.Sequential(
            nn.Linear(proj_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        img_tokens: torch.Tensor,
        txt_tokens: torch.Tensor,
        txt_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            img_tokens:       [B, N_img, encoder_dim]
            txt_tokens:       [B, N_txt, encoder_dim]
            txt_padding_mask: [B, N_txt] bool — True where padding (from tokenizer)
                              Pass (1 - attention_mask).bool()

        Returns:
            shared_repr: [B, shared_dim]  (1024 by default)
        """
        B = img_tokens.size(0)

        # Project each modality to fusion dim
        img = self.img_proj(img_tokens)   # [B, N_img, proj_dim]
        txt = self.txt_proj(txt_tokens)   # [B, N_txt, proj_dim]

        # Expand CLS token for the batch
        cls = self.cls_token.expand(B, -1, -1)   # [B, 1, proj_dim]

        # Concatenate: [CLS] | image | text
        x = torch.cat([cls, img, txt], dim=1)    # [B, 1+N_img+N_txt, proj_dim]

        # Build padding mask for the concatenated sequence
        # CLS and image tokens are never masked
        N_img = img.size(1)
        N_txt = txt.size(1)
        if txt_padding_mask is not None:
            # txt_padding_mask: [B, N_txt], True = pad
            cls_img_mask = torch.zeros(
                B, 1 + N_img,
                dtype=torch.bool,
                device=x.device,
            )
            full_mask = torch.cat([cls_img_mask, txt_padding_mask], dim=1)
        else:
            full_mask = None

        # Run through fusion blocks
        for block in self.blocks:
            x = block(x, key_padding_mask=full_mask)

        x = self.norm(x)

        # Extract CLS token (position 0) as pooled representation
        cls_out = x[:, 0, :]     # [B, proj_dim]

        # Project to shared_dim
        shared = self.head_proj(cls_out)   # [B, shared_dim]
        return shared