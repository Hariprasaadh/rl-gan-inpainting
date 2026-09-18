"""DeepFillEncoder — state embedding head wrapping the frozen DeepFill backbone.

The encoder taps the CoarseNet bottleneck (after conv10, shape (B, 4*cnum, H/4, W/4))
and projects it to the 256-d latent vector used by the RL agent.

This replaces the old InpaintingEncoder for the new RLInpaintingModel but the
old class is kept in encoder.py for backward compatibility.

Usage
-----
from src.models.encoder_deepfill import DeepFillEncoder

enc = DeepFillEncoder(cnum=48, latent_dim=256)
# enc is trainable; backbone is frozen separately in RLInpaintingModel
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepFillEncoder(nn.Module):
    """Lightweight trainable head on top of frozen DeepFill bottleneck features.

    Architecture:
        AdaptiveAvgPool2d(1,1)
        Linear(4*cnum, latent_dim)
        LayerNorm
        ELU
        Linear(latent_dim, latent_dim)

    Args:
        cnum:       Channel multiplier of the backbone (48 for nipponjo).
        latent_dim: Output embedding dimension (256).
    """

    def __init__(self, cnum: int = 48, latent_dim: int = 256):
        super().__init__()
        bottleneck_channels = 4 * cnum   # 192 for cnum=48
        self.latent_dim = latent_dim

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(bottleneck_channels, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ELU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, bottleneck_feats: torch.Tensor) -> torch.Tensor:
        """Project bottleneck features to 256-d state embedding.

        Args:
            bottleneck_feats: (B, 4*cnum, H', W') -- from backbone.get_bottleneck_features()

        Returns:
            state_embedding: (B, latent_dim)
        """
        pooled = self.pool(bottleneck_feats).flatten(1)   # (B, 4*cnum)
        return self.head(pooled)                           # (B, latent_dim)
