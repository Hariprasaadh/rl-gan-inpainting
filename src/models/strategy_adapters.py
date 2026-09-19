"""Strategy adapters: 4 lightweight specialized heads applied after DeepFill Stage 2.

Each adapter is a small residual network with FiLM conditioning.
Only adapter weights are trainable; backbone is frozen.

Actions:
  0 - GlobalAdapter: overall context reconstruction
  1 - LocalAdapter: hole-region focused (heavy hole loss)
  2 - BoundaryAdapter: edge continuity on dilated-eroded boundary zone
  3 - TextureAdapter: high-frequency perceptual sharpening
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class StrategyModulationLite(nn.Module):
    """FiLM modulation for adapter intermediate features."""

    def __init__(self, channels: int, strategy_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(strategy_dim, channels * 2),
            nn.ELU(inplace=True),
            nn.Linear(channels * 2, channels * 2),
        )

    def forward(self, x: torch.Tensor, strategy_emb: torch.Tensor) -> torch.Tensor:
        params = self.mlp(strategy_emb)  # (B, 2*C)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.view(-1, x.shape[1], 1, 1)
        beta = beta.view(-1, x.shape[1], 1, 1)
        return (1.0 + gamma) * x + beta


class BaseAdapter(nn.Module):
    """Common adapter scaffolding."""

    def __init__(self, strategy_dim: int = 64, hidden: int = 64):
        super().__init__()
        self.strategy_dim = strategy_dim
        self.hidden = hidden

        # 3-layer residual conv block operating on RGB
        self.conv1 = nn.Conv2d(3, hidden, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(hidden)
        self.act1 = nn.ELU(inplace=True)

        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(hidden)
        self.mod = StrategyModulationLite(hidden, strategy_dim)
        self.act2 = nn.ELU(inplace=True)

        self.conv3 = nn.Conv2d(hidden, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward_base(self, x: torch.Tensor, strategy_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Core 3-layer forward (override in subclasses for specialized logic)."""
        h = self.act1(self.norm1(self.conv1(x)))
        if strategy_emb is not None:
            h = self.mod(h, strategy_emb)
        h = self.act2(self.norm2(self.conv2(h)))
        delta = self.conv3(h)
        # Residual + tanh clamping to keep in [-1, 1]
        out = torch.tanh(x + 0.5 * torch.tanh(delta))
        return out


class GlobalAdapter(BaseAdapter):
    """Action 0: Global context pass-through with mild refinement."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor, strategy_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # No mask weighting - global objective
        return self.forward_base(x, strategy_emb)


class LocalAdapter(BaseAdapter):
    """Action 1: Amplifies hole region features.

    Uses mask-weighted attention: applies stronger refinement inside hole.
    Intermediate features are multiplied by (1 + mask) inside hole.
    """

    def forward(self, x: torch.Tensor, mask: torch.Tensor, strategy_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Weight input toward hole region
        # mask: (B,1,H,W)
        masked_input = x * (1.0 + 0.5 * mask)  # slight amplification
        out = self.forward_base(masked_input, strategy_emb)
        # Composite: keep valid region from input, use adapted output only inside hole-ish blended region
        # Learn implicit blending
        return out


class BoundaryAdapter(BaseAdapter):
    """Action 2: Edge-preserving refinement on boundary zone.

    Creates a boundary ring via dilation - erosion to target gradient-preserving
    convolutions on the seam.
    """

    def forward(self, x: torch.Tensor, mask: torch.Tensor, strategy_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Compute boundary mask on the fly (feathered ring)
        # Dilate mask
        kernel = torch.ones((1, 1, 5, 5), device=mask.device, dtype=mask.dtype)
        dilated = (F.conv2d(mask.float(), kernel, padding=2) > 0).float()
        # Erode mask
        # erosion approx via min pooling: invert, dilate, invert
        inv = 1.0 - mask.float()
        dilated_inv = (F.conv2d(inv, kernel, padding=2) > 0).float()
        eroded = 1.0 - dilated_inv
        boundary = torch.clamp(dilated - eroded, 0, 1)  # ring around hole edge

        # Gradient-preserving: add boundary-weighted high-pass component
        # Laplacian kernel for edge emphasis
        # Use boundary to weight refinement delta
        out = self.forward_base(x, strategy_emb)
        # Mix: out is full refinement; blend boundary zone more strongly
        # inside hole near edge we trust adapter more
        blend = torch.clamp(boundary * 2.0 + mask * 0.5, 0, 1)
        return x * (1 - blend * 0.5) + out * (0.5 * blend + 0.5 * (1 - blend) * 0.3 + 0.5)


class TextureAdapter(BaseAdapter):
    """Action 3: High-frequency texture sharpening.

    Applies Laplacian high-pass inside hole to encourage detail synthesis.
    """

    def __init__(self, strategy_dim: int = 64, hidden: int = 64):
        super().__init__(strategy_dim=strategy_dim, hidden=hidden)
        # Additional high-frequency conv branch
        self.hf_conv = nn.Conv2d(3, hidden, kernel_size=3, padding=1)
        self.hf_mod = StrategyModulationLite(hidden, strategy_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, strategy_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Laplacian high-pass on input
        # Kernel: [[0,-1,0],[-1,4,-1],[0,-1,0]]
        lap_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                                  dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        lap_kernel = lap_kernel.repeat(3, 1, 1, 1)  # depthwise per channel
        # Need groups=3 for depthwise
        high_freq = F.conv2d(x, lap_kernel, padding=1, groups=3)
        # Boost HF inside mask
        hf_boost = high_freq * mask

        out = self.forward_base(x + 0.2 * hf_boost, strategy_emb)

        # Optional HF branch mixing for extra texture fidelity
        if strategy_emb is not None:
            hf_feat = self.hf_mod(F.elu(self.hf_conv(high_freq)), strategy_emb)
            # project back via 1x1 implicit through conv3 style? just add small residual
            # Approximate by global pooling scalar mixing
            hf_mix = hf_feat.mean(dim=1, keepdim=True) * 0.02 * mask
            out = out + hf_mix.expand_as(out)

        return torch.tanh(out)
