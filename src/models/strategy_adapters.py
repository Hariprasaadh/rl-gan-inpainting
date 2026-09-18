"""Strategy Adapters — 4 lightweight post-processing adapters.

Each adapter is a small residual network applied after the DeepFill Stage-2
refined output.  The four adapters have genuinely different inductive biases
so that, after strategy-specific pretraining, each adapter specialises on a
different aspect of reconstruction quality.

Architecture overview
---------------------
All adapters:
  - Input:  (B, 3, H, W) in [-1, 1]  (refined output from backbone)
  - Output: (B, 3, H, W) in [-1, 1]
  - Share the same external interface: forward(x, mask, strategy_emb)

FiLM modulation
---------------
StrategyModulation applies:
    out = (1 + gamma(e)) * x + beta(e)
where e is the strategy embedding vector (B, strategy_dim).

Action assignments
------------------
  0  GlobalAdapter  -- context coherence  (L1 full + small perceptual)
  1  LocalAdapter   -- hole accuracy      (heavy L1/SSIM inside hole)
  2  BoundaryAdapter-- edge continuity    (gradient-weighted loss on seam)
  3  TextureAdapter -- high-frequency     (FFT + perceptual loss)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# FiLM modulation (shared)
# ---------------------------------------------------------------------------

class StrategyModulation(nn.Module):
    """Feature-wise Linear Modulation conditioned on a strategy embedding.

    out = (1 + gamma) * x + beta
    where gamma, beta are predicted from the strategy embedding.
    """

    def __init__(self, channels: int, strategy_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(strategy_dim, channels * 2),
            nn.ELU(inplace=True),
            nn.Linear(channels * 2, channels * 2),
        )

    def forward(self, x: torch.Tensor, strategy_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:            (B, C, H, W)
            strategy_emb: (B, strategy_dim)

        Returns:
            (B, C, H, W) FiLM-modulated feature map.
        """
        params = self.mlp(strategy_emb)               # (B, 2*C)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)     # (B, C, 1, 1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * x + beta


# ---------------------------------------------------------------------------
# Shared residual block used by all adapters
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """3-conv residual block with instance norm."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ELU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
        )
        self.act = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


# ---------------------------------------------------------------------------
# Action 0 — GlobalAdapter
# ---------------------------------------------------------------------------

class GlobalAdapter(nn.Module):
    """Global context coherence adapter.

    Objective during pretraining:  L1(full) + 0.05 * Perceptual
    Inductive bias: general reconstruction quality everywhere.
    """

    CHANNELS = 32

    def __init__(self, strategy_dim: int = 64):
        super().__init__()
        c = self.CHANNELS
        self.enc   = nn.Conv2d(3, c, 3, padding=1)
        self.res1  = ResBlock(c)
        self.film  = StrategyModulation(c, strategy_dim)
        self.res2  = ResBlock(c)
        self.dec   = nn.Conv2d(c, 3, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        strategy_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:            (B, 3, H, W) refined backbone output in [-1, 1]
            mask:         (B, 1, H, W) hole mask (1 = hole)
            strategy_emb: (B, strategy_dim) FiLM conditioning

        Returns:
            (B, 3, H, W) in [-1, 1]
        """
        feat = F.elu(self.enc(x))
        feat = self.res1(feat)
        feat = self.film(feat, strategy_emb)
        feat = self.res2(feat)
        delta = self.dec(feat)
        return torch.tanh(x + delta)


# ---------------------------------------------------------------------------
# Action 1 — LocalAdapter
# ---------------------------------------------------------------------------

class LocalAdapter(nn.Module):
    """Hole-region accuracy adapter.

    Objective during pretraining:  6.0 * L1(hole) + 0.5 * SSIM(hole)
    Inductive bias: amplifies features inside the hole, suppresses valid region
    changes.
    """

    CHANNELS = 32

    def __init__(self, strategy_dim: int = 64):
        super().__init__()
        c = self.CHANNELS
        # Encode image + mask together so the adapter knows the hole location
        self.enc   = nn.Conv2d(4, c, 3, padding=1)  # 3 image + 1 mask
        self.res1  = ResBlock(c)
        self.film  = StrategyModulation(c, strategy_dim)
        self.res2  = ResBlock(c)
        # Attention gate for mask region
        self.attn  = nn.Sequential(
            nn.Conv2d(c, 1, 1),
            nn.Sigmoid(),
        )
        self.dec   = nn.Conv2d(c, 3, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        strategy_emb: torch.Tensor,
    ) -> torch.Tensor:
        inp  = torch.cat([x, mask], dim=1)           # (B, 4, H, W)
        feat = F.elu(self.enc(inp))
        feat = self.res1(feat)
        feat = self.film(feat, strategy_emb)
        feat = self.res2(feat)

        # Attention: concentrate refinement on the hole
        attn_gate = self.attn(feat)                  # (B, 1, H, W) in (0,1)
        attn_gate = attn_gate * mask                 # zero outside hole

        delta = self.dec(feat) * attn_gate           # apply only inside hole
        return torch.tanh(x + delta)


# ---------------------------------------------------------------------------
# Action 2 — BoundaryAdapter
# ---------------------------------------------------------------------------

class BoundaryAdapter(nn.Module):
    """Boundary/seam continuity adapter.

    Objective during pretraining:  L1 + 2.0 * boundary_gradient_loss
    Inductive bias: focuses on the narrow transition zone between inpainted
    region and original content.
    """

    CHANNELS = 32

    def __init__(self, strategy_dim: int = 64):
        super().__init__()
        c = self.CHANNELS
        self.enc  = nn.Conv2d(3, c, 3, padding=1)
        self.res1 = ResBlock(c)
        self.film = StrategyModulation(c, strategy_dim)
        self.res2 = ResBlock(c)
        self.dec  = nn.Conv2d(c, 3, 3, padding=1)

    @staticmethod
    def _boundary_map(mask: torch.Tensor, dilation: int = 5) -> torch.Tensor:
        """Compute a soft boundary band around the mask edge.

        Returns a (B, 1, H, W) map that is 1 at the boundary zone, 0 elsewhere.
        """
        kernel = torch.ones(
            (1, 1, dilation, dilation),
            dtype=torch.float32,
            device=mask.device,
        )
        pad = dilation // 2
        dilated = F.conv2d(mask.float(), kernel, padding=pad)
        # Erode by doing dilation on (1 - mask)
        eroded_inv = F.conv2d((1 - mask).float(), kernel, padding=pad)
        # Boundary = dilated hole AND NOT fully in the hole
        boundary = (dilated > 0).float() * (eroded_inv > 0).float()
        return boundary.clamp(0, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        strategy_emb: torch.Tensor,
    ) -> torch.Tensor:
        boundary = self._boundary_map(mask)          # (B, 1, H, W)

        feat  = F.elu(self.enc(x))
        feat  = self.res1(feat)
        feat  = self.film(feat, strategy_emb)
        feat  = self.res2(feat)
        delta = self.dec(feat) * boundary            # focus on boundary zone
        return torch.tanh(x + delta)


# ---------------------------------------------------------------------------
# Action 3 — TextureAdapter
# ---------------------------------------------------------------------------

class TextureAdapter(nn.Module):
    """High-frequency / texture adapter.

    Objective during pretraining:  L1 + 0.1 * FFT_frequency_loss + 0.05 * Perceptual
    Inductive bias: Laplacian high-pass sharpening inside the hole region.
    """

    CHANNELS = 32

    def __init__(self, strategy_dim: int = 64):
        super().__init__()
        c = self.CHANNELS
        self.enc  = nn.Conv2d(3, c, 3, padding=1)
        self.res1 = ResBlock(c)
        self.film = StrategyModulation(c, strategy_dim)
        self.res2 = ResBlock(c)
        # Texture branch: learn to combine high-pass and low-pass features
        self.hf_proj  = nn.Conv2d(3, c, 1)          # project Laplacian to c channels
        self.hf_fuse  = nn.Conv2d(c * 2, c, 1)
        self.dec      = nn.Conv2d(c, 3, 3, padding=1)

    @staticmethod
    def _laplacian(x: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian (high-pass) of x using a fixed 3x3 kernel."""
        kernel = torch.tensor(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
            dtype=torch.float32,
            device=x.device,
        ).view(1, 1, 3, 3).expand(3, 1, 3, 3)       # per-channel (depthwise)
        return F.conv2d(x, kernel, padding=1, groups=3)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        strategy_emb: torch.Tensor,
    ) -> torch.Tensor:
        # High-frequency component of the current output
        hf = self._laplacian(x).detach()             # (B, 3, H, W)

        feat     = F.elu(self.enc(x))
        feat     = self.res1(feat)
        feat     = self.film(feat, strategy_emb)
        feat     = self.res2(feat)

        hf_feat  = F.elu(self.hf_proj(hf))           # (B, c, H, W)
        fused    = self.hf_fuse(torch.cat([feat, hf_feat], dim=1))
        fused    = F.elu(fused)

        delta    = self.dec(fused) * mask             # apply inside hole
        return torch.tanh(x + delta)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTER_CLASSES = {
    0: GlobalAdapter,
    1: LocalAdapter,
    2: BoundaryAdapter,
    3: TextureAdapter,
}

ADAPTER_NAMES = {
    0: "Global",
    1: "Local",
    2: "Boundary",
    3: "Texture",
}


def build_adapters(strategy_dim: int = 64) -> nn.ModuleList:
    """Instantiate all 4 adapters and return as an nn.ModuleList.

    Index ordering matches action space: [Global, Local, Boundary, Texture].
    """
    return nn.ModuleList([
        cls(strategy_dim=strategy_dim)
        for cls in [GlobalAdapter, LocalAdapter, BoundaryAdapter, TextureAdapter]
    ])
