from typing import Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedConv2d(nn.Module):
    """Gated 2D Convolution layer (DeepFill v2 style).

    Applies feature convolution and gating convolution in parallel:
        output = activation(feat_conv(x)) * sigmoid(gate_conv(x))
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        activation: str = "elu",
    ):
        super().__init__()
        self.feature_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.gating_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        if activation == "elu":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "none":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_conv(x)
        feat = self.activation(feat)
        gate = torch.sigmoid(self.gating_conv(x))
        return feat * gate


class InpaintingEncoder(nn.Module):
    """Shared 4-channel encoder backbone.

    Processes [masked_image (3) ⊕ mask (1)] and extracts multi-scale features
    along with a pooled state representation vector (256-dim) for the RL agent.
    """

    def __init__(self, in_channels: int = 4, base_channels: int = 64, latent_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim

        # Multi-scale downsampling stages
        self.stage1 = nn.Sequential(
            GatedConv2d(in_channels, base_channels, kernel_size=5, stride=1, padding=2),
            GatedConv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),  # /2
        )
        self.stage2 = nn.Sequential(
            GatedConv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1),
            GatedConv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1),  # /4
        )
        self.stage3 = nn.Sequential(
            GatedConv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1),
            GatedConv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1),  # /8
        )

        # Pooled state embedding head for RL controller
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.state_head = nn.Sequential(
            nn.Linear(base_channels * 4, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ELU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, masked_image: torch.Tensor, mask: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Forward pass.

        Args:
            masked_image: (B, 3, H, W) in [-1, 1]
            mask: (B, 1, H, W) in {0, 1}

        Returns:
            features: List of multi-scale feature tensors [f1, f2, f3]
            state_embedding: (B, latent_dim) pooled feature vector for RL agent
        """
        x = torch.cat([masked_image, mask], dim=1)  # (B, 4, H, W)

        f1 = self.stage1(x)   # (B, 64, H/2, W/2)
        f2 = self.stage2(f1)  # (B, 128, H/4, W/4)
        f3 = self.stage3(f2)  # (B, 256, H/8, W/8)

        pooled = self.pool(f3).flatten(1)
        state_embedding = self.state_head(pooled)

        return [f1, f2, f3], state_embedding
