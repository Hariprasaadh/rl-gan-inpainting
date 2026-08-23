from typing import Tuple
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class SNPatchGANDiscriminator(nn.Module):
    """Spectral-Normalized PatchGAN Discriminator.

    Takes 4-channel input: [image (3) ⊕ mask (1)].
    Outputs patch-level real/fake prediction logits.
    """

    def __init__(self, in_channels: int = 4, base_channels: int = 64):
        super().__init__()
        self.base_channels = base_channels

        self.net = nn.Sequential(
            # Stage 1: (B, 4, H, W) -> (B, base_channels, H/2, W/2)
            spectral_norm(nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 2: -> (B, base_channels * 2, H/4, W/4)
            spectral_norm(nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 3: -> (B, base_channels * 4, H/8, W/8)
            spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 4: -> (B, base_channels * 8, H/16, W/16)
            spectral_norm(nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 5: -> (B, 1, H/16, W/16)
            spectral_norm(nn.Conv2d(base_channels * 8, 1, kernel_size=3, stride=1, padding=1)),
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute patch logits.

        Args:
            image: (B, 3, H, W) in [-1, 1]
            mask: (B, 1, H, W) in {0, 1}

        Returns:
            logits: (B, 1, H/16, W/16)
        """
        x = torch.cat([image, mask], dim=1)
        return self.net(x)

    def get_confidence(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute scalar realism confidence in [0.0, 1.0] for the RL reward signal."""
        with torch.no_grad():
            logits = self.forward(image, mask)
            probs = torch.sigmoid(logits)
            # Average over patches: (B,)
            return probs.mean(dim=[1, 2, 3])
