import os
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFFCResNetBlock(nn.Module):
    """Fast Fourier Convolution style residual block for LaMa reference wrapper."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return out + res


class LaMaModel(nn.Module):
    """LaMa-style Inpainting Network (ResNet-FFC Architecture)."""

    def __init__(self, in_channels: int = 4, out_channels: int = 3, base_channels: int = 64, num_blocks: int = 6):
        super().__init__()
        # Encoder
        self.in_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, kernel_size=7),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        # Residual Blocks
        blocks = [SimpleFFCResNetBlock(base_channels * 4) for _ in range(num_blocks)]
        self.res_blocks = nn.Sequential(*blocks)

        # Decoder
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_channels, out_channels, kernel_size=7),
            nn.Tanh(),
        )

    def forward(self, masked_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([masked_image, mask], dim=1)
        x = self.in_conv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        out = self.out_conv(x)
        completed = masked_image + out * mask
        return torch.clamp(completed, -1.0, 1.0)


class LaMaInference:
    """Wrapper for LaMa model inference used as an external baseline comparison."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = LaMaModel().to(self.device).eval()

        if checkpoint_path and os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if "model" in state:
                state = state["model"]
            self.model.load_state_dict(state, strict=False)

        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def inpaint(self, masked_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run inpainting inference.

        Args:
            masked_image: (B, 3, H, W) in [-1, 1]
            mask: (B, 1, H, W) in {0, 1}

        Returns:
            completed_image: (B, 3, H, W) in [-1, 1]
        """
        if masked_image.dim() == 3:
            masked_image = masked_image.unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)

        masked_image = masked_image.to(self.device)
        mask = mask.to(self.device)
        return self.model(masked_image, mask)
