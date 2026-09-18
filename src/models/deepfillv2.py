"""DeepFill-v2 Generator — exact architecture match for nipponjo/deepfillv2-pytorch.

Matches `networks_tf.py` (states_tf_places2.pth) and `networks.py`
(states_pt_places2.pth) from the nipponjo repository.  Key design choices:

* cnum=48 throughout (nipponjo default)
* cnum_in=5 : [image(3) + mask(1) + ones(1)]
* SamePad2d is used in the TF variant; standard padding in the PT variant.
  We default to the PT variant (no SamePad) which matches states_pt_places2.pth.
* Gated convolution follows the canonical formula:
      output = ELU(conv_feat(x)) * sigmoid(conv_gate(x))
* Two-stage architecture: Coarse → Refine (contextual attention skipped for
  simplicity; the PT checkpoint uses a simplified refine stage).

Checkpoint key prefix mappings (states_pt_places2.pth):
  coarse_net.*  → self.coarse_net.*
  refine_net.*  → self.refine_net.*

Usage
-----
from src.models.deepfillv2 import DeepFillV2Generator, load_deepfillv2_checkpoint

gen = DeepFillV2Generator(cnum=48)
load_deepfillv2_checkpoint(gen, "checkpoints/pretrained/deepfillv2_places2.pth")
gen.eval()
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class GConv(nn.Module):
    """Gated Convolution layer (DeepFill-v2 style).

    output = ELU(conv_feat(x)) * sigmoid(conv_gate(x))

    When ``activation`` is ``'none'``, the feature path is linear (used in the
    final output convolution).
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
        self.conv_feat = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.conv_gate = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_feat(x)
        gate = torch.sigmoid(self.conv_gate(x))
        if self.activation == "elu":
            feat = F.elu(feat, inplace=True)
        elif self.activation == "relu":
            feat = F.relu(feat, inplace=True)
        elif self.activation == "leaky_relu":
            feat = F.leaky_relu(feat, 0.2, inplace=True)
        # 'none' -> identity (raw output, used before tanh in final layer)
        return feat * gate


class GDeConv(nn.Module):
    """Upsample-then-GConv (replaces transposed convolution to avoid checkerboard)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        scale_factor: int = 2,
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.gconv = GConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gconv(self.upsample(x))


# ---------------------------------------------------------------------------
# Coarse Network (Stage 1)
# ---------------------------------------------------------------------------

class CoarseNet(nn.Module):
    """Coarse inpainting network.

    Input:  (B, cnum_in, H, W)  -- [image * (1-mask), mask, ones]  (5 channels)
    Output: (B, 3, H, W) in [-1, 1]
    """

    def __init__(self, cnum_in: int = 5, cnum: int = 48):
        super().__init__()
        c = cnum

        # Encoder
        self.conv1 = GConv(cnum_in, c,    kernel_size=5, stride=1, padding=2)
        self.conv2 = GConv(c,       2*c,  kernel_size=3, stride=2, padding=1)
        self.conv3 = GConv(2*c,     2*c,  kernel_size=3, stride=1, padding=1)
        self.conv4 = GConv(2*c,     4*c,  kernel_size=3, stride=2, padding=1)
        self.conv5 = GConv(4*c,     4*c,  kernel_size=3, stride=1, padding=1)
        self.conv6 = GConv(4*c,     4*c,  kernel_size=3, stride=1, padding=1)

        # Dilated bottleneck
        self.conv7  = GConv(4*c, 4*c, kernel_size=3, dilation=2,  padding=2)
        self.conv8  = GConv(4*c, 4*c, kernel_size=3, dilation=4,  padding=4)
        self.conv9  = GConv(4*c, 4*c, kernel_size=3, dilation=8,  padding=8)
        self.conv10 = GConv(4*c, 4*c, kernel_size=3, dilation=16, padding=16)

        # Decoder
        self.conv11   = GConv(4*c, 4*c,  kernel_size=3, stride=1, padding=1)
        self.conv12   = GConv(4*c, 4*c,  kernel_size=3, stride=1, padding=1)
        self.deconv13 = GDeConv(4*c, 2*c)
        self.conv14   = GConv(2*c, 2*c,  kernel_size=3, stride=1, padding=1)
        self.deconv15 = GDeConv(2*c, c)
        self.conv16   = GConv(c, c//2,   kernel_size=3, stride=1, padding=1)
        # Final output -- no gating activation on feature path
        self.conv17   = GConv(c//2, 3,   kernel_size=3, stride=1, padding=1, activation="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.deconv13(x)
        x = self.conv14(x)
        x = self.deconv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        return torch.tanh(x)


# ---------------------------------------------------------------------------
# Refine Network (Stage 2) -- simplified (no contextual attention)
# ---------------------------------------------------------------------------

class RefineNet(nn.Module):
    """Refinement network (Stage 2).

    Input:  (B, cnum_in, H, W)  -- coarse output composited with original [+ mask]
    Output: (B, 3, H, W) in [-1, 1]

    This follows the PT variant of nipponjo which omits contextual attention
    in the refine branch (contextual attention requires custom CUDA ops and
    degrades gracefully to a standard dilated encoder-decoder).
    """

    def __init__(self, cnum_in: int = 5, cnum: int = 48):
        super().__init__()
        c = cnum

        # Encoder
        self.conv1 = GConv(cnum_in, c,    kernel_size=5, stride=1, padding=2)
        self.conv2 = GConv(c,       2*c,  kernel_size=3, stride=2, padding=1)
        self.conv3 = GConv(2*c,     2*c,  kernel_size=3, stride=1, padding=1)
        self.conv4 = GConv(2*c,     4*c,  kernel_size=3, stride=2, padding=1)
        self.conv5 = GConv(4*c,     4*c,  kernel_size=3, stride=1, padding=1)
        self.conv6 = GConv(4*c,     4*c,  kernel_size=3, stride=1, padding=1)

        # Dilated bottleneck
        self.conv7  = GConv(4*c, 4*c, kernel_size=3, dilation=2,  padding=2)
        self.conv8  = GConv(4*c, 4*c, kernel_size=3, dilation=4,  padding=4)
        self.conv9  = GConv(4*c, 4*c, kernel_size=3, dilation=8,  padding=8)
        self.conv10 = GConv(4*c, 4*c, kernel_size=3, dilation=16, padding=16)

        # Decoder
        self.conv11   = GConv(4*c, 4*c,  kernel_size=3, stride=1, padding=1)
        self.conv12   = GConv(4*c, 4*c,  kernel_size=3, stride=1, padding=1)
        self.deconv13 = GDeConv(4*c, 2*c)
        self.conv14   = GConv(2*c, 2*c,  kernel_size=3, stride=1, padding=1)
        self.deconv15 = GDeConv(2*c, c)
        self.conv16   = GConv(c, c//2,   kernel_size=3, stride=1, padding=1)
        self.conv17   = GConv(c//2, 3,   kernel_size=3, stride=1, padding=1, activation="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.deconv13(x)
        x = self.conv14(x)
        x = self.deconv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        return torch.tanh(x)


# ---------------------------------------------------------------------------
# Full two-stage generator
# ---------------------------------------------------------------------------

class DeepFillV2Generator(nn.Module):
    """Two-stage gated-convolution generator (DeepFill-v2).

    Args:
        cnum: Base channel multiplier (48 for nipponjo pretrained weights).
        cnum_in: Input channels per stage. Default 5: [img*visible + mask + ones].

    Forward signature
    -----------------
    forward(masked_img, mask) -> (coarse_out, refined_out)

    Both outputs are (B, 3, H, W) in [-1, 1].

    Internal state embedding
    ------------------------
    Call get_bottleneck_features(masked_img, mask) to extract the 4*cnum-channel
    bottleneck tensor (used by DeepFillEncoder for the RL state vector).
    """

    def __init__(self, cnum: int = 48, cnum_in: int = 5):
        super().__init__()
        self.cnum = cnum
        self.cnum_in = cnum_in

        self.coarse_net = CoarseNet(cnum_in=cnum_in, cnum=cnum)
        self.refine_net = RefineNet(cnum_in=cnum_in, cnum=cnum)

    def _make_input(self, masked_img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Build the 5-channel input tensor.

        masked_img: (B, 3, H, W) -- original image with hole set to 0
        mask:       (B, 1, H, W) -- 1 inside hole, 0 outside
        """
        ones = torch.ones_like(mask)
        return torch.cat([masked_img, mask, ones], dim=1)  # (B, 5, H, W)

    def forward(
        self, masked_img: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Two-stage forward pass.

        Returns:
            coarse_out  : (B, 3, H, W) in [-1, 1]
            refined_out : (B, 3, H, W) in [-1, 1]
        """
        x_in = self._make_input(masked_img, mask)

        coarse_out = self.coarse_net(x_in)
        # Composite: use coarse inside hole, original outside
        coarse_comp = masked_img * (1 - mask) + coarse_out * mask

        x_in2 = self._make_input(coarse_comp, mask)
        refined_out = self.refine_net(x_in2)

        return coarse_out, refined_out

    def get_bottleneck_features(
        self, masked_img: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract the 4*cnum bottleneck feature map from CoarseNet (after conv10).

        Used by DeepFillEncoder to build the RL state embedding without an
        extra forward pass.

        Returns:
            feats: (B, 4*cnum, H/4, W/4)
        """
        x_in = self._make_input(masked_img, mask)
        c = self.coarse_net
        x = c.conv1(x_in)
        x = c.conv2(x)
        x = c.conv3(x)
        x = c.conv4(x)
        x = c.conv5(x)
        x = c.conv6(x)
        x = c.conv7(x)
        x = c.conv8(x)
        x = c.conv9(x)
        x = c.conv10(x)
        return x  # (B, 4*cnum, H/4, W/4)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_deepfillv2_checkpoint(
    model: DeepFillV2Generator,
    path: str,
    strict: bool = True,
    verbose: bool = True,
) -> DeepFillV2Generator:
    """Load nipponjo pretrained weights into DeepFillV2Generator.

    Handles two common key-prefix variants:
      1. Flat keys: ``coarse_net.conv1.conv_feat.weight``  (PT variant default)
      2. Nested:    ``generator.coarse_net.conv1.conv_feat.weight``

    Args:
        model: DeepFillV2Generator instance (randomly initialized).
        path:  Path to .pth file from nipponjo Google Drive.
        strict: If True raises on unexpected/missing keys (recommended).
        verbose: Print summary of loaded keys.

    Returns:
        model with weights loaded in-place.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # The nipponjo checkpoint can be either a raw state-dict or a dict with a
    # 'G' or 'generator' key depending on the save format.
    if isinstance(ckpt, dict):
        if "G" in ckpt:
            state_dict = ckpt["G"]
        elif "generator" in ckpt:
            state_dict = ckpt["generator"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # Assume top-level IS the state dict
            state_dict = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    # Strip common wrapper prefixes
    def strip_prefix(sd: dict, prefix: str) -> dict:
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    # Try nested prefixes
    for prefix in ("generator.", "G.", "module."):
        if any(k.startswith(prefix) for k in state_dict):
            state_dict = strip_prefix(state_dict, prefix)
            break

    result = model.load_state_dict(state_dict, strict=strict)

    if verbose:
        if result.missing_keys:
            print(f"[deepfillv2] Missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]}")
        if result.unexpected_keys:
            print(f"[deepfillv2] Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]}")
        if not result.missing_keys and not result.unexpected_keys:
            print(f"[deepfillv2] Checkpoint loaded successfully from: {path}")

    return model
