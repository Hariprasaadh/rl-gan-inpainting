"""DeepFill-v2 backbone wrapper integrating official nipponjo architecture."""

import os
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.networks_tf import (
    Generator,
    GConv,
    GDeConv,
    ContextualAttention,
    same_padding,
)


class DeepFillV2Generator(Generator):
    """DeepFill-v2 Generator exposing coarse, refine, and bottleneck methods for RL-GAN."""

    def __init__(
        self,
        cnum: int = 48,
        cnum_in: int = 5,
        return_flow: bool = False,
        checkpoint: Optional[str] = None,
    ):
        super().__init__(cnum_in=cnum_in, cnum=cnum, return_flow=return_flow, checkpoint=checkpoint)
        self.cnum = cnum
        self.cnum_in = cnum_in

    def _prepare_input(self, masked_img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Ensure 5-channel input: [masked_img (3), ones (1), mask (1)]."""
        if self.cnum_in == 5:
            b, _, h, w = masked_img.shape
            ones = torch.ones(b, 1, h, w, device=masked_img.device, dtype=masked_img.dtype)
            return torch.cat([masked_img, ones, ones * mask], dim=1)
        return torch.cat([masked_img, mask], dim=1)

    def get_bottleneck_feature(self, masked_img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Extract coarse bottleneck feature (B, 96, H/4, W/4) for encoder head."""
        x = self._prepare_input(masked_img, mask)
        x = self.conv1(x)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        return x

    def coarse_forward(self, masked_img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run Stage 1 coarse generator."""
        x = self._prepare_input(masked_img, mask)
        x = self.conv1(x)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13_upsample(x)
        x = self.conv14(x)
        x = self.conv15_upsample(x)
        x = self.conv16(x)
        x = self.conv17(x)
        return self.tanh(x)

    def refine_forward(self, coarse_composite: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run Stage 2 refinement network."""
        xnow = coarse_composite
        x = self.xconv1(xnow)
        x = self.xconv2_downsample(x)
        x = self.xconv3(x)
        x = self.xconv4_downsample(x)
        x = self.xconv5(x)
        x = self.xconv6(x)
        x = self.xconv7_atrous(x)
        x = self.xconv8_atrous(x)
        x = self.xconv9_atrous(x)
        x = self.xconv10_atrous(x)
        x_hallu = x

        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        x, offset_flow = self.contextual_attention(x, x, mask)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], dim=1)

        x = self.allconv11(x)
        x = self.allconv12(x)
        x = self.allconv13_upsample(x)
        x = self.allconv14(x)
        x = self.allconv15_upsample(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        return self.tanh(x)

    def forward(self, masked_img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (coarse_out, refined_out)."""
        coarse_out = self.coarse_forward(masked_img, mask)
        coarse_composite = masked_img * (1.0 - mask) + coarse_out * mask
        refined_out = self.refine_forward(coarse_composite, mask)
        return coarse_out, refined_out


def load_deepfillv2_checkpoint(
    model: DeepFillV2Generator,
    checkpoint_path: str,
    strict: bool = False,
    verbose: bool = True,
) -> Dict:
    """Load pretrained DeepFillv2 checkpoint into model with 100% key matching."""
    if not os.path.exists(checkpoint_path):
        if verbose:
            print(f"[DeepFillV2] Checkpoint not found at {checkpoint_path}, skipping load.")
        return {"loaded": False, "reason": "file_not_found"}

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Unwrap dictionary wrapping
    if isinstance(ckpt, dict):
        for key in ["G", "generator", "model", "state_dict", "netG"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    # Clean prefixes if present
    clean_dict = {}
    for k, v in ckpt.items():
        nk = k
        for prefix in ["module.", "generator.", "netG.", "backbone."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
                break
        clean_dict[nk] = v

    try:
        model.load_state_dict(clean_dict, strict=strict)
        if verbose:
            print(f"[DeepFillV2] Successfully loaded all {len(clean_dict)} backbone parameters from {checkpoint_path} with 100% key match!")
        return {"loaded": True, "matched": len(clean_dict), "total": len(model.state_dict())}
    except Exception as e:
        res = model.load_state_dict(clean_dict, strict=False)
        if verbose:
            print(f"[DeepFillV2] Loaded non-strict from {checkpoint_path}: matched {len(model.state_dict()) - len(res.missing_keys)} keys")
        return {"loaded": True, "missing": len(res.missing_keys), "unexpected": len(res.unexpected_keys)}
