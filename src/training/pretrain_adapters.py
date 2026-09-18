"""Adapter Pretraining — Stage 2 of the rebuild pipeline.

Pretrain all 4 strategy adapters independently on COCO val2017.
Each adapter has a strategy-specific loss function designed to produce
genuinely different specialisations.

Loss functions
--------------
  Action 0  GlobalAdapter   : L1(full) + 0.05 * Perceptual
  Action 1  LocalAdapter    : 6.0 * L1(hole) + 0.5 * SSIM(hole)
  Action 2  BoundaryAdapter : L1(full) + 2.0 * BoundaryGradientLoss
  Action 3  TextureAdapter  : L1(full) + 0.1 * FFTFrequencyLoss + 0.05 * Perceptual

Run
---
  python main.py pretrain-adapters --config configs/pretrain_adapters.yaml

Or directly:
  python src/training/pretrain_adapters.py --config configs/pretrain_adapters.yaml
"""

from __future__ import annotations

import os
import argparse
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

try:
    import lpips as _lpips_lib
    _HAS_LPIPS = True
except ImportError:
    _HAS_LPIPS = False

from src.models.rl_inpainting_model import RLInpaintingModel
from src.models.strategy_adapters import ADAPTER_NAMES
from src.data.dataset import InpaintingDataset
from src.training.trainer_utils import load_config, get_device


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

_LPIPS_NET: Optional[object] = None


def get_lpips(device: torch.device) -> Optional[object]:
    global _LPIPS_NET
    if not _HAS_LPIPS:
        return None
    if _LPIPS_NET is None:
        _LPIPS_NET = _lpips_lib.LPIPS(net="vgg", verbose=False).to(device).eval()
        for p in _LPIPS_NET.parameters():
            p.requires_grad = False
    return _LPIPS_NET


def perceptual_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """LPIPS perceptual loss (VGG). Falls back to MSE if lpips unavailable."""
    lpips_net = get_lpips(device)
    if lpips_net is None:
        return F.mse_loss(pred, target)
    with torch.no_grad():
        dist = lpips_net(pred, target)
    return dist.mean()


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Simple windowed SSIM loss (1 - SSIM), computed in [0,1] space."""
    # Convert [-1,1] -> [0,1]
    p = (pred + 1) / 2
    t = (target + 1) / 2

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    kernel = torch.ones(1, 1, 11, 11, device=pred.device) / 121.0
    # Per-channel
    mu_p  = F.conv2d(p.mean(1, keepdim=True), kernel, padding=5)
    mu_t  = F.conv2d(t.mean(1, keepdim=True), kernel, padding=5)
    sig_p = F.conv2d((p.mean(1, keepdim=True))**2, kernel, padding=5) - mu_p**2
    sig_t = F.conv2d((t.mean(1, keepdim=True))**2, kernel, padding=5) - mu_t**2
    sig_pt= F.conv2d(p.mean(1, keepdim=True)*t.mean(1, keepdim=True), kernel, padding=5) - mu_p*mu_t

    ssim_map = ((2*mu_p*mu_t + c1)*(2*sig_pt + c2)) / ((mu_p**2 + mu_t**2 + c1)*(sig_p + sig_t + c2))
    return 1 - ssim_map.mean()


def boundary_gradient_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    dilation: int = 7,
) -> torch.Tensor:
    """Gradient-magnitude loss focused on the mask boundary zone."""
    # Build boundary map
    kernel = torch.ones(1, 1, dilation, dilation, device=mask.device)
    pad    = dilation // 2
    dilated = F.conv2d(mask.float(), kernel, padding=pad).clamp(0, 1)
    eroded_inv = F.conv2d((1 - mask).float(), kernel, padding=pad).clamp(0, 1)
    boundary = (dilated * eroded_inv).detach()          # (B, 1, H, W)

    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                            dtype=torch.float32, device=pred.device).view(1,1,3,3)
    sobel_y = sobel_x.transpose(2, 3)

    def grad_mag(img: torch.Tensor) -> torch.Tensor:
        gray = img.mean(1, keepdim=True)
        gx   = F.conv2d(gray, sobel_x, padding=1)
        gy   = F.conv2d(gray, sobel_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    pred_grad   = grad_mag(pred)
    target_grad = grad_mag(target)
    loss = (boundary * (pred_grad - target_grad).abs()).sum() / (boundary.sum() + 1e-6)
    return loss


def fft_frequency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 distance in the 2D FFT magnitude spectrum (frequency domain loss)."""
    # Works on float32 images in [-1, 1]
    pred_fft   = torch.fft.fft2(pred, norm="ortho")
    target_fft = torch.fft.fft2(target, norm="ortho")
    pred_mag   = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)
    return F.l1_loss(pred_mag, target_mag)


# ---------------------------------------------------------------------------
# Per-adapter loss functions
# ---------------------------------------------------------------------------

def loss_global(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Action 0 — GlobalAdapter loss."""
    l1   = F.l1_loss(pred, target)
    perc = perceptual_loss(pred, target, device)
    return l1 + 0.05 * perc


def loss_local(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Action 1 — LocalAdapter loss (heavy hole focus)."""
    # Hole region only
    hole_pred   = pred * mask
    hole_target = target * mask
    num_pixels  = mask.sum().clamp(min=1.0)

    l1_hole   = (hole_pred - hole_target).abs().sum() / num_pixels
    ssim_hole = ssim_loss(
        pred * mask + target * (1 - mask),   # only hole region varies
        target,
    )
    return 6.0 * l1_hole + 0.5 * ssim_hole


def loss_boundary(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Action 2 — BoundaryAdapter loss."""
    l1  = F.l1_loss(pred, target)
    bgl = boundary_gradient_loss(pred, target, mask)
    return l1 + 2.0 * bgl


def loss_texture(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Action 3 — TextureAdapter loss."""
    l1   = F.l1_loss(pred, target)
    fft  = fft_frequency_loss(pred, target)
    perc = perceptual_loss(pred, target, device)
    return l1 + 0.1 * fft + 0.05 * perc


LOSS_FNS = {
    0: loss_global,
    1: loss_local,
    2: loss_boundary,
    3: loss_texture,
}


# ---------------------------------------------------------------------------
# Dataset wrapper for adapter pretraining
# ---------------------------------------------------------------------------

class AdapterPretrainDataset(torch.utils.data.Dataset):
    """Wraps InpaintingDataset to return dicts compatible with this script."""

    def __init__(self, image_dir: str, image_size: int = 256):
        self.dataset = InpaintingDataset(
            image_dir=image_dir,
            image_size=image_size,
            is_train=True,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        return {
            "image":        sample["image"],         # (3, H, W) in [-1, 1]
            "mask":         sample["mask"],           # (1, H, W) in {0, 1}
            "masked_image": sample["masked_image"],   # (3, H, W)
        }


# ---------------------------------------------------------------------------
# Train one adapter
# ---------------------------------------------------------------------------

def pretrain_single_adapter(
    action: int,
    model: RLInpaintingModel,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-4,
    save_dir: str = "checkpoints/adapters",
    log_every: int = 50,
) -> None:
    """Pretrain a single adapter for `epochs` epochs."""
    adapter   = model.adapters[action]
    loss_fn   = LOSS_FNS[action]
    name      = ADAPTER_NAMES[action]
    strat_idx = torch.tensor([action], dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(dataloader)
    )

    model.backbone.eval()
    adapter.train()

    print(f"\n{'='*60}")
    print(f"  Pretraining Action {action}: {name}Adapter  ({epochs} epochs)")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(dataloader, 1):
            img    = batch["image"].to(device)
            mask   = batch["mask"].to(device)
            masked = batch["masked_image"].to(device)

            with torch.no_grad():
                # Backbone full forward (frozen)
                ones   = torch.ones_like(mask)
                x_in   = torch.cat([masked, mask, ones], dim=1)
                coarse = model.backbone.coarse_net(x_in)
                comp   = masked * (1 - mask) + coarse * mask
                x_in2  = torch.cat([comp, mask, ones], dim=1)
                backbone_refined = model.backbone.refine_net(x_in2)

            # Strategy embedding
            b = img.shape[0]
            strat_emb = model.strategy_embeddings(
                strat_idx.expand(b)
            )

            pred   = adapter(backbone_refined, mask, strat_emb)
            loss   = loss_fn(pred, img, mask, device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if step % log_every == 0:
                avg = epoch_loss / step
                elapsed = time.time() - t0
                print(
                    f"  [{name}] Epoch {epoch}/{epochs}  "
                    f"Step {step}/{len(dataloader)}  "
                    f"Loss={avg:.4f}  "
                    f"Time={elapsed:.1f}s"
                )

        epoch_loss /= len(dataloader)
        print(f"  [{name}] Epoch {epoch} done — avg loss: {epoch_loss:.4f}")

    print(f"  [{name}] Pretraining complete.")


# ---------------------------------------------------------------------------
# Main pretraining loop
# ---------------------------------------------------------------------------

def pretrain_adapters(config_path: str) -> None:
    """Pretrain all 4 adapters sequentially."""
    cfg    = load_config(config_path)
    device = get_device(cfg["training"].get("device", "cpu"))

    # Build model
    model = RLInpaintingModel(
        cnum        = cfg["model"].get("cnum", 48),
        strategy_dim= cfg["model"].get("strategy_dim", 64),
        latent_dim  = cfg["model"].get("latent_dim", 256),
    ).to(device)

    # Load pretrained backbone
    backbone_ckpt = cfg["model"]["backbone_checkpoint"]
    model.load_pretrained_backbone(backbone_ckpt)
    model.backbone.eval()

    # Dataset
    ds = AdapterPretrainDataset(
        image_dir  = cfg["data"]["image_dir"],
        image_size = cfg["data"].get("image_size", 256),
    )
    loader = DataLoader(
        ds,
        batch_size  = cfg["data"].get("batch_size", 4),
        shuffle     = True,
        num_workers = cfg["data"].get("num_workers", 2),
        pin_memory  = device.type == "cuda",
        drop_last   = True,
    )

    save_dir = cfg["training"].get("save_dir", "checkpoints/adapters")
    os.makedirs(save_dir, exist_ok=True)

    epochs = cfg["training"].get("epochs", 5)
    lr     = cfg["training"].get("lr", 1e-4)

    # Pretrain each adapter independently
    for action in range(4):
        pretrain_single_adapter(
            action     = action,
            model      = model,
            dataloader = loader,
            device     = device,
            epochs     = epochs,
            lr         = lr,
            save_dir   = save_dir,
        )

    # Save all adapters together
    final_path = os.path.join(save_dir, "adapters_final.pt")
    model.save_adapters(final_path)
    print(f"\nAll adapters saved to: {final_path}")
    print("Run: python main.py verify-strategies   to validate specialisation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain 4 Strategy Adapters")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain_adapters.yaml",
        help="Path to adapter pretraining config",
    )
    args = parser.parse_args()
    pretrain_adapters(args.config)
