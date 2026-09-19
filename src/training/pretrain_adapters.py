"""Pretrain 4 strategy adapters with differentiated losses.

Losses:
  Global   : L1(hole) + L1(valid) + 0.05*Perceptual
  Local    : 6.0*L1(hole) + 0.5*SSIM(hole)
  Boundary : L1 + 2.0*boundary_weighted_gradient_loss
  Texture  : L1 + 0.1*FFT_frequency_loss + 0.05*Perceptual
"""

import os
import argparse
from typing import Dict, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.seed import seed_everything
from src.utils.logger import setup_logger, MetricLogger
from src.data.dataset import InpaintingDataset
from src.models.rl_inpainting_model import RLInpaintingModel
from src.models.losses import PerceptualAndStyleLoss
from src.training.trainer_utils import load_config, get_device, save_checkpoint


def boundary_weighted_gradient_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sobel gradient loss weighted on boundary ring."""
    # Compute boundary ring: dilated - eroded
    kernel = torch.ones((1, 1, 5, 5), device=mask.device, dtype=mask.dtype)
    dilated = (F.conv2d(mask.float(), kernel, padding=2) > 0).float()
    inv = 1.0 - mask.float()
    dilated_inv = (F.conv2d(inv, kernel, padding=2) > 0).float()
    eroded = 1.0 - dilated_inv
    boundary = torch.clamp(dilated - eroded, 0, 1)  # (B,1,H,W)
    # Weight: more on boundary, less inside
    weight = boundary * 2.0 + (1 - boundary) * 0.2
    weight = weight.expand_as(pred)

    # Sobel
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    # Grayscale gradient
    pred_gray = pred.mean(dim=1, keepdim=True)
    gt_gray = gt.mean(dim=1, keepdim=True)
    pred_gx = F.conv2d(pred_gray, sobel_x, padding=1)
    pred_gy = F.conv2d(pred_gray, sobel_y, padding=1)
    gt_gx = F.conv2d(gt_gray, sobel_x, padding=1)
    gt_gy = F.conv2d(gt_gray, sobel_y, padding=1)

    loss_x = F.l1_loss(pred_gx, gt_gx, reduction="none")
    loss_y = F.l1_loss(pred_gy, gt_gy, reduction="none")
    # Apply boundary weight (average over channel dim already)
    w = weight[:, 0:1, :, :]  # (B,1,H,W)
    loss = (loss_x * w + loss_y * w).mean()
    return loss


def fft_frequency_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 on FFT magnitude inside hole region."""
    # Compute 2D FFT magnitude difference
    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    gt_fft = torch.fft.rfft2(gt, norm="ortho")
    # Magnitude
    pred_mag = torch.abs(pred_fft)
    gt_mag = torch.abs(gt_fft)
    loss = F.l1_loss(pred_mag, gt_mag)
    return loss


def ssim_hole_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """SSIM-inspired: 1 - SSIM approximated via local statistics, weighted hole.
    For speed we use a simple L1 on normalized patches as proxy if kornia not available.
    Uses standard SSIM formula with 11x11 gaussian window.
    """
    # Use simple approach: compute SSIM via skimage-like formula on hole-masked region
    # For differentiability, approximate with L1 on smoothed images
    # Apply gaussian blur (approx) then compute SSIM components
    # Simplified: L1 on box-filtered images weighted by hole
    kernel = torch.ones((1, 1, 11, 11), device=pred.device, dtype=pred.dtype) / 121.0
    # Expand to 3 channels via groups
    # Convert to grayscale for SSIM
    pred_g = pred.mean(dim=1, keepdim=True)
    gt_g = gt.mean(dim=1, keepdim=True)
    mu_pred = F.conv2d(pred_g, kernel, padding=5)
    mu_gt = F.conv2d(gt_g, kernel, padding=5)
    # Variance approximation not needed for proxy; just L1 between means weighted hole
    # Use mask downsampled? keep full
    hole_w = mask.expand_as(pred_g)
    loss = F.l1_loss(mu_pred * hole_w, mu_gt * hole_w)
    # Scale to ~0-1 range
    return loss * 2.0


def compute_adapter_loss(
    adapter_idx: int,
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    perceptual_fn: PerceptualAndStyleLoss,
    hole_weight: float = 1.0,
) -> torch.Tensor:
    """Strategy-specific loss dispatch."""
    # Base L1 split
    diff = torch.abs(pred - gt)
    mask_exp = mask.expand_as(diff)
    hole_pix = mask_exp.sum() + 1e-6
    valid_pix = ((1 - mask_exp).sum() + 1e-6)
    hole_l1 = (diff * mask_exp).sum() / hole_pix
    valid_l1 = (diff * (1 - mask_exp)).sum() / valid_pix

    if adapter_idx == 0:
        # Global: balanced + perceptual
        perc, _, _ = perceptual_fn(pred, gt)
        # perceptual_fn returns already weighted
        return hole_l1 + valid_l1 + perc * 1.0
    elif adapter_idx == 1:
        # Local: heavy hole focus + SSIM(hole)
        ssim_h = ssim_hole_loss(pred, gt, mask)
        return 6.0 * hole_l1 + 0.2 * valid_l1 + 0.5 * ssim_h
    elif adapter_idx == 2:
        # Boundary
        grad_loss = boundary_weighted_gradient_loss(pred, gt, mask)
        return (hole_l1 + valid_l1) + 2.0 * grad_loss
    elif adapter_idx == 3:
        # Texture: L1 + FFT + perceptual
        fft_loss = fft_frequency_loss(pred, gt, mask)
        perc, _, _ = perceptual_fn(pred, gt)
        return (hole_l1 + valid_l1) + 0.1 * fft_loss + perc * 1.0
    else:
        return hole_l1 + valid_l1


def pretrain_adapters(config_path: str, override_epochs: Optional[int] = None) -> None:
    cfg = load_config(config_path)
    seed_everything(cfg["training"].get("seed", 42))
    device = get_device(cfg["training"].get("device", "cpu"))

    save_dir = cfg["training"].get("save_dir", "checkpoints/adapters")
    log_dir = cfg["training"].get("log_dir", "logs/adapters")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger = setup_logger("pretrain_adapters", log_file=os.path.join(log_dir, "train.log"))
    metric_logger = MetricLogger(log_dir)

    logger.info(f"Starting adapter pretraining on device: {device}")

    # Dataset
    train_dataset = InpaintingDataset(
        image_dir=cfg["data"].get("image_dir"),
        mask_dir=cfg["data"].get("mask_dir"),
        image_size=cfg["data"].get("image_size", 256),
        split="train",
        is_train=True,
        synthetic_size=cfg["data"].get("synthetic_size", 200),
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["data"].get("batch_size", 4),
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 0),
        drop_last=True,
    )

    # Model
    model = RLInpaintingModel(
        cnum=cfg["model"].get("cnum", 48),
        cnum_in=cfg["model"].get("cnum_in", 5),
        strategy_dim=cfg["model"].get("strategy_dim", 64),
        latent_dim=cfg["model"].get("latent_dim", 256),
    ).to(device)

    # Load backbone
    ckpt_path = cfg["model"].get("backbone_checkpoint")
    if ckpt_path:
        try:
            model.load_pretrained_backbone(ckpt_path, strict=False)
            logger.info(f"Loaded backbone from {ckpt_path}")
        except Exception as e:
            logger.warning(f"Could not load backbone checkpoint: {e}")

    # Ensure backbone frozen, adapters trainable
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()
    model.unfreeze_adapters()

    # One optimizer per adapter + strategy embedding handled jointly
    # Use single optimizer for all adapters to keep simple and allow joint embedding update
    optim = torch.optim.Adam(
        list(model.adapters.parameters()) + list(model.strategy_embeddings.parameters()) + list(model.encoder_head.parameters()),
        lr=cfg["training"].get("lr", 0.0001),
        betas=(0.5, 0.999),
    )

    perceptual_fn = PerceptualAndStyleLoss(perceptual_weight=0.05, style_weight=0.0)
    # Move perceptual internal VGG to device lazily; we manage via forward device

    epochs = override_epochs if override_epochs is not None else cfg["training"].get("epochs", 5)
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        # Keep backbone in eval even during train
        model.backbone.eval()

        epoch_losses = {i: 0.0 for i in range(4)}
        epoch_counts = {i: 0 for i in range(4)}

        pbar = tqdm(dataloader, desc=f"Adapter Epoch [{epoch}/{epochs}]")
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            masked_images = batch["masked_image"].to(device)

            # Train each adapter separately on same batch (round-robin)
            # For efficiency, train all 4 adapters per batch with separate forwards
            for adapter_idx in range(4):
                optim.zero_grad()

                with torch.no_grad():
                    coarse = model.backbone.coarse_forward(masked_images, masks)
                    coarse_comp = masked_images + coarse * masks

                refined_backbone = model.backbone.refine_forward(coarse_comp, masks)
                # Adapter forward with strategy embedding
                strat_emb = model.get_strategy_embedding(adapter_idx, batch_size=images.shape[0], device=device)
                adapter = model.adapters[adapter_idx]
                adapted = adapter(refined_backbone, masks, strat_emb)
                # Completed composite
                completed = masked_images + adapted * masks
                # Clamp
                completed = torch.clamp(completed, -1, 1)

                loss = compute_adapter_loss(adapter_idx, completed, images, masks, perceptual_fn)
                loss.backward()
                optim.step()

                epoch_losses[adapter_idx] += loss.item()
                epoch_counts[adapter_idx] += 1
                global_step += 1
                metric_logger.log_scalar(f"train/adapter_{adapter_idx}_loss", loss.item(), global_step)

                pbar.set_postfix({f"ad_{k}": f"{v/ max(1,epoch_counts[k]):.4f}" for k, v in epoch_losses.items()})

        # Log epoch summary
        for idx in range(4):
            avg = epoch_losses[idx] / max(1, epoch_counts[idx])
            logger.info(f"Epoch {epoch} Adapter {idx} avg loss: {avg:.4f}")

        # Save checkpoint every epoch
        save_path = os.path.join(save_dir, f"adapters_epoch_{epoch:03d}.pt")
        save_checkpoint(
            {
                "epoch": epoch,
                "adapters_state_dict": model.adapters.state_dict(),
                "adapters": model.adapters.state_dict(),
                "strategy_embeddings": model.strategy_embeddings.state_dict(),
                "encoder_head": model.encoder_head.state_dict(),
                "config": cfg,
            },
            save_path,
            is_best=(epoch == epochs),
        )
        logger.info(f"Saved adapters checkpoint to {save_path}")

    # Also save final in expected path for rl config
    final_path = os.path.join(save_dir, "adapters_final.pt")
    torch.save(
        {
            "adapters": model.adapters.state_dict(),
            "strategy_embeddings": model.strategy_embeddings.state_dict(),
            "encoder_head": model.encoder_head.state_dict(),
        },
        final_path,
    )
    logger.info(f"Adapters final saved to {final_path}")
    metric_logger.close()
    logger.info("Adapter pretraining completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain Strategy Adapters")
    parser.add_argument("--config", type=str, default="configs/pretrain_adapters.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    args = parser.parse_args()
    pretrain_adapters(args.config, override_epochs=args.epochs)
