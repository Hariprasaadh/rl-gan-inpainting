import os
import argparse
from typing import Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.seed import seed_everything
from src.utils.logger import setup_logger, MetricLogger
from src.data.dataset import InpaintingDataset
from src.models.generator import InpaintingGenerator
from src.models.discriminator import SNPatchGANDiscriminator
from src.models.losses import MaskedL1Loss, PerceptualAndStyleLoss, HingeAdversarialLoss
from src.training.trainer_utils import load_config, get_device, save_checkpoint


def train_gan_baseline(config_path: str, override_epochs: Optional[int] = None) -> None:
    """Train GAN baseline (Experiment A) with neutral fixed strategy."""
    cfg = load_config(config_path)
    seed = cfg["training"].get("seed", 42)
    seed_everything(seed)

    device = get_device(cfg["training"].get("device", "cpu"))
    logger = setup_logger("pretrain_gan", log_file=os.path.join(cfg["training"]["log_dir"], "train.log"))
    metric_logger = MetricLogger(cfg["training"]["log_dir"])

    logger.info(f"Starting GAN baseline pretraining on device: {device}")

    # Dataset & DataLoader
    train_dataset = InpaintingDataset(
        image_dir=cfg["data"].get("image_dir"),
        mask_dir=cfg["data"].get("mask_dir"),
        image_size=cfg["data"].get("image_size", 128),
        split="train",
        is_train=True,
        synthetic_size=cfg["data"].get("synthetic_size", 100),
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["data"].get("batch_size", 8),
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 0),
        drop_last=True,
    )

    # Models
    generator = InpaintingGenerator(
        base_channels=cfg["model"].get("base_channels", 32),
        strategy_dim=cfg["model"].get("strategy_dim", 64),
        latent_dim=cfg["model"].get("latent_dim", 256),
    ).to(device)

    discriminator = SNPatchGANDiscriminator(
        base_channels=cfg["model"].get("base_channels", 32) * 2,
    ).to(device)

    # Losses
    l1_loss_fn = MaskedL1Loss(
        hole_weight=cfg["loss"].get("hole_weight", 6.0),
        valid_weight=cfg["loss"].get("valid_weight", 1.0),
    )
    perc_style_loss_fn = PerceptualAndStyleLoss(
        perceptual_weight=cfg["loss"].get("perceptual_weight", 0.05),
        style_weight=cfg["loss"].get("style_weight", 120.0),
    )
    adv_loss_fn = HingeAdversarialLoss(
        adv_weight=cfg["loss"].get("adv_weight", 0.001),
    )

    # Optimizers
    opt_g = torch.optim.Adam(
        generator.parameters(),
        lr=cfg["training"].get("lr_g", 0.0002),
        betas=(cfg["training"].get("beta1", 0.5), cfg["training"].get("beta2", 0.999)),
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=cfg["training"].get("lr_d", 0.0002),
        betas=(cfg["training"].get("beta1", 0.5), cfg["training"].get("beta2", 0.999)),
    )

    epochs = override_epochs if override_epochs is not None else cfg["training"].get("epochs", 5)
    global_step = 0

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        epoch_loss_g, epoch_loss_d = 0.0, 0.0

        pbar = tqdm(dataloader, desc=f"Epoch [{epoch}/{epochs}]")
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            masked_images = batch["masked_image"].to(device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            opt_d.zero_grad()
            with torch.no_grad():
                out = generator(masked_images, masks)
                fake_images = out["completed"]

            real_logits = discriminator(images, masks)
            fake_logits = discriminator(fake_images.detach(), masks)

            loss_d = adv_loss_fn.discriminator_loss(real_logits, fake_logits)
            loss_d.backward()
            opt_d.step()

            # ---------------------
            # Train Generator
            # ---------------------
            opt_g.zero_grad()
            out = generator(masked_images, masks)
            coarse_img = out["coarse"]
            refined_img = out["refined"]
            completed_img = out["completed"]

            # L1 losses
            loss_l1_coarse, _ = l1_loss_fn(coarse_img, images, masks)
            loss_l1_refined, _ = l1_loss_fn(refined_img, images, masks)
            loss_l1 = loss_l1_coarse + loss_l1_refined

            # Perceptual & style losses
            loss_perc, loss_style, _ = perc_style_loss_fn(completed_img, images)

            # Adversarial generator loss
            fake_logits_g = discriminator(completed_img, masks)
            loss_adv_g = adv_loss_fn.generator_loss(fake_logits_g)

            loss_g = loss_l1 + loss_perc + loss_style + loss_adv_g
            loss_g.backward()
            opt_g.step()

            epoch_loss_g += loss_g.item()
            epoch_loss_d += loss_d.item()
            global_step += 1

            pbar.set_postfix({
                "loss_g": f"{loss_g.item():.4f}",
                "loss_d": f"{loss_d.item():.4f}",
                "l1": f"{loss_l1.item():.4f}",
            })

            metric_logger.log_scalar("train/loss_g", loss_g.item(), global_step)
            metric_logger.log_scalar("train/loss_d", loss_d.item(), global_step)
            metric_logger.log_scalar("train/loss_l1", loss_l1.item(), global_step)

        avg_g = epoch_loss_g / len(dataloader)
        avg_d = epoch_loss_d / len(dataloader)
        logger.info(f"Epoch {epoch}/{epochs} complete — Avg Loss G: {avg_g:.4f}, Avg Loss D: {avg_d:.4f}")

        # Save checkpoint
        save_path = os.path.join(cfg["training"]["save_dir"], f"gan_epoch_{epoch:03d}.pt")
        save_checkpoint(
            {
                "epoch": epoch,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "opt_g_state_dict": opt_g.state_dict(),
                "opt_d_state_dict": opt_d.state_dict(),
                "config": cfg,
            },
            save_path,
            is_best=(epoch == epochs),
        )

    logger.info("GAN Baseline Pretraining completed successfully!")
    metric_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain GAN Baseline (Experiment A)")
    parser.add_argument("--config", type=str, default="configs/pretrain_gan.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    args = parser.parse_args()
    train_gan_baseline(args.config, args.epochs)
