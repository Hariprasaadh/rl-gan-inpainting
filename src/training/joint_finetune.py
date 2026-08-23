import os
import argparse
from typing import Optional
import torch
from torch.utils.data import DataLoader
from stable_baselines3 import PPO
from tqdm import tqdm

from src.utils.seed import seed_everything
from src.utils.logger import setup_logger, MetricLogger
from src.data.dataset import InpaintingDataset
from src.models.generator import InpaintingGenerator
from src.models.discriminator import SNPatchGANDiscriminator
from src.models.losses import MaskedL1Loss, HingeAdversarialLoss
from src.rl.env_bandit import InpaintingBanditEnv
from src.training.trainer_utils import load_config, get_device, save_checkpoint, load_checkpoint


def train_joint_finetune(
    config_path: str = "configs/joint_finetune.yaml",
    override_epochs: Optional[int] = None,
) -> None:
    """Run optional Experiment E: Alternating Generator Fine-Tuning and PPO updates with stability guards."""
    cfg = load_config(config_path)
    seed = cfg["training"].get("seed", 42)
    seed_everything(seed)
    device = get_device(cfg["training"].get("device", "cpu"))

    os.makedirs(cfg["training"]["save_dir"], exist_ok=True)
    os.makedirs(cfg["training"]["log_dir"], exist_ok=True)
    logger = setup_logger("joint_finetune", log_file=os.path.join(cfg["training"]["log_dir"], "train.log"))
    metric_logger = MetricLogger(cfg["training"]["log_dir"])

    logger.info("=== Starting Experiment E: Joint GAN + RL Fine-Tuning (Optional / Advanced) ===")

    dataset = InpaintingDataset(
        image_dir=cfg["data"].get("image_dir"),
        mask_dir=cfg["data"].get("mask_dir"),
        image_size=cfg["data"].get("image_size", 128),
        is_train=True,
        synthetic_size=cfg["data"].get("synthetic_size", 200),
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    generator = InpaintingGenerator(
        base_channels=cfg["model"].get("base_channels", 32),
        strategy_dim=cfg["model"].get("strategy_dim", 64),
        latent_dim=cfg["model"].get("latent_dim", 256),
    ).to(device)

    discriminator = SNPatchGANDiscriminator(
        base_channels=cfg["model"].get("base_channels", 32) * 2,
    ).to(device)

    # Load pretrained weights
    ckpt_path = cfg["model"].get("checkpoint_gan")
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = load_checkpoint(ckpt_path, device=device)
        generator.load_state_dict(ckpt["generator_state_dict"], strict=False)
        discriminator.load_state_dict(ckpt["discriminator_state_dict"], strict=False)

    l1_fn = MaskedL1Loss(hole_weight=6.0, valid_weight=1.0)
    adv_fn = HingeAdversarialLoss(adv_weight=0.001)

    opt_g = torch.optim.Adam(generator.parameters(), lr=cfg["training"].get("lr_g", 0.00005))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg["training"].get("lr_d", 0.00005))

    env = InpaintingBanditEnv(
        dataset=dataset,
        generator=generator,
        discriminator=discriminator,
        device=str(device),
        compute_lpips=False,
    )

    rl_path = cfg["model"].get("checkpoint_rl")
    if rl_path and os.path.exists(rl_path):
        ppo_agent = PPO.load(rl_path, env=env, device=device)
    else:
        ppo_agent = PPO("MlpPolicy", env, n_steps=64, batch_size=32, gamma=0.0, seed=seed)

    epochs = override_epochs if override_epochs is not None else cfg["training"].get("epochs", 3)
    rl_steps = cfg["training"].get("rl_steps_per_epoch", 256)
    max_div_loss = cfg["training"].get("max_divergence_loss", 25.0)

    for epoch in range(1, epochs + 1):
        logger.info(f"\n--- Joint Epoch {epoch}/{epochs} ---")

        # Phase 1: Train Generator & Discriminator with RL-guided policy actions
        generator.train()
        discriminator.train()
        epoch_g_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch} - GAN Step"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            masked_images = batch["masked_image"].to(device)

            # Get RL suggested action for batch item
            with torch.no_grad():
                latent_emb = generator.extract_state_embedding(masked_images, masks)
                obs = env.state_builder.build_state(latent_emb[0], batch["stats"])
                action, _ = ppo_agent.predict(obs, deterministic=False)

            # Generator Forward
            opt_g.zero_grad()
            out = generator(masked_images, masks, strategy=int(action))
            loss_l1, _ = l1_fn(out["completed"], images, masks)

            # Discriminator check
            fake_logits = discriminator(out["completed"], masks)
            loss_adv = adv_fn.generator_loss(fake_logits)
            loss_g = loss_l1 + loss_adv

            # Stability Guardrail
            if torch.isnan(loss_g) or loss_g.item() > max_div_loss:
                logger.warning(f"Loss instability detected ({loss_g.item():.4f})! Skipping update.")
                continue

            loss_g.backward()
            opt_g.step()
            epoch_g_loss += loss_g.item()

        # Phase 2: Update RL Agent with updated Generator
        logger.info(f"Phase 2: Updating PPO controller ({rl_steps} steps)...")
        generator.eval()
        ppo_agent.learn(total_timesteps=rl_steps, reset_num_timesteps=False)

        save_path = os.path.join(cfg["training"]["save_dir"], f"joint_epoch_{epoch:03d}.pt")
        save_checkpoint({
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
        }, save_path)

    ppo_agent.save(os.path.join(cfg["training"]["save_dir"], "ppo_joint_final.zip"))
    logger.info("Joint Fine-Tuning complete.")
    metric_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint Fine-Tuning (Experiment E)")
    parser.add_argument("--config", type=str, default="configs/joint_finetune.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    train_joint_finetune(args.config, args.epochs)
