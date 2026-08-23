import os
import argparse
from typing import Optional
import torch
from stable_baselines3 import PPO

from src.utils.seed import seed_everything
from src.utils.logger import setup_logger
from src.data.dataset import InpaintingDataset
from src.models.generator import InpaintingGenerator
from src.models.discriminator import SNPatchGANDiscriminator
from src.rl.env_bandit import InpaintingBanditEnv
from src.training.trainer_utils import load_config, get_device, load_checkpoint
from src.training.train_rl_bandit import RLBanditLoggingCallback


def train_ablation_c(
    config_path: str = "configs/rl_agent_bandit.yaml",
    timesteps: int = 5000,
    save_dir: str = "checkpoints/ablation_c_no_mask_stats",
    log_dir: str = "logs/ablation_c_no_mask_stats",
) -> PPO:
    """Experiment C: GAN + PPO without mask geometric statistics in observation."""
    cfg = load_config(config_path)
    seed = cfg["rl"].get("seed", 42)
    seed_everything(seed)
    device = get_device(cfg["rl"].get("device", "cpu"))
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger = setup_logger("ablation_c", log_file=os.path.join(log_dir, "train.log"))
    logger.info("=== Experiment C: Ablation WITHOUT Mask Stats ===")

    dataset = InpaintingDataset(
        image_dir=cfg["data"].get("image_dir"),
        mask_dir=cfg["data"].get("mask_dir"),
        image_size=cfg["data"].get("image_size", 128),
        is_train=True,
        synthetic_size=cfg["data"].get("synthetic_size", 200),
    )

    generator = InpaintingGenerator(
        base_channels=cfg["model"].get("base_channels", 32),
        strategy_dim=cfg["model"].get("strategy_dim", 64),
        latent_dim=cfg["model"].get("latent_dim", 256),
    ).to(device)

    discriminator = SNPatchGANDiscriminator(
        base_channels=cfg["model"].get("base_channels", 32) * 2,
    ).to(device)

    # Disable mask geometric statistics in observation
    env = InpaintingBanditEnv(
        dataset=dataset,
        generator=generator,
        discriminator=discriminator,
        device=str(device),
        include_mask_stats=False,  # <--- Ablation C
        compute_lpips=False,
    )

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=128,
        batch_size=32,
        gamma=0.0,
        tensorboard_log=log_dir,
        seed=seed,
    )

    callback = RLBanditLoggingCallback(check_freq=128)
    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(os.path.join(save_dir, "ppo_ablation_c.zip"))
    logger.info(f"Experiment C complete. Distribution:\n{callback.get_distribution_summary()}")
    return model


def train_ablation_d(
    config_path: str = "configs/rl_agent_bandit.yaml",
    timesteps: int = 5000,
    save_dir: str = "checkpoints/ablation_d_no_perc_reward",
    log_dir: str = "logs/ablation_d_no_perc_reward",
) -> PPO:
    """Experiment D: GAN + PPO without perceptual reward (gamma=0.0 in reward)."""
    cfg = load_config(config_path)
    seed = cfg["rl"].get("seed", 42)
    seed_everything(seed)
    device = get_device(cfg["rl"].get("device", "cpu"))
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger = setup_logger("ablation_d", log_file=os.path.join(log_dir, "train.log"))
    logger.info("=== Experiment D: Ablation WITHOUT Perceptual Reward ===")

    dataset = InpaintingDataset(
        image_dir=cfg["data"].get("image_dir"),
        mask_dir=cfg["data"].get("mask_dir"),
        image_size=cfg["data"].get("image_size", 128),
        is_train=True,
        synthetic_size=cfg["data"].get("synthetic_size", 200),
    )

    generator = InpaintingGenerator(
        base_channels=cfg["model"].get("base_channels", 32),
        strategy_dim=cfg["model"].get("strategy_dim", 64),
        latent_dim=cfg["model"].get("latent_dim", 256),
    ).to(device)

    discriminator = SNPatchGANDiscriminator(
        base_channels=cfg["model"].get("base_channels", 32) * 2,
    ).to(device)

    # gamma=0.0 in reward weights (alpha, beta, gamma, delta, lambda)
    reward_weights = (1.0, 1.0, 0.0, 0.5, 0.3)  # <--- Ablation D: gamma=0.0
    env = InpaintingBanditEnv(
        dataset=dataset,
        generator=generator,
        discriminator=discriminator,
        device=str(device),
        reward_weights=reward_weights,
        include_mask_stats=True,
        compute_lpips=False,
    )

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=128,
        batch_size=32,
        gamma=0.0,
        tensorboard_log=log_dir,
        seed=seed,
    )

    callback = RLBanditLoggingCallback(check_freq=128)
    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(os.path.join(save_dir, "ppo_ablation_d.zip"))
    logger.info(f"Experiment D complete. Distribution:\n{callback.get_distribution_summary()}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL Ablation Experiments (C & D)")
    parser.add_argument("--mode", type=str, default="all", choices=["c", "d", "all"], help="Ablation to run")
    parser.add_argument("--timesteps", type=int, default=2000, help="Timesteps per ablation")
    args = parser.parse_args()

    if args.mode in ("c", "all"):
        train_ablation_c(timesteps=args.timesteps)
    if args.mode in ("d", "all"):
        train_ablation_d(timesteps=args.timesteps)
